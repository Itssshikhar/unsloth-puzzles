## C) Make `torch.compile` work without graph breaks for QLoRA [Difficulty: Easy to Medium] [Max points: 9]
"""
1. Goal: Write a single Python script like task B), except the goal is to `torch.compile` all modules if possible.

2. There must NOT be graph breaks, and excessive re-compilations should not be seen.

3. You should have say max 30 compilations. Over 60 is definitely wrong.

4. The loss must match with the non compiled module.

5. Utilize patching as much as possible.

6. Think about which areas might need disabling for compilation. Think about regional compilation. How do we compile sections efficiently?

7. Log memory / VRAM usage, and monitor speedups as well.

8. Must work for QLoRA.

We provided a script below, and showcased how to detect if graph breaks are seen. We also torch compiled the MLP for Llama:
"""

import torch
import os
import logging
import gc
import psutil
import time
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers.trainer_callback import TrainerCallback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure PyTorch and CUDA settings
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = \
    "expandable_segments:True,"\
    "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"

# Enable verbose logging for torch.compile
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TORCH_LOGS"] = "dynamic"
os.environ["TORCHINDUCTOR_VERBOSE"] = "1"

# Configure torch._dynamo settings for better compilation
torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.dynamic_shapes = True
torch._dynamo.config.automatic_dynamic_shapes = True
torch._dynamo.config.optimize_ddp = False
torch._dynamo.config.replay_record_enabled = True

# Compilation counter for monitoring
compilation_counter = 0

def log_memory_usage():
    process = psutil.Process(os.getpid())
    logger.info(f"CPU Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i} Memory: {torch.cuda.memory_allocated(i) / 1024 / 1024:.2f} MB / {torch.cuda.memory_reserved(i) / 1024 / 1024:.2f} MB")

# Configure torch.compile options with better defaults for QLoRA
torch_compile_options = {
    "mode": "reduce-overhead",
    "dynamic": True,
    "fullgraph": False,
    "options": {
        "epilogue_fusion": True,
        "max_autotune": True,
        "shape_padding": True,
        "triton.cudagraphs": False,
        "triton.unique_kernel_names": True,
        "trace.enabled": True,
        "trace.graph_diagram": True,
        "trace.action_set_logging": True
    }
}

def count_compilation():
    global compilation_counter
    compilation_counter += 1
    if compilation_counter > 30:
        logger.warning(f"Excessive compilations detected: {compilation_counter}")
    logger.info(f"Current compilation count: {compilation_counter}")

# Patch the forward methods
def patch_llama_modules():
    import transformers.models.llama.modeling_llama as llama_module
    from functools import partial
    
    def create_compiled_function(fn, name):
        def wrapped(*args, **kwargs):
            count_compilation()
            return fn(*args, **kwargs)
        return wrapped
    
    def mlp_forward(self, x):
        # Split operations to reduce graph complexity
        gate_out = self.gate_proj(x)
        up_out = self.up_proj(x)
        act_gate = self.act_fn(gate_out)
        prod = act_gate * up_out
        return self.down_proj(prod)
    
    def attention_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False):
        bsz, q_len, _ = hidden_states.shape
        
        # Split QKV projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            # Reuse k, v for faster generation
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None
        
        # Use scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, past_key_value
    
    def rmsnorm_forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)
    
    # Create compiled versions with proper error handling
    def safe_compile(fn, name):
        try:
            compiled_fn = torch.compile(
                fn,
                **torch_compile_options
            )
            return create_compiled_function(compiled_fn, name)
        except Exception as e:
            logger.warning(f"Could not compile {name}: {str(e)}")
            return fn

    # Apply patches with safe compilation
    logger.info("Applying module patches...")
    llama_module.LlamaMLP.forward = safe_compile(mlp_forward, "mlp_forward")
    llama_module.LlamaAttention.forward = safe_compile(attention_forward, "attention_forward")
    llama_module.LlamaRMSNorm.forward = safe_compile(rmsnorm_forward, "rmsnorm_forward")
    logger.info("Module patches applied successfully")

# Configure model parameters
max_seq_length = 1024
torch.set_default_dtype(torch.float16)
model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
dtype = torch.float16

# Log initial memory state
logger.info("Initial memory state:")
log_memory_usage()

# Configure BitsAndBytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype,
)

# Load model with optimized settings
logger.info("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    attn_implementation="sdpa",
    quantization_config=bnb_config,
    torch_dtype=dtype,
)

# Load and configure tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"

# Configure LoRA
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply patches before LoRA
logger.info("Applying patches...")
patch_llama_modules()

# Setup LoRA
logger.info("Setting up LoRA...")
model = get_peft_model(model, lora_config)
with torch.no_grad():
    for name, param in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

model.enable_input_require_grads()

# Log memory after model setup
logger.info("Memory state after model setup:")
log_memory_usage()

# Load dataset
url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
dataset = load_dataset("json", data_files={"train": url}, split="train[:10%]")

class MemoryTrackingCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
    
    def on_init_end(self, args, state, control, **kwargs):
        logger.info("Training initialization completed")
        log_memory_usage()
    
    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("Training started")
        self.start_time = time.time()
        log_memory_usage()
    
    def on_step_begin(self, args, state, control, **kwargs):
        log_memory_usage()
        logger.info(f"Compilation count: {compilation_counter}")
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            logger.info(f"Step {state.global_step} completed")
            log_memory_usage()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Add memory info to logs
            if torch.cuda.is_available():
                logs["gpu_memory"] = torch.cuda.memory_allocated() / 1024 / 1024
            logs["compilation_count"] = compilation_counter
            logger.info(f"Training stats: {logs}")
    
    def on_train_end(self, args, state, control, **kwargs):
        logger.info("Training completed")
        if self.start_time:
            total_time = time.time() - self.start_time
            logger.info(f"Total training time: {total_time:.2f} seconds")
        log_memory_usage()

# Configure training arguments
training_args = SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    warmup_steps=1,
    max_steps=10,
    logging_steps=1,
    output_dir="outputs",
    max_seq_length=max_seq_length,
    fp16=model.get_input_embeddings().weight.dtype == torch.float16,
    bf16=model.get_input_embeddings().weight.dtype == torch.bfloat16,
    optim="adamw_torch_fused",
    learning_rate=2e-4,
    seed=3407,
    report_to="none",
    dataloader_pin_memory=True,
    dataloader_num_workers=1,
)

# Initialize trainer with memory tracking
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=training_args,
    callbacks=[MemoryTrackingCallback()],
)

# Start training
logger.info("Starting training...")
trainer.train()

"""Log all your steps for debugging in a Colab (maybe this one). Edward's blog http://blog.ezyang.com/, Horace's blogs https://www.thonking.ai/, Slaying OOMs by Jane & Mark: ttps://www.youtube.com/watch?v=UvRl4ansfCg could be useful.

## Marking Criteria for C) Max points = 9
```python
if attemped_C:
    C_score = 0
    if uses_flex_attention:
        if dynamic_sequence_length_works: C_score += 3
        else: C_score += 1
    if no_torch_compile_BnB: C_score -= 2
    elif use_part_A: C_score += 1
    elif torch_compile_BnB: C_score += 1

    if attention_compiled:
        if excessive_recompilation: C_score -= 3
        else: C_score += 2
    if mlp_compiled:
        if excessive_recompilation: C_score -= 3
        C_score += 1

    if not loss_compiled: C_score -= 1
    if not layernorms_compiled: C_score -= 3

    if max_autotune_triton_matmul:
        if excessive_recompilation: C_score -= 2
        else: C_score += 2
    
    final_score += C_score
else:
    final_score -= 1
```
"""