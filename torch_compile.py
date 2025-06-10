import transformers.models.llama.modeling_llama
import logging
import os
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import functools

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log PyTorch version and compilation settings
logger.info(f"PyTorch Version: {torch.__version__}")
logger.info(f"Using CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"CUDA Device: {torch.cuda.get_device_name()}")

# Configure torch.compile settings
torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.dynamic_shapes = True

# Patch the forward methods directly without using torch.compile
def patch_llama_modules():
    def patched_mlp_forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    
    def patched_attention_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False):
        bsz, q_len, _ = hidden_states.shape
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=True
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
    
    def patched_rmsnorm_forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

    # Apply the patches
    transformers.models.llama.modeling_llama.LlamaMLP.forward = patched_mlp_forward
    transformers.models.llama.modeling_llama.LlamaAttention.forward = patched_attention_forward
    transformers.models.llama.modeling_llama.LlamaRMSNorm.forward = patched_rmsnorm_forward

# Apply patches before model loading
patch_llama_modules()

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = \
    "expandable_segments:True,"\
    "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"

max_seq_length = 1024
torch.set_default_dtype(torch.float16)
model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
dtype = torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype,
)

# Load model with flash attention
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    attn_implementation="flash_attention_2",
    quantization_config=bnb_config,
    torch_dtype=dtype,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"

# Disable features that might cause graph breaks
model.config.use_cache = False

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Get LoRA and setup model
model = get_peft_model(model, lora_config)
with torch.no_grad():
    for name, param in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

model.enable_input_require_grads()

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

# Get dataset
url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
dataset = load_dataset("json", data_files={"train": url}, split="train[:10%]")

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=training_args,
)

# Add memory tracking
import psutil
import GPUtil
def log_memory_usage():
    gpu = GPUtil.getGPUs()[0]
    process = psutil.Process()
    print(f"GPU Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
    print(f"CPU Memory: {process.memory_info().rss / 1024 / 1024:.2f}MB")

class MemoryCallback(transformers.TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            log_memory_usage()

trainer.add_callback(MemoryCallback())

# Start training
trainer.train()
