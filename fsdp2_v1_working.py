# !pip install bitsandbytes>=0.43.0 transformers>=4.38.0 peft>=0.7.0 accelerate>=0.25.0 trl>=0.7.4 datasets>=2.14.0

import sys
import os
import logging
import torch
import torch.distributed as dist
import importlib.util
import subprocess
from accelerate import notebook_launcher
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    CPUOffload,
    BackwardPrefetch,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
    OptimStateDictConfig,
    FullOptimStateDictConfig,
)
import transformers
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding
import math
import torch.nn.init as init
import torch.nn as nn
from functools import partial
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import enum

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Debug print helper function - define it early so it's available throughout
def debug_print(message):
    print(f"DEBUG: {message}")
    with open("fsdp_debug.log", "a") as f:
        f.write(f"{message}\n")

# Check if a package is installed
def is_package_installed(package_name):
    """Check if a package is installed"""
    return importlib.util.find_spec(package_name) is not None

# Install missing packages
def install_missing_packages():
    """Install required packages if missing"""
    required_packages = [
        "bitsandbytes>=0.43.0",
        "transformers>=4.38.0",
        "peft>=0.7.0",
        "accelerate>=0.25.0",
        "trl>=0.7.4",
        "datasets>=2.14.0",
    ]
    
    for package in required_packages:
        package_name = package.split(">=")[0]
        if not is_package_installed(package_name):
            logger.info(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {package}: {e}")
                if package_name == "bitsandbytes":
                    logger.info("Attempting to install bitsandbytes with specific CUDA version...")
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes-cuda118"])
                    except:
                        logger.error("Failed to install bitsandbytes-cuda118. Please install manually.")

# Set up environment variables
def setup_environment():
    """Set up environment variables for training"""
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
    os.environ["ACCELERATE_USE_FSDP"] = "1"
    os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "1"
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    logger.info("Environment variables set")

# Move the wrap policy function to module level for picklability
# Define as a separate top-level function rather than a nested function
def peft_lora_wrap_policy(module, recurse, unwrapped_params, **kwargs):
    """Custom wrap policy that avoids wrapping LoRA modules in FSDP"""
    # Explicitly avoid wrapping LoRA modules
    if isinstance(module, (nn.Linear, nn.Embedding)) and \
       any("lora" in name for name, _ in module.named_parameters(recurse=False)):
        return False

    # Force-wrap transformer layers
    if isinstance(module, LlamaDecoderLayer):
        return True

    # Default to size-based policy for other modules
    return size_based_auto_wrap_policy(
        module, recurse, unwrapped_params,
        min_num_params=100_000_000  # Only wrap large modules
    )

# Global formatting function needed by the trainer
def formatting_func(example):
    return [example["text"]]

# Main training function for distributed execution
def training_function():
    """Main training function to be executed by notebook_launcher"""
    # Get local rank for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    logger.info(f"Starting process with local_rank: {local_rank}, world_size: {world_size}")
    
    # Set device and initialize process group
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        logger.info("Initializing process group for distributed training...")
        dist.init_process_group(backend="nccl")
        logger.info(f"Process group initialized: rank={dist.get_rank()}, world_size={dist.get_world_size()}")

    # Log system information
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"Current CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    # Verify bitsandbytes installation
    try:
        import bitsandbytes as bnb
        logger.info(f"Successfully imported bitsandbytes version: {bnb.__version__}")
    except ImportError:
        logger.error("Failed to import bitsandbytes. Attempting reinstallation...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "bitsandbytes>=0.43.0"])
            import bitsandbytes as bnb
            logger.info(f"Reinstalled and imported bitsandbytes version: {bnb.__version__}")
        except:
            logger.error("Failed to reinstall bitsandbytes. Training may fail.")
    
    # Setup model and tokenizer
    logger.info("Setting up model and tokenizer...")
    dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_storage=dtype,  # For FSDP compatibility
    )
    
    logger.info("Loading model...")
    model_name = "unsloth/meta-Llama-3.1-8B-Instruct-bnb-4bit"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=dtype,
            device_map=None,  # Important for FSDP
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        try:
            logger.info("Attempting to load with alternate configuration...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=None,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            )
        except Exception as e2:
            logger.error(f"Failed with alternate config too: {e2}")
            return
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    
    # Define mixed precision and CPU offloading policies
    mixed_precision_policy = MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )
    cpu_offload = CPUOffload(offload_params=True)
    
    # Set up LoRA
    logger.info("Setting up LoRA configuration...")
    lora_config = LoraConfig(
        r=32,              # Reduced for T4 GPUs
        lora_alpha=64,     # Reduced for T4 compatibility
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    logger.info("Applying LoRA to model...")
    model = get_peft_model(model, lora_config)
    
    # Set only LoRA parameters to trainable
    logger.info("Setting trainable parameters...")
    trainable_param_count, total_param_count = 0, 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            total_param_count += param.numel()
            if ".lora_A." in name or ".lora_B." in name:
                param.requires_grad_(True)
                trainable_param_count += param.numel()
            else:
                param.requires_grad_(False)
    logger.info(f"Trainable parameters: {trainable_param_count:,} ({trainable_param_count/total_param_count:.2%} of total {total_param_count:,})")
    
    # Enable gradient checkpointing
    logger.info("Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # Patch LlamaRMSNorm and LlamaRotaryEmbedding if needed
    if not hasattr(LlamaRMSNorm, 'reset_parameters'):
        def reset_parameters(self):
            pass
        LlamaRMSNorm.reset_parameters = reset_parameters
        logger.info("Successfully added reset_parameters method to LlamaRMSNorm")
    if not hasattr(LlamaRotaryEmbedding, 'reset_parameters'):
        def reset_parameters(self):
            pass
        LlamaRotaryEmbedding.reset_parameters = reset_parameters
        logger.info("Successfully added reset_parameters method to LlamaRotaryEmbedding")
    
    # Patch Linear layers to safely handle quantized weights
    def patch_module_with_quantized_safety(module_class):
        if hasattr(module_class, 'reset_parameters'):
            original_reset_parameters = module_class.reset_parameters
            def safe_reset_parameters(self):
                skip_init = False
                if hasattr(self, 'weight') and self.weight is not None:
                    if self.weight.dtype in [torch.uint8, torch.int8, torch.quint8, torch.qint8]:
                        logger.info(f"Skipping initialization for quantized weights in {self}")
                        skip_init = True
                if not skip_init:
                    original_reset_parameters(self)
            module_class.reset_parameters = safe_reset_parameters
            return True
        else:
            def empty_reset_parameters(self):
                pass
            module_class.reset_parameters = empty_reset_parameters
            return False

    for module_class in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding]:
        had_reset = patch_module_with_quantized_safety(module_class)
        if had_reset:
            logger.info(f"Patched {module_class.__name__}.reset_parameters to handle quantized weights")
        else:
            logger.info(f"Added empty reset_parameters to {module_class.__name__}")
    
    # Convert frozen quantized parameters to buffers
    logger.info("Converting frozen quantized parameters to buffers...")
    for module in model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            if not param.requires_grad and param.dtype in [torch.uint8, torch.int8, torch.quint8, torch.qint8]:
                logger.info(f"Converting {name} in {module} from parameter to buffer")
                module._parameters.pop(name)
                module.register_buffer(name, param)
    
    # Mark quantized parameters to be ignored by FSDP
    for name, param in model.named_parameters():
        if param.dtype in [torch.uint8, torch.int8, torch.quint8, torch.qint8]:
            setattr(param, '_fsdp_ignore', True)
    
    logger.info("Casting model to torch.float16 for uniform dtypes...")
    model = model.half()
    
    max_seq_length = 512  # Reduced for T4 GPUs
    logger.info(f"Using max sequence length: {max_seq_length}")
    
    # Load and prepare dataset
    logger.info("Loading dataset...")
    try:
        url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
        dataset = load_dataset("json", data_files={"train": url}, split="train[:2%]")
        def format_text(example):
            text = example["text"]
            if isinstance(text, list):
                return {"text": " ".join([str(t) for t in text])}
            return {"text": str(text)}
        dataset = dataset.map(format_text, remove_columns=[col for col in dataset.column_names if col != "text"])
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing dataset",
        )
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # Configure training with FSDP2 settings using the global wrap policy
    debug_print("Setting up FSDP2 configuration...")
    
    # Using our top-level function for picklability
    debug_print("Using top-level peft_lora_wrap_policy function for picklability")
    
    fsdp_config = {
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "fsdp_sharding_strategy": ShardingStrategy.HYBRID_SHARD,
        "fsdp_offload_params": False,
        "fsdp_activation_checkpointing": True,
        "fsdp_use_orig_params": True,
        "fsdp_auto_wrap_policy": peft_lora_wrap_policy,  # Use the module-level function
        "fsdp_sync_module_states": True,
        "fsdp_cpu_ram_efficient_loading": True,
        "fsdp_state_dict_type": StateDictType.SHARDED_STATE_DICT,
        "state_dict_config": FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        "optim_state_dict_config": FullOptimStateDictConfig(offload_to_cpu=True),
    }
    
    debug_print("FSDP config created, checking if wrap policy is picklable...")
    import pickle
    try:
        pickle.dumps(fsdp_config)
        debug_print("FSDP config is picklable")
    except Exception as e:
        debug_print(f"FSDP config is not picklable: {e}")
        debug_print("Trying to identify unpicklable components...")
        for key, value in fsdp_config.items():
            try:
                pickle.dumps(value)
                debug_print(f"  - {key} is picklable")
            except Exception as e:
                debug_print(f"  - {key} is not picklable: {e}")
    
    training_args = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=1,
        max_steps=20,
        logging_steps=1,
        output_dir="outputs",
        seed=3407,
        max_seq_length=max_seq_length,
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        report_to="none",
        save_steps=10,
        save_total_limit=2,
        save_strategy="steps",
        fsdp="full_shard auto_wrap offload",
        fsdp_config=fsdp_config,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        disable_tqdm=False,
        logging_first_step=True,
        dataset_num_proc=1,
        dataloader_drop_last=False,
        dataloader_pin_memory=False,
        auto_find_batch_size=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        fp16_full_eval=False,
        half_precision_backend="cuda_amp",
    )

    debug_print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        ),
        formatting_func=formatting_func,
    )

    # Debug: Inspect model parameters and FSDP structure
    debug_print("===== CHECKING MODEL PARAMETERS BEFORE TRAINING =====")
    total_params = 0
    trainable_params = 0
    buffer_count = 0
    quantized_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if param.requires_grad:
            trainable_params += 1
        if hasattr(param, '_fsdp_ignore') and param._fsdp_ignore:
            debug_print(f"Parameter marked for FSDP ignore: {name}")
        if param.dtype in [torch.uint8, torch.int8, torch.quint8, torch.qint8]:
            quantized_params += 1
            debug_print(f"Quantized parameter found: {name} with dtype {param.dtype}")
    for name, buffer in model.named_buffers():
        buffer_count += 1
        if buffer.dtype in [torch.uint8, torch.int8, torch.quint8, torch.qint8]:
            debug_print(f"Quantized buffer found: {name} with dtype {buffer.dtype}")
    debug_print(f"Total parameters: {total_params}, Trainable: {trainable_params}, Buffers: {buffer_count}, Quantized params: {quantized_params}")
    
    # Completely overhaul the checkpoint saving approach
    original_save_checkpoint = trainer._save_checkpoint
    
    def custom_save_checkpoint(model, trial, metrics=None):
        """Custom checkpoint saving function that handles FSDP state dict issues"""
        debug_print("===== CUSTOM CHECKPOINT SAVING =====")
        
        # Only process on the main process
        if dist.get_rank() != 0:
            debug_print("Not rank 0, skipping checkpoint save")
            return
            
        debug_print("Saving checkpoint on rank 0")
        output_dir = os.path.join(trainer.args.output_dir, f"checkpoint-{trainer.state.global_step}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save the training arguments WITHOUT the wrap policy
        debug_print("Saving training arguments without wrap policy")
        args_dict = trainer.args.to_dict()
        
        # Convert non-serializable objects to strings
        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, enum.Enum):
                return str(obj.name)  # Convert enum to string
            elif hasattr(obj, '__name__'):
                return obj.__name__  # Function or class name
            elif hasattr(obj, '__class__'):
                return str(obj)  # General fallback for objects
            else:
                return obj
        
        # Process the args dict to make it serializable
        processed_args = make_json_serializable(args_dict)
        debug_print("Processed training args to make them JSON serializable")
        
        import json
        with open(os.path.join(output_dir, "training_args.json"), "w") as f:
            json.dump(processed_args, f, indent=2)
        debug_print("Saved training arguments to JSON")
            
        # 2. Save the model state itself - directly use PEFT for simplicity
        try:
            debug_print("Saving model using PEFT's save_pretrained")
            model.save_pretrained(output_dir)
            debug_print("Successfully saved model state using PEFT")
        except Exception as e:
            debug_print(f"Error saving model state: {e}")
            import traceback
            debug_print(traceback.format_exc())
            
        # 3. Save tokenizer and other components
        trainer.tokenizer.save_pretrained(output_dir)
        debug_print("Saved tokenizer")
        
        # 4. Save training state (simplified to avoid serialization issues)
        if hasattr(trainer, "state"):
            # Convert state to dict and make it serializable
            state_dict = {k: v for k, v in trainer.state.__dict__.items()}
            state_dict = make_json_serializable(state_dict)
            with open(os.path.join(output_dir, "trainer_state.json"), "w") as f:
                json.dump(state_dict, f, indent=2)
            debug_print("Saved trainer state")
        
        # 5. Save metrics if provided
        if metrics:
            metrics = make_json_serializable(metrics)
            with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            debug_print("Saved metrics")
        
        debug_print("Checkpoint saving complete")
        return output_dir

    # Replace the trainer's checkpoint save method with our custom one
    debug_print("Registering custom checkpoint saving method")
    trainer._save_checkpoint = custom_save_checkpoint

    debug_print("===== STARTING TRAINING =====")
    try:
        trainer.train()
        if dist.get_rank() == 0:
            debug_print("Training complete! Saving final model...")
            # Use PEFT's save_pretrained which handles LoRA weights correctly
            model.save_pretrained(os.path.join(trainer.args.output_dir, "final_model"))
            tokenizer.save_pretrained(os.path.join(trainer.args.output_dir, "final_model"))
            debug_print("Final model saved successfully.")
    except Exception as e:
        debug_print(f"Error during training: {e}")
        import traceback
        debug_print(traceback.format_exc())
        raise

# Main execution script for Kaggle notebook
def main():
    install_missing_packages()
    setup_environment()
    try:
        from accelerate import notebook_launcher
    except ImportError:
        logger.error("accelerate library is missing. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate>=0.25.0"])
        from accelerate import notebook_launcher
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    logger.info("Launching training across 2 GPUs...")
    notebook_launcher(training_function, num_processes=2)
    logger.info("Training complete!")

if __name__ == "__main__":
    main()