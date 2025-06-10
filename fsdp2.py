# Kaggle Notebook Implementation for FSDP2 + QLoRA Training
# Run this in a cell to install dependencies
# !pip install -q transformers>=4.38.0 peft>=0.7.0 accelerate>=0.25.0 trl>=0.7.4 datasets>=2.14.0 bitsandbytes>=0.43.0

# HELPFUL functions to undo Unsloth patches:
import sys

def remove_patched_module(package_name):
    """Remove any patched modules to ensure clean imports"""
    modules_to_delete = [
        name for name in sys.modules
        if name == package_name or name.startswith(package_name + ".")
    ]
    for name in modules_to_delete: del sys.modules[name]

# Ensure clean imports by removing any patched modules
remove_patched_module("trl")
remove_patched_module("transformers")
remove_patched_module("peft")
remove_patched_module("bitsandbytes")

# Import necessary libraries
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
)
import transformers
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding
import math
import torch.nn.init as init
import torch.nn as nn
from functools import partial
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

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

# Main training function for distributed execution
def training_function():
    """Main training function to be executed by notebook_launcher"""
    # Get local rank for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    logger.info(f"Starting process with local_rank: {local_rank}, world_size: {world_size}")
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    # IMPORTANT: Initialize the process group for distributed training
    # This is required before using FSDP
    if not dist.is_initialized():
        logger.info("Initializing process group for distributed training...")
        # If using notebook_launcher, it should set the required env vars
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
    
    # For T4 GPUs, we need to be careful with memory usage
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
    
    # Define mixed precision policy - must match quantization storage type
    mixed_precision_policy = MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )
    
    # CPU offloading for memory efficiency on limited GPUs
    cpu_offload = CPUOffload(offload_params=True)
    
    # Set up LoRA
    logger.info("Setting up LoRA configuration...")
    lora_config = LoraConfig(
        r=32,  # Reduced from 64 to save memory on T4 GPUs
        lora_alpha=64,  # Reduced from 128 for better T4 compatibility
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    logger.info("Applying LoRA to model...")
    model = get_peft_model(model, lora_config)
    
    # Set only LoRA parameters to trainable
    logger.info("Setting trainable parameters...")
    trainable_param_count = 0
    total_param_count = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            total_param_count += param.numel()
            if ".lora_A." in name or ".lora_B." in name:
                param.requires_grad_(True)
                trainable_param_count += param.numel()
            else:
                param.requires_grad_(False)
    logger.info(f"Trainable parameters: {trainable_param_count:,} ({trainable_param_count/total_param_count:.2%} of total {total_param_count:,})")
    
    # Enable gradient checkpointing for memory efficiency
    logger.info("Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # Patch LlamaRMSNorm to add reset_parameters method
    logger.info("Patching LlamaRMSNorm to add reset_parameters method...")
    
    # Add reset_parameters method to LlamaRMSNorm if it doesn't exist already
    if not hasattr(LlamaRMSNorm, 'reset_parameters'):
        def reset_parameters(self):
            # RMSNorm doesn't need parameter initialization beyond what's done in __init__
            pass
        
        # Add the method to the class
        LlamaRMSNorm.reset_parameters = reset_parameters
        logger.info("Successfully added reset_parameters method to LlamaRMSNorm")
    
    # Patch LlamaRotaryEmbedding to add reset_parameters method
    logger.info("Patching LlamaRotaryEmbedding to add reset_parameters method...")
    
    # Add reset_parameters method to LlamaRotaryEmbedding if it doesn't exist already
    if not hasattr(LlamaRotaryEmbedding, 'reset_parameters'):
        def reset_parameters(self):
            # RotaryEmbedding doesn't need parameter initialization beyond what's done in __init__
            pass
        
        # Add the method to the class
        LlamaRotaryEmbedding.reset_parameters = reset_parameters
        logger.info("Successfully added reset_parameters method to LlamaRotaryEmbedding")
    
    # Patch Linear layers to safely handle quantized weights
    logger.info("Patching modules to safely handle quantized weights...")
    
    # Create a patch for any module with weight tensors
    def patch_module_with_quantized_safety(module_class):
        if hasattr(module_class, 'reset_parameters'):
            # Store the original reset_parameters method
            original_reset_parameters = module_class.reset_parameters
            
            def safe_reset_parameters(self):
                """A safe reset_parameters that checks for quantized weights first"""
                skip_init = False
                
                # Check if weights are quantized (byte, int8, etc.)
                if hasattr(self, 'weight') and self.weight is not None:
                    if self.weight.dtype in [torch.uint8, torch.int8, torch.quint8, torch.qint8]:
                        logger.info(f"Skipping initialization for quantized weights in {self}")
                        skip_init = True
                
                if not skip_init:
                    # Call the original initialization for non-quantized weights
                    original_reset_parameters(self)
            
            # Replace the method
            module_class.reset_parameters = safe_reset_parameters
            return True
        else:
            # For modules without reset_parameters, add an empty one
            def empty_reset_parameters(self):
                pass
            
            module_class.reset_parameters = empty_reset_parameters
            return False
    
    # Patch common module types
    for module_class in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding]:
        had_reset = patch_module_with_quantized_safety(module_class)
        if had_reset:
            logger.info(f"Patched {module_class.__name__}.reset_parameters to handle quantized weights")
        else:
            logger.info(f"Added empty reset_parameters to {module_class.__name__}")
    
    # Add special handling for embedding layer
    logger.info("Ensuring embedding layer integrity...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            logger.info(f"Preserving 2D structure for embedding layer: {name}")
            # Reshape if needed and ensure proper parameter registration
            if module.weight.dim() == 1:
                logger.info(f"Reshaping 1D embedding weight to 2D for {name}")
                module.weight = nn.Parameter(module.weight.view(module.num_embeddings, -1))
            # Ensure embedding weights maintain 2D shape
            if hasattr(module.weight, '_fsdp_flat_param'):
                logger.info(f"Reconstructing 2D structure from FSDP flat param for {name}")
                module.weight = nn.Parameter(module.weight.data.view(module.num_embeddings, -1))
    
    # Modify parameter conversion to buffers to exclude embeddings
    logger.info("Converting frozen quantized parameters to buffers...")
    for module in model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            if not param.requires_grad and \
               param.dtype in [torch.uint8, torch.int8, torch.quint8, torch.qint8] and \
               not isinstance(module, (nn.Embedding, LlamaRMSNorm)) and \
               "embed_tokens" not in name:  # Additional safeguard for embeddings
                logger.info(f"Converting {name} in {module} from parameter to buffer")
                module._parameters.pop(name)
                module.register_buffer(name, param)
    
    # Special handling for quantized params
    for name, param in model.named_parameters():
        if param.dtype in [torch.uint8, torch.int8, torch.quint8, torch.qint8]:
            setattr(param, '_fsdp_ignore', True)
    
    # Cast the entire model to torch.float16 to ensure uniform dtypes for FSDP flattening
    logger.info("Casting model to torch.float16 for uniform dtypes...")
    model = model.half()
    
    # Reduce dataset size and sequence length further for T4 GPUs
    max_seq_length = 512  # Reduced for faster training
    logger.info(f"Using max sequence length: {max_seq_length}")
    
    # Load and prepare dataset
    logger.info("Loading dataset...")
    try:
        url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
        dataset = load_dataset("json", data_files={"train": url}, split="train[:2%]")
        
        # Ensure text field is properly formatted
        def format_text(example):
            text = example["text"]
            # Handle nested lists and ensure string format
            if isinstance(text, list):
                return {"text": " ".join([str(t) for t in text])}
            return {"text": str(text)}
        
        dataset = dataset.map(format_text, remove_columns=[col for col in dataset.column_names if col != "text"])
        
        # Tokenize dataset upfront with proper padding/truncation
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
    
    # After loading the model
    logger.info("Materializing model parameters on device...")
    device = torch.device(f"cuda:{local_rank}")
    
    # Materialize meta tensors using to_empty
    with torch.device(device):
        model = model.to_empty(device=device)
    
    # Initialize weights for modules still using meta device
    def init_meta_weights(module):
        if any(p.is_meta for p in module.parameters()):
            logger.info(f"Initializing meta weights for {module.__class__.__name__}")
            module.to_empty(device=device)
            try:
                module.reset_parameters()
            except AttributeError:
                # Handle modules without reset_parameters
                for param in module.parameters():
                    if param.is_meta:
                        init.kaiming_normal_(param)

    model.apply(init_meta_weights)

    # Modified FSDP configuration
    fsdp_config = {
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "fsdp_sharding_strategy": 0,  # Changed to FULL_SHARD for better compatibility
        "fsdp_offload_params": True,
        "fsdp_activation_checkpointing": True,
        "fsdp_use_orig_params": True,
        "fsdp_sync_module_states": True,
        "fsdp_cpu_ram_efficient_loading": True,
        "fsdp_state_dict_type": "FULL_STATE_DICT",  # Added for checkpoint compatibility
        "mixed_precision": mixed_precision_policy,
    }

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
        # FSDP2 configuration
        fsdp="full_shard auto_wrap offload",
        fsdp_config=fsdp_config,
        # Critical data processing settings
        remove_unused_columns=False,  # Prevent column removal
        dataloader_num_workers=0,
        disable_tqdm=False,
        logging_first_step=True,
        dataset_num_proc=1,
        dataloader_drop_last=False,
        dataloader_pin_memory=False,
        auto_find_batch_size=True,
    )

    # Initialize trainer with proper data handling
    logger.info("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        ),
    )

    # Modified FSDP initialization
    logger.info("Preparing model for FSDP...")
    
    # Update custom wrapper policy to match FSDP's expected signature
    def custom_auto_wrap_policy(
        module: nn.Module,
        recurse: bool,
        nonwrapped_numel: int,  # Add required FSDP parameter
        min_num_params: int = 100,
        **kwargs,  # Capture additional arguments to future-proof
    ):
        """Custom wrapper policy that preserves embedding layers"""
        # Prevent wrapping of embedding layers
        if isinstance(module, nn.Embedding):
            return False
        # Use size-based policy for other modules
        return size_based_auto_wrap_policy(
            module=module,
            recurse=recurse,
            nonwrapped_numel=nonwrapped_numel,
            min_num_params=min_num_params,
        )
    
    # Update param_init_fn to handle all module types
    def fsdp_param_init_fn(module):
        if any(p.is_meta for p in module.parameters()):
            logger.info(f"FSDP initializing meta weights for {module.__class__.__name__}")
            module.to_empty(device=device)
            try:
                module.reset_parameters()
            except AttributeError:
                for param in module.parameters():
                    if param.is_meta:
                        init.kaiming_normal_(param)
    
    model = FSDP(
        model,
        auto_wrap_policy=partial(
            custom_auto_wrap_policy,
            min_num_params=100,
        ),
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        param_init_fn=fsdp_param_init_fn,
    )

    # Add this after converting parameters to buffers
    logger.info("Final parameter requirements check...")
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    # Run training
    logger.info("Starting training...")
    try:
        trainer.train()
        # Safe checkpoint saving
        with model.summon_full_params():
            model = model.float()  # Convert to stable dtype
            trainer.save_model()
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

# Main execution script for Kaggle notebook
def main():
    # Install required packages
    install_missing_packages()
    
    # Set up environment variables for T4 GPUs
    setup_environment()
    
    # Make sure accelerate is imported
    try:
        from accelerate import notebook_launcher
    except ImportError:
        logger.error("accelerate library is missing. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate>=0.25.0"])
        from accelerate import notebook_launcher
    
    # Set required environment variables for PyTorch Distributed
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    # Launch training with notebook_launcher
    logger.info("Launching training across 2 GPUs...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    notebook_launcher(training_function, num_processes=2)
    logger.info("Training complete!")

# Call main() to execute
main()