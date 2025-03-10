# FSDP2 with QLoRA for LLM Fine-tuning

This directory contains a working implementation of Fully Sharded Data Parallel v2 (FSDP2) combined with Quantized Low-Rank Adaptation (QLoRA) for efficient fine-tuning of large language models.

## Pros and Cons

### Advantages

1. **Memory Efficiency**: Combines sharding, quantization, and LoRA to fit large models on limited hardware
2. **Scalability**: Scales to multiple GPUs with efficient communication patterns
3. **Training Stability**: Handles quantized weights properly to avoid numerical issues
4. **Checkpoint Compatibility**: Custom saving ensures checkpoints can be loaded correctly
5. **Resource Utilization**: Maximizes GPU utilization while minimizing memory footprint

### Challenges

1. **Implementation Complexity**: Requires careful coordination between FSDP, quantization, and LoRA
2. **Serialization Issues**: Custom handling needed for non-serializable components in checkpoints
3. **Debugging Difficulty**: Distributed training errors can be harder to diagnose
4. **Version Dependencies**: Sensitive to specific versions of PyTorch, PEFT, and other libraries
5. **Performance Overhead**: Some communication overhead compared to single-GPU training

## Implementation Notes

1. **Custom Wrap Policy**: The implementation uses a module-level function for FSDP wrapping to ensure picklability
2. **Patched Reset Parameters**: Adds `reset_parameters` methods to model components that lack them
3. **Quantized Weight Handling**: Converts quantized parameters to buffers to avoid FSDP errors
4. **Gradient Checkpointing**: Disables `use_cache` for compatibility with gradient checkpointing
5. **Custom Checkpoint Saving**: Completely overhauls the checkpoint saving approach for FSDP compatibility
