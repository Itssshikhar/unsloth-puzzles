# Unsloth Puzzles

A collection of optimized solutions for large language model (LLM) training and inference challenges.

## Components

- **[nf4_triton](nf4_triton/)**: High-performance NF4 dequantization using Triton kernels, achieving >1.15x speedup over standard implementations.

- **[graph_breaks](graph_breaks/)**: Solutions for handling PyTorch compilation breaks, enabling more efficient model execution through better graph optimization.

- **[fsdp2_qlora](fsdp2_qlora/)**: Implementation of Fully Sharded Data Parallel v2 with Quantized LoRA for memory-efficient distributed training of large models.

- **[efficient_backprop](efficient_backprop/)**: Memory-efficient backpropagation techniques that reduce VRAM usage by >60% for models with large vocabulary sizes.

## Use Cases

- **Memory Optimization**: Train larger models on limited hardware
- **Training Speedups**: Accelerate fine-tuning and pre-training workflows
- **Inference Optimization**: Improve throughput for deployed models
- **Distributed Training**: Scale efficiently across multiple GPUs

## Getting Started

Each directory contains a standalone notebook demonstrating the implementation and usage of the specific optimization technique. See the individual README files for detailed explanations and benchmarks.
