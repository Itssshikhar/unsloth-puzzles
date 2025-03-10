# NF4 Dequantization with Triton

This directory contains a high-performance implementation of NF4 (Normalized Float 4-bit) dequantization using Triton kernels.

## Pros

1. **High Performance**: Uses Triton kernels for GPU acceleration, targeting >1.15x speedup over reference implementations.

2. **Adaptive Kernel Selection**: Automatically chooses between two kernel implementations based on matrix size:
   - `dequantize_nf4_kernel_simple`: Row-parallel kernel for smaller matrices
   - `dequantize_nf4_kernel_fast`: 1D grid kernel for large matrices (>10M elements)

3. **Low-Level Optimization**: Uses custom PTX assembly for efficient nibble extraction, avoiding slower bit manipulation operations.

4. **Memory Efficiency**: Reuses the NF4 lookup table tensor across calls to reduce memory allocations.

5. **Cache-Aware Design**: Uses cache hints (`.ca`, `.cs`) for optimized memory access patterns.

## Cons

1. **Scaling Issue**: Currently multiplies by `absmax` directly, but NF4 dequantization should use `absmax / 8.0` for correct scaling. This causes numerical discrepancies.

2. **Numerical Stability**: Limited handling of edge cases like NaN or Inf values in the output tensor.

3. **Error Diagnostics**: Limited error information when dequantization fails, making debugging challenging.

4. **Precision Handling**: May not correctly handle mixed precision operations, especially with bfloat16 tensors.

5. **Quantization State Extraction**: Assumes specific structure of the quantization state, which may not be robust to changes in underlying libraries.

## Areas for Improvement

1. **Correct Scaling**: Modify both kernels to use `values * (absmax / 8.0)` instead of `values * absmax`.

2. **Robust Error Handling**: Add more comprehensive error checking and diagnostics.

3. **Numerical Stability**: Add `torch.nan_to_num` to handle NaN/Inf values in the output.

4. **Dtype Consistency**: Ensure LUT tensor uses the same dtype as the computation for better precision.
