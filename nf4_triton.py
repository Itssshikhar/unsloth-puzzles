import torch
import triton
import triton.language as tl

# NF4 quantization constants from the spec
NF4_LUT = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
], dtype=torch.float32, device="cuda")

@triton.jit
def _nf4_dequant_kernel(
    output_ptr,
    packed_weights_ptr,
    absmax_ptr,
    num_blocks,
    block_size: tl.constexpr,
    output_dtype: tl.constexpr,
):
    pid = tl.program_id(0)
    n_blocks_per_program = 4  # Process 4 blocks per program for better utilization
    
    # Calculate block indices this program handles
    block_start = pid * n_blocks_per_program
    block_end = tl.minimum((pid + 1) * n_blocks_per_program, num_blocks)
    
    # Load NF4 LUT into fast memory
    nf4_lut = tl.load(NF4_LUT.data_ptr() + tl.arange(0, 16), cache_modifier=".cg")
    
    for block_idx in range(block_start, block_end):
        # Calculate offset for this block
        offset = block_idx * block_size // 2  # 4 bits per value, 2 per byte
        absmax = tl.load(absmax_ptr + block_idx)
        
        # Load 32 bytes (64 4-bit values) at once
        packed = tl.load(packed_weights_ptr + offset + tl.arange(0, 32), mask=offset + tl.arange(0, 32) < num_blocks * block_size // 2)
        
        # Unpack 4-bit values using bitwise operations
        shifts = tl.arange(0, 8) * 4
        shifts = tl.expand_dims(shifts, 1)
        vals = (packed >> shifts) & 0x0F
        vals = tl.reshape(vals, (64,))
        
        # Convert to FP32 using LUT and scale
        dequant_vals = tl.load(nf4_lut + vals, mask=vals >= 0) * absmax
        
        # Convert to target dtype and store
        output_offset = block_idx * block_size
        tl.store(output_ptr + output_offset + tl.arange(0, 64), 
                dequant_vals.to(output_dtype),
                mask=output_offset + tl.arange(0, 64) < num_blocks * block_size)

def _your_dequantize_nf4(weight, quant_state):
    absmax, shape, dtype, blocksize = quant_state
    num_blocks = absmax.numel()
    
    # Allocate output tensor
    output = torch.empty(shape, dtype=dtype, device="cuda")
    
    # Configure kernel launch
    grid = lambda meta: (triton.cdiv(num_blocks, meta['n_blocks_per_program']),)
    
    _nf4_dequant_kernel[grid](
        output,
        weight,
        absmax,
        num_blocks,
        block_size=blocksize,
        output_dtype=tl.float16 if dtype == torch.float16 else tl.bfloat16,
        n_blocks_per_program=4,
        num_warps=4,
    )
    return output

def your_dequantize_nf4(weight):
    return _your_dequantize_nf4(weight.weight.data, weight.weight.quant_state)
