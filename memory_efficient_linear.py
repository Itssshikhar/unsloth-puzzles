import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# MemoryEfficientLinear:
# A custom autograd Function that avoids storing large intermediate logits.
class MemoryEfficientLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, weight, bias, labels, chunk_size, transformation_fn):
        ctx.save_for_backward(X, weight, bias, labels)
        ctx.chunk_size = chunk_size
        ctx.transformation_fn = transformation_fn

        # Keep original dtype without upcasting
        X_flat = X.view(-1, X.size(-1)) if X.dim() > 2 else X
        labels_flat = labels.view(-1) if labels.dim() > 1 else labels
        V = weight.size(0)

        # Add label bounds check
        if (labels_flat >= V).any():
            raise RuntimeError(f"Label values exceed vocabulary size {V}")

        total_loss = 0.0
        for start in range(0, V, chunk_size):
            end = min(start + chunk_size, V)
            print(f"[DEBUG Forward] Processing chunk {start}-{end}")
            with torch.no_grad():
                # Maintain input dtype throughout
                logits = F.linear(
                    X_flat, 
                    weight[start:end].to(X.dtype), 
                    bias[start:end].to(X.dtype) if bias is not None else None
                )
            
            chunk_loss = transformation_fn(logits, labels_flat, start, end)
            # Log the computed per-chunk loss (if scalar)
            loss_val = chunk_loss.item() if chunk_loss.numel() == 1 else chunk_loss
            print(f"[DEBUG Forward] Chunk {start}-{end}, Loss: {loss_val}")
            # Proper normalization using actual valid labels count
            total_loss += chunk_loss * (chunk_loss.numel() / labels_flat.numel())

            del logits
            if start % (2*chunk_size) == 0:
                torch.cuda.empty_cache()

        return total_loss

    @staticmethod
    def backward(ctx, grad_output):
        X, weight, bias, labels = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        transformation_fn = ctx.transformation_fn

        X_flat = X.view(-1, X.size(-1)) if X.dim() > 2 else X
        labels_flat = labels.view(-1)
        V = weight.size(0)

        grad_X = torch.zeros_like(X_flat)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias) if bias is not None else None

        for start in range(0, V, chunk_size):
            end = min(start + chunk_size, V)
            print(f"[DEBUG Backward] Processing chunk {start}-{end}")
            weight_chunk = weight[start:end].detach().requires_grad_(True)
            bias_chunk = bias[start:end].detach().requires_grad_(True) if bias is not None else None
            
            with torch.enable_grad():
                logits = F.linear(X_flat, weight_chunk.to(X.dtype), bias_chunk.to(X.dtype))
                chunk_loss = transformation_fn(logits, labels_flat, start, end)
                # Normalize loss in backward similar to forward:
                chunk_loss = chunk_loss.mean() * (chunk_loss.numel() / labels_flat.numel())
                chunk_loss.backward()

            grad_X += X_flat.grad.to(X.dtype) if X_flat.grad is not None else 0
            grad_weight[start:end] += weight_chunk.grad.to(weight.dtype)
            if grad_bias is not None:
                grad_bias[start:end] += bias_chunk.grad.to(bias.dtype)

            del weight_chunk, bias_chunk, logits, chunk_loss
            torch.cuda.empty_cache()

        if X.dim() > 2:
            grad_X = grad_X.view(*X.shape)

        return (
            grad_X * grad_output if X.requires_grad else None,
            grad_weight * grad_output,
            grad_bias * grad_output if grad_bias is not None else None,
            None, None, None
        )

# ----------------------------------------------------------------------
# A default transformation function that materializes full logits.
# (For smaller inputs only.)
def transformation_function(X, linear, labels):
    def ce_transform(logits, labels, start, end):
        mask = (labels >= start) & (labels < end)
        valid_count = mask.sum().item()
        print(f"[DEBUG ce_transform] Chunk {start}-{end}: valid_count = {valid_count}/{labels.numel()}")
        if valid_count == 0:
            print(f"[DEBUG ce_transform] Chunk {start}-{end}: No valid labels. Returning 0 loss.")
            return torch.tensor(0.0, device=logits.device)
        
        chunk_labels = labels[mask] - start
        print(f"[DEBUG ce_transform] Chunk {start}-{end}:")
        print(f"  Original labels range: [{labels.min().item()}, {labels.max().item()}]")
        print(f"  Masked labels range: [{labels[mask].min().item()}, {labels[mask].max().item()}]")
        print(f"  Adjusted labels range: [{chunk_labels.min().item()}, {chunk_labels.max().item()}]")
        print(f"  Chunk size: {end - start}, Selected logits shape: {logits[mask].shape}")
        
        # Validate that all offset labels are within [0, end - start)
        if (chunk_labels < 0).any() or (chunk_labels >= (end - start)).any():
            print(f"[ERROR ce_transform] Chunk {start}-{end}: Found invalid labels: {chunk_labels}")
            raise ValueError(f"Invalid labels in chunk {start}-{end}")
        
        return F.cross_entropy(
            logits[mask],
            chunk_labels,
            reduction='mean'
        )

    return MemoryEfficientLinear.apply(
        X,
        linear.weight,
        linear.bias,
        labels,
        4096,
        ce_transform
    )

def validate_efficiency():
    import gc
    from torch.nn import CrossEntropyLoss
    
    # Test configuration
    bsz, qlen, hd, vocab = 4, 4096, 4096, 128000
    X = torch.randn(bsz, qlen, hd, dtype=torch.bfloat16).cuda().requires_grad_(True)
    linear = nn.Linear(hd, vocab).cuda().bfloat16()
    labels = torch.randint(0, vocab-1, (bsz, qlen)).cuda()  # Exclusive upper bound

    # Before tensor creation
    print(f"CUDA bfloat16 support: {torch.cuda.is_bf16_supported()}")
    print(f"Available CUDA memory: {torch.cuda.mem_get_info()[0]/1e9:.2f}GB")
    
    # Test with smaller tensors first
    try:
        test_tensor = torch.randn(2, 2, 2, dtype=torch.bfloat16, device='cuda')
        print("Small tensor creation: ✓")
        del test_tensor
    except Exception as e:
        print(f"Small tensor creation failed: {str(e)}")

    # Check label bounds
    print(f"Label bounds: [{labels.min()}, {labels.max()}] (vocab={vocab})")
    
    # Verify tensor dimensions
    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"Weight shape: {linear.weight.shape}")

    # Add label validation
    print(f"Label validation: [{labels.min()}, {labels.max()}] (vocab={vocab})")
    assert labels.max() < vocab, "Invalid label generation"
    
    # Baseline measurement - FIXED DIMENSIONS
    torch.cuda.reset_peak_memory_stats()
    logits_normal = linear(X).float().view(-1, vocab)  # Reshape to [bsz*qlen, vocab]
    loss_normal = F.cross_entropy(logits_normal, labels.view(-1))  # Targets [bsz*qlen]
    loss_normal.backward()
    mem_normal = torch.cuda.max_memory_allocated()
    grad_normal = linear.weight.grad.clone()
    linear.zero_grad()
    
    # Our implementation - FIXED DIMENSIONS
    torch.cuda.reset_peak_memory_stats()
    loss_efficient = transformation_function(
        X.view(bsz, qlen, hd),  # Ensure original dimensions
        linear,
        labels.view(bsz, qlen)  # Ensure original dimensions
    )
    loss_efficient.backward()
    mem_efficient = torch.cuda.max_memory_allocated()
    grad_efficient = linear.weight.grad.clone()
    linear.zero_grad()

    # Criterion 1: VRAM reduction
    vram_reduction = (mem_normal - mem_efficient) / mem_normal
    print(f"VRAM reduction: {vram_reduction:.1%} (Target >50%)")
    
    # Criterion 2: No float32 upcast
    print(f"Using dtype: {loss_efficient.dtype} (Should be bfloat16)")
    
    # Criterion 3: CE Loss works
    loss_diff = torch.abs(loss_normal - loss_efficient).item()
    print(f"Loss difference: {loss_diff:.2e} (Should be <1e-3)")
    
    # Test MSE compatibility with scalar output
    try:
        def mse_transform(logits, labels, start, end):
            targets = torch.randn_like(logits)
            return F.mse_loss(logits, targets, reduction='mean')
            
        loss = MemoryEfficientLinear.apply(
            X.view(-1, X.size(-1)),
            linear.weight,
            linear.bias if linear.bias is not None else None,  # Handle None case
            labels.view(-1),
            8192,
            mse_transform
        )
        loss.backward()
        print("MSE Compatibility: ✓") 
    except Exception as e:
        print(f"MSE Compatibility: ✗ ({str(e)})")

    # Fix dynamic chunk test: Updated lambda transformation function
    try:
        for chunk_size in [4096, 8192, 16384]:
            MemoryEfficientLinear.apply(
                X, linear.weight, None, labels, chunk_size,
                lambda logits, labels, s, e: F.cross_entropy(
                    # Correctly filter to include only labels in [s, e)
                    logits[(labels >= s) & (labels < e)],
                    (labels[(labels >= s) & (labels < e)] - s),
                    reduction='mean'
                )
            )
        print("Dynamic chunks: ✓")
    except Exception as e:
        print(f"Dynamic chunks: ✗ ({str(e)})")

    # Criterion 6: Gradient matching
    grad_error = torch.max(torch.abs(grad_normal - grad_efficient)).item()
    print(f"Gradient error: {grad_error:.2e} (Should be <1e-3)")

    # Cleanup
    del X, linear, labels
    gc.collect()
    torch.cuda.empty_cache()

# Run validation
validate_efficiency()
