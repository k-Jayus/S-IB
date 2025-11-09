# S-IB: Spatial Information Bottleneck Training Method

This repository contains 20 training scripts implementing our Disentangled Information Bottleneck (S-IB) method across 5 datasets and 4 model architectures. All scripts share identical core S-IB logic.

## Core Method Components

Our S-IB method consists of 3 key components:

### 1. VJP-based Gradient Reconstruction

```python
# Compute Jacobian-vector product (gradient of output w.r.t. input)
p = logits.softmax(dim=1)
p2_sum = (p ** 2).sum() * 0.5
R = torch.autograd.grad(p2_sum, x, create_graph=True)[0]
```

**Purpose**: Backproject model decisions to input space to obtain reconstruction map R capturing model attention.

**Key Points**:
- Must enable `x.requires_grad_(True)`
- `create_graph=True` maintains computation graph for second-order gradients
- R shape: `[B, C, H, W]`, same as input image

### 2. Differentiable Adaptive Mask Generation

```python
M = batch_integral_density_mask_v2_differentiable(R).unsqueeze(1)
```

**Implementation Details**:

```python
def differentiable_quantile(x, q, dim=-1, temperature=1.0):
    """
    Differentiable quantile approximation
    x: input tensor
    q: quantile value (between 0-1)
    dim: dimension to compute quantile
    temperature: temperature parameter, smaller values closer to true quantile
    """
    # Sort the input
    sorted_x, _ = torch.sort(x, dim=dim)
    
    # Calculate index position for quantile
    n = x.shape[dim]
    target_idx = q * (n - 1)
    
    # Convert target_idx to tensor
    target_idx = torch.tensor(target_idx, device=x.device, dtype=torch.float32)
    
    # Use softmax weighting to approximate selection
    indices = torch.arange(n, device=x.device, dtype=torch.float32)
    
    if dim != -1:
        # Adjust indices shape to match target dimension
        indices_shape = [1] * len(x.shape)
        indices_shape[dim] = n
        indices = indices.view(indices_shape)
        
        # Adjust target_idx shape
        target_shape = [1] * len(x.shape)
        target_shape[dim] = 1
        target_idx = target_idx.view(target_shape)
    
    # Calculate distances and apply softmax
    distances = -torch.abs(indices - target_idx) / temperature
    weights = F.softmax(distances, dim=dim)
    
    # Weighted sum to get quantile approximation
    quantile_approx = torch.sum(sorted_x * weights, dim=dim, keepdim=True)
    
    return quantile_approx


def batch_integral_density_mask_v2_differentiable(image_tensor, window_size=31, device='cuda',
                                                  temperature=0.1, slope=10.0):
    """
    Differentiable version of adaptive threshold function
    temperature: controls quantile approximation precision, smaller is more precise
    slope: controls sigmoid threshold steepness, larger is closer to hard threshold
    """
    if not isinstance(image_tensor, torch.Tensor):
        image_tensor = torch.tensor(image_tensor, device=device, dtype=torch.float32)
    else:
        image_tensor = image_tensor.to(device)
    
    batch_size, channels, h, w = image_tensor.shape
    
    # Convert to grayscale
    gray = torch.mean(image_tensor, dim=1)
    gray_abs = torch.abs(gray)
    
    # Batch normalization - using differentiable quantile
    gray_flat = gray_abs.view(batch_size, -1)
    gray_min = differentiable_quantile(gray_flat, 0.05, dim=1, temperature=temperature).view(batch_size, 1, 1)
    gray_max = differentiable_quantile(gray_flat, 0.95, dim=1, temperature=temperature).view(batch_size, 1, 1)
    gray_norm = (gray_abs - gray_min) / (gray_max - gray_min + 1e-8)
    
    # Adaptive threshold based on 80% quantile - using differentiable version
    gray_norm_flat = gray_norm.view(batch_size, -1)
    adaptive_threshold = differentiable_quantile(gray_norm_flat, 0.8, dim=1, temperature=temperature).view(batch_size, 1, 1)
    
    # High intensity regions - using sigmoid soft threshold instead of hard threshold
    high_intensity = torch.sigmoid((gray_norm - adaptive_threshold) * slope)
    
    # Compute integral density
    input_tensor = high_intensity.unsqueeze(1)
    density = F.avg_pool2d(input_tensor,
                           kernel_size=window_size,
                           stride=1,
                           padding=window_size // 2)
    density = density.squeeze(1)
    
    # Use differentiable quantile to compute threshold
    density_flat = density.view(batch_size, -1)
    thresholds = differentiable_quantile(density_flat, 0.8, dim=1, temperature=temperature).view(batch_size, 1, 1)
    
    # Use sigmoid soft threshold instead of hard threshold
    mask = torch.sigmoid((density - thresholds) * slope)
    
    return mask
```

**Key Features**:
- Fully differentiable (no `.detach()` operations)
- Uses differentiable quantile approximation via softmax weighting
- Smooth sigmoid produces binary-like masks while maintaining gradients
- Adaptive thresholding based on 80th percentile of intensity distribution

### 3. Foreground/Background Information Loss

```python
# Foreground: maximize alignment between reconstruction and input
fore_R = F.normalize((R * M).flatten(1), dim=1)
fore_x = F.normalize((x * M).flatten(1), dim=1)

if mi_type == "hsic":
    if hsic_rand_feat is not None:
        # Optional random projection for efficiency
        proj = torch.randn(fore_R.size(1), hsic_rand_feat, device='cuda')
        fore_R = F.normalize(fore_R @ proj, dim=1)
        fore_x = F.normalize(fore_x @ proj, dim=1)
    align_loss = hsic_unbiased(fore_R, fore_x)

# Background: minimize information variance
mi_bg = torch.mean(torch.var((R * (1 - M)), dim=(1, 2, 3)))
```

**HSIC Implementation**:

```python
def hsic_unbiased(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Unbiased HSIC with linear kernel, O(B²) complexity
    
    Args:
        z1: First feature matrix [B, D]
        z2: Second feature matrix [B, D]
    
    Returns:
        Unbiased HSIC estimate
    """
    B = z1.size(0)
    K = z1 @ z1.T  # Gram matrix for z1
    L = z2 @ z2.T  # Gram matrix for z2
    H = torch.eye(B, device=z1.device) - 1. / B  # Centering matrix
    Kc = H @ K @ H  # Centered K
    Lc = H @ L @ H  # Centered L
    return (Kc * Lc).sum() / (B - 1) ** 2
```

**Objective**:
- **Foreground Loss**: Maximize HSIC between masked reconstruction R and masked input x, encouraging the model to capture relevant information from foreground regions
- **Background Loss**: Minimize variance of reconstruction in background regions, suppressing spurious information extraction

**Mathematical Formulation**:
```
L_fg = -HSIC(R ⊙ M, X ⊙ M)      (maximize foreground alignment)
L_bg = E[Var(R ⊙ (1-M))]        (minimize background variance)
```

where ⊙ denotes element-wise multiplication and M is the binary mask.

## Usage

### Hyperparameter Configuration

```python
# S-IB hyperparameters
lambda_fg = 0.01          # Foreground loss weight
lambda_bg = 1000          # Background loss weight
window_size = 31         # Integral density window
temperature = 0.1        # Quantile approximation temperature
slope = 10.0             # Sigmoid slope for soft thresholding
mi_type = "hsic"         # Mutual information estimator
hsic_rand_feat = None    # Optional: use random projection (e.g., 512)
```

## Requirements

```
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.19.0

```
