# Vision Transformer (ViT) Module for TRACE

This module provides Vision Transformer models as an alternative backbone to ResNet50 for the TRACE table recognition system.

## Features

✅ **Drop-in replacement for ResNet50** - Same output interface (h5, h4, h3, h2, h1)  
✅ **Multiple model sizes** - Small, Base, and Large variants  
✅ **Flexible patch sizes** - Support for patch_size 16 and 32  
✅ **Pretrained weights** - Automatic loading from torchvision or timm  
✅ **Position embedding interpolation** - Works with any image size  
✅ **Multi-scale features** - Extracts features at 5 different depths  

## Quick Start

### Basic Usage

```python
from basenet.vit import vit_base

# Create ViT-Base with pretrained ImageNet weights
model = vit_base(pretrained=True, img_size=768, patch_size=16)

# Forward pass
import torch
x = torch.randn(2, 3, 768, 768)
output = model(x)

# Access multi-scale features (same as ResNet)
print(output.h5.shape)  # [2, 2048, 24, 24]   - deepest features
print(output.h4.shape)  # [2, 1024, 48, 48]   - intermediate
print(output.h3.shape)  # [2, 512, 96, 96]    - intermediate
print(output.h2.shape)  # [2, 256, 192, 192]  - intermediate
print(output.h1.shape)  # [2, 64, 384, 384]   - shallowest
```

### Model Variants

```python
from basenet.vit import vit_small, vit_base, vit_large

# Small model (384 dim, 12 layers, 6 heads) - ~22M params
model = vit_small(pretrained=True, img_size=768, patch_size=16)

# Base model (768 dim, 12 layers, 12 heads) - ~86M params
model = vit_base(pretrained=True, img_size=768, patch_size=16)

# Large model (1024 dim, 24 layers, 16 heads) - ~304M params
model = vit_large(pretrained=True, img_size=768, patch_size=16)
```

### Different Patch Sizes

```python
# Patch size 16 - More fine-grained (default)
# 768x768 image -> 48x48 patches
model = vit_base(pretrained=True, patch_size=16)

# Patch size 32 - Fewer patches, faster
# 768x768 image -> 24x24 patches
model = vit_base(pretrained=True, patch_size=32)
```

### Custom Configuration

```python
from basenet.vit import VisionTransformer

model = VisionTransformer(
    img_size=768,
    patch_size=16,
    embed_dim=512,       # Custom embedding dimension
    depth=8,             # Number of transformer blocks
    num_heads=8,         # Number of attention heads
    mlp_ratio=4.0,       # MLP hidden dim = embed_dim * mlp_ratio
    qkv_bias=True,       # Bias in attention QKV projection
    drop_rate=0.0,       # Dropout rate
    attn_drop_rate=0.0,  # Attention dropout rate
    pretrained=False,
    freeze=False
)
```

## Integration with TRACE

### Option 1: Modify model.py

```python
# In model.py
from basenet.resnet50 import resnet50, resnet50d
from basenet.vit import vit_base, vit_large

class TraceModel(nn.Module):
    def __init__(
        self,
        output_ch=5,
        pretrained=False,
        freeze=False,
        backbone='resnet50',  # NEW: add backbone parameter
        **kwargs
    ):
        super(TraceModel, self).__init__()
        
        # Select backbone
        if backbone == 'vit_base':
            self.basenet = vit_base(
                pretrained=pretrained,
                freeze=freeze,
                img_size=768,  # or make this configurable
                patch_size=16
            )
        elif backbone == 'vit_large':
            self.basenet = vit_large(
                pretrained=pretrained,
                freeze=freeze,
                img_size=768,
                patch_size=16
            )
        elif backbone == 'resnet50d':
            self.basenet = resnet50d(pretrained, freeze)
        else:  # default resnet50
            self.basenet = resnet50(pretrained, freeze)
        
        # Rest of the model remains unchanged!
        self.upconv1 = double_conv(2048, 1024, 512)
        # ... etc
```

### Option 2: Update config files

Add backbone configuration to your JSON config files:

```json
{
  "name": "TRACE_ViT",
  "model": {
    "backbone": "vit_base",
    "output_ch": 5,
    "pretrained": true,
    "freeze": false
  },
  ...
}
```

## Pretrained Weights

### Automatic Loading

The module automatically loads pretrained weights based on your configuration:

| Configuration | Torchvision Model | Timm Model (fallback) |
|--------------|-------------------|----------------------|
| `vit_base(patch_size=16)` | `vit_b_16` | `vit_base_patch16_224` |
| `vit_base(patch_size=32)` | `vit_b_32` | `vit_base_patch32_224` |
| `vit_large(patch_size=16)` | `vit_l_16` | `vit_large_patch16_224` |
| `vit_large(patch_size=32)` | `vit_l_32` | `vit_large_patch32_224` |
| `vit_small(patch_size=16)` | `vit_b_16`* | `vit_small_patch16_224` |
| `vit_small(patch_size=32)` | `vit_b_32`* | `vit_small_patch32_224` |

*Uses base model as fallback for small variant in torchvision

### Position Embedding Interpolation

When using pretrained weights trained on 224x224 images with larger input sizes (e.g., 768x768), the position embeddings are automatically interpolated using bicubic interpolation:

```python
# Pretrained on 224x224, but using 768x768
model = vit_base(pretrained=True, img_size=768, patch_size=16)
# Position embeddings automatically interpolated from 14x14 to 48x48
```

## Architecture Details

### Multi-Scale Feature Extraction

Unlike standard ViT which outputs a single representation, this implementation extracts features at 5 different depths to match ResNet's multi-scale outputs:

- **Stage 1** (after 20% of blocks) → projected to h1 (64 channels)
- **Stage 2** (after 40% of blocks) → projected to h2 (256 channels)  
- **Stage 3** (after 60% of blocks) → projected to h3 (512 channels)
- **Stage 4** (after 80% of blocks) → projected to h4 (1024 channels)
- **Stage 5** (after 100% of blocks) → projected to h5 (2048 channels)

Each stage's features are:
1. Extracted from transformer blocks at different depths
2. Reshaped from sequence to spatial format
3. Projected to match ResNet channel dimensions
4. Interpolated to match ResNet spatial dimensions

### Computational Complexity

| Model | Params | GFLOPs (768x768) | Memory (batch=1) |
|-------|--------|------------------|------------------|
| ResNet50 | ~25M | ~16 | ~1.2GB |
| ViT-Small (p16) | ~22M | ~18 | ~1.5GB |
| ViT-Base (p16) | ~86M | ~55 | ~3.2GB |
| ViT-Large (p16) | ~304M | ~191 | ~8.5GB |

*Estimated values for 768x768 input

### Performance Considerations

**Advantages of ViT:**
- Better at capturing long-range dependencies
- More parameter-efficient at larger scales
- Better transferability to downstream tasks
- Can handle variable input sizes

**Advantages of ResNet:**
- Faster inference (especially on CPU)
- Lower memory footprint
- Better inductive bias for small datasets
- More established training recipes

## Testing

Run the example script to verify installation:

```bash
python basenet/vit_example.py
```

## Requirements

```
torch >= 1.12.0
torchvision >= 0.13.0  # For pretrained weights
timm >= 0.6.0          # Optional fallback for pretrained weights
```

## Troubleshooting

### Issue: Position embedding shape mismatch
```
Solution: This is automatically handled by position embedding interpolation.
The model will print: "Interpolating position embeddings from [shape1] to [shape2]"
```

### Issue: Out of memory
```
Solution 1: Use smaller model (vit_small)
Solution 2: Use larger patch_size (32 instead of 16)
Solution 3: Reduce batch size
Solution 4: Use gradient checkpointing (not implemented yet)
```

### Issue: Pretrained weights not loading
```
Solution: The model tries torchvision first, then timm. If neither is available,
it will print a warning but continue with random initialization.
Install: pip install timm
```

## Future Enhancements

- [ ] Gradient checkpointing for memory efficiency
- [ ] Mixed precision training support
- [ ] More ViT variants (DeiT, Swin Transformer, etc.)
- [ ] Flash Attention integration
- [ ] LoRA fine-tuning support

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [PyTorch Vision Transformer Implementation](https://pytorch.org/vision/stable/models/vision_transformer.html)
- [timm: PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)

