from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer with multi-scale feature extraction"""
    def __init__(
        self,
        img_size=768,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        pretrained=False,
        freeze=False,
        freeze_mode='full',  # 'full', 'partial', 'patch_only', or 'none'
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Feature projection layers to match ResNet channel dimensions
        # ResNet outputs: h5(2048), h4(1024), h3(512), h2(256), h1(64)
        self.proj5 = nn.Conv2d(embed_dim, 2048, kernel_size=1)
        self.proj4 = nn.Conv2d(embed_dim, 1024, kernel_size=1)
        self.proj3 = nn.Conv2d(embed_dim, 512, kernel_size=1)
        self.proj2 = nn.Conv2d(embed_dim, 256, kernel_size=1)
        self.proj1 = nn.Conv2d(embed_dim, 64, kernel_size=1)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
        if pretrained:
            self.load_pretrained_weights()
            
        # Apply freezing strategy
        if freeze:
            if freeze_mode == 'full':
                self.freeze_backbone()
                print("Frozen entire ViT backbone. Only projection layers are trainable.")
            elif freeze_mode == 'partial':
                self.freeze_backbone_partial()
                print(f"Frozen first {len(self.blocks) // 2} transformer blocks.")
            elif freeze_mode == 'patch_only':
                self.freeze_patch_embed()
                print("Frozen patch embedding only.")
            else:  # 'none'
                print("No freezing applied.")
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def freeze_patch_embed(self):
        """Freeze patch embedding layer only"""
        for param in self.patch_embed.parameters():
            param.requires_grad = False
    
    def freeze_backbone(self):
        """
        Freeze the entire ViT backbone (patch_embed, blocks, norm).
        Only projection layers remain trainable for adaptation to new task.
        This is the recommended approach for transfer learning with limited data.
        """
        # Freeze patch embedding
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        
        # Freeze position embeddings and class token
        self.pos_embed.requires_grad = False
        self.cls_token.requires_grad = False
        
        # Freeze all transformer blocks
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False
        
        # Freeze final norm
        for param in self.norm.parameters():
            param.requires_grad = False
        
        # Keep projection layers trainable (proj1-5)
        # These adapt ViT features to ResNet-compatible dimensions
    
    def freeze_backbone_partial(self, num_layers_to_freeze=None):
        """
        Freeze only the first N transformer blocks.
        Useful for gradual fine-tuning strategy.
        
        Args:
            num_layers_to_freeze: Number of blocks to freeze from the start.
                                 If None, freezes half of the blocks.
        """
        if num_layers_to_freeze is None:
            num_layers_to_freeze = len(self.blocks) // 2
        
        # Freeze patch embedding
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        
        # Freeze early transformer blocks
        for i, block in enumerate(self.blocks):
            if i < num_layers_to_freeze:
                for param in block.parameters():
                    param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning"""
        for param in self.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters(self):
        """
        Returns statistics about trainable vs frozen parameters.
        Useful for verifying freeze strategy.
        """
        trainable_params = 0
        frozen_params = 0
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()
        
        total_params = trainable_params + frozen_params
        
        return {
            'trainable': trainable_params,
            'frozen': frozen_params,
            'total': total_params,
            'trainable_pct': 100.0 * trainable_params / total_params if total_params > 0 else 0
        }
    
    def print_trainable_parameters(self):
        """Print detailed breakdown of trainable parameters"""
        stats = self.get_trainable_parameters()
        print(f"\n{'='*60}")
        print(f"ViT Parameter Statistics:")
        print(f"{'='*60}")
        print(f"Total parameters:     {stats['total']:>15,}")
        print(f"Trainable parameters: {stats['trainable']:>15,} ({stats['trainable_pct']:.2f}%)")
        print(f"Frozen parameters:    {stats['frozen']:>15,} ({100-stats['trainable_pct']:.2f}%)")
        print(f"{'='*60}\n")
    
    def interpolate_pos_encoding(self, pos_embed, H, W):
        """
        Interpolate position embeddings for arbitrary image sizes
        Args:
            pos_embed: position embeddings [1, num_patches + 1, embed_dim]
            H, W: target height and width in patches
        """
        num_patches = H * W
        N = pos_embed.shape[1] - 1  # Exclude class token
        
        if num_patches == N:
            return pos_embed
        
        class_pos_embed = pos_embed[:, 0:1]
        patch_pos_embed = pos_embed[:, 1:]
        
        dim = pos_embed.shape[-1]
        
        # Get original grid size (assuming square)
        h0 = w0 = int(N ** 0.5)
        
        # Reshape to grid
        patch_pos_embed = patch_pos_embed.reshape(1, h0, w0, dim).permute(0, 3, 1, 2)
        
        # Interpolate
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(H, W),
            mode='bicubic',
            align_corners=False
        )
        
        # Reshape back
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, H * W, dim)
        
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)
    
    def load_pretrained_weights(self):
        """Load pretrained weights from torchvision or timm"""
        try:
            # Try torchvision first (PyTorch >= 1.12)
            from torchvision import models
            
            # Map model config to torchvision models
            model_mapping = {
                (768, 16): ('vit_b_16', models.ViT_B_16_Weights.IMAGENET1K_V1),
                (768, 32): ('vit_b_32', models.ViT_B_32_Weights.IMAGENET1K_V1),
                (1024, 16): ('vit_l_16', models.ViT_L_16_Weights.IMAGENET1K_V1),
                (1024, 32): ('vit_l_32', models.ViT_L_32_Weights.IMAGENET1K_V1),
                (384, 16): ('vit_b_16', models.ViT_B_16_Weights.IMAGENET1K_V1),  # Use base for small
            }
            
            model_key = (self.embed_dim, self.patch_size)
            if model_key not in model_mapping:
                raise ValueError(f"No pretrained model available for embed_dim={self.embed_dim}, patch_size={self.patch_size}")
            
            model_name, weights = model_mapping[model_key]
            print(f"Loading pretrained ViT weights from torchvision ({model_name}, patch_size={self.patch_size})...")
            
            model_fn = getattr(models, model_name)
            pretrained_model = model_fn(weights=weights)
            
            # Load compatible weights
            model_dict = self.state_dict()
            pretrained_dict = pretrained_model.state_dict()
            
            # Map torchvision keys to our model keys
            # torchvision uses different naming conventions
            key_mapping = {
                'conv_proj': 'patch_embed.proj',
                'class_token': 'cls_token',
                'encoder.pos_embedding': 'pos_embed',
            }
            
            mapped_dict = {}
            for k, v in pretrained_dict.items():
                # Map keys from torchvision to our naming
                new_k = k
                for old_prefix, new_prefix in key_mapping.items():
                    if k.startswith(old_prefix):
                        new_k = k.replace(old_prefix, new_prefix)
                        break
                
                # Map encoder layers
                if 'encoder.layers.encoder_layer_' in k:
                    new_k = k.replace('encoder.layers.encoder_layer_', 'blocks.')
                    new_k = new_k.replace('.ln_1', '.norm1')
                    new_k = new_k.replace('.ln_2', '.norm2')
                    new_k = new_k.replace('.self_attention', '.attn')
                    new_k = new_k.replace('.mlp.linear_1', '.mlp.fc1')
                    new_k = new_k.replace('.mlp.linear_2', '.mlp.fc2')
                    new_k = new_k.replace('self_attention.in_proj_', 'attn.qkv.')
                    new_k = new_k.replace('self_attention.out_proj', 'attn.proj')
                
                # Only add if key exists in our model and shapes match
                if new_k in model_dict and v.shape == model_dict[new_k].shape:
                    mapped_dict[new_k] = v
            
            model_dict.update(mapped_dict)
            self.load_state_dict(model_dict, strict=False)
            print(f"Loaded {len(mapped_dict)} pretrained weights from torchvision")
            
            # Interpolate position embeddings if needed
            if 'pos_embed' in mapped_dict:
                pretrained_pos_embed = mapped_dict['pos_embed']
                current_pos_embed = self.pos_embed
                if pretrained_pos_embed.shape != current_pos_embed.shape:
                    print(f"Interpolating position embeddings from {pretrained_pos_embed.shape} to {current_pos_embed.shape}")
                    H = W = int(self.num_patches ** 0.5)
                    interpolated = self.interpolate_pos_encoding(pretrained_pos_embed, H, W)
                    self.pos_embed.data.copy_(interpolated)
            
        except (ImportError, AttributeError, ValueError) as e:
            print(f"torchvision ViT not available ({e}), trying timm...")
            try:
                import timm
                
                # Map to timm model names based on embed_dim and patch_size
                timm_model_mapping = {
                    (768, 16): 'vit_base_patch16_224',
                    (768, 32): 'vit_base_patch32_224',
                    (1024, 16): 'vit_large_patch16_224',
                    (1024, 32): 'vit_large_patch32_224',
                    (384, 16): 'vit_small_patch16_224',
                    (384, 32): 'vit_small_patch32_224',
                }
                
                model_key = (self.embed_dim, self.patch_size)
                if model_key not in timm_model_mapping:
                    raise ValueError(f"No timm pretrained model for embed_dim={self.embed_dim}, patch_size={self.patch_size}")
                
                timm_model_name = timm_model_mapping[model_key]
                print(f"Loading pretrained weights from timm ({timm_model_name})...")
                pretrained_model = timm.create_model(timm_model_name, pretrained=True)
                
                # Load compatible weights
                model_dict = self.state_dict()
                pretrained_dict = pretrained_model.state_dict()
                
                # Filter out incompatible keys (due to different image size or projection layers)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                 if k in model_dict and v.shape == model_dict[k].shape}
                
                model_dict.update(pretrained_dict)
                self.load_state_dict(model_dict, strict=False)
                print(f"Loaded {len(pretrained_dict)} pretrained weights from timm")
                
                # Interpolate position embeddings if needed
                if 'pos_embed' in pretrained_dict:
                    pretrained_pos_embed = pretrained_dict['pos_embed']
                    current_pos_embed = self.pos_embed
                    if pretrained_pos_embed.shape != current_pos_embed.shape:
                        print(f"Interpolating position embeddings from {pretrained_pos_embed.shape} to {current_pos_embed.shape}")
                        H = W = int(self.num_patches ** 0.5)
                        interpolated = self.interpolate_pos_encoding(pretrained_pos_embed, H, W)
                        self.pos_embed.data.copy_(interpolated)
            except ImportError:
                print("Neither torchvision nor timm available, skipping pretrained weights loading")
            except Exception as e:
                print(f"Could not load pretrained weights: {e}")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
    
    def forward_features(self, x):
        """Forward pass through transformer blocks, returning intermediate features"""
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # B, embed_dim, H//patch_size, W//patch_size
        
        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Store intermediate features at different depths
        features = []
        depth = len(self.blocks)
        
        # Extract features at 5 different stages (similar to ResNet)
        # Divide blocks into 5 stages
        stage_indices = [
            depth // 5,           # Early stage
            depth * 2 // 5,       # Stage 2
            depth * 3 // 5,       # Stage 3
            depth * 4 // 5,       # Stage 4
            depth                  # Final stage
        ]
        
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i + 1 in stage_indices:
                features.append(x)
        
        return features
    
    def reshape_features(self, x, H, W):
        """Reshape features from sequence to spatial format"""
        B = x.shape[0]
        # Remove class token
        x = x[:, 1:, :]
        # Reshape to spatial
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        return x
    
    def forward(self, x):
        """
        Forward pass returning multi-scale features compatible with ResNet output
        Returns namedtuple with h5, h4, h3, h2, h1 (from deep to shallow)
        """
        B, C, H, W = x.shape
        
        # Calculate spatial dimensions after patch embedding
        feat_H = H // self.patch_size
        feat_W = W // self.patch_size
        
        # Get intermediate features from transformer
        features = self.forward_features(x)
        
        # Reshape and project features to match ResNet dimensions
        # ResNet feature map sizes (assuming 768x768 input):
        # h5: /32 (24x24, 2048 channels)
        # h4: /16 (48x48, 1024 channels)  
        # h3: /8  (96x96, 512 channels)
        # h2: /4  (192x192, 256 channels)
        # h1: /2  (384x384, 64 channels)
        
        # All ViT features have same spatial size, so we project and resize them
        h5_feat = self.reshape_features(features[4], feat_H, feat_W)  # Deepest
        h5 = self.proj5(h5_feat)
        
        h4_feat = self.reshape_features(features[3], feat_H, feat_W)
        h4 = self.proj4(h4_feat)
        h4 = F.interpolate(h4, scale_factor=2, mode='bilinear', align_corners=False)
        
        h3_feat = self.reshape_features(features[2], feat_H, feat_W)
        h3 = self.proj3(h3_feat)
        h3 = F.interpolate(h3, scale_factor=4, mode='bilinear', align_corners=False)
        
        h2_feat = self.reshape_features(features[1], feat_H, feat_W)
        h2 = self.proj2(h2_feat)
        h2 = F.interpolate(h2, scale_factor=8, mode='bilinear', align_corners=False)
        
        h1_feat = self.reshape_features(features[0], feat_H, feat_W)
        h1 = self.proj1(h1_feat)
        h1 = F.interpolate(h1, scale_factor=16, mode='bilinear', align_corners=False)
        
        # Return in same format as ResNet
        vit_outputs = namedtuple("VitOutputs", ["h5", "h4", "h3", "h2", "h1"])
        out = vit_outputs(h5, h4, h3, h2, h1)
        
        return out


def vit_base(pretrained=False, freeze=False, freeze_mode='full', img_size=768, patch_size=16):
    """
    Vision Transformer Base model
    - embed_dim: 768
    - depth: 12
    - num_heads: 12
    
    Args:
        pretrained: Load pretrained ImageNet weights
        freeze: Whether to freeze parameters
        freeze_mode: 'full' (freeze entire backbone), 'partial' (freeze half), 
                     'patch_only' (freeze patch embed only), 'none'
        img_size: Input image size
        patch_size: Patch size (16 or 32)
    """
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        pretrained=pretrained,
        freeze=freeze,
        freeze_mode=freeze_mode
    )
    return model


def vit_large(pretrained=False, freeze=False, freeze_mode='full', img_size=768, patch_size=16):
    """
    Vision Transformer Large model
    - embed_dim: 1024
    - depth: 24
    - num_heads: 16
    
    Args:
        pretrained: Load pretrained ImageNet weights
        freeze: Whether to freeze parameters
        freeze_mode: 'full' (freeze entire backbone), 'partial' (freeze half), 
                     'patch_only' (freeze patch embed only), 'none'
        img_size: Input image size
        patch_size: Patch size (16 or 32)
    """
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        pretrained=pretrained,
        freeze=freeze,
        freeze_mode=freeze_mode
    )
    return model


def vit_small(pretrained=False, freeze=False, freeze_mode='full', img_size=768, patch_size=16):
    """
    Vision Transformer Small model
    - embed_dim: 384
    - depth: 12
    - num_heads: 6
    
    Args:
        pretrained: Load pretrained ImageNet weights
        freeze: Whether to freeze parameters
        freeze_mode: 'full' (freeze entire backbone), 'partial' (freeze half), 
                     'patch_only' (freeze patch embed only), 'none'
        img_size: Input image size
        patch_size: Patch size (16 or 32)
    """
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        pretrained=pretrained,
        freeze=freeze,
        freeze_mode=freeze_mode
    )
    return model


if __name__ == "__main__":
    # Test the model
    model = vit_base(pretrained=False, freeze=False, img_size=768, patch_size=16)
    x = torch.randn(1, 3, 768, 768)
    output = model(x)
    
    print("ViT Multi-scale Feature Maps:")
    print(f"h5 (deepest): {output.h5.shape}")
    print(f"h4: {output.h4.shape}")
    print(f"h3: {output.h3.shape}")
    print(f"h2: {output.h2.shape}")
    print(f"h1 (shallowest): {output.h1.shape}")
    
    # Expected output shapes for 768x768 input:
    # h5: [1, 2048, 24, 24]  - /32
    # h4: [1, 1024, 48, 48]  - /16
    # h3: [1, 512, 96, 96]   - /8
    # h2: [1, 256, 192, 192] - /4
    # h1: [1, 64, 384, 384]  - /2


