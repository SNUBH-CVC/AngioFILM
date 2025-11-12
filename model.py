"""
References
----------
- Vision Transformer (ViT)
    - Model architecture
- Latent Diffusion Transformer for Video Generation (Latte)
    - Model architecture
    - Positional embedding

Positional embedding
- https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py

Attention / TransformerBlock
- https://github.com/huggingface/pytorch-image-models/blob/e44f14d7d2f557b9f3add82ee4f1ed2beefbb30d/timm/models/vision_transformer.py

Latte
- https://github.com/Vchitect/Latte/blob/07c7e5c64d25f627766fa69bea50f3a27900ec84/models/latte.py
"""

import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp, PatchEmbed

# the xformers lib allows less memory, faster training and inference
try:
    import xformers
    import xformers.ops
except:
    XFORMERS_IS_AVAILBLE = False

# from timm.models.layers.helpers import to_2tuple
# from timm.models.layers.trace_utils import _assert


def modulate(x, shift, scale):
    if shift.ndim != 3:
        shift = shift.unsqueeze(1)
    if scale.ndim != 3:
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


#################################################################################
#               Attention Layers from TIMM                                      #
#################################################################################


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        attention_mode="math",
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.attention_mode = attention_mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.attention_mode == "xformers":  # cause loss nan while using with amp
            # https://github.com/facebookresearch/xformers/blob/e8bd8f932c2f48e3a3171d06749eecbbf1de420c/xformers/ops/fmha/__init__.py#L135
            q_xf = q.transpose(1, 2).contiguous()
            k_xf = k.transpose(1, 2).contiguous()
            v_xf = v.transpose(1, 2).contiguous()
            x = xformers.ops.memory_efficient_attention(q_xf, k_xf, v_xf).reshape(
                B, N, C
            )

        elif self.attention_mode == "flash":
            # cause loss nan while using with amp
            # Optionally use the context manager to ensure one of the fused kerenels is run
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v).reshape(
                    B, N, C
                )  # require pytorch 2.0

        elif self.attention_mode == "math":
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


# https://github.com/lucidrains/imagen-pytorch/blob/fa29d249a8bdd7b97ae0da8da02261fc69292b72/imagen_pytorch/imagen_pytorch.py#L610
class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


class ContinuousTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, learned_sinu_pos_emb_dim=16):
        super().__init__()
        sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1
        self.sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sinu_pos_emb_input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        x = self.mlp(self.sinu_pos_emb(x))
        return x


class EcgEmbedder(nn.Module):
    def __init__(self, signal_length, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(signal_length, hidden_size),   
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


#################################################################################
#                                 Core Latte Model                                #
#################################################################################


class TransformerBlock(nn.Module):
    """
    A Latte tansformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        if gate_mlp.ndim != 3:
            gate_mlp = gate_mlp.unsqueeze(1)
        if gate_msa.ndim != 3:
            gate_msa = gate_msa.unsqueeze(1)
        # The rest is for shape alignment; in attention, we hold batch and one of the temporal/spatial axes fixed and mix the remaining information.
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of Latte.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def interleave(x, y, b):
    """
    x: ((b f), n, d) or ((b f), d) 
    y: ((b f+1), n, d) or ((b f+1), d)
    """
    x = rearrange(x, "(b f) ... -> b f ...", b=b)
    y = rearrange(y, "(b f_p1) ... -> b f_p1 ...", b=b)
    f = x.size(1)
    y_first = y[:, :f]  
    stacked = torch.stack([y_first, x], dim=2)
    interleaved = rearrange(stacked, 'b f two ... -> b (f two) ...')
    remainder = y[:, f:]  
    result = torch.cat([interleaved, remainder], dim=1) 
    result = rearrange(result, "b f2_p1 ... -> (b f2_p1) ...")
    return result


class Latte(nn.Module):
    """
    Flow Matching model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_frames=7,
        mask_cond_prob=0.1,
        ecg_mask_cond_prob=0.1,
        ecg_signal_len=100,
        attention_mode="math",
        use_ecg=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.mask_cond_prob = mask_cond_prob
        self.ecg_mask_cond_prob = ecg_mask_cond_prob
        self.ecg_signal_len = ecg_signal_len
        self.use_ecg = use_ecg

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = ContinuousTimestepEmbedder(hidden_size)
        self.t_min_val = 0.0
        self.t_max_val = 1.0

        self.num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size), requires_grad=False
        )  # same with Latte
        self.temp_embed = nn.Parameter(
            torch.zeros(1, num_frames * 2 + 1, hidden_size), requires_grad=False
        )
        self.hidden_size = hidden_size

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attention_mode=attention_mode,
                )
                for _ in range(depth)
            ]
        )

        # y_ecg embedding
        if self.use_ecg:
            self.ecg_embedder = EcgEmbedder(signal_length=self.ecg_signal_len, hidden_size=hidden_size)

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        temp_embed = get_1d_sincos_temp_embed(
            self.temp_embed.shape[-1], self.temp_embed.shape[-2]
        )
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        if self.use_ecg:
            nn.init.normal_(self.ecg_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.ecg_embedder.mlp[2].weight, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Zero-out adaLN modulation layers in Latte blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def _apply_flow_matching_noise(self, images, timesteps):
        """
        Apply flow matching noise interpolation to images at given timesteps.
        
        Args:
            images: (B*F, C, H, W) tensor of images
            timesteps: (B*F,) tensor of timesteps in [0, 1]
                      t=0: pure noise, t=1: clean data
            
        Returns:
            noised_images: Images with flow matching noise interpolation applied
                          xt = t * data + (1-t) * noise
        """
        noise = torch.randn_like(images)
        # Expand timesteps to match image dimensions for broadcasting
        t_expanded = timesteps.view(-1, 1, 1, 1)
        return t_expanded * images + (1 - t_expanded) * noise

    def _get_timestep_embeddings(self, timesteps, repeat_factor):
        """
        Get timestep embeddings and repeat for target shape.
        
        Args:
            timesteps: (B,) tensor of timesteps
            repeat_factor: Number of times to repeat each batch item
            
        Returns:
            embeddings: (B*repeat_factor, D) tensor of timestep embeddings
        """
        embeddings = self.t_embedder(timesteps)  # (B, D)
        if repeat_factor > 1:
            embeddings = repeat(embeddings, "b d -> (b f) d", f=repeat_factor)
        return embeddings

    def _apply_conditional_mask(self, tensor, mask, replacement):
        """
        Apply conditional masking to tensor.
        
        Args:
            tensor: Input tensor to mask
            mask: Boolean mask (True = replace, False = keep original)
            replacement: Replacement values
            
        Returns:
            masked_tensor: Tensor with mask applied
        """
        # Expand mask to match tensor dimensions
        while mask.ndim < tensor.ndim:
            mask = mask.unsqueeze(-1)
        return torch.where(mask, replacement, tensor)

    def forward(
        self,
        x,
        t_x,
        y_image,
        y_ecg=None,
    ):
        """
        Forward pass of Latte.
        x: (B, F, C, H, W) tensor of video inputs
        t_x: (B,) tensor of flow matching timesteps in [0, 1]
        y_image: (B, F + 1, C, H, W) tensor of conditioning frames
        y_ecg: (B, L) tensor of ECG signals
        """
        batches, frames, channels, height, width = x.shape
        assert y_image.shape[1] == frames + 1, "y_image must have F+1 frames"
        if self.use_ecg:
            assert y_ecg is not None, "y_ecg must be provided when use_ecg is True"
        device = x.device

        x = rearrange(x, "b f c h w -> (b f) c h w")
        # self.pos_embed.shape: 1, n, d
        x = self.x_embedder(x) + self.pos_embed  # (b*f) n d | n: num_patches d: hidden_size
        t_x_emb = self._get_timestep_embeddings(t_x, frames)

        y_image = rearrange(y_image, "b f_p1 c h w -> (b f_p1) c h w", f_p1=frames + 1)

        # Conditional image is at clean data level (t=1 in flow matching)
        t_y_emb = self._get_timestep_embeddings(torch.ones_like(t_x) * self.t_max_val, frames + 1)

        # Classifier-Free Guidance
        drop_mask = None
        if self.mask_cond_prob > 0:
            drop_mask = (
                torch.rand(batches * (frames + 1), device=device) < self.mask_cond_prob
            )

            # For unconditional branch, use pure noise (t=0 in flow matching)
            t_noise = torch.zeros(batches * (frames + 1), device=device)  # t=0 for pure noise
            pure_noise_y_image = self._apply_flow_matching_noise(y_image, t_noise)
            t_null_emb = self._get_timestep_embeddings(torch.ones_like(t_x) * self.t_min_val, frames + 1)
            
            # Apply CFG masking
            y_image = self._apply_conditional_mask(y_image, drop_mask, pure_noise_y_image)
            t_y_emb = self._apply_conditional_mask(t_y_emb, drop_mask, t_null_emb)

        if self.use_ecg:
            ecg_emb = self.ecg_embedder(y_ecg)  # (b, hidden_size)
            if self.ecg_mask_cond_prob > 0:
                drop_ecg_mask = torch.rand(batches, device=device) < self.ecg_mask_cond_prob
                ecg_null_emb = torch.zeros_like(ecg_emb)
                ecg_emb = self._apply_conditional_mask(ecg_emb, drop_ecg_mask, ecg_null_emb)

        y_image = (
            self.x_embedder(y_image) + self.pos_embed
        )  # (b f2_p1) n, d | n: num_patches

        # interleave
        x = interleave(x, y_image, b=batches)
        t_x_emb = interleave(t_x_emb, t_y_emb, b=batches)

        timestep_spatial = t_x_emb  # (b f2_p1) d

        timestep_temp = rearrange(t_x_emb, "(b f2_p1) d -> b f2_p1 d", b=batches)

        # If ECG data is provided, add ECG embedding to the temporal conditioning
        if self.use_ecg:
            ecg_emb_temp = repeat(ecg_emb, "b d -> b f2_p1 d", f2_p1=self.num_frames * 2 + 1)
            timestep_temp = timestep_temp + ecg_emb_temp

            ecg_emb_spatial = repeat(ecg_emb, "b d -> (b f2_p1) d", f2_p1=self.num_frames * 2 + 1)
            timestep_spatial = timestep_spatial + ecg_emb_spatial

        timestep_temp = repeat(
            timestep_temp, "b f2_p1 d -> (b n) f2_p1 d", n=self.pos_embed.shape[1]
        )

        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i : i + 2]
            c = timestep_spatial

            # (b f2_p1) n, d
            x = spatial_block(x, c)

            x = rearrange(x, "(b f2_p1) n d -> (b n) f2_p1 d", b=batches)
            # Add Time Embedding
            if i == 0:
                # self.temp_embed.shape: (1, self.num_frames * 2 + 1, d)
                x = x + self.temp_embed

            c = timestep_temp

            x = temp_block(x, c)  # (b n) f2_p1 d
            x = rearrange(x, "(b n) f2_p1 d -> (b f2_p1) n d", b=batches)

        c = timestep_spatial
        x = self.final_layer(x, c)

        # Select the frames corresponding to the original 'x' inputs (odd indices: 1, 3, ..., 2f-1)
        x = rearrange(x, "(b f2_p1) n d -> b f2_p1 n d", b=batches, f2_p1=self.num_frames * 2 + 1)
        x = x[:, 1::2, :, :]
        x = rearrange(x, "b f n d -> (b f) n d")

        x = self.unpatchify(x)
        x = rearrange(x, "(b f) c h w -> b f c h w", b=batches)
        return x

    def forward_with_cfg(
        self,
        x,
        t,
        y_image=None,
        y_ecg=None,
        cfg_scale=3.0,
    ):
        """
        Forward pass of Latte for classifier-free guidance (CFG). This method constructs a combined
        batch with both conditional and unconditional branches. When conditioning images are provided
        (i.e. use_ecg == True), the unconditional branch is created by replacing the conditioning frames
        with fully masked ones (i.e. isotropic Gaussian noise) and replacing the associated timesteps
        with the learned null token.

        Args:
            x (Tensor): Video inputs of shape (B, f, c, h, w).
            t (Tensor): Flow matching timesteps of shape (B, ...) in [0, 1] range.
            y_image (Tensor, optional): Conditioning frames of shape (B, f, c, h, w).
            y_ecg (Tensor, optional): ECG signals used for conditioning when use_ecg==True.
            cfg_scale (float): Guidance scale.

        Returns:
            Tensor: The final output (after applying classifier-free guidance) with the same shape as
            the standard forward pass.
        """
        # In CFG we want to run both a conditional branch (with the original conditioning)
        # and an unconditional branch (with completely masked conditioning).
        # We assume the input batch 'x' is organized such that the first half is used for duplication.

        # Forward the combined inputs.
        # Temporarily disable all conditioning augmentations for CFG
        original_ecg_mask_cond_prob = self.ecg_mask_cond_prob
        original_mask_cond_prob = self.mask_cond_prob  
        
        self.ecg_mask_cond_prob = 0.0
        self.mask_cond_prob = 0.0
        logits = self.forward(
            x,
            t,
            y_image=y_image,
            y_ecg=y_ecg,
        )
        
        self.ecg_mask_cond_prob = 1.0
        self.mask_cond_prob = 1.0
        null_logits = self.forward(
            x,
            t,
            y_image=y_image,
            y_ecg=y_ecg,
        )
        
        # Restore original values
        self.ecg_mask_cond_prob = original_ecg_mask_cond_prob
        self.mask_cond_prob = original_mask_cond_prob

        update = logits - null_logits
        return logits + update * (cfg_scale - 1)

    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to automatically handle temp_embed resizing
        when loading checkpoints trained with different num_frames.
        """
        # Check if temp_embed exists and has different size
        if 'temp_embed' in state_dict:
            ckpt_temp_embed = state_dict['temp_embed']
            curr_temp_embed = self.temp_embed
            
            if ckpt_temp_embed.shape != curr_temp_embed.shape:
                ckpt_length = ckpt_temp_embed.shape[1]
                curr_length = curr_temp_embed.shape[1]
                ckpt_num_frames = (ckpt_length - 1) // 2
                curr_num_frames = (curr_length - 1) // 2
                
                print(f"temp_embed size mismatch detected:")
                print(f"  Checkpoint: num_frames={ckpt_num_frames} (temp_embed length={ckpt_length})")
                print(f"  Current model: num_frames={curr_num_frames} (temp_embed length={curr_length})")
                
                if ckpt_length != curr_length:
                    print(f"  Interpolating temp_embed from {ckpt_length} to {curr_length}")
                    
                    # Interpolate temp_embed
                    old_temp_embed = ckpt_temp_embed.transpose(1, 2)  # (1, hidden_size, old_length)
                    new_temp_embed = torch.nn.functional.interpolate(
                        old_temp_embed, 
                        size=curr_length, 
                        mode='linear', 
                        align_corners=False
                    )
                    new_temp_embed = new_temp_embed.transpose(1, 2)  # (1, new_length, hidden_size)
                    
                    # Replace temp_embed in state_dict with interpolated version
                    state_dict = state_dict.copy()  # Don't modify original
                    state_dict['temp_embed'] = new_temp_embed
                    print(f"  temp_embed successfully interpolated")
        
        # Call parent load_state_dict
        return super().load_state_dict(state_dict, strict=strict)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_1d_sincos_temp_embed(embed_dim, length):
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


#################################################################################
#                                   Latte Configs                                  #
#################################################################################


def Latte_XL_2(**kwargs):
    return Latte(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def Latte_XL_4(**kwargs):
    return Latte(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def Latte_XL_8(**kwargs):
    return Latte(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def Latte_L_2(**kwargs):
    return Latte(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def Latte_L_4(**kwargs):
    return Latte(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def Latte_L_8(**kwargs):
    return Latte(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def Latte_B_2(**kwargs):
    return Latte(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def Latte_B_4(**kwargs):
    return Latte(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def Latte_B_8(**kwargs):
    return Latte(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def Latte_S_2(**kwargs):
    return Latte(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def Latte_S_4(**kwargs):
    return Latte(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def Latte_S_8(**kwargs):
    return Latte(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


Latte_models = {
    "Latte-XL/2": Latte_XL_2,
    "Latte-XL/4": Latte_XL_4,
    "Latte-XL/8": Latte_XL_8,
    "Latte-L/2": Latte_L_2,
    "Latte-L/4": Latte_L_4,
    "Latte-L/8": Latte_L_8,
    "Latte-B/2": Latte_B_2,
    "Latte-B/4": Latte_B_4,
    "Latte-B/8": Latte_B_8,
    "Latte-S/2": Latte_S_2,
    "Latte-S/4": Latte_S_4,
    "Latte-S/8": Latte_S_8,
}


def get_model(name, input_size: int, in_channels: int, num_frames: int, mask_cond_prob: float, ecg_mask_cond_prob: float, use_ecg: bool, ecg_signal_len: int):
    return Latte_models[name](
        input_size=input_size,
        in_channels=in_channels,
        num_frames=num_frames,
        mask_cond_prob=mask_cond_prob,
        ecg_mask_cond_prob=ecg_mask_cond_prob,
        use_ecg=use_ecg,
        ecg_signal_len=ecg_signal_len,
    )


if __name__ == "__main__":
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = torch.randn(3, 15, 4, 32, 32).to(device)
    t = torch.tensor([0.1, 0.2, 0.3]).to(device)
    y_image = torch.randn(3, 16, 4, 32, 32).to(device)
    y_ecg = torch.randn(3, 100).to(device)
    model = Latte_L_4(use_ecg=True, num_frames=16).to(device)
    output = model(img, t, y_image, y_ecg)
    assert output.shape == (3, 16, 4, 32, 32)
