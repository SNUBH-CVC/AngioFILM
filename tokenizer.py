from abc import abstractmethod

import torch
from cosmos_tokenizer.image_lib import ImageTokenizer
from diffusers.models import AutoencoderKL

from utils import requires_grad


class Tokenizer:
    latent_dim = None
    img_size_divisor = None

    def __init__(self, model):
        self.model = model
        # Freeze vae and text_encoder
        requires_grad(self.model, False)

    @abstractmethod
    def encode(self, x):
        """
        x: (B, C, H, W)
        return: (B, latent_dim, H/8, W/8)
        """
        pass

    @abstractmethod
    def decode(self, x):
        """
        x: (B, latent_dim, H/8, W/8)
        return: (B, C, H, W)
        """
        pass


class CosmosImageTokenizer(Tokenizer):
    latent_dim = 16
    img_size_divisor = 16

    def encode(self, x):
        # only bfloat16 is supported for now
        dtype = x.dtype
        x = x.to(torch.bfloat16)
        # https://github.com/nvidia-cosmos/cosmos-predict1/blob/main/examples/inference_tokenizer.md#encoding-into-continuous-latent-space
        # x should be ranged in [-1, 1]:
        # x = x * 2. - 1. 
        latent = self.model.encode(x)[0]
        return latent.to(dtype)
        
    def decode(self, x):
        dtype = x.dtype
        x = x.to(torch.bfloat16)
        reconstructed_x = self.model.decode(x)
        return reconstructed_x.to(dtype)


class VAESDTokenizer(Tokenizer):
    latent_dim = 4
    img_size_divisor = 8

    def encode(self, x):
        return self.model.encode(x).latent_dist.sample().mul_(0.18215)
    
    def decode(self, x):
        return self.model.decode(x / 0.18215).sample


def vae_sd(device="cuda"):
    # input shape: (B, C, H, W) 
    # latent shape: (B, 4, H/8, W/8)
    model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    return VAESDTokenizer(model)


def cosmos_tokenizer_image_8x8(device="cuda"):
    # input shape: (B, H, W, C)
    # latent shape: (B, 16, H/8, W/8)
    model = ImageTokenizer(
        checkpoint=None,
        checkpoint_enc="pretrained_ckpts/Cosmos-Tokenizer-CI8x8/encoder.jit",
        checkpoint_dec="pretrained_ckpts/Cosmos-Tokenizer-CI8x8/decoder.jit",
        tokenizer_config=None,  # Using JIT mode, so no config needed
        device=device,
        dtype="bfloat16",
    )
    return CosmosImageTokenizer(model)


def cosmos_tokenizer_image_16x16(device="cuda"):
    # input shape: (B, H, W, C)
    # latent shape: (B, 16, H/16, W/16)
    model = ImageTokenizer(
        checkpoint=None,
        checkpoint_enc="pretrained_ckpts/Cosmos-Tokenizer-CI16x16/encoder.jit",
        checkpoint_dec="pretrained_ckpts/Cosmos-Tokenizer-CI16x16/decoder.jit",
        tokenizer_config=None,  # Using JIT mode, so no config needed
        device=device,
        dtype="bfloat16",
    )
    return CosmosImageTokenizer(model)


tokenizers = {
    "vae_sd": vae_sd,
    "cosmos_tokenizer_image_8x8": cosmos_tokenizer_image_8x8,
    "cosmos_tokenizer_image_16x16": cosmos_tokenizer_image_16x16,
    # "cosmos_tokenizer_video": cosmos_tokenizer_video,
}


def get_frozen_tokenizer(name, device):
    tokenizer = tokenizers[name](device)
    return tokenizer
