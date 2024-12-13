import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PretrainedConfig
from einops import rearrange
from accelerate import Accelerator
from accelerate.utils import set_seed
import tqdm
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import safetensors
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttnProcessor, LoRAAttnProcessor, AttnAddedKVProcessor, LoRAAttnAddedKVProcessor
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
import warnings
warnings.filterwarnings('ignore')

check_min_version("0.17.0")

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    from transformers import CLIPTextModel
    return CLIPTextModel

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    max_length = tokenizer_max_length if tokenizer_max_length else tokenizer.model_max_length
    text_inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    return text_inputs

def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=False):
    text_input_ids = input_ids.to(text_encoder.device)
    attention_mask = attention_mask.to(text_encoder.device) if text_encoder_use_attention_mask else None
    prompt_embeds = text_encoder(text_input_ids, attention_mask=attention_mask)
    prompt_embeds = prompt_embeds[0]
    return prompt_embeds

def train_lora(
    image, prompt, save_lora_dir, model_path=None, tokenizer=None, text_encoder=None, vae=None, unet=None,
    noise_scheduler=None, lora_steps=200, lora_lr=2e-4, weight_name="lora_weights.ckpt", safe_serialization=False, progress=tqdm
):
    from accelerate import Accelerator
    from diffusers.optimization import get_scheduler
    from torchvision import transforms
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import os
    import safetensors

    accelerator = Accelerator()
    device = accelerator.device

    # Load defaults if not provided
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", use_fast=False)
    if noise_scheduler is None:
        from diffusers import DDPMScheduler
        noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    if text_encoder is None:
        from transformers import CLIPTextModel
        text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    if vae is None:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    if unet is None:
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")

    vae.to(device).eval().requires_grad_(False)
    text_encoder.to(device).eval().requires_grad_(False)
    unet.to(device).train()

    # Directly gather existing LoRA params
    # We look for parameters ending with lora_A.weight or lora_B.weight
    lora_params = []
    for n, p in unet.named_parameters():
        if "lora_A.weight" in n or "lora_B.weight" in n:
            p.requires_grad_(True)
            lora_params.append(p)

    if len(lora_params) == 0:
        raise ValueError("No existing LoRA parameters found. Your model might not be LoRA-integrated, or naming changed.")

    optimizer = torch.optim.AdamW(lora_params, lr=lora_lr, betas=(0.9,0.999), weight_decay=1e-2)
    lr_scheduler = get_scheduler("constant", optimizer=optimizer, num_warmup_steps=0, num_training_steps=lora_steps)

    unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)

    # Tokenize prompt
    max_length = tokenizer.model_max_length
    text_inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    text_inputs = text_inputs.to(device)
    with torch.no_grad():
        prompt_embeds = text_encoder(text_inputs.input_ids)[0]

    # Preprocess image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image_transforms = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])
    image = image_transforms(image).unsqueeze(0).to(device)
    latents_dist = vae.encode(image).latent_dist

    for step in progress.trange(lora_steps, desc="Training LoRA"):
        model_input = latents_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(model_input)
        bsz = model_input.size(0)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()

        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
        model_pred = unet(noisy_model_input, timesteps, prompt_embeds).sample

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError("Unknown prediction type")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # Save the LoRA weights
    # We'll just dump all parameters with lora_A and lora_B to a dict
    lora_state = {}
    for n, p in unet.named_parameters():
        if "lora_A.weight" in n or "lora_B.weight" in n:
            lora_state[n] = p.detach().cpu()

    out_path = os.path.join(save_lora_dir, weight_name)
    if safe_serialization:
        safetensors.torch.save_file(lora_state, out_path+".safetensors")
    else:
        torch.save(lora_state, out_path)

    return unet

def load_lora(unet, lora_0, lora_1, alpha):
    combined = {}
    keys = set(lora_0.keys()).union(set(lora_1.keys()))
    for k in keys:
        w0 = lora_0.get(k, None)
        w1 = lora_1.get(k, None)
        if w0 is None:
            combined[k] = w1
        elif w1 is None:
            combined[k] = w0
        else:
            combined[k] = (1 - alpha)*w0 + alpha*w1

    for name, module in unet.named_modules():
        if isinstance(module, (LoRAAttnProcessor, LoRAAttnAddedKVProcessor)) and hasattr(module, 'lora_layers'):
            for proj in list(module.lora_layers.keys()):
                A_key = f"{name}.{proj}.lora_A.weight"
                B_key = f"{name}.{proj}.lora_B.weight"
                if A_key in combined and B_key in combined:
                    A = combined[A_key].to(module.lora_layers[proj][0].device)
                    B = combined[B_key].to(module.lora_layers[proj][1].device)
                    module.lora_layers[proj] = (torch.nn.Parameter(A), torch.nn.Parameter(B))
    return unet