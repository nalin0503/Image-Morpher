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
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

check_min_version("0.17.0")


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", revision=revision)
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        try:
            from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
            return RobertaSeriesModelWithTransformation
        except ImportError:
            raise ValueError("Optional alt_diffusion model not available.")
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

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

def train_lora(image, prompt, save_lora_dir, model_path=None, tokenizer=None, text_encoder=None, vae=None, unet=None,
               noise_scheduler=None, lora_steps=200, lora_lr=2e-4, lora_rank=16, weight_name=None, safe_serialization=False, progress=tqdm):
    accelerator = Accelerator(gradient_accumulation_steps=1)
    set_seed(0)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", revision=None, use_fast=False)
    if noise_scheduler is None:
        noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    if text_encoder is None:
        text_encoder_cls = import_model_class_from_model_name_or_path(model_path, revision=None)
        text_encoder = text_encoder_cls.from_pretrained(model_path, subfolder="text_encoder", revision=None)
    if vae is None:
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", revision=None)
    if unet is None:
        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", revision=None)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    unet.to(device)
    vae.to(device)
    text_encoder.to(device)

    # Create LoRA layers using create_lora_layers for given rank
    unet_lora_attn_procs = LoraLoaderMixin.write_lora_layers(unet, lora_rank=lora_rank)
    unet.set_attn_processor(unet_lora_attn_procs)
    unet_lora_layers = AttnProcsLayers(unet.attn_processors)

    params_to_optimize = unet_lora_layers.parameters()
    optimizer = torch.optim.AdamW(params_to_optimize, lr=lora_lr, betas=(0.9,0.999), weight_decay=1e-2, eps=1e-08)
    lr_scheduler = get_scheduler("constant", optimizer=optimizer, num_warmup_steps=0, num_training_steps=lora_steps)

    unet_lora_layers = accelerator.prepare_model(unet_lora_layers)
    optimizer = accelerator.prepare_optimizer(optimizer)
    lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)

    with torch.no_grad():
        text_inputs = tokenize_prompt(tokenizer, prompt)
        text_embedding = encode_prompt(text_encoder, text_inputs.input_ids, text_inputs.attention_mask, False)

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image_transforms = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

    image = image_transforms(image).to(device).unsqueeze(0)
    latents_dist = vae.encode(image).latent_dist

    for _ in progress.tqdm(range(lora_steps), desc="Training LoRA..."):
        unet.train()
        model_input = latents_dist.sample()*vae.config.scaling_factor
        noise = torch.randn_like(model_input)
        bsz,channels,height,width = model_input.shape
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,(bsz,),device=model_input.device).long()

        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
        model_pred = unet(noisy_model_input, timesteps, text_embedding).sample

        if noise_scheduler.config.prediction_type=="epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type=="v_prediction":
            target = noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError("Unknown prediction type")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    LoraLoaderMixin.save_lora_weights(
        save_directory=save_lora_dir,
        unet_lora_layers=unet_lora_layers,
        text_encoder_lora_layers=None,
        weight_name=weight_name,
        safe_serialization=safe_serialization
    )

def load_lora(unet, lora_0, lora_1, alpha):
    lora = {}
    for key in lora_0:
        lora[key] = (1 - alpha)*lora_0[key] + alpha*lora_1[key]
    unet.load_attn_procs(lora)
    return unet