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
from diffusers.models.attention_processor import AttnProcessor, LoRAAttnProcessor
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
    noise_scheduler=None, lora_steps=200, lora_lr=2e-4, lora_rank=16, weight_name=None, safe_serialization=False, progress=tqdm
):
    accelerator = Accelerator(gradient_accumulation_steps=1)
    set_seed(0)

    if tokenizer is None:
        from transformers import AutoTokenizer
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

    print("DEBUG: Printing all UNet parameters before replacement:")
    for n, p in unet.named_parameters():
        print("UNet param:", n)

    # Replace all AttnProcessors with LoRAAttnProcessor (no longer checking 'attn2' in name)
    replaced_count = 0
    for name, module in unet.named_modules():
        if isinstance(module, AttnProcessor):
            parent = module.parent
            new_processor = LoRAAttnProcessor()
            setattr(parent, module.name, new_processor)
            replaced_count += 1

    print(f"DEBUG: Replaced {replaced_count} AttnProcessors with LoRAAttnProcessor.")
    lora_params = []
    found_any = False

    # Try to find LoRA parameters by substring matching
    for name, module in unet.named_modules():
        if isinstance(module, LoRAAttnProcessor):
            attn_parent = module.parent
            parent_params = dict(attn_parent.named_parameters())
            print(f"DEBUG: LoRAAttnProcessor at {name}, parent: {attn_parent.__class__.__name__}")
            print("DEBUG: Parent parameters keys:", list(parent_params.keys()))

            proj_keys = ["to_q", "to_k", "to_v", "to_out", "proj_in", "proj_out", "conv1", "conv2", "time_emb_proj"]
            module.lora_layers = {}

            # Loop through each projection key and search parent_params keys
            for pk in proj_keys:
                matched_keys = [pn for pn in parent_params.keys() if pk in pn]
                if len(matched_keys) > 0:
                    # Normally, we expect one main parameter per proj key (e.g. to_q)
                    # but stable diffusion might structure them as to_q.base_layer.weight, etc.
                    # We will handle the main weight
                    for matched_param in matched_keys:
                        p = parent_params[matched_param]
                        if p.ndim == 2:  # only create LoRA for linear weights
                            out_dim, in_dim = p.shape
                            lora_A = torch.nn.Parameter(torch.zeros((out_dim, lora_rank), device=device))
                            lora_B = torch.nn.Parameter(torch.zeros((lora_rank, in_dim), device=device))
                            torch.nn.init.normal_(lora_A, mean=0.0, std=0.01)
                            torch.nn.init.normal_(lora_B, mean=0.0, std=0.01)
                            # Create a unique key_base by removing everything after a dot
                            # except the main proj name
                            # e.g. "attn1.to_q.base_layer.weight" -> key_base = "to_q"
                            # We'll just reuse pk as the key_base
                            key_base = pk
                            module.lora_layers[key_base] = (lora_A, lora_B)
                            lora_params.append(lora_A)
                            lora_params.append(lora_B)
                            found_any = True
                            # Assuming one set per key, break after first match
                            break

    if not found_any:
        print("DEBUG: No LoRA parameters found with current proj_keys.")
        print("DEBUG: Inspect the printed parameter keys above and adjust code if needed.")
        raise ValueError("No LoRA parameters found. Check naming conventions or update code to match your model.")

    optimizer = torch.optim.AdamW(lora_params, lr=lora_lr, betas=(0.9,0.999), weight_decay=1e-2, eps=1e-08)
    lr_scheduler = get_scheduler("constant", optimizer=optimizer, num_warmup_steps=0, num_training_steps=lora_steps)

    unet = accelerator.prepare_model(unet)
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
        bsz, channels, height, width = model_input.shape
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,),
            device=model_input.device
        ).long()

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

    # Save LoRA weights
    lora_state = {}
    for name, module in unet.named_modules():
        if isinstance(module, LoRAAttnProcessor) and hasattr(module, 'lora_layers'):
            for proj, (A,B) in module.lora_layers.items():
                A_key = f"{name}.{proj}.lora_A.weight"
                B_key = f"{name}.{proj}.lora_B.weight"
                lora_state[A_key] = A.detach().cpu()
                lora_state[B_key] = B.detach().cpu()

    out_path = os.path.join(save_lora_dir, weight_name)
    if safe_serialization:
        safetensors.torch.save_file(lora_state, out_path+".safetensors")
    else:
        torch.save(lora_state, out_path)

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
        if isinstance(module, LoRAAttnProcessor) and hasattr(module, 'lora_layers'):
            for proj in list(module.lora_layers.keys()):
                A_key = f"{name}.{proj}.lora_A.weight"
                B_key = f"{name}.{proj}.lora_B.weight"
                if A_key in combined and B_key in combined:
                    A = combined[A_key].to(module.lora_layers[proj][0].device)
                    B = combined[B_key].to(module.lora_layers[proj][1].device)
                    module.lora_layers[proj] = (torch.nn.Parameter(A), torch.nn.Parameter(B))
    return unet