import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    LCMScheduler,
    StableDiffusionXLPipeline,
)
from diffusers.utils.torch_utils import randn_tensor
import safetensors
from transformers import CLIPTextModel, CLIPTokenizer

############################################
# Utilities: SLERP and Linear Interpolation
############################################
def slerp(v0, v1, alpha):
    """
    Spherical linear interpolation for vectors v0, v1
    See: https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355
    """
    v0 = v0.detach().cpu().numpy().astype(np.float32)
    v1 = v1.detach().cpu().numpy().astype(np.float32)
    dot = np.sum(v0 * v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))
    DOT_THRESHOLD = 0.9995

    if np.abs(dot) > DOT_THRESHOLD:
        # Vectors are close, do a linear interpolation
        result = (1 - alpha) * v0 + alpha * v1
    else:
        # Spherical interpolation
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = alpha * theta_0
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        result = s0 * v0 + s1 * v1
    return torch.from_numpy(result)

def linear_interp(v0, v1, alpha):
    return (1.0 - alpha) * v0 + alpha * v1

############################################
# Utilities: Load & Combine LoRAs in Diffusers
############################################
def load_lora_weights(pipeline, lora_path, adapter_name="lora"):
    """
    Loads LoRA weights (potentially .safetensors) into the pipeline as a named adapter.
    """
    # In diffusers: pipeline.load_lora_weights(checkpoint_or_dir, adapter_name=...)
    pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)

def set_lora_weights(pipeline, lora_adapters, lora_weights):
    """
    lora_adapters: list of adapter names, e.g. ["lcm", "styleA", "styleB"]
    lora_weights: list of floats, same length as lora_adapters, e.g. [1.0, 0.6, 0.4]
    """
    pipeline.set_adapters(lora_adapters, adapter_weights=lora_weights)

############################################
# Training a Style LoRA (placeholder logic)
############################################
def train_style_lora(
    image,
    style_prompt,
    base_model_id="stabilityai/stable-diffusion-xl-base-1.0",
    output_path="./my_style_lora.safetensors",
    steps=200,
    lr=2e-4,
    rank=16
):
    """
    Here you would implement or call your training routine to fine-tune a LoRA on a single image
    or dataset. This is a placeholder function since you already have a method in your code
    (train_lora) that does something similar.

    For your real pipeline, use your existing function to train and save the LoRA weights.
    Return the path to your newly created LoRA file.
    """
    # Pseudocode:
    # 1. Load pipeline
    # 2. Prepare dataset from 'image'
    # 3. LoRA train procedure
    # 4. Save .safetensors

    # We'll just pretend it's done:
    print(f"Training a style LoRA on the image with prompt: '{style_prompt}' ... (mock)")
    # ... do training ...
    # save at output_path
    print(f"LoRA saved to {output_path}")
    return output_path

############################################
# Interpolate Between Two Style LoRAs + LCM
############################################
def generate_interpolated_keyframes(
    base_model_id,
    lcm_lora_id,
    style_lora_a,
    style_lora_b,
    prompt_a,
    prompt_b,
    negative_prompt="",
    num_inference_steps=8,
    guidance_scale=7.5,
    num_interpolation_steps=6,
    height=768,
    width=768,
    device="cuda",
    seed=42,
    do_slerp_prompt=True,
    do_slerp_style=False,
    save_gif=True,
    gif_path="morph_style.gif",
):
    """
    1) Loads base SDXL
    2) Attaches LCM-LoRA (acceleration vector) at weight=1.0
    3) Attaches style LoRA #A and #B, and interpolates between them
    4) Optionally does prompt embedding slerp from prompt A to prompt B
    5) Renders each frame, returns a list of PIL.Image
    """

    torch.manual_seed(seed)

    # -- 1. Load Pipeline
    pipe = DiffusionPipeline.from_pretrained(
        base_model_id, variant="fp16", torch_dtype=torch.float16
    )
    # Replace scheduler with LCM
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    pipe.to(device)

    # -- 2. Load LCM LoRA & keep it at weight=1.0
    load_lora_weights(pipe, lcm_lora_id, adapter_name="lcm")

    # Load style LoRAs
    load_lora_weights(pipe, style_lora_a, adapter_name="styleA")
    load_lora_weights(pipe, style_lora_b, adapter_name="styleB")

    # -- 3. Encode text embeddings for prompt A / B
    #       We'll do optional slerp in text-embedding space
    #       or linear interpolation if do_slerp_prompt=False
    text_embeds_a, text_embeds_b, pooled_a, pooled_b = pipe.encode_prompt(
        [prompt_a, prompt_b],
        do_classifier_free_guidance=False
    )

    # negative prompt
    neg_embeds, _, neg_pooled, _ = pipe.encode_prompt(
        [negative_prompt], do_classifier_free_guidance=False
    )

    # Because we might do CFG, we also need the unconditional version
    # We'll handle that inside pipe if we pass negative_prompt=..., but
    # if you want manual control of embeddings, do it here.

    # We'll create a function to get a range of prompt embeddings:
    def interpolate_text_embeddings(alpha):
        if do_slerp_prompt:
            # Use spherical interpolation
            main_embed = slerp(text_embeds_a, text_embeds_b, alpha)
            pooled_embed = slerp(pooled_a, pooled_b, alpha)
        else:
            # Linear
            main_embed = linear_interp(text_embeds_a, text_embeds_b, alpha)
            pooled_embed = linear_interp(pooled_a, pooled_b, alpha)

        return main_embed.unsqueeze(0), pooled_embed.unsqueeze(0)

    # -- 4. Create frames
    frames = []

    for i in tqdm(range(num_interpolation_steps), desc="Interpolating style & prompt"):
        alpha = i / (num_interpolation_steps - 1) if num_interpolation_steps > 1 else 0.0

        # Interpolate style LoRA weights
        if do_slerp_style:
            # Slerp style weights... (some advanced users try "model merging" at weight-level,
            # but we only have a linear adapter interface in diffusers. 
            # We'll do a linear interpolation of LoRA adapter weights at runtime.)
            wA = alpha
            wB = (1.0 - alpha)
        else:
            # Linear approach
            wA = alpha
            wB = (1.0 - alpha)

        # lcm always 1.0
        # styleA=alpha, styleB=(1-alpha)
        set_lora_weights(pipe, ["lcm", "styleA", "styleB"], [1.0, wA, wB])

        # Interpolate text embeddings
        main_embed, pooled_embed = interpolate_text_embeddings(alpha)

        # Run pipeline
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            image = pipe(
                prompt_embeds=main_embed,
                pooled_prompt_embeds=pooled_embed,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            ).images[0]

        frames.append(image)

    # -- 5. Optionally save as GIF
    if save_gif and len(frames) > 1:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=200,
            loop=0
        )
        print(f"Saved style morph GIF to: {gif_path}")

    return frames

############################################
# Using DDIM Inversion for Real Images
############################################
def ddim_inversion_example(
    real_image_path,
    base_model_id="stabilityai/stable-diffusion-xl-base-1.0",
    lcm_lora_id="latent-consistency/lcm-lora-sdxl",
    style_lora_a="TheLastBen/Papercut_SDXL",
    style_lora_b="SomeUser/AnotherStyleLoRA",
    invert_prompt="a photo of a landscape",  # for demonstration
    negative_prompt="",
    steps=50,
    guidance_scale=1.0,
):
    """
    Demonstrates how you can do a quick DDIM inversion with LCM-loRA pipeline.
    Then you could morph that latent with another latent, or do style interpolation
    from a real image as your starting point. The code is approximate and might
    need refinement for your exact DiffMorpher approach.
    """
    # 1. Load pipeline with LCM scheduler
    pipe = DiffusionPipeline.from_pretrained(
        base_model_id, variant="fp16", torch_dtype=torch.float16
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    # 2. Load LoRAs
    load_lora_weights(pipe, lcm_lora_id, "lcm")
    load_lora_weights(pipe, style_lora_a, "styleA")
    load_lora_weights(pipe, style_lora_b, "styleB")

    # 3. Inversion
    # For a robust approach, consider using a specialized pipeline or your
    # custom code from DiffMorpher. The snippet below is just a placeholder.
    image = Image.open(real_image_path).convert("RGB").resize((512, 512))
    image_tensor = transforms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
    image_tensor = image_tensor.to("cuda", dtype=torch.float16)

    # We must pass through the VAE to get latents
    latents = pipe.vae.encode(image_tensor).latent_dist.mean
    latents = 0.18215 * latents  # scaling factor used by SD

    # 4. Perform some steps of “inverse” DDIM if needed
    # The DiffMorpher approach does something like:
    #   for t in reversed(pipe.scheduler.timesteps):
    #       predict noise, step, etc.
    # We'll skip the full details as you already have them.

    # 5. Now `latents` should approximate the noise you started from.
    # You can pass these latents directly to pipe(..., latents=latents)
    # or do an interpolation with another latent from a second image.

    print("DDIM inversion completed (placeholder).")
    return latents

############################################
# Example Main
############################################
if __name__ == "__main__":
    # --------------------------------------------------------------
    # 1) Suppose you have 2 images: styleA.jpg and styleB.jpg
    #    You train 2 LoRAs:
    # --------------------------------------------------------------
    # styleA_lora = train_style_lora("styleA.jpg", "A cute papercut elephant")
    # styleB_lora = train_style_lora("styleB.jpg", "A stylized swirling planet")
    #
    # For demo, we’ll just use existing LoRAs on HF:
    styleA_lora = "TheLastBen/Papercut_SDXL"
    styleB_lora = "SomeUser/AnotherStyleLoRA"  # replace with your real ID or local file

    # --------------------------------------------------------------
    # 2) We have an LCM-LoRA for acceleration
    # --------------------------------------------------------------
    lcm_lora_id = "latent-consistency/lcm-lora-sdxl"

    # --------------------------------------------------------------
    # 3) Interpolate prompts
    # --------------------------------------------------------------
    prompt_a = "A pink papercut elephant playing in a garden"
    prompt_b = "A swirling cosmic planet swirling in a galaxy, vibrant colors"

    # Negative prompt
    negative_prompt = "blurry, low quality, oversaturated"

    # --------------------------------------------------------------
    # 4) Actually generate interpolation frames
    # --------------------------------------------------------------
    frames = generate_interpolated_keyframes(
        base_model_id="stabilityai/stable-diffusion-xl-base-1.0",
        lcm_lora_id=lcm_lora_id,
        style_lora_a=styleA_lora,
        style_lora_b=styleB_lora,
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        negative_prompt=negative_prompt,
        num_inference_steps=6,   # very few steps, thanks to LCM
        guidance_scale=5.0,
        num_interpolation_steps=8,  # how many frames in the morph
        height=768,
        width=768,
        device="cuda",
        seed=123,
        do_slerp_prompt=True,   # use spherical interpolation for text embeddings
        do_slerp_style=False,   # style LoRA interpolation can be linear
        save_gif=True,
        gif_path="morph_style.gif",
    )

    # Now 'frames' is a list of PIL.Images for your keyframes.
    # You can feed them into FILM or another frame interpolation method to get
    # a smoother, higher-framerate final video.

    print("Done! Keyframes generated.")
