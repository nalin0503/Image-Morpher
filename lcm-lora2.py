import torch
import numpy as np
from diffusers import DiffusionPipeline, LCMScheduler
from PIL import Image
from tqdm import tqdm

############################################
# Utility: Slerp for latents or embeddings
############################################
def slerp(v0, v1, alpha):
    """
    Spherical linear interpolation between two tensors.
    Example usage:
        slerp(latentsA, latentsB, alpha=0.5)
    """
    v0_np = v0.detach().cpu().numpy().astype(np.float32)
    v1_np = v1.detach().cpu().numpy().astype(np.float32)

    dot = (v0_np * v1_np).sum() / (np.linalg.norm(v0_np) * np.linalg.norm(v1_np))
    DOT_THRESHOLD = 0.9995

    if abs(dot) > DOT_THRESHOLD:
        # fallback to linear interpolation
        result = (1 - alpha) * v0_np + alpha * v1_np
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = alpha * theta_0
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        result = s0 * v0_np + s1 * v1_np
    return torch.from_numpy(result).to(v0.device, dtype=v0.dtype)

def linear_interp(v0, v1, alpha):
    """Simple linear interpolation."""
    return (1 - alpha) * v0 + alpha * v1

############################################
# Interpolate style LoRA weights + latents + text
############################################
def generate_triple_interp_keyframes(
    base_model_id: str,
    lcm_lora_id: str,
    style_lora_a: str,
    style_lora_b: str,
    prompt_a: str,
    prompt_b: str,
    negative_prompt: str = "",
    num_interpolation_steps: int = 10,
    latent_res: tuple = (64, 64),  # e.g. (height//8, width//8) for SDXL
    guidance_scale: float = 7.5,
    num_inference_steps: int = 8,
    device: str = "cuda",
    seed: int = 42,
    do_slerp_prompt: bool = True,
    do_slerp_latent: bool = True,
    do_slerp_style: bool = False,
    height: int = 768,
    width: int = 768,
    save_gif: bool = True,
    gif_path: str = "triple_interp.gif",
):
    """
    1) Load base SDXL with LCM scheduler
    2) Keep LCM-LoRA (acceleration vector) at weight=1.0
    3) Interpolate style LoRA #A <-> #B
    4) Interpolate text embeddings from prompt_a <-> prompt_b
    5) Interpolate latents (either from random or real image inversion)

    Returns a list of PIL images. Optionally saves as GIF.
    """

    torch.manual_seed(seed)

    # 1. Load pipeline with LCM scheduler
    pipe = DiffusionPipeline.from_pretrained(
        base_model_id,
        variant="fp16",
        torch_dtype=torch.float16,
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    # 2. Load LoRAs: LCM + styleA + styleB
    pipe.load_lora_weights(lcm_lora_id, adapter_name="lcm")
    pipe.load_lora_weights(style_lora_a, adapter_name="styleA")
    pipe.load_lora_weights(style_lora_b, adapter_name="styleB")

    # 3. Prepare random latents, shape = [2, unet_in_channels, latent_height, latent_width]
    #    or replace with real latents if you invert from real images
    generator = torch.Generator(device=device).manual_seed(seed)
    latent_shape = (
        2,
        pipe.unet.config.in_channels,
        latent_res[0],
        latent_res[1],
    )
    latents_pair = torch.randn(latent_shape, generator=generator, device=device, dtype=torch.float16)

    latent_0 = latents_pair[0]
    latent_1 = latents_pair[1]

    # 4. Encode text embeddings for prompt A & prompt B (no classifier-free guidance here,
    #    we'll rely on negative_prompt argument. But if you want manual CFG embeddings,
    #    you can do so in a more advanced approach.)
    #    We'll do an unconditional version under the hood. So for now, do_classifier_free_guidance=False.
    #    Then pipe(...) with negative_prompt=... will handle negative embeddings automatically.
    text_embeds, neg_embeds, pooled_embeds, neg_pooled_embeds = pipe.encode_prompt(
        [prompt_a, prompt_b],
        do_classifier_free_guidance=False,
    )

    text_embed_0 = text_embeds[0]
    text_embed_1 = text_embeds[1]
    pooled_embed_0 = pooled_embeds[0]
    pooled_embed_1 = pooled_embeds[1]

    # Utility to get the alpha-blended text embeddings
    def get_text_embeds(alpha):
        if do_slerp_prompt:
            # spherical
            main_text = slerp(text_embed_0, text_embed_1, alpha)
            main_pooled = slerp(pooled_embed_0, pooled_embed_1, alpha)
        else:
            # linear
            main_text = linear_interp(text_embed_0, text_embed_1, alpha)
            main_pooled = linear_interp(pooled_embed_0, pooled_embed_1, alpha)

        # shape must be [1, ...]
        main_text = main_text.unsqueeze(0)
        main_pooled = main_pooled.unsqueeze(0)
        return main_text, main_pooled

    # 5. Generate frames by triple interpolation
    frames = []
    for i in tqdm(range(num_interpolation_steps), desc="Generating frames"):
        alpha = i / (num_interpolation_steps - 1) if num_interpolation_steps > 1 else 0.0

        # 5a. Interpolate LoRA style weights (styleA vs styleB)
        # keep LCM at 1.0
        # For style: wA=alpha, wB=1-alpha (linear). 
        if do_slerp_style:
            # naive approach: we rarely do true spherical interpolation on adapter weights,
            # but you could if you had more advanced merges. We'll do linear here:
            wA = alpha
            wB = 1.0 - alpha
        else:
            # also linear
            wA = alpha
            wB = 1.0 - alpha

        pipe.set_adapters(["lcm", "styleA", "styleB"], [1.0, wA, wB])

        # 5b. Interpolate latents
        if do_slerp_latent:
            # spherical
            latent_t = slerp(latent_0, latent_1, alpha)
        else:
            # linear
            latent_t = linear_interp(latent_0, latent_1, alpha)

        # 5c. Interpolate text embeddings
        main_text_embeds, main_pooled_embeds = get_text_embeds(alpha)

        # 5d. Run the pipeline with these latents, text embeddings, style weights
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = pipe(
                prompt_embeds=main_text_embeds,       # custom text embeddings
                pooled_prompt_embeds=main_pooled_embeds,
                negative_prompt=negative_prompt,       # let pipeline handle negative CFG
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                latents=latent_t.unsqueeze(0),
                generator=generator,
            )
        image = out.images[0]
        frames.append(image)

    # 6. (Optional) Save frames as a GIF
    if save_gif and len(frames) > 1:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=150,  # ms per frame
            loop=0
        )
        print(f"Saved triple interpolation GIF to: {gif_path}")

    return frames


############################################
# Example usage
############################################
if __name__ == "__main__":
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"
    lcm_lora_id = "latent-consistency/lcm-lora-sdxl"  # universal "acceleration vector"
    style_lora_a = "TheLastBen/Papercut_SDXL"
    style_lora_b = "SomeUser/AnotherStyleLoRA"  # Replace with an actual LoRA ID or local path

    prompt_a = "A pink papercut elephant in a whimsical garden"
    prompt_b = "A futuristic swirling planet in deep space, trending on artstation"
    negative_prompt = "blurry, low quality, oversaturated"

    # Generate 12 frames total, each at 512x512 latent resolution is 64x64
    frames = generate_triple_interp_keyframes(
        base_model_id=base_model,
        lcm_lora_id=lcm_lora_id,
        style_lora_a=style_lora_a,
        style_lora_b=style_lora_b,
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        negative_prompt=negative_prompt,
        num_interpolation_steps=12,
        latent_res=(64, 64),  # 512/8=64 for SDXL
        guidance_scale=5.0,
        num_inference_steps=6,  # thanks to LCM, we can use fewer steps
        device="cuda",
        seed=42,
        do_slerp_prompt=True,   # SLERP text
        do_slerp_latent=True,   # SLERP latents
        do_slerp_style=False,   # style LoRA weights: linear blend
        height=512,
        width=512,
        save_gif=True,
        gif_path="triple_interp.gif",
    )

    print("Done! Keyframes ready for further interpolation (e.g., FILM).")
