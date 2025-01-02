import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from lora_utility import train_lora

from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    LCMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer


######################################
# Utility: Slerp & Linear Interp
######################################
def slerp(v0, v1, alpha):
    """
    Spherical linear interpolation for 1D or multi-D PyTorch tensors.
    Fallbacks to linear if vectors are nearly parallel.
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
    return (1.0 - alpha) * v0 + alpha * v1


######################################
# Custom LCM-based Inversion Pipeline
######################################
class LCMInversionPipeline(DiffusionPipeline):
    """
    A pipeline that:
      1. Loads a base SDXL model
      2. Uses an LCM scheduler for fewer steps
      3. Provides a method to invert an image to latents (DDIM Inversion),
         referencing your DiffMorpher code
      4. Provides a forward pass that supports negative_prompt for
         classifier-free guidance.
    """

    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler):
        super().__init__()
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler

        # If you're using SDXL, you might also have a second text encoder for
        # refiner or for large prompt embeddings. Adjust as needed.

    @torch.no_grad()
    def image2latent(self, image):
        """
        Convert image [-1..1] into latents for SDXL with factor 0.18215.
        If the input image is 0..255, first convert to [-1..1].
        """
        # If input is PIL, convert to torch:
        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image).unsqueeze(0)  # [1,3,H,W]
            image = 2.0 * image - 1.0  # scale to [-1..1]

        # If not, assume it's already [B,3,H,W], possibly in [-1..1].
        # Send to GPU
        image = image.to(self.device, dtype=torch.float32)

        # VAE encode
        latent_dist = self.vae.encode(image).latent_dist
        latents = latent_dist.mean
        latents = latents * 0.18215
        return latents.half()

    @torch.no_grad()
    def latent2image(self, latents):
        """
        Convert latents back to an image in [0..255] (uint8).
        """
        latents = latents / 0.18215
        image = self.vae.decode(latents.float()).sample  # in [-1..1]
        image = (image / 2 + 0.5).clamp(0,1).mul(255).type(torch.uint8)
        image = image.permute(0,2,3,1).cpu().numpy()
        # return as PIL
        pil_images = [Image.fromarray(img) for img in image]
        return pil_images

    ##################################
    # DiffMorpher-inspired DDIM Inversion
    ##################################
    def inv_step(self, model_output, timestep, x):
        """
        Inverse sampling for DDIM Inversion (from your snippet).
        model_output is the predicted noise at time 'timestep'.
        x is the current noisy latents.
        """
        next_step = timestep
        # step size
        step_size = self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        # example: if there are 1000 training timesteps and 50 steps, step_size=20

        # the next older step in the chain
        timestep = min(timestep - step_size, 999)

        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t

        # pred_x0 from x_t and noise
        pred_x0 = (x - (beta_prod_t**0.5)*model_output) / (alpha_prod_t**0.5)
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir

        return x_next, pred_x0

    @torch.no_grad()
    def invert(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps=50,
        guidance_scale=7.5,
    ):
        """
        Invert a real image into latents using deterministic DDIM inversion,
        referencing your DiffMorpher code. If negative_prompt is provided,
        we do classifier-free guidance in the noise prediction.
        """
        self.scheduler.set_timesteps(num_inference_steps)

        # 1) Convert image to latents
        latents = self.image2latent(image)  # [1, 4, H//8, W//8], half-precision
        batch_size = latents.shape[0]

        # 2) Encode prompts for CF guidance
        text_embeds = self._encode_for_inversion(prompt, negative_prompt, batch_size)

        # 3) DDIM Inversion Loop
        timesteps = list(reversed(self.scheduler.timesteps))
        for i, t in enumerate(tqdm(timesteps, desc="DDIM Inversion")):
            # Expand latents for CFG
            if guidance_scale > 1.0:
                model_input = torch.cat([latents] * 2, dim=0)
            else:
                model_input = latents

            # U-Net noise prediction
            with torch.autocast(self.device, dtype=torch.float16):
                noise_pred = self.unet(model_input, t, encoder_hidden_states=text_embeds).sample

            # Classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + guidance_scale*(noise_pred_text - noise_pred_uncond)

            # Inverse step
            latents, _pred_x0 = self.inv_step(noise_pred, t, latents)

        return latents  # latents corresponding to the original image after inversion

    def _encode_for_inversion(self, prompt, negative_prompt, batch_size=1):
        """
        Helper to produce text embeddings for classifier-free guidance.
        Using the main text_encoder. If you have a second text encoder (SDXL),
        you'd adapt accordingly. This is simplified for demonstration.
        """
        device = self.device
        # 1) Encode the main prompt
        text_input = self.tokenizer(
            [prompt]*batch_size,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]

        # 2) Negative prompt
        if guidance_scale := 7.5:  # i.e. if > 1
            if negative_prompt:
                neg_input = self.tokenizer(
                    [negative_prompt]*batch_size,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt"
                )
                neg_embeddings = self.text_encoder(neg_input.input_ids.to(device))[0]
            else:
                # an empty string for unconditional
                neg_input = self.tokenizer(
                    ["" for _ in range(batch_size)],
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt"
                )
                neg_embeddings = self.text_encoder(neg_input.input_ids.to(device))[0]

            # cat unconditional + conditional
            text_embeddings = torch.cat([neg_embeddings, text_embeddings], dim=0)

        return text_embeddings

    ###############################################
    # Forward Generation (with negative_prompt)
    ###############################################
    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        negative_prompt=None,
        latents=None,
        num_inference_steps=50,
        guidance_scale=7.5,
        height=512,
        width=512,
        **kwargs
    ):
        """
        Forward pass to generate images from latents, using negative_prompt for CF guidance.
        We assume LCM scheduler is set for fewer steps.
        If `latents` is not None, we start from those latents (like after we invert or do interpolation).
        If `prompt` is not None, we do the normal text encoder path for CFG.
        """
        self.scheduler.set_timesteps(num_inference_steps)
        device = self.device

        # If user provided latents, use them; else random
        if latents is None:
            # create random latents
            # (B,4,H//8,W//8) for SD. For XL base = 4 channels as well
            latents = torch.randn(
                (1, self.unet.config.in_channels, height//8, width//8),
                device=device, dtype=torch.float16
            )
        else:
            # shape check
            if latents.ndim == 3:
                latents = latents.unsqueeze(0)  # [1,4,H//8,W//8]
            latents = latents.to(device, dtype=torch.float16)

        # Encode text for CFG
        if prompt is None:
            prompt = ""
        text_embeddings = self._encode_for_inversion(prompt, negative_prompt, latents.shape[0])

        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Forward Sampling")):
            if guidance_scale > 1.0:
                model_input = torch.cat([latents]*2, dim=0)
            else:
                model_input = latents

            with torch.autocast(device_type=device, dtype=torch.float16):
                noise_pred = self.unet(
                    model_input, t, encoder_hidden_states=text_embeddings
                ).sample

            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + guidance_scale*(noise_pred_text - noise_pred_uncond)

            # Standard forward DDIM step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Convert final latents to images
        images = self.latent2image(latents)
        return images


############################################################
# Putting it all together: LCM + LoRAs + negative_prompt + interpolation
############################################################
def main():
    """
    1) Load base model
    2) Attach LCM scheduler
    3) Load LCM-LoRA and two style LoRAs
    4) Invert two real images to latents (DDIM)
    5) Interpolate latents, then forward sample with negative_prompt
    """
    device = "cuda"

    # (A) Load a base SDXL pipeline in half precision
    base_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = DiffusionPipeline.from_pretrained(
        base_id,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)
    # replace scheduler with LCM
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # Create our custom LCMInversionPipeline
    # (We extract sub-components from `pipe`.)
    custom_pipe = LCMInversionPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
    )
    custom_pipe.to(device)

    # (B) Load LoRAs (LCM + styleA + styleB)
    # Example:
    #   pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm")
    #   pipe.load_lora_weights("TheLastBen/Papercut_SDXL", adapter_name="styleA")
    #   pipe.load_lora_weights("SomeUser/AnotherStyleLoRA", adapter_name="styleB")
    #
    # Because we want them in the `unet`, we do it on `pipe`:
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm")
    # pipe.load_lora_weights("TheLastBen/Papercut_SDXL", adapter_name="styleA")
    pipe.load_lora_weights("path/to/saved_lora", adapter_name="style_lora_1")
    pipe.load_lora_weights("path/to/saved_lora", adapter_name="style_lora_2")


    # pipe.load_lora_weights("SomeUser/AnotherStyleLoRA", adapter_name="styleB")

    # Combine them. Keep LCM=1.0, styleA=0.0, styleB=1.0 for one image, etc.
    # We'll do dynamic re-weighting later
    # pipe.set_adapters(["lcm","styleA","styleB"], [1.0,1.0,0.0])

    # Ensure updated components are used in custom pipeline
    custom_pipe.unet = pipe.unet

    # (C) Suppose we have two real images: "imageA.jpg" and "imageB.jpg"
    # We'll invert each to latents
    # Insert your own paths or images
    image_path_a = "imageA.jpg"
    image_path_b = "imageB.jpg"
    if not os.path.exists(image_path_a) or not os.path.exists(image_path_b):
        # placeholder: handle case if images don't exist
        # for demonstration, we can skip or use random images
        print("WARNING: 'imageA.jpg' or 'imageB.jpg' not found. Using placeholder images.")
        # PSEUDOCODE: you can load placeholders from some URL or local fallback
        # imageA = ...
        # imageB = ...
        # (We won't break the code, just show the logic.)
        pass

    imageA = Image.open(image_path_a).convert("RGB").resize((512,512))
    imageB = Image.open(image_path_b).convert("RGB").resize((512,512))

    # Prompts for inversion. Typically the same descriptive prompt that might reconstruct the image
    promptA = "A photo of image A"
    promptB = "A photo of image B"
    negative_prompt_for_inversion = ""  # or something like "blurry, low quality"

    # invert to latents
    latentsA = custom_pipe.invert(
        image=imageA,
        prompt=promptA,
        negative_prompt=negative_prompt_for_inversion,
        num_inference_steps=50,
        guidance_scale=7.5,
    )
    latentsB = custom_pipe.invert(
        image=imageB,
        prompt=promptB,
        negative_prompt=negative_prompt_for_inversion,
        num_inference_steps=50,
        guidance_scale=7.5,
    )

    # (D) Interpolate latentsA -> latentsB (SLERP or linear)
    def latent_interp(a, b, alpha, use_slerp=True):
        if use_slerp:
            return slerp(a, b, alpha)
        else:
            return linear_interp(a, b, alpha)

    # (E) For each alpha, set LoRA weights & generate
    #     We'll do e.g. 10 frames
    frames = []
    num_frames = 10
    negative_prompt = "blurry, low quality, oversaturated"
    out_dir = "morph_results"
    os.makedirs(out_dir, exist_ok=True)

    for i in range(num_frames):
        alpha = i / (num_frames - 1) if num_frames>1 else 0.0

        # For style LoRAs, e.g. styleA=1-alpha, styleB=alpha
        wA = 1.0 - alpha
        wB = alpha
        pipe.set_adapters(["lcm","styleA","styleB"], [1.0, wA, wB])

        # Interpolate latents
        latentsT = latent_interp(latentsA, latentsB, alpha, use_slerp=True)

        # Now run forward pass with negative prompt (CF guidance)
        images = custom_pipe(
            prompt="some final creative prompt describing the morph",
            negative_prompt=negative_prompt,
            latents=latentsT,
            num_inference_steps=20,  # fewer steps thanks to LCM
            guidance_scale=7.5,
            height=512,
            width=512,
        )

        # We'll just store the first in batch
        frame = images[0]
        frame_path = os.path.join(out_dir, f"frame_{i:03d}.png")
        frame.save(frame_path)
        frames.append(frame)

    # (F) Optionally create a GIF
    frames[0].save(
        os.path.join(out_dir, "morph.gif"),
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0
    )
    print(f"Saved morph GIF at: {os.path.join(out_dir, 'morph.gif')}")


if __name__ == "__main__":
    main()
