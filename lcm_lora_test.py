import torch
# from diffusers import AutoPipelineForText2Image
from lcm_schedule import LCMScheduler
from diffusers import DiffusionPipeline

# Load the SD v1.5 base model
pipe = DiffusionPipeline.from_pretrained(
    # "runwayml/stable-diffusion-v1-5",
    # sd-legacy/stable-diffusion-v1-5"
    # lykon/dreamshaper-7  # seems to produce better results, its a fine-tuned sd1-5
    "lykon/dreamshaper-7", 
    torch_dtype=torch.float16,
    # variant="fp16", # Removing variant since torch_dtype is set
).to("cuda")

# Set the LCMScheduler
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# Load LCM LoRA for SD v1.5
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.fuse_lora() #  Stable Diffusion Pipeline above does not have this... diffusers 0.23 does!

# Ensure the pipeline is on CUDA with the proper dtype
# pipe.to(device="cuda", dtype=torch.float16) # Removed device argument

prompt = "A hyperrealistic portrait of a cat, highly detailed, cute pet"
negative_prompt = "blurry, low quality"
generator = torch.manual_seed(0)  # for reproducibility

# Run inference (using very few steps as typical for LCM accelerated inference)
image = pipe(prompt, 
             negative_prompt=negative_prompt, # this works well for dreamshaper ! best output is with this, num_inference = 8, and guidance =0 on dreamshaper
             num_inference_steps=8, # 8 seems to work best for both... but finetune it. 
             guidance_scale=0, # the dreamshaper model works better with guidance_scale = 0!! disabled essentially. 
             generator=generator).images[0]

# Save the resulting image to disk
image.save("lcm_accelerated_cat_dreamshaper.png")














# model_id = "Lykon/dreamshaper-7"
# adapter_id = "latent-consistency/lcm-lora-sdv1-5"

# pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
# pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
# pipe.to("cuda")

# # load and fuse lcm lora
# pipe.load_lora_weights(adapter_id)
# pipe.fuse_lora()


# prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

# # disable guidance_scale by passing 0
# image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0).images[0]
