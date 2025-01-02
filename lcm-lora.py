import torch
from diffusers import DiffusionPipeline, LCMScheduler

pipe = DiffusionPipeline.from_pretrained( # using diffusionpipeline here, not the latentconsisstency one #TODO
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    torch_dtype=torch.float16
).to("cuda")

# Set scheduler
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# LoRA parameters obtained through LCM-LoRA training ('acceleration vector')
# directly combined with other LoRA parameters ('style vector') obtained by
# fine-tuning on a particular style dataset.

# Load LoRAs
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm")
pipe.load_lora_weights("TheLastBen/Papercut_SDXL", weight_name="papercut.safetensors", adapter_name="papercut")

# Combine LoRAs
pipe.set_adapters(["lcm", "papercut"], adapter_weights=[1.0, 0.8])

pipe.to(device="cuda", dtype=torch.float16)

prompt = "papercut, a cute fox"
negative_prompt = "blurry, low quality, oversaturated, random, incoherent" # huh! so you can have a negative prompt too! #TODO
# maybe this can increase the quality of the image as well. 
generator = torch.manual_seed(0) # what does this do? should just be controlling the randomness for reproducibility... yep
image = pipe(prompt, negative_prompt = negative_prompt,
              num_inference_steps=4, guidance_scale=1, generator=generator).images[0]
image

# what is the generator keyword used for? why is it not used in the standard sdxl here --? 

# The generator parameter, initialized with a fixed seed using torch.manual_seed, 
# is a tool for controlling randomness in your diffusion pipeline. 
# While not always necessary in standard examples, 
# it becomes crucial when reproducibility and consistency are required. 