# main.py
import os
import torch
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from model import DiffMorpherPipeline

parser = ArgumentParser()
parser.add_argument(
    "--model_path", type=str, default="lykon/dreamshaper-7",
    help="Pretrained model to use (default: %(default)s)"
)
# Available SDV1-5 versions (all have lcm-lora support): 
# sd-legacy/stable-diffusion-v1-5
# lykon/dreamshaper-7
# "runwayml/stable-diffusion-v1-5"

# Original DiffMorpher SD (no lcm-lora support): 
# stabilityai/stable-diffusion-2-1-base

parser.add_argument(
    "--image_path_0", type=str, default="",
    help="Path of the first image (default: %(default)s)"
)
parser.add_argument(
    "--prompt_0", type=str, default="",
    help="Prompt of the first image (default: %(default)s)"
)
parser.add_argument(
    "--image_path_1", type=str, default="",
    help="Path of the second image (default: %(default)s)"
)
parser.add_argument(
    "--prompt_1", type=str, default="",
    help="Prompt of the second image (default: %(default)s)"
)
parser.add_argument(
    "--output_path", type=str, default="./results",
    help="Path of the output image (default: %(default)s)"
)
parser.add_argument(
    "--save_lora_dir", type=str, default="./lora",
    help="Path for saving LoRA weights (default: %(default)s)"
)
parser.add_argument(
    "--load_lora_path_0", type=str, default="",
    help="Path of the LoRA weights for the first image (default: %(default)s)"
)
parser.add_argument(
    "--load_lora_path_1", type=str, default="",
    help="Path of the LoRA weights for the second image (default: %(default)s)"
)
parser.add_argument("--use_adain", action="store_true", help="Use AdaIN (default: %(default)s)")
parser.add_argument("--use_reschedule",  action="store_true", help="Use reschedule sampling (default: %(default)s)")
parser.add_argument("--lamb",  type=float, default=0.6, help="Lambda for self-attention replacement (default: %(default)s)")
parser.add_argument("--fix_lora_value", type=float, default=None, help="Fix lora value (default: LoRA Interp., not fixed)")
parser.add_argument("--save_inter", action="store_true", help="Save intermediate results (default: %(default)s)")
parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to generate (default: %(default)s)")
parser.add_argument("--duration", type=int, default=100, help="Duration of each frame (default: %(default)s ms)")
parser.add_argument("--no_lora", action="store_true", help="Disable style LoRA (default: %(default)s)")

# New argument for LCMS LoRA acceleration
parser.add_argument("--use_lcm", action="store_true", help="Enable LCM-LoRA acceleration for faster sampling")

args = parser.parse_args()
os.makedirs(args.output_path, exist_ok=True)

# Create the pipeline from the given model path
pipeline = DiffMorpherPipeline.from_pretrained(args.model_path, torch_dtype=torch.float32)
pipeline.to("cuda") 

# Integrate LCMS-LORA if flagged, OUTSIDE any of the style LoRA loading / training steps.
if args.use_lcm:
    from lcm_schedule import LCMScheduler
    # Replace scheduler using LCMS's configuration
    pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
    # Load the LCMS LoRA weights (LCMS provides an add-on network; use your local path or default)
    pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5") ## This is working correctly! 
    # Set the lcm_inference_steps
    args.num_inference_steps = 8  # Override with LCM-recommended steps

# Run the pipeline inference using existing parameters
# Note to self: pipeline is a callable class.
images = pipeline(
    img_path_0=args.image_path_0,
    img_path_1=args.image_path_1,
    prompt_0=args.prompt_0,
    prompt_1=args.prompt_1,
    save_lora_dir=args.save_lora_dir,
    load_lora_path_0=args.load_lora_path_0,
    load_lora_path_1=args.load_lora_path_1,
    use_adain=args.use_adain,
    use_reschedule=args.use_reschedule,
    lamd=args.lamb,
    output_path=args.output_path,
    num_frames=args.num_frames,
    num_inference_steps = args.num_inference_steps, # enforce when LCM enabled
    fix_lora=args.fix_lora_value,
    save_intermediates=args.save_inter,
    use_lora=not args.no_lora
)

# Save the resulting GIF output from the sequence of images
images[0].save(f"{args.output_path}/output.gif", save_all=True,
               append_images=images[1:], duration=args.duration, loop=0)