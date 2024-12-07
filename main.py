import os
import torch
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from model import DiffMorpherPipeline

parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                    help="Base Stable Diffusion model (default: xl)")
parser.add_argument("--image_path_0", type=str, default="",
                    help="Path of the first image")
parser.add_argument("--prompt_0", type=str, default="",
                    help="Prompt describing the first image")
parser.add_argument("--image_path_1", type=str, default="",
                    help="Path of the second image")
parser.add_argument("--prompt_1", type=str, default="",
                    help="Prompt describing the second image")
parser.add_argument("--output_path", type=str, default="./results",
                    help="Output directory")
parser.add_argument("--save_lora_dir", type=str, default="./lora",
                    help="Directory to save trained LoRAs")
parser.add_argument("--load_lora_path_0", type=str, default="",
                    help="LoRA weights path for first image if already trained")
parser.add_argument("--load_lora_path_1", type=str, default="",
                    help="LoRA weights path for second image if already trained")
parser.add_argument("--use_adain", action="store_true",
                    help="Use AdaIN blending in latent space")
parser.add_argument("--use_reschedule", action="store_true",
                    help="Use alpha rescheduling after initial morph")
parser.add_argument("--lamb", type=float, default=0.6,
                    help="Lambda for self-attention replacement")
parser.add_argument("--fix_lora_value", type=float, default=None,
                    help="Fix LoRA interpolation value")
parser.add_argument("--save_inter", action="store_true",
                    help="Save intermediate frames")
parser.add_argument("--num_frames", type=int, default=16,
                    help="Number of keyframes to generate")
parser.add_argument("--duration", type=int, default=100,
                    help="Frame duration in ms for the GIF")
parser.add_argument("--no_lora", action="store_true",
                    help="Disable LoRA usage")
parser.add_argument("--num_inference_steps", type=int, default=4,
                    help="Number of inference steps with LCM-LoRA (~2-4 recommended)")
parser.add_argument("--guidance_scale", type=float, default=1.0,
                    help="LCM-LoRA recommended guidance ~1.0")
parser.add_argument("--lora_steps", type=int, default=200,
                    help="LoRA training steps if needed")
parser.add_argument("--lora_lr", type=float, default=2e-4,
                    help="LoRA training learning rate")
parser.add_argument("--lora_rank", type=int, default=16,
                    help="LoRA rank")
parser.add_argument("--lcm_lora_path", type=str, default="latent-consistency/lcm-lora-sdxl",
                    help="Path or HF Hub ID for LCM-LoRA weights")

args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)
pipeline = DiffMorpherPipeline.from_pretrained(
    args.model_path, torch_dtype=torch.float32
)
pipeline.to("cuda")

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
    fix_lora=args.fix_lora_value,
    save_intermediates=args.save_inter,
    use_lora=not args.no_lora,
    num_inference_steps=args.num_inference_steps,
    guidance_scale=args.guidance_scale,
    lora_steps=args.lora_steps,
    lora_lr=args.lora_lr,
    lora_rank=args.lora_rank,
    lcm_lora_path=args.lcm_lora_path
)

images[0].save(f"{args.output_path}/output.gif", save_all=True,
               append_images=images[1:], duration=args.duration, loop=0)

print("Morphing completed. Output GIF saved.")