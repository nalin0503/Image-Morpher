import os
import torch
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from model import DiffMorpherPipeline

parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="runwayml/stable-diffusion-v1-5")
parser.add_argument("--image_path_0", type=str, default="")
parser.add_argument("--prompt_0", type=str, default="")
parser.add_argument("--image_path_1", type=str, default="")
parser.add_argument("--prompt_1", type=str, default="")
parser.add_argument("--output_path", type=str, default="./results")
parser.add_argument("--save_lora_dir", type=str, default="./lora")
parser.add_argument("--load_lora_path_0", type=str, default="")
parser.add_argument("--load_lora_path_1", type=str, default="")
parser.add_argument("--use_adain", action="store_true")
parser.add_argument("--use_reschedule", action="store_true")
parser.add_argument("--lamb", type=float, default=0.6)
parser.add_argument("--fix_lora_value", type=float, default=None)
parser.add_argument("--save_inter", action="store_true")
parser.add_argument("--num_frames", type=int, default=16)
parser.add_argument("--duration", type=int, default=100)
parser.add_argument("--no_lora", action="store_true")
parser.add_argument("--num_inference_steps", type=int, default=4)
parser.add_argument("--guidance_scale", type=float, default=1.0)
parser.add_argument("--lora_steps", type=int, default=200)
parser.add_argument("--lora_lr", type=float, default=2e-4)
parser.add_argument("--lora_rank", type=int, default=16)
parser.add_argument("--lcm_lora_path", type=str, default="latent-consistency/lcm-lora-sdv1-5")

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