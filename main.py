"""DiffMorpher with LCM-LoRA support, parameter additions, logging, and performance optimizations.

This script implements image morphing functionality using the DiffMorpher pipeline,
with additional support for LCM-LoRA acceleration and various optimizations.
"""

import os
import gc
import time
import logging
from typing import List, Optional, Tuple

import torch
import numpy as np
from PIL import Image
from argparse import ArgumentParser

from model import DiffMorpherPipeline

# Create logs directory
logs_folder = "keyframe_gen_logs"
os.makedirs(logs_folder, exist_ok=True)

# Create a unique log filename using the current time
log_filename = os.path.join(logs_folder, f"keyframegen_{time.strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

start_time = time.time()

parser = ArgumentParser()
parser.add_argument(
    "--model_path", 
    type=str, 
    default="stabilityai/stable-diffusion-2-1-base",
    help="Pretrained model to use (default: %(default)s)"
)
# Available SDV1-5 versions: 
# sd-legacy/stable-diffusion-v1-5
# lykon/dreamshaper-7

# Original DiffMorpher SD: 
# stabilityai/stable-diffusion-2-1-base

# Quantized models to try (non-functional, possible extension for future)
# DarkFlameUniverse/Stable-Diffusion-2-1-Base-8bit
# Xerox32/SD2.1-base-Int8

parser.add_argument(
    "--image_path_0", 
    type=str, 
    default="",
    help="Path of the first image (default: %(default)s)"
)
parser.add_argument(
    "--prompt_0", 
    type=str, 
    default="",
    help="Prompt of the first image (default: %(default)s)"
)
parser.add_argument(
    "--image_path_1", 
    type=str, 
    default="",
    help="Path of the second image (default: %(default)s)"
)
parser.add_argument(
    "--prompt_1", 
    type=str, 
    default="",
    help="Prompt of the second image (default: %(default)s)"
)
parser.add_argument(
    "--output_path", 
    type=str, 
    default="./results",
    help="Path of the output image (default: %(default)s)"
)
parser.add_argument(
    "--save_lora_dir", 
    type=str, 
    default="./lora",
    help="Path for saving LoRA weights (default: %(default)s)"
)
parser.add_argument(
    "--load_lora_path_0", 
    type=str, 
    default="",
    help="Path of the LoRA weights for the first image (default: %(default)s)"
)
parser.add_argument(
    "--load_lora_path_1", 
    type=str, 
    default="",
    help="Path of the LoRA weights for the second image (default: %(default)s)"
)
parser.add_argument(
    "--num_inference_steps", 
    type=int, 
    default=50, 
    help="Number of inference steps (default: %(default)s)"
)
parser.add_argument(
    "--guidance_scale", 
    type=float, 
    default=1,  # To match current diffmorpher
    help="Guidance scale for classifier-free guidance (default: %(default)s)"
)
parser.add_argument(
    "--use_adain", 
    action="store_true", 
    help="Use AdaIN (default: %(default)s)"
)
parser.add_argument(
    "--use_reschedule", 
    action="store_true", 
    help="Use reschedule sampling (default: %(default)s)"
)
parser.add_argument(
    "--lamb", 
    type=float, 
    default=0.6, 
    help="Lambda for self-attention replacement (default: %(default)s)"
)
parser.add_argument(
    "--fix_lora_value", 
    type=float, 
    default=None, 
    help="Fix lora value (default: LoRA Interp., not fixed)"
)
parser.add_argument(
    "--save_inter", 
    action="store_true", 
    help="Save intermediate results (default: %(default)s)"
)
parser.add_argument(
    "--num_frames", 
    type=int, 
    default=16, 
    help="Number of frames to generate (default: %(default)s)"
)
parser.add_argument(
    "--duration", 
    type=int, 
    default=100, 
    help="Duration of each frame (default: %(default)s ms)"
)
parser.add_argument(
    "--no_lora", 
    action="store_true", 
    help="Disable style LoRA (default: %(default)s)"
)
# New argument for LCM LoRA acceleration
parser.add_argument(
    "--use_lcm", 
    action="store_true", 
    help="Enable LCM-LoRA acceleration for faster sampling"
)

args = parser.parse_args()
os.makedirs(args.output_path, exist_ok=True)

# Set environment variable for memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def create_pipeline(model_path: str) -> DiffMorpherPipeline:
    """Create and configure the DiffMorpher pipeline.
    
    Args:
        model_path: Path to the pretrained model.
        
    Returns:
        A configured DiffMorpherPipeline ready for inference.
    """
    pipeline = DiffMorpherPipeline.from_pretrained(model_path, torch_dtype=torch.float32)
    
    # Memory optimizations for vae and attention slicing
    pipeline.enable_vae_slicing()
    pipeline.enable_attention_slicing()
    
    pipeline.to("cuda")
    
    # Performance optimizations
    torch.backends.cudnn.benchmark = True  # Find efficient convolution algorithms
    torch.set_float32_matmul_precision("high")  # Better for modern GPUs
    
    return pipeline


def apply_lcm_lora(pipeline: DiffMorpherPipeline) -> Tuple[DiffMorpherPipeline, int, float]:
    """Apply Latent Consistency Model LoRA for acceleration.
    
    Args:
        pipeline: The DiffMorpher pipeline to modify.
        
    Returns:
        Tuple containing:
            - Modified pipeline with LCM scheduler and weights
            - Recommended inference steps for LCM
            - Recommended guidance scale for LCM
    """
    from lcm_lora.lcm_schedule import LCMScheduler
    
    # Replace scheduler using LCM's configuration
    pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
    
    # Load the LCM LoRA weights
    pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    
    # LCM-recommended values
    inference_steps = 8
    guidance_scale = 1
    
    return pipeline, inference_steps, guidance_scale


def main() -> None:
    """Main execution function for DiffMorpher."""
    # Create the pipeline from the given model path
    pipeline = create_pipeline(args.model_path)
    
    # Integrate LCM-LoRA if flagged
    if args.use_lcm:
        pipeline, lcm_steps, lcm_guidance = apply_lcm_lora(pipeline)
        args.num_inference_steps = lcm_steps
        args.guidance_scale = lcm_guidance
    
    # Run the pipeline inference using existing parameters
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
        num_inference_steps=args.num_inference_steps,
        fix_lora=args.fix_lora_value,
        save_intermediates=args.save_inter,
        use_lora=not args.no_lora,
        use_lcm=args.use_lcm,
        guidance_scale=args.guidance_scale,
    )
    
    # Save the resulting GIF output from the sequence of images
    images[0].save(
        f"{args.output_path}/output.gif", 
        save_all=True,
        append_images=images[1:], 
        duration=args.duration, 
        loop=0
    )
    
    # Ensure memory is freed after completion
    pipeline = None
    torch.cuda.empty_cache()
    gc.collect()
    
    # Log execution details
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logging.info(f"Total execution time: {elapsed_time:.2f} seconds")
    logging.info(f"Model Path: {args.model_path}")
    logging.info(f"Image Path 0: {args.image_path_0}")
    logging.info(f"Image Path 1: {args.image_path_1}")
    logging.info(f"Use LCM: {args.use_lcm}")
    logging.info(f"Number of inference steps: {args.num_inference_steps}")
    logging.info(f"Guidance scale: {args.guidance_scale}")
    
    print(f"Total execution time: {elapsed_time:.2f} seconds, log file saved as {log_filename}")


if __name__ == "__main__":
    main()