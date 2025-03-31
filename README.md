# DiffMorpher + LCM-LoRA: Optimized Keyframe Generator

This repository contains an optimized version of [DiffMorpher](https://github.com/Kevin-thu/DiffMorpher) with LCM-LoRA acceleration for efficient keyframe generation in image morphing pipelines.

## Overview

This keyframe generator serves as a critical component in a two-phase image morphing architecture. It creates high-quality intermediate frames between source and target images while maintaining semantic consistency, even when the images are semantically divergent. See upstream [repository](https://github.com/nalin0503/FYP_ImageMorpher) for further details.

## Key Features

- **LCM-LoRA Acceleration**: Integrates Latent Consistency Model with Low-Rank Adaptation to reduce sampling steps from 50 to 4-8 with minimal quality loss
- **Memory Optimization**: Implements various techniques including:
  - Lazy loading mechanism
  - VAE and attention slicing
  - PyTorch 2.0 optimizations
  - Explicit CUDA memory management
- **Performance**: 40-50% reduction in runtime compared to standard DiffMorpher (from ~170-185s to ~90-100s)
- **Multi-Model Support**: Compatible with Stable Diffusion 1.5, 2.1, and Dreamshaper-7

## Usage

```bash
python main.py \
  --model_path "stabilityai/stable-diffusion-2-1-base" \
  --image_path_0 "path/to/source.jpg" \
  --image_path_1 "path/to/target.jpg" \
  --prompt_0 "description of source image" \
  --prompt_1 "description of target image" \
  --output_path "./results" \
  --use_lcm  # Enable LCM-LoRA acceleration
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Pretrained model to use | stabilityai/stable-diffusion-2-1-base |
| `--image_path_0` | Path of the first image | "" |
| `--prompt_0` | Prompt for the first image | "" |
| `--image_path_1` | Path of the second image | "" |
| `--prompt_1` | Prompt for the second image | "" |
| `--output_path` | Output directory | ./results |
| `--num_inference_steps` | Number of inference steps | 50 (8 when LCM enabled) |
| `--guidance_scale` | Guidance scale | 1.0 |
| `--use_lcm` | Enable LCM-LoRA acceleration | False |
| `--num_frames` | Number of frames to generate | 16 |
| `--duration` | Duration of each frame (ms) | 100 |

## LCM-LoRA Integration

The implementation uses LCM-LoRA as a "universal acceleration module" while preserving the style adaptations from DiffMorpher:

1. Replaces DDIM scheduler with LCMScheduler
2. Loads LCM-LoRA weights on top of any style LoRA weights
3. Reduces inference steps while maintaining semantic consistency

## Acknowledgements

- [DiffMorpher](https://github.com/Kevin-thu/DiffMorpher) for the base implementation
- [Latent Consistency Models](https://github.com/luosiallen/latent-consistency-model) for the acceleration technique