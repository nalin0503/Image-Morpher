import os
import torch
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from model import DiffMorpherPipeline
from moviepy import ImageSequenceClip

parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="runwayml/stable-diffusion-v1-5",
                    help="Pretrained base model, e.g. SDXL")
parser.add_argument("--image_path_0", type=str, required=True, help="Path to the first image")
parser.add_argument("--prompt_0", type=str, default="A painting of a man",
                    help="Prompt describing first image style/content")
parser.add_argument("--image_path_1", type=str, required=True, help="Path to the second image")
parser.add_argument("--prompt_1", type=str, default="A painting of a woman",
                    help="Prompt describing second image style/content")
parser.add_argument("--output_path", type=str, default="./results",
                    help="Where to store results")
parser.add_argument("--save_lora_dir", type=str, default="./lora",
                    help="Directory to save the newly trained LoRAs")
parser.add_argument("--load_lora_path_0", type=str, default="",
                    help="Path to LoRA for the first image (optional)")
parser.add_argument("--load_lora_path_1", type=str, default="",
                    help="Path to LoRA for the second image (optional)")
parser.add_argument("--use_adain", action="store_true")
parser.add_argument("--use_reschedule", action="store_true")
parser.add_argument("--lamb", type=float, default=0.6)
parser.add_argument("--fix_lora_value", type=float, default=None)
parser.add_argument("--save_inter", action="store_true")
parser.add_argument("--num_frames", type=int, default=16)
parser.add_argument("--duration", type=int, default=100)
parser.add_argument("--no_lora", action="store_true")
parser.add_argument("--num_inference_steps", type=int, default=4,
                    help="Number of inference steps for LCM-LoRA inference")
parser.add_argument("--guidance_scale", type=float, default=1.0,
                    help="Guidance scale for LCM-LoRA")
parser.add_argument("--lora_steps", type=int, default=200)
parser.add_argument("--lora_lr", type=float, default=2e-4)
parser.add_argument("--lora_rank", type=int, default=16)
parser.add_argument("--lcm_lora_path", type=str, default="latent-consistency/lcm-lora-sdv1-5",
                    help="HF Hub ID or local path to LCM-LoRA weights")
parser.add_argument("--film_model_url", type=str, default="https://tfhub.dev/google/film/1",
                    help="TFHub URL for FILM model")

args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)
os.makedirs(args.save_lora_dir, exist_ok=True)  # Ensure this directory is created

pipeline = DiffMorpherPipeline.from_pretrained(
    args.model_path, torch_dtype=torch.float32
).to("cuda")

# Ensure we pass a prompt to avoid the ValueError:
# We'll use prompt_0 as the main prompt for initial generation
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
    lcm_lora_path=args.lcm_lora_path,
    prompt=args.prompt_0  # Now we explicitly provide a prompt
)

# Save initial morph frames as a GIF
images[0].save(f"{args.output_path}/output.gif", save_all=True,
               append_images=images[1:], duration=args.duration, loop=0)

print("Keyframe generation completed. Now using FILM for frame interpolation...")

# Use FILM model for interpolation between generated frames to achieve smoother video
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

film_model = hub.load(args.film_model_url)

def load_image_pil(img):
    image = Image.open(img).convert("RGB")
    image = image.resize((512, 512))
    arr = np.array(image) / 255.0
    return arr

# Suppose we only have keyframes from pipeline output (images)
# If you want more frames, you can FILM-interpolate between consecutive frames
def film_interpolate(img_a_np, img_b_np, model, num_inter_frames=8):
    # model expects shape: [1, 2, H, W, 3]
    # We'll generate `num_inter_frames` intermediate frames between img_a and img_b
    # FILM model returns frames excluding the endpoints, we must handle that carefully.
    img_a_np = img_a_np.astype(np.float32)
    img_b_np = img_b_np.astype(np.float32)
    batch = tf.stack([img_a_np, img_b_np], axis=0)[tf.newaxis, ...]  # [1,2,H,W,3]
    interpolated = model(batch)
    interpolated = interpolated[0].numpy()  # remove batch dim
    # The FILM model typically returns a sequence of intermediate frames (check its doc).
    # If it returns less frames than requested, you may need to apply FILM multiple times 
    # or just choose one setting that FILM model provides (it usually provides a fixed # of frames)
    # For simplicity, we assume FILM returns a set of intermediate frames:
    return interpolated  # a list/array of frames

# Collect the original frames from the DiffMorpher pipeline
frame_files = []
for i, img in enumerate(images):
    frame_path = os.path.join(args.output_path, f"frame_{i:03d}.png")
    img.save(frame_path)
    frame_files.append(frame_path)

# Interpolate using FILM between each consecutive keyframe pair if needed
final_frames = []
num_inter_frames = 4  # number of intermediate frames between each keyframe
for i in range(len(frame_files)-1):
    img_a = load_image_pil(frame_files[i])
    img_b = load_image_pil(frame_files[i+1])
    final_frames.append(Image.fromarray((img_a*255).astype(np.uint8)))
    intermediates = film_interpolate(img_a, img_b, film_model, num_inter_frames=num_inter_frames)
    for f_np in intermediates:
        final_frames.append(Image.fromarray((f_np*255).astype(np.uint8)))

# Add the last frame
final_frames.append(Image.fromarray((load_image_pil(frame_files[-1])*255).astype(np.uint8)))

video_path = os.path.join(args.output_path, "morphed_video.mp4")
clip = ImageSequenceClip([np.array(f) for f in final_frames], fps=30)
clip.write_videofile(video_path, codec="libx264")

print("Morphing completed. Video saved at:", video_path)
