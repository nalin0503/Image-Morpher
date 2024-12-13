import os
import torch
import torch.nn.functional as F
import tqdm
import numpy as np
import safetensors
from PIL import Image
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers import (StableDiffusionPipeline, LCMScheduler, 
                       AutoencoderKL, UNet2DConditionModel)
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from argparse import ArgumentParser
import inspect

from utils.model_utils import get_img, slerp, do_replace_attn
from utils.lora_utils import train_lora, load_lora
from utils.alpha_scheduler import AlphaScheduler


class StoreProcessor():
    def __init__(self, original_processor, value_dict, name):
        self.original_processor = original_processor
        self.value_dict = value_dict
        self.name = name
        self.value_dict[self.name] = dict()
        self.id = 0

    def __call__(self, attn, hidden_states, *args, encoder_hidden_states=None, attention_mask=None, **kwargs):
        if encoder_hidden_states is None:
            self.value_dict[self.name][self.id] = hidden_states.detach()
            self.id += 1
        return self.original_processor(attn, hidden_states, *args,
                                       encoder_hidden_states=encoder_hidden_states,
                                       attention_mask=attention_mask,
                                       **kwargs)


class LoadProcessor():
    def __init__(self, original_processor, name, img0_dict, img1_dict, alpha, beta=0, lamd=0.6):
        super().__init__()
        self.original_processor = original_processor
        self.name = name
        self.img0_dict = img0_dict
        self.img1_dict = img1_dict
        self.alpha = alpha
        self.beta = beta
        self.lamd = lamd
        self.id = 0

    def __call__(self, attn, hidden_states, *args, encoder_hidden_states=None, attention_mask=None, **kwargs):
        if encoder_hidden_states is None:
            if self.id < 50 * self.lamd:
                map0 = self.img0_dict[self.name][self.id]
                map1 = self.img1_dict[self.name][self.id]
                cross_map = self.beta * hidden_states + (1 - self.beta)*((1 - self.alpha)*map0 + self.alpha*map1)
                res = self.original_processor(
                    attn, hidden_states, *args,
                    encoder_hidden_states=cross_map,
                    attention_mask=attention_mask,
                    **kwargs)
            else:
                res = self.original_processor(
                    attn, hidden_states, *args,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    **kwargs)
            self.id += 1
            if self.id == len(self.img0_dict[self.name]):
                self.id = 0
        else:
            res = self.original_processor(attn, hidden_states, *args,
                                          encoder_hidden_states=encoder_hidden_states,
                                          attention_mask=attention_mask,
                                          **kwargs)
        return res


class DiffMorpherPipeline(StableDiffusionPipeline):

    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: UNet2DConditionModel,
                 scheduler: LCMScheduler,
                 safety_checker: StableDiffusionSafetyChecker,
                 feature_extractor: CLIPImageProcessor,
                 image_encoder=None,
                 requires_safety_checker=True):
        # LCMScheduler replaced stable schedulers for LCM-LoRA
        sig = inspect.signature(super().__init__)
        params = sig.parameters
        if 'image_encoder' in params:
            super().__init__(vae, text_encoder, tokenizer, unet, scheduler,
                             safety_checker, feature_extractor, image_encoder, requires_safety_checker)
        else:
            super().__init__(vae, text_encoder, tokenizer, unet, scheduler,
                             safety_checker, feature_extractor, requires_safety_checker)
        self.img0_dict = {}
        self.img1_dict = {}
        self.use_lora = False
        self.use_adain = False
        self.use_reschedule = False
        self.output_path = "./results"

    def image2latent(self, image):
        DEVICE = self.device
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = torch.from_numpy(image).float()/127.5-1
            image = image.permute(2,0,1).unsqueeze(0)
        latents = self.vae.encode(image.to(DEVICE))['latent_dist'].mean
        latents = latents*0.18215
        return latents

    def latent2image(self, latents, return_type='np'):
        latents = 1/0.18215*latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type=='np':
            image = (image/2+0.5).clamp(0,1)
            image = image.cpu().permute(0,2,3,1).numpy()[0]
            image = (image*255).astype(np.uint8)
        return image

    def ddim_inversion(self, latent, cond):
        # Using standard DDIM inversion if needed:
        timesteps = reversed(self.scheduler.timesteps)
        with torch.autocast(device_type='cuda',dtype=torch.float32):
            for i,t in enumerate(tqdm.tqdm(timesteps,desc="DDIM inversion")):
                cond_batch = cond.repeat(latent.shape[0],1,1)
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                if i>0:
                    alpha_prod_t_prev = self.scheduler.alphas_cumprod[list(timesteps)[i-1]]
                else:
                    alpha_prod_t_prev = self.scheduler.final_alpha_cumprod

                mu = alpha_prod_t**0.5
                mu_prev = alpha_prod_t_prev**0.5
                sigma = (1 - alpha_prod_t)**0.5
                sigma_prev = (1 - alpha_prod_t_prev)**0.5

                eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample
                pred_x0 = (latent - sigma_prev * eps)/mu_prev
                latent = mu*pred_x0 + sigma*eps
        return latent

    def get_text_embeddings(self, prompt, guidance_scale, neg_prompt, batch_size):
        DEVICE = self.device
        text_input = self.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        if guidance_scale > 1.:
            uc_text = neg_prompt if neg_prompt else ""
            unconditional_input = self.tokenizer([uc_text]*batch_size, padding="max_length", max_length=77, return_tensors="pt")
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings,text_embeddings],dim=0)
        return text_embeddings

    def cal_latent(self,num_inference_steps,guidance_scale,unconditioning,
                   img_noise_0,img_noise_1,text_embeddings_0,text_embeddings_1,
                   lora_0,lora_1,alpha,use_lora,fix_lora=None):
        latents = slerp(img_noise_0,img_noise_1,alpha,self.use_adain)
        text_embeddings = (1-alpha)*text_embeddings_0 + alpha*text_embeddings_1
        self.scheduler.set_timesteps(num_inference_steps)
        if use_lora and lora_0 is not None and lora_1 is not None:
            if fix_lora is not None:
                self.unet = load_lora(self.unet,lora_0,lora_1,fix_lora)
            else:
                self.unet = load_lora(self.unet,lora_0,lora_1,alpha)

        for i,t in enumerate(tqdm.tqdm(self.scheduler.timesteps, desc=f"Sampling, alpha={alpha}")):
            if guidance_scale>1.0:
                model_inputs = torch.cat([latents]*2)
            else:
                model_inputs = latents
            if unconditioning is not None and isinstance(unconditioning,list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape),text_embeddings])

            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            if guidance_scale>1.0:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2,dim=0)
                noise_pred = noise_pred_uncon + guidance_scale*(noise_pred_con-noise_pred_uncon)
            latents = self.scheduler.step(noise_pred,t,latents,return_dict=False)[0]
        return latents

    def __call__(self,
                 img_0=None,
                 img_1=None,
                 img_path_0=None,
                 img_path_1=None,
                 prompt_0="",
                 prompt_1="",
                 save_lora_dir="./lora",
                 load_lora_path_0=None,
                 load_lora_path_1=None,
                 lora_steps=200,
                 lora_lr=2e-4,
                 lora_rank=16,
                 batch_size=1,
                 height=512,
                 width=512,
                 num_inference_steps=4,
                 guidance_scale=1.0,
                 attn_beta=0,
                 lamd=0.6,
                 use_lora=True,
                 use_adain=True,
                 use_reschedule=True,
                 output_path="./results",
                 num_frames=50,
                 fix_lora=None,
                 progress=tqdm,
                 unconditioning=None,
                 neg_prompt=None,
                 save_intermediates=False,
                 lcm_lora_path=None,
                 prompt="A painting",
                 **kwds):

        self.use_lora = use_lora
        self.use_adain = use_adain
        self.use_reschedule = use_reschedule
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

        # Switch to LCMScheduler for LCM-LoRA acceleration
        self.scheduler = LCMScheduler.from_config(self.scheduler.config)

        # Load LCM-LoRA if provided
        if lcm_lora_path is not None:
            print(f"Loading LCM-LoRA from {lcm_lora_path}...")
            self.load_lora_weights(lcm_lora_path)
            print("LCM-LoRA loaded successfully.")

        if img_0 is None:
            img_0 = Image.open(img_path_0).convert("RGB")
        if img_1 is None:
            img_1 = Image.open(img_path_1).convert("RGB")

        # Train LoRAs if needed
        if self.use_lora:
            if not load_lora_path_0:
                weight_name = f"{output_path.split('/')[-1]}_lora_0.ckpt"
                load_lora_path_0 = os.path.join(save_lora_dir, weight_name)
                if not os.path.exists(load_lora_path_0):
                    # Train LoRA for the first image
                    train_lora(
                        img_0,
                        prompt_0,
                        save_lora_dir,
                        model_path=self.tokenizer.name_or_path,
                        tokenizer=self.tokenizer,
                        text_encoder=self.text_encoder,
                        vae=self.vae,
                        unet=self.unet,
                        noise_scheduler=self.scheduler,
                        lora_steps=lora_steps,
                        lora_lr=lora_lr,
                        weight_name=weight_name
                    )
                    # Free GPU memory
                    del self.unet
                    torch.cuda.empty_cache()

                    # Reload base UNet
                    self.unet = UNet2DConditionModel.from_pretrained(
                        self.tokenizer.name_or_path, subfolder="unet"
                    ).to(self.device)

            # Load LoRA_0 from ckpt file
            if load_lora_path_0.endswith(".safetensors"):
                lora_0 = safetensors.torch.load_file(load_lora_path_0, device="cpu")
            else:
                lora_0 = torch.load(load_lora_path_0, map_location="cpu")

            # Load lora_0 into unet
            self.unet = load_lora(self.unet, lora_0, lora_0, 0.0)  # Just load, alpha=0 means original weights + lora_0
            torch.cuda.empty_cache()

            if not load_lora_path_1:
                weight_name = f"{output_path.split('/')[-1]}_lora_1.ckpt"
                load_lora_path_1 = os.path.join(save_lora_dir, weight_name)
                if not os.path.exists(load_lora_path_1):
                    # Train LoRA for the second image
                    train_lora(
                        img_1,
                        prompt_1,
                        save_lora_dir,
                        model_path=self.tokenizer.name_or_path,
                        tokenizer=self.tokenizer,
                        text_encoder=self.text_encoder,
                        vae=self.vae,
                        unet=self.unet,
                        noise_scheduler=self.scheduler,
                        lora_steps=lora_steps,
                        lora_lr=lora_lr,
                        weight_name=weight_name
                    )

                    # Free GPU memory after training second LoRA
                    del self.unet
                    torch.cuda.empty_cache()

                    # Reload base UNet
                    self.unet = UNet2DConditionModel.from_pretrained(
                        self.tokenizer.name_or_path, subfolder="unet"
                    ).to(self.device)

            # Load LoRA_1 from ckpt file
            if load_lora_path_1.endswith(".safetensors"):
                lora_1 = safetensors.torch.load_file(load_lora_path_1, device="cpu")
            else:
                lora_1 = torch.load(load_lora_path_1, map_location="cpu")

            # Both LoRAs are now available as lora_0, lora_1 and can be used during DDIM inversion and sampling
        else:
            lora_0 = lora_1 = None

        text_embeddings_0 = self.get_text_embeddings(prompt_0, guidance_scale, neg_prompt, batch_size)
        text_embeddings_1 = self.get_text_embeddings(prompt_1, guidance_scale, neg_prompt, batch_size)
        img_0 = get_img(img_0)
        img_1 = get_img(img_1)

        # Perform DDIM inversion if needed (only if you want real-image morphing)
        if self.use_lora:
            self.unet = load_lora(self.unet, lora_0, lora_1, 0)
        img_noise_0 = self.ddim_inversion(self.image2latent(img_0), text_embeddings_0)
        if self.use_lora:
            self.unet = load_lora(self.unet, lora_0, lora_1, 1)
        img_noise_1 = self.ddim_inversion(self.image2latent(img_1), text_embeddings_1)

        alpha_list = list(torch.linspace(0, 1, num_frames))
        images = self._morph(alpha_list, progress, "Sampling...", lora_0, lora_1,
                             text_embeddings_0, text_embeddings_1, img_noise_0, img_noise_1,
                             num_inference_steps, guidance_scale, None,
                             attn_beta, lamd, save_intermediates, fix_lora)

        return images