import os
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers, LCMScheduler
import torch
import torch.nn.functional as F
import tqdm
import numpy as np
import safetensors
from PIL import Image
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
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
            max_idx = len(self.img0_dict[self.name])
            if self.id < int(50*self.lamd) and self.id < max_idx:
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
            if self.id >= max_idx:
                self.id = 0
        else:
            res = self.original_processor(
                attn, hidden_states, *args,
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
                 scheduler: KarrasDiffusionSchedulers,
                 safety_checker: StableDiffusionSafetyChecker,
                 feature_extractor: CLIPImageProcessor,
                 image_encoder=None,
                 requires_safety_checker=True):
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

    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = self.device
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = torch.from_numpy(image).float()/127.5-1
            image = image.permute(2,0,1).unsqueeze(0)
        latents = self.vae.encode(image.to(DEVICE))['latent_dist'].mean
        latents = latents*0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1/0.18215*latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type=='np':
            image = (image/2+0.5).clamp(0,1)
            image = image.cpu().permute(0,2,3,1).numpy()[0]
            image = (image*255).astype(np.uint8)
        return image

    @torch.no_grad()
    def ddim_inversion(self, latent, cond):
        timesteps = reversed(self.scheduler.timesteps)
        with torch.autocast(device_type='cuda',dtype=torch.float32):
            for i,t in enumerate(tqdm.tqdm(timesteps,desc="DDIM inversion")):
                cond_batch = cond.repeat(latent.shape[0],1,1)
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (self.scheduler.alphas_cumprod[timesteps[i-1]] if i>0 else self.scheduler.final_alpha_cumprod)
                mu = alpha_prod_t**0.5
                mu_prev = alpha_prod_t_prev**0.5
                sigma = (1 - alpha_prod_t)**0.5
                sigma_prev = (1 - alpha_prod_t_prev)**0.5

                eps = self.unet(latent,t,encoder_hidden_states=cond_batch).sample
                pred_x0 = (latent - sigma_prev*eps)/mu_prev
                latent = mu*pred_x0 + sigma*eps
        return latent

    @torch.no_grad()
    def get_text_embeddings(self, prompt, guidance_scale, neg_prompt, batch_size):
        DEVICE = self.device
        text_input = self.tokenizer(prompt, padding="max_length", max_length=77,return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]

        if guidance_scale>1.:
            uc_text = neg_prompt if neg_prompt else ""
            unconditional_input = self.tokenizer([uc_text]*batch_size,padding="max_length",max_length=77,return_tensors="pt")
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings,text_embeddings],dim=0)
        return text_embeddings

    @torch.no_grad()
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

            noise_pred = self.unet(model_inputs,t,encoder_hidden_states=text_embeddings).sample
            if guidance_scale>1.0:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2,dim=0)
                noise_pred = noise_pred_uncon + guidance_scale*(noise_pred_con-noise_pred_uncon)

            latents = self.scheduler.step(noise_pred,t,latents,return_dict=False)[0]
        return latents

    def _morph(self, alpha_list, progress, desc, lora_0, lora_1, text_embeddings_0, text_embeddings_1,
               img_noise_0, img_noise_1, num_inference_steps, guidance_scale, unconditioning,
               attn_beta, lamd, save_intermediates, fix_lora):
        original_processor = list(self.unet.attn_processors.values())[0]
        images = []

        if attn_beta is not None:
            # Store for first frame
            if self.use_lora and lora_0 is not None and lora_1 is not None:
                self.unet = load_lora(self.unet,lora_0,lora_1,0 if fix_lora is None else fix_lora)

            attn_processor_dict = {}
            for k in self.unet.attn_processors.keys():
                if do_replace_attn(k):
                    if self.use_lora:
                        attn_processor_dict[k] = StoreProcessor(self.unet.attn_processors[k],self.img0_dict,k)
                    else:
                        attn_processor_dict[k] = StoreProcessor(original_processor,self.img0_dict,k)
                else:
                    attn_processor_dict[k] = self.unet.attn_processors[k]
            self.unet.set_attn_processor(attn_processor_dict)

            latents = self.cal_latent(num_inference_steps,guidance_scale,unconditioning,
                                      img_noise_0,img_noise_1,text_embeddings_0,text_embeddings_1,
                                      lora_0,lora_1,alpha_list[0],False,fix_lora)
            first_image = self.latent2image(latents)
            first_image = Image.fromarray(first_image)
            if save_intermediates:
                first_image.save(f"{self.output_path}/{0:02d}.png")

            # Store for last frame
            if self.use_lora and lora_0 is not None and lora_1 is not None:
                self.unet = load_lora(self.unet,lora_0,lora_1,1 if fix_lora is None else fix_lora)
            attn_processor_dict = {}
            for k in self.unet.attn_processors.keys():
                if do_replace_attn(k):
                    if self.use_lora:
                        attn_processor_dict[k] = StoreProcessor(self.unet.attn_processors[k],self.img1_dict,k)
                    else:
                        attn_processor_dict[k] = StoreProcessor(original_processor,self.img1_dict,k)
                else:
                    attn_processor_dict[k] = self.unet.attn_processors[k]
            self.unet.set_attn_processor(attn_processor_dict)

            latents = self.cal_latent(num_inference_steps,guidance_scale,unconditioning,
                                      img_noise_0,img_noise_1,text_embeddings_0,text_embeddings_1,
                                      lora_0,lora_1,alpha_list[-1],False,fix_lora)
            last_image = self.latent2image(latents)
            last_image = Image.fromarray(last_image)
            if save_intermediates:
                last_image.save(f"{self.output_path}/{len(alpha_list)-1:02d}.png")

            # Intermediates
            for i in progress.tqdm(range(1,len(alpha_list)-1),desc=desc):
                alpha = alpha_list[i]
                if self.use_lora and lora_0 is not None and lora_1 is not None:
                    self.unet = load_lora(self.unet,lora_0,lora_1,alpha if fix_lora is None else fix_lora)

                attn_processor_dict = {}
                for k in self.unet.attn_processors.keys():
                    if do_replace_attn(k):
                        attn_processor_dict[k] = LoadProcessor(self.unet.attn_processors[k],
                                                               k,self.img0_dict,self.img1_dict,alpha,attn_beta,lamd)
                    else:
                        attn_processor_dict[k] = self.unet.attn_processors[k]

                self.unet.set_attn_processor(attn_processor_dict)

                latents = self.cal_latent(num_inference_steps,guidance_scale,unconditioning,
                                          img_noise_0,img_noise_1,text_embeddings_0,text_embeddings_1,
                                          lora_0,lora_1,alpha,False,fix_lora)
                image = self.latent2image(latents)
                image = Image.fromarray(image)
                if save_intermediates:
                    image.save(f"{self.output_path}/{i:02d}.png")
                images.append(image)

            images = [first_image]+images+[last_image]
        else:
            # No attention blending
            for k,alpha in enumerate(alpha_list):
                latents = self.cal_latent(num_inference_steps,guidance_scale,unconditioning,
                                          img_noise_0,img_noise_1,
                                          text_embeddings_0,text_embeddings_1,
                                          lora_0,lora_1,alpha,self.use_lora,fix_lora)
                image = self.latent2image(latents)
                image = Image.fromarray(image)
                if save_intermediates:
                    image.save(f"{self.output_path}/{k:02d}.png")
                images.append(image)
        return images

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
                 num_frames=16,
                 fix_lora=None,
                 progress=tqdm,
                 unconditioning=None,
                 neg_prompt=None,
                 save_intermediates=False,
                 lcm_lora_path="latent-consistency/lcm-lora-sdv1-5",
                 **kwds):

        self.use_lora = use_lora
        self.use_adain = use_adain
        self.use_reschedule = use_reschedule
        self.output_path = output_path
        os.makedirs(output_path,exist_ok=True)

        # LCM-LoRA for fast generation
        self.scheduler = LCMScheduler.from_config(self.scheduler.config)
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
                    train_lora(img_0, prompt_0, save_lora_dir, self.tokenizer.name_or_path,
                               self.tokenizer, self.text_encoder, self.vae, self.unet,
                               self.scheduler, lora_steps, lora_lr, lora_rank, weight_name=weight_name)
            if load_lora_path_0.endswith(".safetensors"):
                lora_0 = safetensors.torch.load_file(load_lora_path_0, device="cpu")
            else:
                lora_0 = torch.load(load_lora_path_0, map_location="cpu")

            if not load_lora_path_1:
                weight_name = f"{output_path.split('/')[-1]}_lora_1.ckpt"
                load_lora_path_1 = os.path.join(save_lora_dir, weight_name)
                if not os.path.exists(load_lora_path_1):
                    train_lora(img_1, prompt_1, save_lora_dir, self.tokenizer.name_or_path,
                               self.tokenizer, self.text_encoder, self.vae, self.unet,
                               self.scheduler, lora_steps, lora_lr, lora_rank, weight_name=weight_name)
            if load_lora_path_1.endswith(".safetensors"):
                lora_1 = safetensors.torch.load_file(load_lora_path_1, device="cpu")
            else:
                lora_1 = torch.load(load_lora_path_1, map_location="cpu")
        else:
            lora_0 = lora_1 = None

        text_embeddings_0 = self.get_text_embeddings(prompt_0, guidance_scale, neg_prompt, batch_size)
        text_embeddings_1 = self.get_text_embeddings(prompt_1, guidance_scale, neg_prompt, batch_size)

        img_0 = get_img(img_0)
        img_1 = get_img(img_1)

        if self.use_lora and lora_0 is not None and lora_1 is not None:
            self.unet = load_lora(self.unet,lora_0,lora_1,0)
        img_noise_0 = self.ddim_inversion(self.image2latent(img_0), text_embeddings_0)
        if self.use_lora and lora_0 is not None and lora_1 is not None:
            self.unet = load_lora(self.unet,lora_0,lora_1,1)
        img_noise_1 = self.ddim_inversion(self.image2latent(img_1), text_embeddings_1)

        alpha_list = list(torch.linspace(0,1,num_frames))
        # Initial morph
        images_pt = self._morph(alpha_list, progress, "Sampling...", lora_0, lora_1,
                                text_embeddings_0, text_embeddings_1,
                                img_noise_0, img_noise_1,
                                num_inference_steps, guidance_scale, None,
                                attn_beta, lamd, save_intermediates, fix_lora)

        if self.use_reschedule:
            alpha_scheduler = AlphaScheduler()
            images_pt_tensors = [transforms.ToTensor()(img).unsqueeze(0) for img in images_pt]
            alpha_scheduler.from_imgs(images_pt_tensors)
            alpha_list = alpha_scheduler.get_list()
            print(alpha_list)

            # Reset dicts
            self.img0_dict = {}
            self.img1_dict = {}

            images = self._morph(alpha_list, progress, "Reschedule...",
                                 lora_0, lora_1, text_embeddings_0, text_embeddings_1,
                                 img_noise_0, img_noise_1,
                                 num_inference_steps, guidance_scale, None,
                                 attn_beta, lamd, save_intermediates, fix_lora)
        else:
            images = images_pt

        return images