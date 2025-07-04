import inspect
import os
from typing import Union

import PIL
import numpy as np
import torch
import tqdm
from accelerate import load_checkpoint_in_model
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.models import ImageProjection
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.utils.torch_utils import randn_tensor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from safetensors import safe_open
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from huggingface_hub import snapshot_download
from transformers import CLIPImageProcessor

from model.attn_processor import SkipAttnProcessor
from model.utils import get_trainable_module, init_adapter
from utils import (compute_vae_encodings, numpy_to_pil, prepare_image,
                   prepare_mask_image, resize_and_crop, resize_and_padding)




class CatVTONPipeline():  # 修改
    def __init__(
            self,
            base_ckpt,
            attn_ckpt,
            unet_ckpt,  # 新增权重地址
            attn_ckpt_version="mix",
            weight_dtype=torch.float32,
            device='cuda',
            compile=False,
            skip_safety_check=False,
            use_tf32=True,
    ):
        self.device = device
        self.weight_dtype = weight_dtype
        self.skip_safety_check = skip_safety_check

        self.noise_scheduler = DDIMScheduler.from_pretrained(base_ckpt, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained("/mnt/sda2/lry/image_ckpt/sd-vae-ft-mse").to(device,
                                                                                              dtype=weight_dtype)
        if not skip_safety_check:  # 初始化图像特征提取器和内容安全检查器，用于确保生成内容的安全性
            self.feature_extractor = CLIPImageProcessor.from_pretrained(base_ckpt, subfolder="feature_extractor")
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(base_ckpt,
                                                                               subfolder="safety_checker").to(device,
                                                                                                              dtype=weight_dtype)

        '''#############这里修改一下 将模型的权重读入进来'''
        self.unet = UNet2DConditionModel.from_pretrained(base_ckpt, subfolder="unet").to(device, dtype=weight_dtype)
        # 新的UNet加载流程
        from safetensors.torch import load_file  # 确保已安装safetensors库

        # 1. 先加载基础配置创建模型结构
        # self.unet = UNet2DConditionModel.from_pretrained(
        #     base_ckpt,
        #     subfolder="unet",
        #     # 添加以下参数确保只加载配置
        #     #pretrained_model_name_or_path=None,
        #     ignore_mismatched_sizes=True
        # )

        # 添加IP-Adapter初始化
        init_adapter(self.unet, cross_attn_cls=SkipAttnProcessor)  # Skip Cross-Attention

        # 2. 加载safetensors权重文件
        try:
            unet_state_dict = load_file(unet_ckpt)
        except Exception as e:
            raise ValueError(f"加载UNet权重文件失败: {str(e)}") from e


        # 3. 转换权重格式（如果需要）
        '''权重内容
        conv_in.bias: torch.Size([320])
        conv_in.weight: torch.Size([320, 9, 3, 3])
        conv_norm_out.bias: torch.Size([320])
        conv_norm_out.weight: torch.Size([320])
        conv_out.bias: torch.Size([4])
        '''
        # 检查是否需要移除前缀（根据具体模型结构调整）
        if any(k.startswith("unet.") for k in unet_state_dict.keys()):
            unet_state_dict = {k.replace("unet.", ""): v for k, v in unet_state_dict.items()}
        #
        # 4. 加载权重到模型
        missing_keys, unexpected_keys = self.unet.load_state_dict(unet_state_dict, strict=False)

        # 5. 处理权重加载异常
        if len(missing_keys) > 0:
            print(f"[警告] 缺失的权重键: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"[警告] 意外的权重键: {unexpected_keys}")
        '''####################'''


        '''去除额外加载自注意力层权重的代码'''
        # self.attn_modules = get_trainable_module(self.unet, "attention")
        # self.auto_attn_ckpt_load(attn_ckpt, attn_ckpt_version)

        # Pytorch 2.0 Compile
        if compile:
            self.unet = torch.compile(self.unet)
            self.vae = torch.compile(self.vae, mode="reduce-overhead")

        # Enable TF32 for faster training on Ampere GPUs (A100 and RTX 30 series).
        if use_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

    def auto_attn_ckpt_load(self, attn_ckpt, version):
        sub_folder = {
            "mix": "mix-48k-1024",
            "vitonhd": "vitonhd-16k-512",
            "dresscode": "dresscode-16k-512",
        }[version]
        # 添加IP-Adapter权重加载，新增
        if hasattr(self, "ip_adapter"):
            ip_adapter_path = os.path.join(attn_ckpt, "ip_adapter")
            self.ip_adapter.load_state_dict(torch.load(ip_adapter_path))
        if os.path.exists(attn_ckpt):
            load_checkpoint_in_model(self.attn_modules, os.path.join(attn_ckpt, sub_folder, 'attention'))
        else:
            repo_path = snapshot_download(repo_id=attn_ckpt)
            print(f"Downloaded {attn_ckpt} to {repo_path}")
            load_checkpoint_in_model(self.attn_modules, os.path.join(repo_path, sub_folder, 'attention'))

    def run_safety_checker(self, image):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(self.device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(self.weight_dtype)
            )
        return image, has_nsfw_concept

    def check_inputs(self, image, condition_image, mask, width, height):
        if isinstance(image, torch.Tensor) and isinstance(condition_image, torch.Tensor) and isinstance(mask,
                                                                                                        torch.Tensor):
            return image, condition_image, mask
        assert image.size == mask.size, "Image and mask must have the same size"
        image = resize_and_crop(image, (width, height))
        mask = resize_and_crop(mask, (width, height))
        condition_image = resize_and_padding(condition_image, (width, height))
        return image, condition_image, mask

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
            self,
            image: Union[PIL.Image.Image, torch.Tensor],
            condition_image: Union[PIL.Image.Image, torch.Tensor],
            mask: Union[PIL.Image.Image, torch.Tensor],
            num_inference_steps: int = 50,
            guidance_scale: float = 9,
            height: int = 512,
            width: int = 368,
            generator=None,
            eta=1.0,
            **kwargs
    ):
        concat_dim = -2  # FIXME: y axis concat
        #print(image.shape)
        # Prepare inputs to Tensor
        image, condition_image, mask = self.check_inputs(image, condition_image, mask, width, height)
        image = prepare_image(image).to(self.device, dtype=self.weight_dtype)
        condition_image = prepare_image(condition_image).to(self.device, dtype=self.weight_dtype)
        mask = prepare_mask_image(mask).to(self.device, dtype=self.weight_dtype)

        masked_image = image * (mask < 0.5)  # 对于掩码中每个像素值小于0.5的位置，保留图像的对应像素值。对于掩码中每个像素值大于等于0.5的位置，将图像的对应像素值置为0
        # VAE encoding
        masked_latent = compute_vae_encodings(masked_image, self.vae)
        condition_latent = compute_vae_encodings(condition_image, self.vae)  # 衣服图像
        mask_latent = torch.nn.functional.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")
        del image, mask, condition_image
        # Concatenate latents
        masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
        mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)
        # Prepare noise
        latents = randn_tensor(
            masked_latent_concat.shape,
            generator=generator,
            device=masked_latent_concat.device,
            dtype=self.weight_dtype,
        )
        # Prepare timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        latents = latents * self.noise_scheduler.init_noise_sigma
        # Classifier-Free Guidance
        if do_classifier_free_guidance := (guidance_scale > 1.0):
            masked_latent_concat = torch.cat(
                [
                    torch.cat([masked_latent, torch.zeros_like(condition_latent)], dim=concat_dim),
                    masked_latent_concat,
                ]
            )
            mask_latent_concat = torch.cat([mask_latent_concat] * 2)

        # Denoising loop
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = (len(timesteps) - num_inference_steps * self.noise_scheduler.order)
        with tqdm.tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                non_inpainting_latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
                non_inpainting_latent_model_input = self.noise_scheduler.scale_model_input(
                    non_inpainting_latent_model_input, t)
                # prepare the input for the inpainting model
                inpainting_latent_model_input = torch.cat(
                    [non_inpainting_latent_model_input, mask_latent_concat, masked_latent_concat], dim=1)
                # predict the noise residual
                noise_pred = self.unet(
                    inpainting_latent_model_input,
                    t.to(self.device),
                    # encoder_hidden_states=ip_image_embeds,  # 传入图像嵌入,修改
                    encoder_hidden_states=None,  # FIXME
                    return_dict=False,
                )[0]
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                    )
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps
                        and (i + 1) % self.noise_scheduler.order == 0
                ):
                    progress_bar.update()

        # Decode the final latents
        latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.device, dtype=self.weight_dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = numpy_to_pil(image)

        # Safety Check
        if not self.skip_safety_check:
            current_script_directory = os.path.dirname(os.path.realpath(__file__))
            nsfw_image = os.path.join(os.path.dirname(current_script_directory), 'resource', 'img', 'NSFW.jpg')
            nsfw_image = PIL.Image.open(nsfw_image).resize(image[0].size)
            image_np = np.array(image)
            _, has_nsfw_concept = self.run_safety_checker(image=image_np)
            for i, not_safe in enumerate(has_nsfw_concept):
                if not_safe:
                    image[i] = nsfw_image
        return image


class CatVTONPix2PixPipeline(CatVTONPipeline):
    def auto_attn_ckpt_load(self, attn_ckpt, version):
        # TODO: Temperal fix for the model version
        if os.path.exists(attn_ckpt):
            load_checkpoint_in_model(self.attn_modules, os.path.join(attn_ckpt, version, 'attention'))
        else:
            repo_path = snapshot_download(repo_id=attn_ckpt)
            print(f"Downloaded {attn_ckpt} to {repo_path}")
            load_checkpoint_in_model(self.attn_modules, os.path.join(repo_path, version, 'attention'))

    def check_inputs(self, image, condition_image, width, height):
        if isinstance(image, torch.Tensor) and isinstance(condition_image, torch.Tensor) and isinstance(torch.Tensor):
            return image, condition_image
        image = resize_and_crop(image, (width, height))
        condition_image = resize_and_padding(condition_image, (width, height))
        return image, condition_image

    @torch.no_grad()
    def __call__(
            self,
            image: Union[PIL.Image.Image, torch.Tensor],
            condition_image: Union[PIL.Image.Image, torch.Tensor],
            num_inference_steps: int = 50,
            guidance_scale: float = 2.5,
            height: int = 1024,
            width: int = 768,
            generator=None,
            eta=1.0,
            **kwargs
    ):
        concat_dim = -1
        # Prepare inputs to Tensor
        image, condition_image = self.check_inputs(image, condition_image, width, height)
        image = prepare_image(image).to(self.device, dtype=self.weight_dtype)
        condition_image = prepare_image(condition_image).to(self.device, dtype=self.weight_dtype)
        # VAE encoding
        image_latent = compute_vae_encodings(image, self.vae)
        condition_latent = compute_vae_encodings(condition_image, self.vae)
        del image, condition_image
        # Concatenate latents
        condition_latent_concat = torch.cat([image_latent, condition_latent], dim=concat_dim)
        # Prepare noise
        latents = randn_tensor(
            condition_latent_concat.shape,
            generator=generator,
            device=condition_latent_concat.device,
            dtype=self.weight_dtype,
        )
        # Prepare timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        latents = latents * self.noise_scheduler.init_noise_sigma
        # Classifier-Free Guidance
        if do_classifier_free_guidance := (guidance_scale > 1.0):
            condition_latent_concat = torch.cat(
                [
                    torch.cat([image_latent, torch.zeros_like(condition_latent)], dim=concat_dim),
                    condition_latent_concat,
                ]
            )

        # Denoising loop
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = (len(timesteps) - num_inference_steps * self.noise_scheduler.order)
        with tqdm.tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
                latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)
                # prepare the input for the inpainting model
                p2p_latent_model_input = torch.cat([latent_model_input, condition_latent_concat], dim=1)
                # predict the noise residual
                noise_pred = self.unet(
                    p2p_latent_model_input,
                    t.to(self.device),
                    encoder_hidden_states=None,
                    return_dict=False,
                )[0]
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                    )
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps
                        and (i + 1) % self.noise_scheduler.order == 0
                ):
                    progress_bar.update()

        # Decode the final latents
        latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.device, dtype=self.weight_dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = numpy_to_pil(image)

        # Safety Check
        if not self.skip_safety_check:
            current_script_directory = os.path.dirname(os.path.realpath(__file__))
            nsfw_image = os.path.join(os.path.dirname(current_script_directory), 'resource', 'img', 'NSFW.jpg')
            nsfw_image = PIL.Image.open(nsfw_image).resize(image[0].size)
            image_np = np.array(image)
            _, has_nsfw_concept = self.run_safety_checker(image=image_np)
            for i, not_safe in enumerate(has_nsfw_concept):
                if not_safe:
                    image[i] = nsfw_image
        return image
