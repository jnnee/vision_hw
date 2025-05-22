import inspect
import os
from typing import Union
from safetensors import safe_open
import PIL
import torch.nn as nn
import numpy as np
import torch
import tqdm
from accelerate import load_checkpoint_in_model
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download
from transformers import CLIPImageProcessor
from transformers import CLIPVisionModelWithProjection
from model.attn_processor import SkipAttnProcessor
from model.utils import get_trainable_module, init_adapter
from utils import (compute_vae_encodings, numpy_to_pil, prepare_image,
                   prepare_mask_image, resize_and_crop, resize_and_padding)
from model.attn_processor import SkipAttnProcessor, AttnProcessor2_0

from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
def get_module_by_path(model, path):
    """
    根据点分路径访问嵌套子模块，例如：
    path = "down_blocks.0.attentions.0.transformer_blocks.0.attn2"
    """
    modules = path.split(".")
    current_module = model
    for part in modules:
        # 处理数字索引（如列表中的第0项）
        if part.isdigit():
            current_module = current_module[int(part)]
        # 处理属性名（如 "down_blocks"）
        else:
            current_module = getattr(current_module, part)
    return current_module
def init_unified_adapter(unet):
    # attn_procs = {}  # 使用 ModuleDict 替代普通字典
    # ip_dim = image_encoder.config.projection_dim
    # #unet_sd = unet.state_dict()
    # ckpt_path = "/mnt/sda2/lry/catViton_modelckpt/checkpoint-26000/model.safetensors"
    # with safe_open(ckpt_path, framework="pt") as f:
    #     state_dict = {key: f.get_tensor(key) for key in f.keys()}
    #
    #     # 加载 UNet 参数（假设权重名称以 "unet." 开头）
    #     unet_dict = {k.replace("unet.", ""): v for k, v in state_dict.items() if k.startswith("unet.")}
    #
    # # 在初始化适配器前打印所有键
    # print("UNet state_dict keys:")
    # for key in unet_dict.keys():
    #     if "to_k_ip" in key or "to_v_ip" in key:
    #         print(key)
    # for name in unet.attn_processors.keys(): # 根据层名分配处理器
    #     # print(name)
    #     if name.startswith("mid_block"):
    #         hidden_size = unet.config.block_out_channels[-1]
    #     elif name.startswith("up_blocks"):
    #         block_id = int(name[len("up_blocks.")])
    #         hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
    #     elif name.startswith("down_blocks"):
    #         block_id = int(name[len("down_blocks.")])
    #         hidden_size = unet.config.block_out_channels[block_id]
    #
    #     if name.endswith("attn1.processor"):  # 自注意力层
    #         # 提取层前缀（如 "down_blocks.0.attentions.0.transformer_blocks.0.attn1"）
    #         layer_prefix = name.rsplit(".processor", 1)[0]
    #         weights = {
    #             "to_k_ip.weight": unet_dict[f"{layer_prefix}.processor.to_k_ip.weight"],
    #             "to_v_ip.weight": unet_dict[f"{layer_prefix}.processor.to_v_ip.weight"],
    #         }
    #         attn_procs[name] = SelfAttnWithIPProcessor(
    #             ip_dim=ip_dim,
    #             hidden_size = hidden_size,
    #             skip_mode=False  # 强制启用IP特征拼接
    #         )
    #         attn_procs[name].load_state_dict(weights)
    #         # layer_name = name.split(".processor")[0]
    #         # weights = {
    #         #     "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
    #         #     "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
    #         # }
    #         # # attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
    #         # attn_procs[name].load_state_dict(weights)
    #     elif name.endswith("attn2.processor"):  # 交叉注意力层
    #         attn_procs[name] = SkipCrossAttnProcessor(
    #             skip_mode=skip_cross_attn  # 根据参数决定是否跳过
    #         )
    #     else:
    #         attn_procs[name] = BaseAttnProcessor()  # 其他层保持默认
    #
    #
    # unet.set_attn_processor(attn_procs)
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        # 遍历每一层的name
        # print(name)
        # 用来判断是自注意力还是交叉注意力，
        # cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.endswith("attn1.processor"):
            cross_attention_dim = None
        else:
            cross_attention_dim = unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            # 自注意力跳过
            # attn_procs[name] = SkipAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
            #                                     )  # 跳过自注意力
            # midblock层自注意力替换
            if name.startswith("mid_block"):
                attn_procs[name] = SkipAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
                # unet.set_attn_processor(attn_procs)
            # 其他层不换
            else:
                # retain the original attn processor
                attn_procs[name] = AttnProcessor2_0(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            # attn_procs[name] = AttnProcessor2_0(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            # print("跳过自注意力", name)
            # attn_procs[name] = AttnProcessor()
        # elif cross_attention_dim is None:
        else:
            if name.startswith("mid_block"):
                # 交叉注意力
                layer_name = name.split(".processor")[0]
                layer = get_module_by_path(unet, layer_name)
                # if hasattr(layer, "to_k"):
                #     del layer.to_k
                # else:
                #     print("没删")
                # del layer.to_k  # 删除 to_k 线性层
                # del layer.to_v  # 删除 to_v 线性层
                # print(layer_name)layer_name指向自注意力层
                layer_name = layer_name.replace("attn2", "attn1")

                # print(layer_name)
                # 获取自注意力的权重
                weights_1 = {
                    # "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    # "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                    "to_q.weight": unet_sd[layer_name + ".to_q.weight"],
                    # "to_k.weight": unet_sd[layer_name + ".to_k.weight"],
                    # "to_v.weight": unet_sd[layer_name + ".to_v.weight"],
                    "to_out.0.weight": unet_sd[layer_name + ".to_out.0.weight"],
                    "to_out.0.bias": unet_sd[layer_name + ".to_out.0.bias"]
                }
                weights_2 = {
                    "to_k.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v.weight": unet_sd[layer_name + ".to_v.weight"],
                    "to_kv_ip.weight": torch.zeros((hidden_size, 768), dtype=torch.float32, device="cuda"),
                }
                # layer_name = name.split(".processor")[0]
                # weights = {
                #     "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                #     "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                # }
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
                # print(attn_procs[name])
                layer_name.replace("attn1", "attn2")
                layer = get_module_by_path(unet, layer_name)
                # print(layer)
                layer.load_state_dict(weights_1, strict=False)
                # unet[layer_name].load_state_dict(weights)
                attn_procs[name].load_state_dict(weights_2)
            else:
                attn_procs[name] = SkipAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

        # if name.endswith("attn2.processor") :

    unet.set_attn_processor(attn_procs)
    #adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    return torch.nn.ModuleList(unet.attn_processors.values())
# class IPAdapter(torch.nn.Module):
#     """IP-Adapter"""
#     def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
#         super().__init__()
#         self.unet = unet
#         self.image_proj_model = image_proj_model
#         self.adapter_modules = adapter_modules
#
#         if ckpt_path is not None:
#             self.load_from_checkpoint(ckpt_path)
#
#     def forward(self, noisy_latents, timesteps, image_embeds):
#         ip_tokens = self.image_proj_model(image_embeds)
#         #encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
#         # Predict the noise residual
#         noise_pred = self.unet(noisy_latents, cross_attention_kwargs={
#         "ip_features": ip_tokens  # ✅ 传递特征
#     },      # 传递给自注意力层
#          timestep=timesteps,
#          encoder_hidden_states=None ,  # 交叉注意力层被跳过timesteps
#        ).sample
#         return noise_pred
#
#     def load_from_checkpoint(self, ckpt_path: str):
#         # # Calculate original checksums
#         # orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
#         # orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
#         #
#         # state_dict = torch.load(ckpt_path, map_location="cpu")
#         #
#         # self.unet.load_state_dict(state_dict["unet"], strict=True)
#         #
#         # # Load state dict for image_proj_model and adapter_modules
#         # self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
#         # # self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)
#         #
#         # # Calculate new checksums
#         # new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
#         # # new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
#         #
#         # # Verify if the weights have changed
#         # assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
#         # # assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"
#         #
#         # print(f"Successfully loaded weights from checkpoint {ckpt_path}")
#         # 计算原始权重校验和
#         orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
#         # orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
#
#         try:
#             # 使用 safetensors 加载权重
#             with safe_open(ckpt_path, framework="pt", device="cpu") as f:
#                 # 构建完整 state_dict
#                 state_dict = {key: f.get_tensor(key) for key in f.keys()}
#
#                 # 加载 UNet 参数（假设权重名称以 "unet." 开头）
#                 unet_dict = {k.replace("unet.", ""): v for k, v in state_dict.items() if k.startswith("unet.")}
#                 self.unet.load_state_dict(unet_dict, strict=True)
#
#                 # 加载 image_proj 参数（假设权重名称以 "image_proj_model." 开头）
#                 image_proj_dict = {k.replace("image_proj_model.", ""): v for k, v in state_dict.items() if
#                                    k.startswith("image_proj_model.")}
#                 self.image_proj_model.load_state_dict(image_proj_dict, strict=True)
#
#                 # 加载 adapter_modules 参数（假设权重名称以 "adapter_modules." 开头）
#                 # adapter_dict = {k.replace("adapter_modules.", ""): v for k, v in state_dict.items() if
#                 #                 k.startswith("adapter_modules.")}
#                 # self.adapter_modules.load_state_dict(adapter_dict, strict=True)
#
#         except Exception as e:
#             raise ValueError(f"Failed to load safetensors checkpoint: {str(e)}")
#
#         # 验证权重更新
#         new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
#         # new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
#
#         assert not torch.allclose(orig_ip_proj_sum, new_ip_proj_sum, atol=1e-6), "image_proj_model 权重未变化！"
#         # assert not torch.allclose(orig_adapter_sum, new_adapter_sum, atol=1e-6), "adapter_modules 权重未变化！"
#
#         print(f"成功从 {ckpt_path} 加载检查点")
class IPAdapter(torch.nn.Module):
    """IP-Adapter"""

    def __init__(self, unet, image_proj_model,  ckpt_path=None):
        device="cuda"
        super().__init__()
        self.unet = unet.to(device)
        self.image_proj_model = image_proj_model.to(device)
        #self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        # encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # 把ip_token数据直接作为encoder——hidden-states
        encoder_hidden_states = ip_tokens
        # 在IPAdapter的forward方法中添加：
        #print("ip_tokens.shape:", ip_tokens.shape)  # 预期 (batch_size, sequence_length, hidden_size)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        #print("noise_pred.shape_ip:", noise_pred.shape)(1,4,128,48)
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        #orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        #state_dict = torch.load(ckpt_path, map_location="cpu")
        # 使用 safetensors 加载
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            state_dict = {}
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
            # 打印检查点中 proj 权重的形状
            proj_weight = f.get_tensor("image_proj_model.proj.weight")
            print("[调试] 检查点中 proj.weight 的形状:", proj_weight.shape)
           # print("state_dict keys:", state_dict.keys())

        # Load state dict for image_proj_model and adapter_modules
        #self.image_proj_model.load_state_dict(state_dict["image_proj_model"], strict=True)    # 提取 image_proj_model 的参数并去除前缀
        image_proj_dict = {
            k.replace("image_proj_model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("image_proj_model.")
        }
        #print("[调试] 检查点中 proj.weight 的形状:", proj_weight.shape)

        # 加载 image_proj_model
        self.image_proj_model.load_state_dict(image_proj_dict, strict=True)

        #self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)
        #self.unet.load_state_dict(state_dict["unet"], strict=True)
        # 加载 UNet（如果需要）
        unet_dict = {
            k.replace("unet.", ""): v
            for k, v in state_dict.items()
            if k.startswith("unet.")
        }
        self.unet.load_state_dict(unet_dict, strict=True)  # 根据实际情况调整 strict
        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        #new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        #assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")
class MyCatVTONPipeline:
    #base_ckpt传训练权重
    def __init__(
            self,
            base_ckpt,
            # attn_ckpt,
            # attn_ckpt_version="mix",
            unet_ckpt,  # 新增权重地址
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
        self.vae = AutoencoderKL.from_pretrained("/mnt/sda2/lry/image_ckpt/sd-vae-ft-mse").to(device, dtype=weight_dtype)
        if not skip_safety_check:
            self.feature_extractor = CLIPImageProcessor.from_pretrained(base_ckpt, subfolder="feature_extractor")
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(base_ckpt,
                                                                               subfolder="safety_checker").to(device, dtype=weight_dtype)
        #这一步仅从 base_ckpt 中读取 UNet 的结构配置（如层数、通道数等），不会加载权重
        config = UNet2DConditionModel.load_config(
            base_ckpt,
            subfolder="unet"
        )
        # ckpt_path="/mnt/sda2/lry/catViton_modelckpt/checkpoint-4000/model.safetensors"
        # with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        #     # 构建完整 state_dict
        #     state_dict = {key: f.get_tensor(key) for key in f.keys()}
        #     # 加载 UNet 参数（假设权重名称以 "unet." 开头）
        #     unet_dict = {k.replace("unet.", ""): v for k, v in state_dict.items() if k.startswith("unet.")}
        #     #self.unet.load_state_dict(unet_dict, strict=True)

        # . 创建 UNet ,from_config 会根据 config 创建一个新的 UNet 模型，其权重是随机初始化的，而非来自 base_ckpt 或其他预训练权重。
        self.unet = UNet2DConditionModel.from_config(config).to(device, dtype=weight_dtype)
        #self.unet.load_state_dict(unet_dict, strict=True)
        #self.unet = UNet2DConditionModel.from_pretrained(base_ckpt, subfolder="unet").to(device, dtype=weight_dtype)

        '#########增加初始化代码'
        init_unified_adapter(self.unet)  # 初始化adapter
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained('/mnt/sda2/lry/catvton_ckpt/IP-Adapter/models/image_encoder').to(device, dtype=weight_dtype)
        # load ip-adapter
        image_proj_model = ImageProjModel(
                cross_attention_dim=self.unet.config.cross_attention_dim,
                clip_embeddings_dim=self.image_encoder.config.projection_dim,
                clip_extra_context_tokens=4,
            )
        #ip_model = IPAdapter(self.unet, image_proj_model=ImageProjModel, ckpt_path=base_ckpt, device=device)
        self.ip_adapter=IPAdapter(self.unet, image_proj_model=image_proj_model, ckpt_path=unet_ckpt)
        '#########增加初始化代码'

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

    # def auto_attn_ckpt_load(self, attn_ckpt, version):
    #     sub_folder = {
    #         "mix": "mix-48k-1024",
    #         "vitonhd": "vitonhd-16k-512",
    #         "dresscode": "dresscode-16k-512",
    #     }[version]
    #     if os.path.exists(attn_ckpt):
    #         load_checkpoint_in_model(self.attn_modules, os.path.join(attn_ckpt, sub_folder, 'attention'))
    #     else:
    #         repo_path = snapshot_download(repo_id=attn_ckpt)
    #         print(f"Downloaded {attn_ckpt} to {repo_path}")
    #         load_checkpoint_in_model(self.attn_modules, os.path.join(repo_path, sub_folder, 'attention'))

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
            # 对ipadapter的无分类引导
            clip_image,
            #drop_image_embeds,

            num_inference_steps: int = 50,
            guidance_scale: float = 2.5,
            height: int = 1024,
            width: int = 768,
            generator=None,
            eta=1.0,
            **kwargs
    ):
        concat_dim = -2  # FIXME: y axis concat
        # Prepare inputs to Tensor
        image, condition_image, mask = self.check_inputs(image, condition_image, mask, width, height)
        image = prepare_image(image).to(self.device, dtype=self.weight_dtype)
        condition_image = prepare_image(condition_image).to(self.device, dtype=self.weight_dtype)
        mask = prepare_mask_image(mask).to(self.device, dtype=self.weight_dtype)
        #print(111111,clip_image.shape)
        clip_image = prepare_image(clip_image).to(self.device, dtype=self.weight_dtype)
        #print(22222, clip_image.shape)
        # Mask image
        masked_image = image * (mask < 0.5)
        # VAE encoding
        # 加了掩码的image
        masked_latent = compute_vae_encodings(masked_image, self.vae)
        # 衣服
        condition_latent = compute_vae_encodings(condition_image, self.vae)
        # 掩码
        mask_latent = torch.nn.functional.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")
        del image, mask, condition_image
        #clipimage encoding
        # 转移输入到 GPU 并匹配数据类型
        clip_image = clip_image.to(self.device, dtype=self.weight_dtype)
        clip_latent = self.image_encoder(clip_image).image_embeds
        #print(33333, clip_latent.shape)
        #clip_latent = self.image_encoder(clip_image).to(self.device, dtype=self.weight_dtype)

        # Concatenate latents
        masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)  # 4维
        mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)  # 1维

        '######################模型架构修改'
        # ip-adapter,在init里加载
        # image_proj_model = ImageProjModel(
        #     cross_attention_dim=self.unet.config.cross_attention_dim,
        #     clip_embeddings_dim=self.image_encoder.config.projection_dim,
        #     clip_extra_context_tokens=4,
        # )
        # # 初始化适配器（跳过交叉注意力）
        # adapter_modules = init_unified_adapter(self.unet, self.image_encoder, skip_cross_attn=True)
        #
        # ip_adapter = IPAdapter(self.unet, image_proj_model, adapter_modules, '/mnt/sda2/lry/catViton_modelckpt/checkpoint-26000/model.safetensors')
        # 重新将我们自己训练的权重导入进来
        # ip_adapter.load_state_dict(torch.load('/mnt/sda2/lry/catViton_modelckpt/checkpoint-2000/model.safetensors'))
        '######################模型架构修改'

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

        # Classifier-Free Guidance先是无条件，再是有条件
        if do_classifier_free_guidance := (guidance_scale > 1.0):
            #unet条件样本增加
            masked_latent_concat = torch.cat(
                [
                    torch.cat([masked_latent, torch.zeros_like(condition_latent)], dim=concat_dim),
                    masked_latent_concat,
                ]
            )
            mask_latent_concat = torch.cat([mask_latent_concat] * 2)
            #image encoder样本增加
            clip_latent = torch.cat([torch.zeros_like(clip_latent),clip_latent])

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
                #print("inpainting_latent_model_input",inpainting_latent_model_input.shape)
                # predict the noise residual
                '''################unet替换维ip_adapter'''
                # noise_pred = self.unet(
                #     inpainting_latent_model_input,
                #     t.to(self.device),
                #     encoder_hidden_states=None,  # FIXME
                #     return_dict=False,
                # )[0]
                '传进来的数据都是batch[****],将训练的修改成如下处理'
                #image_embeds = self.image_encoder(clip_image).image_embeds
                image_embeds=clip_latent
                print("image_embeds",image_embeds
                      .shape)
                #image_embeds_ = []
                # for image_embed, drop_image_embed in zip(image_embeds, drop_image_embeds):
                #     if do_classifier_free_guidance:
                #         # ✅ 正确：生成条件和无条件嵌入并分别加入列表
                #         image_embeds_.append(image_embed)  # 条件嵌入 [1, 1024]
                #         image_embeds_.append(torch.zeros_like(image_embed))  # 无条件嵌入 [1, 1024]
                #     else:
                #         image_embeds_.append(image_embed)
                #
                # image_embeds = torch.stack(image_embeds_)  # 最终形状 [2*batch_size, 1024]
                # print(image_embeds.shape)
                # print(inpainting_latent_model_input.shape)
                # print(timesteps.shape)
                noise_pred = self.ip_adapter(inpainting_latent_model_input, t.to(self.device), image_embeds)
                '''################unet替换维ip_adapter'''
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
        #print("image.shape",image.shpae)

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