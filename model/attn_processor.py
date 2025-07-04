import torch
# from flash_attn import flash_attn_func
import torch.nn as nn
import torch.nn.functional as F
class SkipAttnProcessor(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        # print('执行跳过操作')
        return hidden_states

#自注意力操作
class CatAttnProcessor(torch.nn.Module):
    def __init__(self, in_channels,hidden_size, num_tokens=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens


    def __call__(self, attn, hidden_states, encoder_hidden_states=None,attention_mask=None,
        temb=None,k_tokens=None,v_tokens=None,**kwargs):
        residual=hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # 生成 Q
        query = attn.to_q(hidden_states)  # [B, L, D]

        # 生成 K/V
        if encoder_hidden_states is None:
            # 自注意力模式：K/V 来自输入 + 额外 token
            key = attn.to_k(hidden_states)  # [B, L, D]
            value = attn.to_v(hidden_states)  # [B, L, D]
            # 拼接额外 token (扩展到 batch 维度)
            key = torch.cat([
                key,
                k_tokens  # [B, N, D]
            ], dim=1)

            value = torch.cat([
                value,
               v_tokens # [B, N, D]
            ], dim=1)
        else:
            # 交叉注意力模式：保持原始逻辑
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        # 调整维度结构（示例参数）
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # [B, H, L, D/H]
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # [B, H, L+N, D/H]
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # [B, H, L+N, D/H]

        # 计算注意力
        hidden_states = F.scaled_dot_product_attention(
            query, key, value,  dropout_p=0.0, is_causal=False
        )
        # hidden_states = flash_attn_func(
        #     query, key, value, dropout_p=0.0, causal=False
        # )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        # 输出投影
        return hidden_states
class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        **kwargs
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        # print(attn.to_q.weight.shape)
        # print(attn.to_k.weight.shape)
        # print(attn.to_v.weight.shape)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            #print(111)
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            #print(222)
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        # hidden_states = flash_attn_func(
        #     query, key, value, dropout_p=0.0, causal=False
        # )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

   