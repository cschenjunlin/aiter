# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Tuple
import torch
import triton
import triton.language as tl

from aiter.ops.triton.mha_fwd import _flash_attn_forward
from aiter.ops.triton.mha_onekernel_bwd import flash_attn_onekernel_backward
from aiter.ops.triton.mha_fused_bwd import flash_attn_fused_backward
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton._triton_kernels.mha_fwd import _attn_fwd, _get_config

_LOGGER = AiterTritonLogger()

global _USE_FUSED_BWD_KERNEL
_USE_FUSED_BWD_KERNEL = False


def mha_set_use_fused_bwd_kernel(value: bool):
    global _USE_FUSED_BWD_KERNEL
    _USE_FUSED_BWD_KERNEL = value


_USE_INT64_STRIDES = True


def mha_set_use_int64_strides(value: bool):
    """Use 64-bit integer strides to prevent integer overflows with very large tensors."""
    global _USE_INT64_STRIDES
    _USE_INT64_STRIDES = value


class _FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        is_grad_enabled,
        config=None,
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, philox_seed, philox_offset = (
            _flash_attn_forward(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size_left=int(window_size[0]),
                window_size_right=int(window_size[1]),
                bias=bias,
                alibi_slopes=alibi_slopes,
                return_lse=return_lse,
                return_softmax=return_softmax and dropout_p > 0,
                max_seqlen_q=q.shape[1],
                max_seqlen_k=k.shape[1],
                config=config,
            )
        )

        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse)
            ctx.philox_seed = philox_seed
            ctx.philox_offset = philox_offset
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.bias = bias
            ctx.window_size = window_size
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic

        out = out_padded[..., :head_size_og]
        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        bias = ctx.bias
        dbias = torch.empty_like(bias) if bias is not None else None
        dq, dk, dv = torch.zeros_like(q), torch.empty_like(k), torch.empty_like(v)
        head_size_v_og = do.size(3)
        do_padded = do
        if head_size_v_og % 8 != 0:
            do_padded = torch.nn.functional.pad(do, [0, 8 - head_size_v_og % 8])

        if _USE_FUSED_BWD_KERNEL:
            flash_attn_fused_backward(
                do_padded,
                q,
                k,
                v,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                dbias,
                ctx.softmax_scale,
                ctx.alibi_slopes,
                ctx.causal,
                None,
                None,
                max_seqlen_q=q.shape[1],
                max_seqlen_k=k.shape[1],
                dropout_p=ctx.dropout_p,
                philox_seed=ctx.philox_seed,
                philox_offset=ctx.philox_offset,
                USE_INT64_STRIDES=_USE_INT64_STRIDES,
            )
        else:
            flash_attn_onekernel_backward(
                do_padded,
                q,
                k,
                v,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                dbias,
                ctx.softmax_scale,
                ctx.alibi_slopes,
                ctx.causal,
                None,
                None,
                max_seqlen_q=q.shape[1],
                max_seqlen_k=k.shape[1],
                dropout_p=ctx.dropout_p,
                philox_seed=ctx.philox_seed,
                philox_offset=ctx.philox_offset,
                USE_INT64_STRIDES=_USE_INT64_STRIDES,
            )

        dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k.shape[-1]]
        dv = dv[..., : v.shape[-1]]
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            dbias,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    bias=None,
    alibi_slopes=None,
    deterministic=True,
    return_lse=False,
    return_attn_probs=False,
    config: Optional[dict[str, any]] = None,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim_q).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        bias: (seqlen_q, seqlen_k)
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    _LOGGER.info(
        f"FLASH_ATTN:  q={tuple(q.shape)}  k={tuple(k.shape)}  v={tuple(v.shape)}"
    )
    return _FlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        torch.is_grad_enabled(),
        config,
    )
