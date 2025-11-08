# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import math
from typing import Optional, Dict
import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from aiter.ops.triton._triton_kernels.mha_onekernel_bwd import (
    _bwd_preprocess,
    _bwd_kernel_causal,
    _bwd_kernel_noncausal,
    _get_config,
)
from aiter.test_mha_common import (
    attention_ref,
)


def flash_attn_onekernel_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    dbias: torch.Tensor,
    sm_scale: float,
    alibi_slopes: Optional[torch.Tensor],
    causal: bool,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    philox_seed: Optional[int] = 0,
    philox_offset: Optional[int] = 0,
    USE_INT64_STRIDES: Optional[bool] = False,
    config: Optional[Dict[str, any]] = None,
):
    if dbias is not None:
        raise ValueError("Bias is not supported yet in the Triton Backend")

    use_alibi, (stride_az, stride_ah) = (
        (True, alibi_slopes.stride()) if alibi_slopes is not None else (False, (0, 0))
    )

    IS_VARLEN = True if cu_seqlens_q is not None else False

    # get strides and shape
    if IS_VARLEN:
        # Layout is thd.
        # q and k are [total_tokens, num_head, head_dim_qk].
        # v is [total_tokens, num_head, head_dim_v].
        batch, seqlen_q, num_q_heads = (
            len(cu_seqlens_q) - 1,
            max_seqlen_q,
            q.shape[1],
        )
        _, num_k_heads = max_seqlen_k, k.shape[1]
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
        dq_strides = (0, dq.stride(1), dq.stride(0), dq.stride(2))
        dk_strides = (0, dk.stride(1), dk.stride(0), dk.stride(2))
        dv_strides = (0, dv.stride(1), dv.stride(0), dv.stride(2))
        do_strides = (0, do.stride(1), do.stride(0), do.stride(2))
    else:
        # Layout is bshd.
        # q and k are [batch, seq_len, num_head, head_dim_qk].
        # v is [batch, seq_len, num_head, head_dim_v]
        batch, seqlen_q, num_q_heads = q.shape[:-1]
        _, num_k_heads = k.shape[1], k.shape[2]
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))
        dq_strides = (dq.stride(0), dq.stride(2), dq.stride(1), dq.stride(3))
        dk_strides = (dk.stride(0), dk.stride(2), dk.stride(1), dk.stride(3))
        dv_strides = (dv.stride(0), dv.stride(2), dv.stride(1), dv.stride(3))
        do_strides = (do.stride(0), do.stride(2), do.stride(1), do.stride(3))

    qk_head_dim = q.shape[-1]
    v_head_dim = v.shape[-1]
    pe_head_dim = qk_head_dim - v_head_dim
    # BLOCK_D_MODEL, BLOCK_D_MODEL_POW2
    # padding for head_dim. Power of 2 or 16
    BLOCK_D_MODEL_POW2 = max(triton.next_power_of_2(v_head_dim), 16)
    BLOCK_D_MODEL_PE_POW2 = (
        0 if pe_head_dim == 0 else max(triton.next_power_of_2(pe_head_dim), 16)
    )
    assert (pe_head_dim == 0 and BLOCK_D_MODEL_PE_POW2 == 0) or (
        v_head_dim == BLOCK_D_MODEL_POW2 and pe_head_dim == BLOCK_D_MODEL_PE_POW2
    ), "Positional encoding support requires NOPE and PE head sizes to be unpadded powers of 2."

    # Configs
    if config is None:
        config = _get_config()

    # init delta
    delta = torch.zeros_like(softmax_lse)
    if IS_VARLEN:
        # [total_tokens, num_q_heads, seqlen_q]
        delta_strides = (0, delta.stride(1), delta.stride(0))
    else:
        # [batch, num_q_heads, seqlen_q]
        delta_strides = delta.stride()

    # preprocess
    # compute D(delta) = rowsum(dO*O). Note, multiplication is element-wise.
    pre_grid = (
        triton.cdiv(max_seqlen_q, config["preprocess_kernel"]["PRE_BLOCK"]),
        batch,
        num_q_heads,
    )
    _bwd_preprocess[pre_grid](
        o,
        do,
        delta,
        *o_strides,
        *delta_strides,
        cu_seqlens_q,
        max_seqlen_q,
        BLOCK_M=config["preprocess_kernel"]["PRE_BLOCK"],
        BLOCK_D_MODEL=v_head_dim,
        BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
        IS_VARLEN=IS_VARLEN,
    )

    # dropout_mask
    use_dropout = dropout_p > 0.0
    if use_dropout:
        dropout_mask = torch.zeros(
            (batch, num_q_heads, max_seqlen_q, max_seqlen_k),
            device=q.device,
            dtype=torch.float32,
        )
        dropout_strides = dropout_mask.stride()
    else:
        dropout_mask = None
        dropout_strides = (0, 0, 0, 0)

    seqlen = max(max_seqlen_q, max_seqlen_k)

    # "onekernel_pe" is for Positional Encoding (PE) causal case, it's going to be
    # used if present. Otherwise, fallback to default "onekernel" config.
    config_onekernel = (
        config["onekernel_pe"]
        if (pe_head_dim > 0 and causal and "onekernel_pe" in config)
        else config["onekernel"]
    )
    grid = (
        num_k_heads,
        triton.cdiv(seqlen, config_onekernel["BLOCK_N1"]),
        batch,
    )

    if causal:
        _bwd_kernel_causal[grid](
            q,
            k,
            v,
            sm_scale,
            do,
            dq,
            dk,
            dv,
            softmax_lse,
            delta,
            *q_strides,
            *k_strides,
            *v_strides,
            *dq_strides,
            *dk_strides,
            *dv_strides,
            *delta_strides,
            *do_strides,
            *dropout_strides,
            stride_az,
            stride_ah,
            num_q_heads,
            num_k_heads,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_mask,
            dropout_p,
            philox_seed,
            philox_offset,
            alibi_slopes,
            HEAD_DIM=v_head_dim,
            ACTUAL_HEAD_DIM=BLOCK_D_MODEL_POW2,
            PE_HEAD_DIM=pe_head_dim,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            USE_ALIBI=use_alibi,
            USE_EXP2=True,
            DEBUG_TRITON=False,
            DEBUG_TRITON_DETAIL=False,
            USE_INT64_STRIDES=USE_INT64_STRIDES,
            **config_onekernel,
        )
    else:
        _bwd_kernel_noncausal[grid](
            q,
            k,
            v,
            sm_scale,
            do,
            dq,
            dk,
            dv,
            softmax_lse,
            delta,
            *q_strides,
            *k_strides,
            *v_strides,
            *dq_strides,
            *dk_strides,
            *dv_strides,
            *delta_strides,
            *do_strides,
            *dropout_strides,
            stride_az,
            stride_ah,
            num_q_heads,
            num_k_heads,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_mask,
            dropout_p,
            philox_seed,
            philox_offset,
            alibi_slopes,
            HEAD_DIM=v_head_dim,
            ACTUAL_HEAD_DIM=BLOCK_D_MODEL_POW2,
            PE_HEAD_DIM=pe_head_dim,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            USE_ALIBI=use_alibi,
            USE_EXP2=True,
            DEBUG_TRITON=False,
            DEBUG_TRITON_DETAIL=False,
            USE_INT64_STRIDES=USE_INT64_STRIDES,
            **config_onekernel,
        )

    return dq, dk, dv


BATCH_SIZE: int = 1
SEQ_LEN: int = 128
NUM_HEADS: int = 16
HEAD_SIZE: int = 32
MHA_SHAPE: tuple[int, int, int, int] = (BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_SIZE)
assert all(dim > 0 for dim in MHA_SHAPE)
dtype = torch.float32


def main(unused_argv):
    torch.cuda.empty_cache()
    torch.manual_seed(42)

    q = torch.randn(MHA_SHAPE, device="cuda", dtype=dtype)
    k = torch.randn(MHA_SHAPE, device="cuda", dtype=dtype)
    v = torch.randn(MHA_SHAPE, device="cuda", dtype=dtype)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    bias = None

    do = torch.randn_like(q)
    dq = torch.zeros_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dbias = torch.empty_like(bias) if bias is not None else None

    # configurations
    softmax_scale = q.shape[-1] ** (-0.5)
    alibi_slopes = None
    causal = True
    cu_seqlens_q = None
    cu_seqlens_k = None
    max_seqlen_q = SEQ_LEN
    max_seqlen_k = SEQ_LEN

    dropout_p = 0.0
    if dropout_p > 0.0:
        dropout_mask = sd_mask >= 0
    else:
        dropout_mask = None

    # reference attention_fwd
    with torch.enable_grad():
        out, attn, lse = attention_ref(
            q, k, v,
            dropout_p=dropout_p,
            dropout_mask=dropout_mask,
            causal=causal,
        )

    # reference attention_bwd
    torch_dq, torch_dk, torch_dv = torch.autograd.grad(
        out, (q, k, v), do
    )

    # triton attention_bwd
    with torch.enable_grad():
        triton_dq, triton_dk, triton_dv = flash_attn_onekernel_backward(
            do,
            q, k, v,
            out, lse,
            dq, dk, dv,
            dbias,
            softmax_scale,
            alibi_slopes,
            causal,
            None,
            None,
            max_seqlen_q=q.shape[1],
            max_seqlen_k=k.shape[1],
            dropout_p=dropout_p,
            # philox_seed=philox_seed,
            # philox_offset=philox_offset,
            # USE_INT64_STRIDES=_USE_INT64_STRIDES,
        )

    # numeric check
    torch.testing.assert_close(
        triton_dq, torch_dq.to(out.dtype), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        triton_dk, torch_dk.to(out.dtype), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        triton_dv, torch_dv.to(out.dtype), atol=1e-2, rtol=1e-2
    )


if __name__ == "__main__":
    from absl import app
    app.run(main)
