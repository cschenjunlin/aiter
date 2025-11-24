# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Dict
import torch
import triton
import triton.language as tl  # type: ignore

from aiter.ops.triton._triton_kernels.mha_fused_bwd import (
    _bwd_preprocess,
    _bwd_kernel_dkdvdq_causal,
    _bwd_kernel_dkdvdq_noncausal,
    _get_config,
)


def flash_attn_fused_backward(
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
    if q.shape[-1] == k.shape[-1] and k.shape[-1] > v.shape[-1]:
        raise ValueError(
            "'Fused' backward doesn't support Positional Encoding (PE). Please use 'one kernel' backward implementation for PE."
        )

    IS_VARLEN = True if cu_seqlens_q is not None else False

    # get strides and shape
    if IS_VARLEN:
        # Layout for q,k,v is thd ie [total tokens, num_head, head_dim]
        batch, seqlen_q, num_q_heads, head_sz = (
            len(cu_seqlens_q) - 1,
            max_seqlen_q,
            q.shape[1],
            q.shape[2],
        )
        _, num_k_heads = max_seqlen_k, k.shape[1]
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
        dq_strides = (0, dq.stride(1), dq.stride(0), dq.stride(2))
        dk_strides = (0, dk.stride(1), dk.stride(0), dk.stride(2))
        do_strides = (0, do.stride(1), do.stride(0), do.stride(2))
    else:
        # Layout for q,k,v is bshd ie [batch, seq_len, num_head, head_dim]
        batch, seqlen_q, num_q_heads, head_sz = q.shape
        _, num_k_heads = k.shape[1], k.shape[2]
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))
        dq_strides = (dq.stride(0), dq.stride(2), dq.stride(1), dq.stride(3))
        dk_strides = (dk.stride(0), dk.stride(2), dk.stride(1), dk.stride(3))
        do_strides = (do.stride(0), do.stride(2), do.stride(1), do.stride(3))

    # BLOCK_D_MODEL, BLOCK_D_MODEL_POW2
    # padding for head_dim. Power of 2 or 16
    BLOCK_D_MODEL_POW2 = triton.next_power_of_2(head_sz)
    BLOCK_D_MODEL_POW2 = max(BLOCK_D_MODEL_POW2, 16)

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
    if config is None:
        config = _get_config()

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
        BLOCK_D_MODEL=head_sz,
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

    # Fuses dk,dv and dq computations into one kernel using atomics
    if BLOCK_D_MODEL_POW2 > 160 or q.dtype == torch.float32:
        config_dkdvdq = config["dkdvdq_kernel_N64"]
    else:
        config_dkdvdq = config["dkdvdq_kernel_N128"]

    num_k_pids = (max_seqlen_k + config_dkdvdq["BLOCK_N"] - 1) // config_dkdvdq[
        "BLOCK_N"
    ]
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    if causal:
        grid_dkdvdq = (batch * num_q_heads * num_k_pids,)

        _bwd_kernel_dkdvdq_causal[grid_dkdvdq](
            q,
            k,
            v,
            sm_scale,
            do,
            dk,
            dv,
            dq,
            softmax_lse,
            delta,
            *q_strides,
            *k_strides,
            *v_strides,
            *dk_strides,
            *dq_strides,
            *delta_strides,
            *do_strides,
            *dropout_strides,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_mask,
            dropout_p,
            philox_seed,
            philox_offset,
            NUM_Q_HEADS=num_q_heads,
            NUM_K_HEADS=num_k_heads,
            BATCH=batch,
            NUM_K_PIDS=num_k_pids,
            BLOCK_D_MODEL=head_sz,
            BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            NUM_SMS=NUM_SMS,
            USE_INT64_STRIDES=USE_INT64_STRIDES,
            NUM_XCD=8,
            **config_dkdvdq,
        )
    else:
        # in non causal inner loop over grouped q heads
        grid_dkdvdq = (batch * num_k_heads * num_k_pids,)
        _bwd_kernel_dkdvdq_noncausal[grid_dkdvdq](
            q,
            k,
            v,
            sm_scale,
            do,
            dk,
            dv,
            dq,
            softmax_lse,
            delta,
            *q_strides,
            *k_strides,
            *v_strides,
            *dk_strides,
            *dq_strides,
            *delta_strides,
            *do_strides,
            *dropout_strides,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_mask,
            dropout_p,
            philox_seed,
            philox_offset,
            NUM_Q_HEADS=num_q_heads,
            NUM_K_HEADS=num_k_heads,
            BATCH=batch,
            NUM_K_PIDS=num_k_pids,
            BLOCK_D_MODEL=head_sz,
            BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            NUM_SMS=NUM_SMS,
            USE_INT64_STRIDES=USE_INT64_STRIDES,
            **config_dkdvdq,
        )

    return dq, dk, dv
