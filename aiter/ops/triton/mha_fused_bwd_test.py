# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops.triton.mha import flash_attn_func
from aiter.ops.triton.mha_fused_bwd import flash_attn_fused_backward
from aiter.test_mha_common import attention_ref

_USE_INT64_STRIDES = True


BATCH_SIZE: int = 1
SEQ_LEN: int = 128
NUM_HEADS: int = 16
HEAD_SIZE: int = 32
MHA_SHAPE: tuple[int, int, int, int] = (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_SIZE)
assert all(dim > 0 for dim in MHA_SHAPE)
dtype = torch.float16


def main(unused_argv):
    # create input data
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
    philox_seed = 0
    philox_offset = 0

    # reference attention_fwd
    with torch.enable_grad():
        out, _, lse = attention_ref(
            q, k, v,
            dropout_p=dropout_p,
            dropout_mask=dropout_mask,
            causal=causal,
        )

    # 1. reference attention_bwd
    torch_dq, torch_dk, torch_dv = torch.autograd.grad(
        out, (q, k, v), do
    )

    # Triton attention_bwd
    with torch.enable_grad():
        triton_out, triton_lse, _ = flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            causal=causal,
            return_lse=True,
            return_attn_probs=True,
        )

    # 2. Triton attention_bwd, use torch.autograd
    triton_auto_dq, triton_auto_dk, triton_auto_dv = torch.autograd.grad(
        triton_out, (q, k, v), do.clone()
    )

    # 3. Triton attention_bwd, directly call
    triton_dq, triton_dk, triton_dv = flash_attn_fused_backward(
        do,
        q, k, v,
        triton_out, triton_lse,
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
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        USE_INT64_STRIDES=_USE_INT64_STRIDES,
    )
    
    # numeric check
    torch.testing.assert_close(
        dq, triton_dq.to(dq.dtype), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        dk, triton_dk.to(dk.dtype), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        dv, triton_dv.to(dv.dtype), atol=1e-2, rtol=1e-2
    )


if __name__ == "__main__":
    from absl import app
    app.run(main)
