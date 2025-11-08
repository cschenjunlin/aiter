# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
import logging
import numpy as np
from aiter.ops.triton.mha import (
    flash_attn_func,
    mha_set_use_fused_bwd_kernel,
)
from aiter.test_mha_common import (
    attention_ref,
    generate_qkv,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
DEBUG_MODE = False


@pytest.mark.parametrize("BATCH", [1, 4, 57, 128])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(1, 1), (4, 4), (128, 128), (2, 1), (1, 2), (32, 16), (64, 128)],
)
@pytest.mark.parametrize("DROPOUT, CAUSAL", [(0.0, False), (0.0, True), (0.2, False)])
# @pytest.mark.parametrize('DROPOUT, CAUSAL',[(0.0, False),(0.0, True),(0.2, False),(0.2, True)]) #Debug Causal + Dropout. fails for seq >= 64
@pytest.mark.parametrize(
    "NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (16, 16), (2, 1), (48, 8)]
)
@pytest.mark.parametrize("HEAD_SZ", [8, 32, 128])
@pytest.mark.parametrize("FUSED", [False, True])
def test_mha_backward(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    DROPOUT: float,
    CAUSAL: bool,
    FUSED: bool,
    dtype=torch.float16,
):
    torch.cuda.empty_cache()
    torch.manual_seed(20)

    # pytest.skip("Backward accuracy issues due to Triton compiler")
    if FUSED and CAUSAL:
        pytest.skip("FUSED+CAUSAL results in NaNs")
    mha_set_use_fused_bwd_kernel(FUSED)
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

    do = torch.randn_like(q)

    if DEBUG_MODE:
        print("--------------Triton----------------")
        print(f"q.shape={q.shape} q={q}")
        print(f"k.shape={k.shape} k={k}")
        print(f"v.shape={v.shape} v={v}")
        print(f"do.shape={do.shape} do={do}")

    with torch.enable_grad():
        triton_out = flash_attn_func(
            q,
            k,
            v,
            dropout_p=DROPOUT,
            causal=CAUSAL,
            return_lse=True,
            return_attn_probs=True,
        )

    assert len(triton_out) == 3
    triton_out, lse, sd_mask = triton_out[0], triton_out[1], triton_out[2]

    if DROPOUT > 0.0:
        dropout_mask = sd_mask >= 0
    else:
        dropout_mask = None

    triton_dq, triton_dk, triton_dv = torch.autograd.grad(
        triton_out, (q, k, v), do.clone()
    )

    if DEBUG_MODE:
        print(f"triton_out={triton_out}")
        print(f"triton_lse={lse}")
        print(f"sd_mask={sd_mask}")
        print(f"triton_dq.shape={triton_dq.shape} triton_dq={triton_dq}")
        print(f"triton_dk.shape={triton_dk.shape} triton_dk={triton_dk}")
        print(f"triton_dv.shape={triton_dv.shape} triton_dv={triton_dv}")
        print(f"dropout_mask={dropout_mask}")

    if DEBUG_MODE:
        print("--------------Torch----------------")
        print(f"q.shape={q.shape} q={q}")
        print(f"k.shape={k.shape} k={k}")
        print(f"v.shape={v.shape} v={v}")
        print(f"do.shape={do.shape} do={do}")
    with torch.enable_grad():
        torch_out = attention_ref(
            q, k, v, dropout_p=DROPOUT, dropout_mask=dropout_mask, causal=CAUSAL
        )
    torch_out, attention_scores, lse = torch_out

    torch.testing.assert_close(
        triton_out, torch_out.to(triton_out.dtype), atol=1e-2, rtol=1e-2
    )

    torch_dq, torch_dk, torch_dv = torch.autograd.grad(torch_out, (q, k, v), do)

    if DEBUG_MODE:
        print(f"torch_out={torch_out}")
        print(f"torch_attn_scores={attention_scores}")
        print(f"torch_dq.shape={torch_dq.shape} torch_dq={torch_dq}")
        print(f"torch_dk.shape={torch_dk.shape} torch_dk={torch_dk}")
        print(f"torch_dv.shape={torch_dv.shape} torch_dv={torch_dv}")

    torch.testing.assert_close(
        triton_dq, torch_dq.to(triton_out.dtype), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        triton_dk, torch_dk.to(triton_out.dtype), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        triton_dv, torch_dv.to(triton_out.dtype), atol=1e-2, rtol=1e-2
    )
