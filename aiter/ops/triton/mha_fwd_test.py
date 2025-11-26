# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops.triton.mha import flash_attn_func
from aiter.ops.triton.mha_fwd import _flash_attn_forward
from aiter.test_mha_common import attention_ref

_USE_INT64_STRIDES = True


BATCH_SIZE: int = 1
SEQ_LEN: int = 128
NUM_HEADS: int = 16
HEAD_SIZE: int = 32
MHA_SHAPE: tuple[int, int, int, int] = (BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_SIZE)
assert all(dim > 0 for dim in MHA_SHAPE)
dtype = torch.float16


def main(unused_argv):
    # create input data
    torch.cuda.empty_cache()
    torch.manual_seed(42)

    q = torch.randn(MHA_SHAPE, device="cuda", dtype=dtype)
    k = torch.randn(MHA_SHAPE, device="cuda", dtype=dtype)
    v = torch.randn(MHA_SHAPE, device="cuda", dtype=dtype)
    bias = None

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

    # # reference attention_fwd
    # out, _, lse = attention_ref(
    #     q, k, v,
    #     dropout_p=dropout_p,
    #     dropout_mask=dropout_mask,
    #     causal=causal,
    # )

    for i in range(100):
        # Triton attention_fwd
        triton_out, triton_lse, _, _, _ = _flash_attn_forward(
            q, k, v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=-1,
            window_size_right=-1,
            bias=bias,
            alibi_slopes=alibi_slopes,
            return_lse=True,
            return_softmax=True,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
        )

    # # numeric check
    # torch.testing.assert_close(
    #     out, triton_out.to(out.dtype), atol=1e-2, rtol=1e-2
    # )
    # torch.testing.assert_close(
    #     lse, triton_lse.to(lse.dtype), atol=1e-2, rtol=1e-2
    # )
    # print("Test passed!")


if __name__ == "__main__":
    from absl import app
    app.run(main)
