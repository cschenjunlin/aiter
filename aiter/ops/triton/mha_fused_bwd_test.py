# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops.triton.mha_onekernel_bwd import flash_attn_fused_backward
from aiter.test_mha_common import attention_ref


BATCH_SIZE: int = 1
SEQ_LEN: int = 128
NUM_HEADS: int = 16
HEAD_SIZE: int = 32
MHA_SHAPE: tuple[int, int, int, int] = (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_SIZE)
assert all(dim > 0 for dim in MHA_SHAPE)
MHA_DTYPE = torch.float32
RNG_SEED = 42


def main(unused_argv):
    # generate input data, causal = True
    torch.manual_seed(RNG_SEED)

    q = torch.randn(MHA_SHAPE, dtype=MHA_DTYPE)
    k = torch.randn(MHA_SHAPE, dtype=MHA_DTYPE)
    v = torch.randn(MHA_SHAPE, dtype=MHA_DTYPE)

    # configurations
    sm_scale = HEAD_SIZE ** -0.5
    causal = True
    alibi_slopes = None
    cu_seqlens_q = cu_seqlens_k = None
    max_seqlen_q = max_seqlen_k = SEQ_LEN
    dropout_p = 0.0

    # save fwd outputs for bwd
    o, softmax_lse = mha_fwd_reference(q, k, v, sm_scale=sm_scale, causal=causal)

    # random upstream gradient
    do = torch.randn_like(q)  # only when D_v == D_q

    # bwd outputs
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    # move all tensors to device
    device = torch.device('cuda')  # in rocm, maps to the hip
    do, q, k, v, o, softmax_lse = [x.to(device) for x in (do, q, k, v, o, softmax_lse)]

    # Triton results
    dq, dk, dv = flash_attn_fused_backward(
        # Input tensors
        do=do,
        q=q,
        k=k,
        v=v,
        o=o,
        softmax_lse=softmax_lse,
        # Output tensors
        dq=dq,
        dk=dk,
        dv=dv,
        dbias=None,
        # Configurations
        sm_scale=sm_scale,
        causal=causal,
        alibi_slopes=alibi_slopes,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
    )
    
    dq.block_until_ready()
    print(dq)


if __name__ == "__main__":
    from absl import app
    app.run(main)
