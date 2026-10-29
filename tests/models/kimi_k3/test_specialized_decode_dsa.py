# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode (decode_backend=latency_specialized): the DSA
layer kernels against the stock cache writers, quantizers and a torch sparse
attention reference."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the specialized decode kernels need SM100/SM103",
)

UNIT, MLA_ROW, INDEX_ROW = 64, 656, 132  # indexer page tokens; cache row bytes
GAP = 36  # rows of the other layers' pages between two blocks of a packed cache
FAR = 5 << 30  # bytes a far packed cache spans: its later blocks lie past 2 GiB
HEADS = 12  # TP8
GATE = HEADS * 128
PITCH = 2112 + GATE + 128 + 32  # packed [q_a | kv_a | gate | wk | wv] row
EPS = 1e-6


def packed_cache(blocks, block, row, gap=GAP):
    """One layer's view of a packed (BLHNC) cache: ``blocks`` blocks of ``block``
    rows, ``gap`` rows of other layers' pages between two consecutive blocks
    (``FAR``: as many as spread the blocks over ``FAR`` bytes)."""
    stride = FAR // blocks // 256 * 256 if gap == FAR else (block + gap) * row
    raw = torch.zeros(blocks * stride, dtype=torch.uint8, device="cuda")
    return torch.as_strided(raw, (blocks, block, row), (stride, row, 1))


def index_blocks(index):
    """Each indexer page's bytes: its FP8 keys, then their FP32 scales."""
    return torch.as_strided(
        index, (index.shape[0], index.shape[1] * INDEX_ROW), (index.stride(0), 1)
    )


def rms_norm(x, weight):
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return (x * weight.float()).to(torch.bfloat16)


def kernels(batch, mla, index, *weights):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        layout,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.prepared import (
        compile_core,
    )

    specs = tuple(layout(w) for w in (*weights, mla, index))
    return compile_core(
        batch, HEADS, PITCH, specs, torch.accelerator.current_device_index(), 1
    )


@pytest.mark.parametrize(
    "block,gap", [(64, GAP), (1024, GAP), (64, FAR)], ids=["64", "1024", "far"]
)
@pytest.mark.parametrize("batch", [1, 2, 5, 16, 64])
def test_query_and_cache_insert(batch, block, gap):
    """BF16 query rows (W_UK absorbed), normalized q_c and the fp8_ds_mla KV row
    as the stock path computes and writes them, into the packed MLA blocks."""
    import cutlass

    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )

    torch.manual_seed(batch)
    bf, dev = torch.bfloat16, "cuda"
    projected = torch.randn(batch, PITCH, device=dev, dtype=bf)
    q_norm = torch.rand(1536, device=dev, dtype=bf) + 0.5
    kv_norm = torch.rand(512, device=dev, dtype=bf) + 0.5
    q_b = torch.randn(HEADS * 192, 1536, device=dev, dtype=bf) / 40
    w_uk = torch.randn(HEADS, 128, 512, device=dev, dtype=bf) / 12
    w_uv = torch.randn(HEADS, 128, 512, device=dev, dtype=bf).transpose(1, 2)
    blocks = 4 if gap == FAR else 2
    mla = packed_cache(blocks, block, MLA_ROW, gap)
    index = packed_cache(8, UNIT, INDEX_ROW)
    slots = torch.randperm(blocks * block - 1, device=dev)[:batch]
    slots[0] = blocks * block - 1  # the last row
    if batch > 2:
        slots[1] = -1  # a padded row writes nothing
    weights = q_norm, kv_norm, q_b, w_uk, w_uv
    query_kernel, _, _ = kernels(batch, mla, index, *weights)
    query = torch.zeros(batch, 64, 576, device=dev, dtype=bf)
    q_c = torch.zeros(batch, 1536, device=dev, dtype=bf)
    unit = torch.ones(1, device=dev, dtype=torch.float32)
    args = (to_cute(projected), *(to_cute(w) for w in weights[:4]), to_cute(mla))
    args += (to_cute(slots, 8), to_cute(unit, 4), to_cute(unit, 4))
    query_kernel(*args, cutlass.Float32(EPS), to_cute(query), to_cute(q_c), stream(dev))

    ref_q_c = rms_norm(projected[:, :1536], q_norm)
    torch.testing.assert_close(q_c, ref_q_c, atol=1e-2, rtol=1e-2)
    q = (ref_q_c @ q_b.T).view(batch, HEADS, 192)
    latent = torch.bmm(q[..., :128].transpose(0, 1), w_uk).transpose(0, 1)
    torch.testing.assert_close(
        query[:, :HEADS], torch.cat([latent, q[..., 128:]], -1), atol=2e-2, rtol=2e-2
    )
    assert (query[:, HEADS:] == 0).all()

    stock = packed_cache(blocks, block, MLA_ROW, gap)
    kv = rms_norm(projected[:, 1536:2048], kv_norm)
    ops.concat_and_cache_mla(
        kv, projected[:, 2048:2112], stock, slots, "fp8_ds_mla", unit
    )
    live = slots[slots >= 0]
    ours, ref = (c.reshape(-1, MLA_ROW)[live] for c in (mla, stock))
    assert torch.equal(ours[:, 512:], ref[:, 512:])  # tile scales and k_pe
    # Latent bytes differ only where the RMSNorm rounds differently.
    assert (ours[:, :512] != ref[:, :512]).float().mean() < 0.01


@pytest.mark.parametrize("gap", [0, GAP, FAR])
@pytest.mark.parametrize("batch", [1, 3, 64])
def test_indexer_inputs(batch, gap):
    """The FP8 indexer query, its folded head weights and the indexer key insert
    match per_token_group_quant_fp8 (ue8m0) and indexer_k_quant_and_cache, into
    64-entry indexer pages, contiguous or packed between other layers' pages."""
    import cutlass

    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )

    torch.manual_seed(batch)
    bf, dev = torch.bfloat16, "cuda"
    projected = torch.randn(batch, PITCH, device=dev, dtype=bf)
    iq = torch.randn(batch, 32 * 128, device=dev, dtype=bf)
    norm_w = torch.rand(128, device=dev) + 0.5
    norm_b = torch.randn(128, device=dev) / 10
    mla, index = packed_cache(8, UNIT, MLA_ROW), packed_cache(8, UNIT, INDEX_ROW, gap)
    slots = torch.randperm(8 * UNIT - 1, device=dev)[:batch]
    slots[0] = 8 * UNIT - 1  # the last entry
    if batch > 2:
        slots[1] = -1  # a padded row or, under DCP, a token another rank owns
    weights = (
        torch.ones(1536, device=dev, dtype=bf),
        torch.ones(512, device=dev, dtype=bf),
        torch.zeros(HEADS * 192, 1536, device=dev, dtype=bf),
        torch.zeros(HEADS, 128, 512, device=dev, dtype=bf),
        torch.zeros(HEADS, 128, 512, device=dev, dtype=bf).transpose(1, 2),
    )
    _, index_kernel, _ = kernels(batch, mla, index, *weights)
    q_fp8 = torch.zeros(batch, 32, 128, device=dev, dtype=torch.uint8)
    head_w = torch.zeros(batch, 32, device=dev, dtype=torch.float32)
    args = (to_cute(projected), to_cute(iq), to_cute(norm_w, 4), to_cute(norm_b, 4))
    args += (to_cute(index), to_cute(slots, 8), to_cute(q_fp8), to_cute(head_w, 4))
    index_kernel(*args, cutlass.Float32(EPS), stream(dev))

    ref_q, ref_scale = per_token_group_quant_fp8(
        iq.view(-1, 128), 128, column_major_scales=False, use_ue8m0=True
    )
    assert torch.equal(q_fp8.view(-1, 128), ref_q.view(torch.uint8))
    key = 2112 + GATE
    wv = projected[:, key + 128 : key + 160] * 32**-0.5
    ref_w = wv.float() * ref_scale.view(batch, 32) * 128**-0.5
    torch.testing.assert_close(head_w, ref_w, atol=0, rtol=0)

    stock = packed_cache(8, UNIT, INDEX_ROW, gap)
    k = torch.nn.functional.layer_norm(
        projected[:, key : key + 128].float(), (128,), norm_w, norm_b, EPS
    ).to(bf)
    ops.indexer_k_quant_and_cache(k, stock, slots, 128, "ue8m0")
    ours, ref = index_blocks(index), index_blocks(stock)
    assert (ours != ref).float().mean() < 1e-3


def fill_cache(mla):
    (blocks, block, _), bf = mla.shape, torch.bfloat16
    latent = torch.randn(blocks, block, 512, device="cuda") * 2
    scales = 2.0 ** torch.randint(-6, -2, (blocks, block, 4), device="cuda").float()
    k_pe = (torch.randn(blocks, block, 64, device="cuda") * 0.1).to(bf)
    mla[:, :, :512] = latent.to(torch.float8_e4m3fn).view(torch.uint8)
    mla[:, :, 512:528] = scales.view(torch.uint8).view(blocks, block, 16)
    mla[:, :, 528:] = k_pe.view(torch.uint8).view(blocks, block, 128)


def sparse_reference(query, topk, table, mla, scale, q):
    rows, block = mla.reshape(-1, MLA_ROW), mla.shape[1]  # a copy of the rows
    out = torch.zeros(query.shape[0], HEADS, 512, device="cuda")
    for t in range(query.shape[0]):
        index = topk[t][topk[t] >= 0].long()
        if index.numel() == 0:
            continue
        row = rows[table[t // q, index // block].long() * block + index % block]
        latent = row[:, :512].view(torch.float8_e4m3fn).float().view(-1, 4, 128)
        scales = row[:, 512:528].view(torch.float32).view(-1, 4, 1)
        latent = (latent * scales).view(-1, 512)
        key = torch.cat([latent, row[:, 528:].view(torch.bfloat16).float()], -1)
        p = torch.softmax(query[t, :HEADS].float() @ key.T * scale, -1)
        out[t] = p @ latent
    return out


@pytest.mark.parametrize(
    "block,gap", [(64, GAP), (1024, GAP), (64, FAR)], ids=["64", "1024", "far"]
)
@pytest.mark.parametrize("batch,q", [(1, 1), (8, 1), (32, 1), (64, 1), (8, 4)])
def test_sparse_decode(batch, q, block, gap):
    """SparseMlaDecode over logical top-k entries (q rows per request, -1 padded)
    against softmax attention over the dequantized rows of packed MLA blocks."""
    import cutlass

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        dynamic_view,
        layout,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.prepared import (
        compile_sparse,
    )

    torch.manual_seed(batch)
    dev, reqs = "cuda", batch // q
    blocks = 3 * reqs + 8
    mla = packed_cache(blocks, block, MLA_ROW, gap)
    fill_cache(mla)
    # A block table whose pitch is not a multiple of 64 entries.
    table = torch.zeros(reqs + 3, 37, device=dev, dtype=torch.int32)[:reqs]
    table[:, :3] = torch.randperm(blocks, device=dev)[: reqs * 3].view(reqs, 3)
    lengths = torch.randint(q, min(3 * block, 2048), (reqs,), device=dev)
    topk = torch.full((batch, 2048), -1, device=dev, dtype=torch.int32)
    for t in range(batch):
        n = int(lengths[t // q]) - (q - 1 - t % q)
        topk[t, torch.randperm(2048, device=dev)[:n]] = torch.randperm(
            n, device=dev, dtype=torch.int32
        )
    if batch > 2:
        topk[1] = -1  # a row without entries is zero
    query = torch.zeros(batch, 64, 576, device=dev, dtype=torch.bfloat16)
    query[:, :HEADS] = torch.randn(batch, HEADS, 576, device=dev, dtype=torch.bfloat16)
    latent = torch.zeros(batch, 64, 512, device=dev, dtype=torch.bfloat16)
    scale = 192**-0.5
    kernel = compile_sparse(
        batch, HEADS, layout(mla), torch.accelerator.current_device_index(), q
    )
    args = (to_cute(query), to_cute(topk, 4), dynamic_view(table, 4), to_cute(mla))
    kernel(*args, to_cute(latent[:, :HEADS]), cutlass.Float32(scale), stream(dev))
    ref = sparse_reference(query, topk, table, mla, scale, q)
    torch.testing.assert_close(latent[:, :HEADS].float(), ref, atol=3e-2, rtol=3e-2)
    assert ((latent[:, :HEADS].float() - ref).norm() / ref.norm()) < 5e-3


def test_sparse_decode_ignores_masked_rows():
    """-1 top-k entries (contexts shorter than the top-k) contribute nothing, even
    when the cache row they are read from holds NaN codes."""
    import cutlass

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        dynamic_view,
        layout,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.prepared import (
        compile_sparse,
    )

    torch.manual_seed(0)
    dev, batch = "cuda", 2
    mla = packed_cache(8, UNIT, MLA_ROW)
    fill_cache(mla)
    mla[0] = 0xFF  # block 0 (slot 0, read for masked entries): NaN codes
    table = torch.arange(1, 3, device=dev, dtype=torch.int32).repeat(batch, 1)
    topk = torch.full((batch, 2048), -1, device=dev, dtype=torch.int32)
    topk[:, :100] = torch.arange(100, device=dev, dtype=torch.int32)
    query = torch.zeros(batch, 64, 576, device=dev, dtype=torch.bfloat16)
    query[:, :HEADS] = torch.randn(batch, HEADS, 576, device=dev, dtype=torch.bfloat16)
    latent = torch.zeros(batch, 64, 512, device=dev, dtype=torch.bfloat16)
    device = torch.accelerator.current_device_index()
    kernel = compile_sparse(batch, HEADS, layout(mla), device, 1)
    args = (to_cute(query), to_cute(topk, 4), dynamic_view(table, 4), to_cute(mla))
    kernel(*args, to_cute(latent[:, :HEADS]), cutlass.Float32(0.07), stream(dev))
    ref = sparse_reference(query, topk, table, mla, 0.07, 1)
    assert latent.isfinite().all()
    torch.testing.assert_close(latent[:, :HEADS].float(), ref, atol=3e-2, rtol=3e-2)
