# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Priming the fast incremental detokenizer with a short prompt suffix
(VLLM_DETOKENIZE_PRIME_SUFFIX)."""

import pytest
import tokenizers.decoders
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import TokenizersBackend

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import (
    FastIncrementalDetokenizer,
    IncrementalDetokenizer,
    prompt_suffix_to_prime,
)


def byte_tokenizer() -> TokenizersBackend:
    """One token per UTF-8 byte, so a token inside a multi-byte character
    decodes to a replacement character on its own."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokenizer = Tokenizer(
        models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[])
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    tokenizer.decoder = decoders.ByteLevel()
    return TokenizersBackend(tokenizer_object=tokenizer)


TOKENIZER = byte_tokenizer()


def encode(text: str) -> list[int]:
    return TOKENIZER._tokenizer.encode(text).ids


def make_request(prompt_token_ids: list[int]) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="test",
        external_req_id="test-ext",
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=SamplingParams(skip_special_tokens=True),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def stream(prompt_token_ids: list[int], output_token_ids: list[int]) -> list[str]:
    """The text deltas streamed for the output tokens, one token per step."""
    detokenizer = IncrementalDetokenizer.from_new_request(
        TOKENIZER, make_request(prompt_token_ids)
    )
    assert isinstance(detokenizer, FastIncrementalDetokenizer)
    deltas = []
    for token_id in output_token_ids:
        detokenizer.update([token_id], stop_terminated=False)
        deltas.append(detokenizer.get_next_output_text(finished=False, delta=True))
    return deltas


def test_suffix_is_the_shortest_that_decodes_cleanly():
    tokenizer = TOKENIZER._tokenizer
    ascii_prompt = encode("the quick brown fox")
    assert prompt_suffix_to_prime(tokenizer, ascii_prompt, True) == ascii_prompt[-4:]

    # The four-token suffix starts inside a three-byte character.
    split = encode("abc\u4e16de")
    assert prompt_suffix_to_prime(tokenizer, split, True) == split[-5:]

    # A prompt that ends inside a character has no clean suffix.
    dangling = encode("abcdef \u4e16")[:-1]
    assert prompt_suffix_to_prime(tokenizer, dangling, True) == dangling

    # Nor has one whose last 23 tokens are all continuation bytes.
    continuation = encode("\u00e9")[-1]
    no_boundary = encode("abc") + [continuation] * 30
    assert prompt_suffix_to_prime(tokenizer, no_boundary, True) == no_boundary

    short = encode("abcd")
    assert prompt_suffix_to_prime(tokenizer, short, True) == short
    assert prompt_suffix_to_prime(tokenizer, None, True) is None


@pytest.mark.parametrize(
    ("prompt", "output"),
    [
        (
            encode("Hello, w\u00f6rld! "),
            encode("Gr\u00fc\u00dfe aus \u6771\u4eac \U0001f642"),
        ),
        (encode("\u4fa1\u683c\u306f"), encode("1,000\u5186\u3067\u3059")),
        (encode("emoji \U0001f642\U0001f642"), encode("\U0001f642 done")),
        # The character that crosses the prompt's end is primed in full.
        (encode("ab \u6771")[:-1], encode("\u6771")[-1:] + encode("\u4eac and more")),
        (encode("x" * 40), encode("\u00e9t\u00e9")),
    ],
)
def test_streamed_text_is_the_same_with_either_priming(monkeypatch, prompt, output):
    monkeypatch.delenv("VLLM_DETOKENIZE_PRIME_SUFFIX", raising=False)
    whole_prompt = stream(prompt, output)
    monkeypatch.setenv("VLLM_DETOKENIZE_PRIME_SUFFIX", "1")
    suffix = stream(prompt, output)
    assert suffix == whole_prompt
    full = TOKENIZER._tokenizer.decode(prompt + output)
    assert full.endswith("".join(whole_prompt))


def test_only_the_switch_shortens_the_primed_tokens(monkeypatch):
    primed: list[list[int]] = []
    decode_stream = tokenizers.decoders.DecodeStream

    def recording_decode_stream(ids=None, skip_special_tokens=False):
        primed.append(list(ids or []))
        return decode_stream(ids=ids, skip_special_tokens=skip_special_tokens)

    monkeypatch.setattr(tokenizers.decoders, "DecodeStream", recording_decode_stream)
    prompt = encode("a prompt of a few dozen tokens, primed one way or the other")

    monkeypatch.delenv("VLLM_DETOKENIZE_PRIME_SUFFIX", raising=False)
    stream(prompt, encode("!"))
    assert primed[-1] == prompt

    monkeypatch.setenv("VLLM_DETOKENIZE_PRIME_SUFFIX", "1")
    stream(prompt, encode("!"))
    assert primed[-1] == prompt[-4:]
