# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mm import KimiK3FlashForConditionalGeneration
    from .model import KimiK3FlashForCausalLM
    from .mtp import KimiK3FlashMTP

__all__ = [
    "KimiK3FlashForCausalLM",
    "KimiK3FlashForConditionalGeneration",
    "KimiK3FlashMTP",
]


def __getattr__(name: str):
    if name == "KimiK3FlashForCausalLM":
        from .model import KimiK3FlashForCausalLM

        return KimiK3FlashForCausalLM
    if name == "KimiK3FlashForConditionalGeneration":
        from .mm import KimiK3FlashForConditionalGeneration

        return KimiK3FlashForConditionalGeneration
    if name == "KimiK3FlashMTP":
        from .mtp import KimiK3FlashMTP

        return KimiK3FlashMTP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
