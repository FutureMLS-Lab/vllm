# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The FlashKDA build that stores prefill checkpoints from registers is an
optional extension: without it, in-chunk exports of KDA states go through
the regular FlashKDA build instead of failing at startup."""

import sys
from types import ModuleType

import pytest

from vllm.models.glm5next.common import kda

pytestmark = pytest.mark.cpu_test


def test_a_missing_export_build_falls_back(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "vllm._flashkda_export_C", None)
    assert not kda.flashkda_export_build_available()


def test_an_installed_export_build_is_used(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(
        sys.modules, "vllm._flashkda_export_C", ModuleType("vllm._flashkda_export_C")
    )
    assert kda.flashkda_export_build_available()
