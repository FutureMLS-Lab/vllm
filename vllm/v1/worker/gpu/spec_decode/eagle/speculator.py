# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
    TargetDependentARSpeculator,
)


class EagleSpeculator(TargetDependentARSpeculator):
    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        model = load_eagle_model(target_model, self.vllm_config)
        # Lets the target fit the drafter to its own KV cache layout.
        if (adapt := getattr(target_model, "adapt_drafter", None)) is not None:
            adapt(model, self.vllm_config)
        return model
