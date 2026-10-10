# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 Flash multimodal model.

The multimodal stack follows the Kimi-K3 pattern: a vLLM-native MoonViT3d
vision tower (``vllm.model_executor.models.kimi_k25_vit``) built from
``vision_config``, a patchmerger projector, the standard ``image`` modality
with the checkpoint's own image processor (remote code), and
resolution-aware prompt updates. The text backbone is ``KimiK3FlashModel``
(MLA + MoE with DSA sparse attention).
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

import torch
from torch import nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions, ImageDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors,
)
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.kimi_k25 import KimiK25MediaPixelInputs
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    MoonViT3dPretrainedModel,
    vision_tower_forward,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.model_executor.models.vision import is_vit_use_data_parallel
from vllm.models.kimi_k3.common.mm_preprocess import navit_resize_image
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    NestedTensors,
)
from vllm.multimodal.parse import (
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processor import cached_get_image_processor

from .configs import KimiK3FlashConfig
from .model import KimiK3FlashForCausalLM
from .processor import KimiK3FlashProcessor

logger = init_logger(__name__)


class KimiK3FlashProcessingInfo(BaseProcessingInfo):
    """Processing information for the image-only Kimi K3 Flash model.

    Uses the standard ``image`` modality and resolves the ``<|media_pad|>``
    token id the same way Kimi-K3 does.
    """

    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__(ctx)

        self.hf_config = hf_config = self.get_hf_config()

        tokenizer = self.get_tokenizer()
        image_processor = cached_get_image_processor(
            self.ctx.model_config.model,
            revision=self.ctx.model_config.revision,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
        )

        # Resolve token ID from the tokenizer because transformers v5
        # may remap token IDs vs config.json.
        config_token_id = hf_config.media_placeholder_token_id
        resolved_token_id = tokenizer.convert_tokens_to_ids("<|media_pad|>")
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        is_valid_resolved = isinstance(resolved_token_id, int) and (
            unk_token_id is None or resolved_token_id != unk_token_id
        )
        if is_valid_resolved and resolved_token_id != config_token_id:
            logger.warning_once(
                "Kimi K3 Flash config.media_placeholder_token_id (%d) disagrees "
                "with tokenizer mapping for <|media_pad|> (%d). "
                "Using tokenizer value.",
                config_token_id,
                resolved_token_id,
            )
            media_token_id = resolved_token_id
            # Patch config so downstream code also sees the correct ID.
            hf_config.media_placeholder_token_id = resolved_token_id
        else:
            media_token_id = config_token_id

        self.media_token_id = media_token_id
        self.media_token = tokenizer.decode(media_token_id)

        self.image_processor = image_processor
        self.hf_processor = KimiK3FlashProcessor(
            tokenizer=tokenizer,
            image_processor=image_processor,
        )
        self.media_tokens_calculator = image_processor.media_tokens_calculator

    def get_hf_processor(self, **kwargs: object) -> KimiK3FlashProcessor:
        return self.hf_processor

    def get_hf_config(self) -> KimiK3FlashConfig:
        return self.ctx.get_hf_config(KimiK3FlashConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # None means unlimited
        return {"image": None}

    @classmethod
    def get_max_image_size(
        cls,
        patch_size: int,
        merge_kernel_size: int,
        in_patch_limit: int,
        patch_limit_on_one_side: int,
        fixed_output_tokens: int | None,
    ) -> ImageSize:
        max_side = patch_limit_on_one_side * patch_size
        best_score = (-1, -1)
        best_size = (max_side, max_side)

        for width_patches in range(patch_limit_on_one_side + 1):
            width = min((width_patches + 1) * patch_size - 1, max_side)
            for height_patches in range(width_patches, patch_limit_on_one_side + 1):
                height = min((height_patches + 1) * patch_size - 1, max_side)
                resize_config = navit_resize_image(
                    width,
                    height,
                    patch_size,
                    merge_kernel_size,
                    in_patch_limit,
                    patch_limit_on_one_side,
                    fixed_output_tokens,
                )
                padded_width = resize_config["new_width"] + resize_config["pad_width"]
                padded_height = (
                    resize_config["new_height"] + resize_config["pad_height"]
                )
                num_patches = padded_width // patch_size * (padded_height // patch_size)
                score = (resize_config["num_tokens"], num_patches)
                if score > best_score:
                    best_score = score
                    best_size = (width, height)
        return ImageSize(width=best_size[0], height=best_size[1])


class KimiK3FlashDummyInputsBuilder(BaseDummyInputsBuilder[KimiK3FlashProcessingInfo]):
    """Builds image-based dummy inputs for Kimi K3 Flash profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return self.info.get_hf_config().image_placeholder * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        media_proc_cfg = self.info.image_processor.media_proc_cfg
        max_size = self.info.get_max_image_size(
            media_proc_cfg["patch_size"],
            media_proc_cfg["merge_kernel_size"],
            media_proc_cfg["in_patch_limit"],
            media_proc_cfg["patch_limit_on_one_side"],
            media_proc_cfg["fixed_output_tokens"],
        )
        num_images = mm_counts.get("image", 0)
        image_overrides = cast(
            ImageDummyOptions | None,
            mm_options.get("image") if mm_options else None,
        )
        return {
            "image": self._get_dummy_images(
                width=max_size.width,
                height=max_size.height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class KimiK3FlashMultiModalProcessor(
    BaseMultiModalProcessor[KimiK3FlashProcessingInfo]
):
    """Image-only multi-modal processor for Kimi K3 Flash."""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Override so the base always routes through the text+mm path
        # (`KimiK3FlashProcessor.__call__`). Otherwise the mm-only fast path calls
        # the checkpoint image processor directly with bare PIL images, but it
        # requires `{"type": "image", "image": PIL}` media dicts that only our
        # wrapper builds.
        return super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Slice the flattened patch tensor back into per-image items.

        ``pixel_values`` holds all patches from every image concatenated; each
        image's patch count is ``prod(grid_thws[i])``. ``grid_thws`` is one
        ``[N_t, N_h, N_w]`` row per image.
        """
        grid_thws = hf_inputs.get("grid_thws", torch.empty((0, 3)))
        grid_sizes = grid_thws.prod(-1)

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", grid_sizes),
            grid_thws=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Expand each image placeholder into a resolution-aware update.

        The prompt carries a single image placeholder token per image. This
        replaces that token with
        ``<|media_begin|>image {w}x{h}<|media_content|>{pads}<|media_end|>``,
        embedding the per-image resolution in the prompt and marking only the
        ``<|media_pad|>`` positions as embedding slots (the number of pads is
        the feature size returned by ``media_tokens_calculator``).
        """
        media_token_id = self.info.media_token_id
        tokenizer = self.info.get_tokenizer()
        image_placeholder = self.info.get_hf_config().image_placeholder
        placeholder_ids = tokenizer.encode(image_placeholder, add_special_tokens=False)

        def get_replacement(item_idx: int) -> PromptUpdateDetails:
            images = mm_items.get_items("image", ImageProcessorItems)
            image = images.get(item_idx)
            if image is None:
                raise ValueError(f"Missing image data at index {item_idx}")

            # The checkpoint image processor works on media dicts, so wrap the
            # PIL image before asking it for the token count.
            num_media_token = self.info.media_tokens_calculator(
                {"type": "image", "image": image}
            )

            # NOTE: `width`/`height` are the ORIGINAL upload dimensions, not
            # the post-preprocess (smart-resized) ones. `image` comes from the
            # untouched parsed `mm_items`; the checkpoint image processor only
            # produces new tensors via `image.resize(...)` and never mutates
            # the stored PIL. The resize is reflected only in the pad count.
            width, height = images.get_image_size(item_idx)
            prefix_ids = tokenizer.encode(
                f"<|media_begin|>image {width}x{height}<|media_content|>",
                add_special_tokens=False,
            )
            suffix_ids = tokenizer.encode("<|media_end|>", add_special_tokens=False)
            full = prefix_ids + [media_token_id] * num_media_token + suffix_ids

            return PromptUpdateDetails.select_token_id(full, media_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=placeholder_ids,
                replacement=get_replacement,
            ),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    KimiK3FlashMultiModalProcessor,
    info=KimiK3FlashProcessingInfo,
    dummy_inputs=KimiK3FlashDummyInputsBuilder,
)
class KimiK3FlashForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    """Kimi K3 Flash model with MoonViT3d vision and the KimiK3Flash text backbone."""

    supports_encoder_tp_data = True

    # The checkpoint names the patchmerger linears `mm_projector.proj.{0,2}`;
    # the language-model subtree (language_model.model.* /
    # language_model.lm_head.*) matches KimiK3FlashForCausalLM as-is.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "mm_projector.proj.0.": "mm_projector.linear_1.",
            "mm_projector.proj.2.": "mm_projector.linear_2.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return "<|kimi_image_placeholder|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config: KimiK3FlashConfig = model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config

        multimodal_config = model_config.multimodal_config
        assert multimodal_config is not None
        self.use_data_parallel = is_vit_use_data_parallel(
            config.vision_config.num_attention_heads
        )
        self.device = current_platform.current_device()

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = MoonViT3dPretrainedModel(
                config.vision_config,
                quant_config=self._maybe_ignore_quant_config(quant_config),
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            if self._maybe_ignore_quant_config(quant_config) is not None:
                self.vision_tower = self.vision_tower.to(device=self.device)
            else:
                self.vision_tower = self.vision_tower.to(
                    device=self.device, dtype=model_config.dtype
                )
            self.mm_projector = KimiK25MultiModalProjector(
                config=config.vision_config,
                use_data_parallel=self.use_data_parallel,
                quant_config=self._maybe_ignore_quant_config(quant_config),
                prefix=maybe_prefix(prefix, "mm_projector"),
            )
            self.mm_projector = self.mm_projector.to(
                device=self.device, dtype=model_config.dtype
            )

        with self._mark_language_model(vllm_config):
            self.language_model = KimiK3FlashForCausalLM(
                vllm_config=vllm_config.with_hf_config(config.text_config),
                prefix=maybe_prefix(prefix, "language_model"),
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self.media_placeholder: int = self.config.media_placeholder_token_id

    def _maybe_ignore_quant_config(
        self, quant_config: QuantizationConfig | None
    ) -> QuantizationConfig | None:
        if isinstance(quant_config, compressed_tensors.CompressedTensorsConfig):
            return None
        return quant_config

    def _parse_and_validate_media_input(
        self, **kwargs: object
    ) -> KimiK25MediaPixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        grid_thws = kwargs.pop("grid_thws", None)
        if pixel_values is None:
            return None

        if isinstance(pixel_values, list):
            pixel_values = torch.cat(cast(list[torch.Tensor], pixel_values), dim=0)
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError(
                "pixel_values must be a tensor or a list of tensors, "
                f"got {type(pixel_values)}"
            )

        if len(pixel_values.shape) == 5 or len(pixel_values.shape) == 3:
            pixel_values = pixel_values.reshape(
                pixel_values.shape[0] * pixel_values.shape[1], *pixel_values.shape[2:]
            )

        target_dtype = next(self.vision_tower.parameters()).dtype
        pixel_values = pixel_values.to(target_dtype)
        assert isinstance(grid_thws, torch.Tensor), (
            f"expect grid_thws to be a tensor, got {type(grid_thws)}"
        )
        grid_thws = grid_thws.reshape(-1, grid_thws.shape[-1])
        assert grid_thws.ndim == 2 and grid_thws.size(1) == 3, (
            f"unexpected shape for grid_thws: {grid_thws.shape}"
        )

        return KimiK25MediaPixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            grid_thws=grid_thws,
        )

    def _process_media_input(
        self, media_input: KimiK25MediaPixelInputs
    ) -> list[torch.Tensor]:
        media_features = vision_tower_forward(
            self.vision_tower,
            media_input["pixel_values"],
            media_input["grid_thws"],
            mm_projector=self.mm_projector,
            use_data_parallel=self.use_data_parallel,
        )
        return media_features

    def embed_multimodal(self, **kwargs: object) -> NestedTensors | None:
        media_input = self._parse_and_validate_media_input(**kwargs)
        if media_input is None:
            return None
        return self._process_media_input(media_input)

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The language_model child (KimiK3FlashForCausalLM) loads the text
        # backbone itself (stacked/expert mappings, DSA indexer renames and
        # fp8 dequant); AutoWeightsLoader delegates to it and loads the
        # vision tower and projector after applying hf_to_vllm_mapper.
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
