# SPDX-License-Identifier: AGPL-3.0-only
"""Let's Infer generic persistent prefix cache as a vLLM v1 KV connector.

Out-of-tree connector for the pinned SparkInfer image (vLLM 0.15.1+befbc472):

    --kv-transfer-config '{"kv_connector": "LetsInferPrefixConnector",
                           "kv_role": "kv_both",
                           "kv_connector_module_path":
                               "letsinfer_prefix_connector.connector"}'

Persistence and lookup live in the Rust `letsinfer_prefix_store` (PyO3):
exact-token authority, CRC'd page-aligned records, atomic commits,
byte-LRU + TTL, writer admission.

Hybrid-correct capture policy
-----------------------------
For hybrid attention/recurrent models, recurrent state exists only at the
position the engine has actually processed, so a record is valid
only if its attention KV and its SSM state describe the SAME boundary.
The only places both materialize together during prefill are the
chunked-prefill step boundaries. Therefore:

- capture happens at the last configured INTERMEDIATE chunked-prefill
  boundary P (P = num_computed + num_scheduled, P < prompt_len, and the
  remaining suffix is at most `capture_tail_tokens`): the pages then hold
  exactly KV[0..P) and SSM@P without writing every earlier chunk;
- P is the request's actual scheduler-step boundary. Under concurrency it may
  fall inside an attention superpage; that partial page is retained and vLLM's
  existing copy-on-write path isolates it before the suffix is appended;
- P < prompt_len guarantees a full-prompt repeat can restore the
  record (the scheduler requires at least one token left to compute);
- on the one-sequence MTP-off lane, or the explicitly opted-in native-MTP
  qualification lane, the final boundary also captures the final prompt
  hidden row. A repeat reports one synthetic token to satisfy the scheduler,
  restores complete KV/recurrent/draft state, and supplies that hidden row through
  the verified connector hook instead of executing the target model;
- full resident attention blocks remain zero-copy.

Regions per record: one per attention layer and one per recurrent-state
tensor per recurrent layer in sorted-layer-name order. Attention regions are
named "a<group>:<ordinal>"; recurrent regions add ":<state>". The final
"meta" region echoes the boundary for post-restore assertions. Short
ordinal names fit the store's 12-byte name field, and both sides derive
the same order from the pinned KV-cache configuration.

Known v1 limits (documented, revisit by measurement): the one tail capture is
synchronous at the boundary step's end; preemption is not handled (release
gates skip cells that exceed the live KV pool); the scheduler-process store
handle is reopened on miss so same-lifetime captures become visible for
warm-repeat hits.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import MambaSpec

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

# Use vLLM's logger namespace so out-of-tree lifecycle evidence follows the
# server's configured INFO policy and is recorded in benchmark logs.
logger = init_logger("vllm.letsinfer_prefix_connector")

ABI_VERSION = 4
DEFAULT_ROOT = "/root/.cache/letsinfer-prefix-store"
DEFAULT_CAPACITY_BYTES = 64 * (1024**3)
DEFAULT_RESIDENT_CAPACITY_BYTES = 8 * (1024**3)
DEFAULT_NATIVE_CAPACITY_BYTES = 8 * (1024**3)
DEFAULT_TTL_SECONDS = 7 * 24 * 3600
DEFAULT_MIN_TOKENS = 4096
DEFAULT_CAPTURE_TAIL_TOKENS = 2048


def _exact_capsules_enabled(extra: dict[str, Any], vllm_config: VllmConfig) -> bool:
    """Gate final-hidden capsules to the configurations qualified by Let's Infer."""
    if not bool(extra.get("exact_capsules", True)):
        return False
    speculative_config = vllm_config.speculative_config
    speculation_supported = speculative_config is None or (
        bool(extra.get("exact_capsules_with_mtp", False))
        and getattr(speculative_config, "method", None) == "mtp"
    )
    return bool(
        speculation_supported
        and vllm_config.scheduler_config.max_num_seqs == 1
        and vllm_config.parallel_config.pipeline_parallel_size == 1
    )


@dataclass
class _ReqPlan:
    """One capture or restore action for one request in this step."""

    req_id: str
    kind: str  # "capture" | "restore"
    boundary: int  # token count P
    token_ids: list[int]  # exact token ids [:P]
    # Per-KV-group full block id lists for this request.
    group_block_ids: list[list[int]]
    # A final-boundary record also carries the final prompt hidden state.
    has_hidden: bool = False
    # Restore the capsule's hidden state instead of executing a synthetic token.
    skip_forward: bool = False


@dataclass(frozen=True)
class _RegionPlan:
    """One record region mapped to one engine-owned cache tensor."""

    name: str
    layer_name: str
    block_ids: list[int]
    # None for an attention KV tensor; 0/1/... for list-valued Mamba state.
    state_index: int | None


@dataclass(frozen=True)
class _PendingRestore:
    boundary: int
    computed_boundary: int
    token_ids: list[int]
    source: str  # "durable" | "native"
    native_key: bytes
    has_hidden: bool = False
    skip_forward: bool = False


@dataclass
class _SchedulerNativeEntry:
    boundary: int
    token_ids: list[int]
    group_block_ids: tuple[tuple[int, ...], ...]
    record_bytes: int
    has_hidden: bool = False


@dataclass
class _WorkerNativeEntry:
    boundary: int
    token_ids: list[int]
    group_block_ids: tuple[tuple[int, ...], ...]
    recurrent: dict[str, torch.Tensor]
    record_bytes: int
    hidden: torch.Tensor | None = None
    has_hidden: bool = False


@dataclass
class LetsInferPrefixConnectorMetadata(KVConnectorMetadata):
    plans: list[_ReqPlan] = field(default_factory=list)


class LetsInferPrefixConnector(KVConnectorBase_V1, SupportsHMA):
    """Generic prefix cache: paged KV plus recurrent state, restart-durable."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        self._store_args = (
            str(extra.get("store_root", DEFAULT_ROOT)),
            int(extra.get("capacity_bytes", DEFAULT_CAPACITY_BYTES)),
            int(extra.get("ttl_seconds", DEFAULT_TTL_SECONDS)),
            int(extra.get("min_tokens", DEFAULT_MIN_TOKENS)),
            int(
                extra.get(
                    "resident_capacity_bytes", DEFAULT_RESIDENT_CAPACITY_BYTES
                )
            ),
            bool(extra.get("direct_reads", True)),
        )
        self._native_capacity_bytes = int(
            extra.get("native_capacity_bytes", DEFAULT_NATIVE_CAPACITY_BYTES)
        )
        self._capture_tail_tokens = int(
            extra.get("capture_tail_tokens", DEFAULT_CAPTURE_TAIL_TOKENS)
        )
        if self._capture_tail_tokens <= 0:
            raise ValueError("capture_tail_tokens must be positive")
        exact_requested = bool(extra.get("exact_capsules", True))
        self._exact_enabled = _exact_capsules_enabled(extra, vllm_config)
        text_config = getattr(
            vllm_config.model_config,
            "hf_text_config",
            vllm_config.model_config.hf_config,
        )
        hidden_size = int(getattr(text_config, "hidden_size", 0))
        model_dtype = vllm_config.model_config.dtype
        self._hidden_bytes = hidden_size * torch.empty(
            (), dtype=model_dtype
        ).element_size()
        self._store = self._open_store()
        self._min_tokens = self._store_args[3]
        self._requires_scale_basis = str(
            vllm_config.cache_config.cache_dtype
        ).startswith("fp8")
        self._scale_basis: dict[str, dict[str, dict[str, Any]]] = {}
        self._fingerprint = self._compute_fingerprint(vllm_config, kv_cache_config)
        if role.name == "SCHEDULER":
            fingerprint_path = Path(self._store_args[0]) / ".active-fingerprint"
            temporary = fingerprint_path.with_name(
                f"{fingerprint_path.name}.{os.getpid()}.tmp"
            )
            temporary.write_text(self._fingerprint.hex() + "\n", encoding="ascii")
            os.replace(temporary, fingerprint_path)

        # Group geometry shared by both roles.
        self._groups = list(kv_cache_config.kv_cache_groups)
        self._num_cache_blocks = kv_cache_config.num_blocks
        attention_block_sizes = {
            group.kv_cache_spec.block_size
            for group in self._groups
            if not isinstance(group.kv_cache_spec, MambaSpec)
        }
        if len(attention_block_sizes) != 1:
            raise ValueError(
                "Let's Infer requires one attention block size, got "
                f"{sorted(attention_block_sizes)}"
            )
        self._attn_block_size = attention_block_sizes.pop()
        self._layer_to_group: dict[str, int] = {}
        for group_index, group in enumerate(self._groups):
            for layer_name in group.layer_names:
                self._layer_to_group[layer_name] = group_index
        self._attention_group_indices = {
            index
            for index, group in enumerate(self._groups)
            if not isinstance(group.kv_cache_spec, MambaSpec)
        }
        self._group_block_bytes = self._derive_group_block_bytes(kv_cache_config)

        # Scheduler-role state.
        self._req_tokens: dict[str, list[int]] = {}
        self._req_computed: dict[str, int] = {}
        self._req_blocks: dict[str, list[list[int]]] = {}
        self._req_captured: dict[str, set[int]] = {}
        self._pending_restore: dict[str, _PendingRestore] = {}
        self._gpu_block_pool: BlockPool | None = None
        self._scheduler_native: OrderedDict[bytes, _SchedulerNativeEntry] = (
            OrderedDict()
        )
        self._scheduler_native_bytes = 0

        # Worker-role state.
        self._kv_caches: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        self._worker_native: OrderedDict[bytes, _WorkerNativeEntry] = OrderedDict()
        self._worker_native_bytes = 0
        self._attention_rows_per_block: dict[int, int] = {}
        self._capture_hidden: dict[str, torch.Tensor] = {}
        self._restore_hidden: dict[str, torch.Tensor] = {}

        logger.info(
            "LetsInferPrefixConnector(role=%s): %d KV groups, "
            "attn_block_size=%d, min_tokens=%d, capture_tail_tokens=%d, "
            "fingerprint=%s",
            role.name,
            len(self._groups),
            self._attn_block_size,
            self._min_tokens,
            self._capture_tail_tokens,
            self._fingerprint.hex()[:16],
        )
        if exact_requested and not self._exact_enabled:
            logger.info(
                "Let's Infer exact capsules disabled: require max_num_seqs=1, "
                "pipeline_parallel_size=1, and either speculation off or "
                "explicit native-MTP qualification"
            )

    @staticmethod
    def _derive_group_block_bytes(kv_cache_config: "KVCacheConfig") -> list[int]:
        """Physical bytes represented by one block ID in each cache group."""
        num_blocks = kv_cache_config.num_blocks
        result: list[int] = []
        for group in kv_cache_config.kv_cache_groups:
            layer_names = set(group.layer_names)
            total = 0
            for tensor in kv_cache_config.kv_cache_tensors:
                if layer_names.intersection(tensor.shared_by):
                    total += tensor.size // num_blocks
            result.append(total)
        return result

    def _native_key(self, token_ids: list[int]) -> bytes:
        digest = hashlib.sha256(self._fingerprint)
        for start in range(0, len(token_ids), 4096):
            chunk = token_ids[start : start + 4096]
            digest.update(struct.pack(f"<{len(chunk)}I", *chunk))
        return digest.digest()

    def _native_record_bytes(
        self, group_block_ids: list[list[int]], has_hidden: bool = False
    ) -> int:
        return sum(
            len(block_ids) * self._group_block_bytes[index]
            for index, block_ids in enumerate(group_block_ids)
        ) + (self._hidden_bytes if has_hidden else 0)

    def _resident_group_ids(self, plan: _ReqPlan) -> list[list[int]]:
        """Lease only state that belongs to the exact prefix boundary."""
        result: list[list[int]] = []
        for group, block_ids in zip(
            self._groups, plan.group_block_ids, strict=True
        ):
            spec = group.kv_cache_spec
            if isinstance(spec, MambaSpec):
                result.append(list(block_ids))
            else:
                count = (plan.boundary + spec.block_size - 1) // spec.block_size
                result.append(list(block_ids[:count]))
        return result

    def _open_store(self):
        import letsinfer_prefix_store

        return letsinfer_prefix_store.PrefixStore(*self._store_args)

    @staticmethod
    def _compute_fingerprint(
        vllm_config: VllmConfig, kv_cache_config: "KVCacheConfig"
    ) -> bytes:
        """State-compatibility digest: any mismatch must be a miss."""
        model_config = vllm_config.model_config
        parts: list[Any] = [
            ABI_VERSION,
            model_config.model,
            getattr(model_config, "revision", None),
            getattr(model_config, "tokenizer", None),
            getattr(model_config, "tokenizer_revision", None),
            getattr(model_config, "tokenizer_mode", None),
            str(model_config.dtype),
            str(vllm_config.cache_config.cache_dtype),
            vllm_config.cache_config.calculate_kv_scales,
            "record-verified-static-kv-scale-basis-v1",
            vllm_config.cache_config.block_size,
            vllm_config.parallel_config.world_size,
        ]
        for group in kv_cache_config.kv_cache_groups:
            parts.append(sorted(group.layer_names))
            parts.append(repr(group.kv_cache_spec))
        return hashlib.sha256(json.dumps(parts, default=str).encode("utf-8")).digest()

    # ------------------------------------------------------------------
    # Shared layout: region name -> engine layer/state tensor + block ids
    # ------------------------------------------------------------------

    def _plan_layout(self, plan: _ReqPlan) -> list[_RegionPlan]:
        """Derive the deterministic record-to-engine tensor mapping.

        vLLM represents each GDN layer as a list containing its convolution
        and recurrent tensors. Those must be separate regions: their dtypes
        and shapes differ and both are required for restart-exact restore.
        """
        layout: list[_RegionPlan] = []
        for ordinal, layer_name in enumerate(sorted(self._layer_to_group)):
            group_index = self._layer_to_group[layer_name]
            block_ids = plan.group_block_ids[group_index]
            spec = self._groups[group_index].kv_cache_spec
            if isinstance(spec, MambaSpec):
                ids = list(block_ids)
                for state_index in range(len(spec.shapes)):
                    layout.append(
                        _RegionPlan(
                            name=f"m{group_index}:{ordinal:03d}:{state_index}",
                            layer_name=layer_name,
                            block_ids=ids,
                            state_index=state_index,
                        )
                    )
            else:
                count = (plan.boundary + spec.block_size - 1) // spec.block_size
                ids = list(block_ids[:count])
                if ids:
                    layout.append(
                        _RegionPlan(
                            name=f"a{group_index}:{ordinal:03d}",
                            layer_name=layer_name,
                            block_ids=ids,
                            state_index=None,
                        )
                    )
        return layout

    def _region_tensor(self, region: _RegionPlan) -> torch.Tensor:
        cache = self._kv_caches.get(region.layer_name)
        if cache is None:
            raise RuntimeError(
                f"Let's Infer KV cache has no layer {region.layer_name!r}"
            )
        if region.state_index is None:
            if not isinstance(cache, torch.Tensor):
                raise RuntimeError(
                    f"Let's Infer attention layer {region.layer_name!r} is list-valued"
                )
            return cache
        if not isinstance(cache, (list, tuple)):
            raise RuntimeError(
                f"Let's Infer recurrent layer {region.layer_name!r} is tensor-valued"
            )
        try:
            return cache[region.state_index]
        except IndexError as error:
            raise RuntimeError(
                f"Let's Infer recurrent layer {region.layer_name!r} has no state "
                f"index {region.state_index}"
            ) from error

    def _region_tensor_block_ids(
        self, region: _RegionPlan, tensor: torch.Tensor
    ) -> list[int]:
        """Map scheduler superpage IDs to physical attention rows.

        Hybrid vLLM enlarges an attention scheduler block (2,096 tokens in
        the MTP-off lane) while FlashInfer keeps its 16-token physical row.
        The registered attention tensor therefore has multiple physical rows
        per scheduler block. Recurrent tensors remain one row per block.
        """
        if region.state_index is not None:
            return region.block_ids
        group_index = self._layer_to_group[region.layer_name]
        rows_per_block = self._attention_rows_per_block.get(group_index)
        if rows_per_block is None:
            if tensor.shape[0] % self._num_cache_blocks != 0:
                raise RuntimeError(
                    f"Let's Infer attention tensor {region.layer_name!r} has "
                    f"{tensor.shape[0]} rows for {self._num_cache_blocks} "
                    "scheduler blocks"
                )
            rows_per_block = tensor.shape[0] // self._num_cache_blocks
            if rows_per_block <= 0:
                raise RuntimeError("Let's Infer attention row multiplier is zero")
            self._attention_rows_per_block[group_index] = rows_per_block
        return [
            block_id * rows_per_block + row
            for block_id in region.block_ids
            for row in range(rows_per_block)
        ]

    # ------------------------------------------------------------------
    # Scheduler role
    # ------------------------------------------------------------------

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        self._gpu_block_pool = gpu_block_pool

    def _scheduler_native_lookup(
        self, prompt: list[int]
    ) -> tuple[bytes, _SchedulerNativeEntry] | None:
        best: tuple[bytes, _SchedulerNativeEntry] | None = None
        for key, entry in self._scheduler_native.items():
            if (
                (
                    entry.boundary < len(prompt)
                    or (
                        self._exact_enabled
                        and entry.has_hidden
                        and entry.boundary == len(prompt)
                    )
                )
                and entry.boundary >= self._min_tokens
                and (best is None or entry.boundary > best[1].boundary)
                and entry.token_ids == prompt[: entry.boundary]
            ):
                best = (key, entry)
        if best is not None:
            self._scheduler_native.move_to_end(best[0])
        return best

    def _insert_scheduler_native(self, plan: _ReqPlan) -> bool:
        pool = self._gpu_block_pool
        if pool is None or self._native_capacity_bytes == 0:
            return False
        key = self._native_key(plan.token_ids)
        if key in self._scheduler_native:
            existing = self._scheduler_native[key]
            if not plan.has_hidden or existing.has_hidden:
                self._scheduler_native.move_to_end(key)
                return True
            self._scheduler_native.pop(key)
            resident_ids = {
                block_id
                for group_index in self._attention_group_indices
                for block_id in existing.group_block_ids[group_index]
            }
            pool.free_blocks([pool.blocks[block_id] for block_id in resident_ids])
            self._scheduler_native_bytes -= existing.record_bytes
        resident_group_ids = self._resident_group_ids(plan)
        record_bytes = self._native_record_bytes(
            resident_group_ids, plan.has_hidden
        )
        if record_bytes > self._native_capacity_bytes:
            return False
        while (
            self._scheduler_native
            and self._scheduler_native_bytes + record_bytes
            > self._native_capacity_bytes
        ):
            _, victim = self._scheduler_native.popitem(last=False)
            resident_ids = {
                block_id
                for group_index in self._attention_group_indices
                for block_id in victim.group_block_ids[group_index]
            }
            pool.free_blocks([pool.blocks[block_id] for block_id in resident_ids])
            self._scheduler_native_bytes -= victim.record_bytes

        group_ids = tuple(tuple(ids) for ids in resident_group_ids)
        resident_ids = {
            block_id
            for group_index in self._attention_group_indices
            for block_id in group_ids[group_index]
        }
        blocks = [pool.blocks[block_id] for block_id in resident_ids]
        if not blocks or any(block.ref_cnt <= 0 for block in blocks):
            return False
        # Baseline connector ownership. vLLM's patched adoption path adds and
        # later releases a separate request reference.
        pool.touch(blocks)
        self._scheduler_native[key] = _SchedulerNativeEntry(
            boundary=plan.boundary,
            token_ids=list(plan.token_ids),
            group_block_ids=group_ids,
            record_bytes=record_bytes,
            has_hidden=plan.has_hidden,
        )
        self._scheduler_native_bytes += record_bytes
        logger.info(
            "Let's Infer native scheduler PROMOTED boundary=%d bytes=%d "
            "entries=%d attention_leases=%s",
            plan.boundary,
            record_bytes,
            len(self._scheduler_native),
            {
                index: list(group_ids[index])
                for index in sorted(self._attention_group_indices)
            },
        )
        return True

    def get_resident_block_ids(
        self, request: "Request", num_external_tokens: int
    ) -> tuple[list[int], ...] | None:
        pending = self._pending_restore.get(request.request_id)
        if (
            pending is None
            or pending.source != "native"
            or pending.computed_boundary != num_external_tokens
        ):
            return None
        entry = self._scheduler_native.get(pending.native_key)
        if entry is None:
            return None
        return tuple(
            list(ids) if index in self._attention_group_indices else []
            for index, ids in enumerate(entry.group_block_ids)
        )

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        prompt = list(request.prompt_token_ids or [])
        if len(prompt) < self._min_tokens:
            return 0, False
        native = self._scheduler_native_lookup(prompt)
        if native is not None:
            key, entry = native
            skip_forward = bool(
                entry.has_hidden
                and entry.boundary == len(prompt)
                and self._exact_enabled
            )
            computed_boundary = entry.boundary - 1 if skip_forward else entry.boundary
            if computed_boundary > num_computed_tokens:
                self._pending_restore[request.request_id] = _PendingRestore(
                    boundary=entry.boundary,
                    computed_boundary=computed_boundary,
                    token_ids=list(entry.token_ids),
                    source="native",
                    native_key=key,
                    has_hidden=entry.has_hidden,
                    skip_forward=skip_forward,
                )
                logger.info(
                    "Let's Infer prefix NATIVE HIT req=%s boundary=%d "
                    "computed=%d prompt=%d exact=%s",
                    request.request_id,
                    entry.boundary,
                    computed_boundary,
                    len(prompt),
                    skip_forward,
                )
                return computed_boundary - num_computed_tokens, False
        reader = self._store.longest_prefix(
            self._fingerprint, prompt, self._min_tokens
        )
        if reader is None:
            # Same-lifetime captures were committed by the worker
            # process; reopen to rescan the directory once per miss.
            self._store = self._open_store()
            reader = self._store.longest_prefix(
                self._fingerprint, prompt, self._min_tokens
            )
        if reader is None:
            return 0, False
        boundary = reader.token_count
        has_hidden = "hidden" in reader.region_names
        reader.close()
        skip_forward = bool(
            has_hidden
            and boundary == len(prompt)
            and self._exact_enabled
        )
        # Intermediate records are strictly below the prompt and aligned.
        # Final capsules may be unaligned and report one fewer computed token;
        # the worker supplies the exact final hidden state for that synthetic
        # scheduler step without re-running the model.
        if (
            boundary > len(prompt)
            or (boundary == len(prompt) and not skip_forward)
        ):
            return 0, False
        computed_boundary = boundary - 1 if skip_forward else boundary
        if computed_boundary <= num_computed_tokens:
            return 0, False
        token_ids = prompt[:boundary]
        self._pending_restore[request.request_id] = _PendingRestore(
            boundary=boundary,
            computed_boundary=computed_boundary,
            token_ids=token_ids,
            source="durable",
            native_key=self._native_key(token_ids),
            has_hidden=has_hidden,
            skip_forward=skip_forward,
        )
        logger.info(
            "Let's Infer prefix HIT req=%s boundary=%d computed=%d "
            "prompt=%d exact=%s",
            request.request_id,
            boundary,
            computed_boundary,
            len(prompt),
            skip_forward,
        )
        return computed_boundary - num_computed_tokens, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        pending = self._pending_restore.get(request.request_id)
        if pending is None or num_external_tokens == 0:
            return
        if pending.source == "durable":
            block_ids = blocks.get_block_ids()
            self._insert_scheduler_native(
                _ReqPlan(
                    req_id=request.request_id,
                    kind="restore",
                    boundary=pending.boundary,
                    token_ids=list(pending.token_ids),
                    group_block_ids=[list(ids) for ids in block_ids],
                    has_hidden=pending.has_hidden,
                    skip_forward=pending.skip_forward,
                )
            )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = LetsInferPrefixConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = new_req.req_id
            prompt = list(new_req.prompt_token_ids or [])
            self._req_tokens[req_id] = prompt
            self._req_computed[req_id] = new_req.num_computed_tokens
            self._req_blocks[req_id] = [list(ids) for ids in new_req.block_ids]
            self._req_captured[req_id] = set()
            pending = self._pending_restore.pop(req_id, None)
            if pending is not None:
                plan = _ReqPlan(
                    req_id=req_id,
                    kind=(
                        "restore_native"
                        if pending.source == "native"
                        else "restore"
                    ),
                    boundary=pending.boundary,
                    token_ids=list(pending.token_ids),
                    group_block_ids=self._req_blocks[req_id],
                    has_hidden=pending.has_hidden,
                    skip_forward=pending.skip_forward,
                )
                meta.plans.append(plan)
                self._req_captured[req_id].add(pending.boundary)
            logger.info(
                "Let's Infer new req=%s prompt=%d computed=%d scheduled=%d blocks=%s",
                req_id,
                len(prompt),
                new_req.num_computed_tokens,
                scheduler_output.num_scheduled_tokens.get(req_id, 0),
                [len(ids) for ids in new_req.block_ids],
            )

        cached = scheduler_output.scheduled_cached_reqs
        for index, req_id in enumerate(cached.req_ids):
            new_block_ids = cached.new_block_ids[index]
            if req_id in self._req_blocks and new_block_ids is not None:
                for group_index, ids in enumerate(new_block_ids):
                    self._req_blocks[req_id][group_index].extend(ids)
            if req_id in self._req_computed:
                self._req_computed[req_id] = cached.num_computed_tokens[index]

        # Capture plans: actual intermediate scheduler-step boundaries plus
        # the exact final boundary when the qualified capsule lane is enabled.
        # Concurrent scheduling can end one request inside an attention
        # superpage; the partial page is safe because restore uses vLLM's
        # existing copy-on-write isolation before appending the suffix.
        for req_id, scheduled in scheduler_output.num_scheduled_tokens.items():
            prompt = self._req_tokens.get(req_id)
            if prompt is None:
                continue
            position = self._req_computed.get(req_id, 0) + scheduled
            exact_boundary = self._exact_enabled and position == len(prompt)
            tail_boundary = (
                position < len(prompt)
                and len(prompt) - position <= self._capture_tail_tokens
            )
            if (
                position <= len(prompt)
                and position >= self._min_tokens
                and "letsinfer-prewarm-" not in req_id
                and (exact_boundary or tail_boundary)
                and position not in self._req_captured[req_id]
            ):
                plan = _ReqPlan(
                    req_id=req_id,
                    kind="capture",
                    boundary=position,
                    token_ids=prompt[:position],
                    group_block_ids=[
                        list(ids) for ids in self._req_blocks[req_id]
                    ],
                    has_hidden=exact_boundary,
                )
                meta.plans.append(plan)
                self._insert_scheduler_native(plan)
                self._req_captured[req_id].add(position)
            self._req_computed[req_id] = position

        if meta.plans:
            logger.info(
                "Let's Infer step plans=%s",
                [(plan.kind, plan.req_id, plan.boundary) for plan in meta.plans],
            )

        return meta

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        self._forget(request.request_id)
        return False, None

    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        self._forget(request.request_id)
        return False, None

    def _forget(self, req_id: str) -> None:
        self._req_tokens.pop(req_id, None)
        self._req_computed.pop(req_id, None)
        self._req_blocks.pop(req_id, None)
        self._req_captured.pop(req_id, None)
        self._pending_restore.pop(req_id, None)
        self._capture_hidden.pop(req_id, None)
        self._restore_hidden.pop(req_id, None)

    # ------------------------------------------------------------------
    # Worker role
    # ------------------------------------------------------------------

    def register_kv_caches(
        self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]
    ) -> None:
        self._kv_caches = dict(kv_caches)
        for group_index in self._attention_group_indices:
            group = self._groups[group_index]
            layer_name = group.layer_names[0]
            tensor = self._kv_caches.get(layer_name)
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(
                    f"Let's Infer attention group {group_index} has no tensor"
                )
            if tensor.shape[0] % self._num_cache_blocks != 0:
                raise RuntimeError(
                    f"Let's Infer attention group {group_index} has "
                    f"{tensor.shape[0]} physical rows for "
                    f"{self._num_cache_blocks} scheduler blocks"
                )
            rows_per_block = tensor.shape[0] // self._num_cache_blocks
            if rows_per_block <= 0:
                raise RuntimeError("Let's Infer attention row multiplier is zero")
            self._attention_rows_per_block[group_index] = rows_per_block
        recurrent = sum(isinstance(value, (list, tuple)) for value in kv_caches.values())
        logger.info(
            "LetsInferPrefixConnector: registered %d KV cache layers "
            "(%d recurrent), attention_rows_per_block=%s",
            len(self._kv_caches),
            recurrent,
            self._attention_rows_per_block,
        )

    def register_kv_scale_basis(
        self, scale_basis: dict[str, dict[str, dict[str, Any]]]
    ) -> None:
        """Bind the exact immutable K/V scale bytes used by this process."""
        expected = {
            layer_name
            for group_index in self._attention_group_indices
            for layer_name in self._groups[group_index].layer_names
        }
        received = set(scale_basis)
        if self._requires_scale_basis and received != expected:
            missing = sorted(expected - received)
            extra = sorted(received - expected)
            raise RuntimeError(
                "Let's Infer FP8 KV scale basis does not match attention layers: "
                f"missing={missing}, extra={extra}"
            )
        for layer_name, scales in scale_basis.items():
            if set(scales) != {"k", "v"}:
                raise RuntimeError(
                    f"Let's Infer scale basis for {layer_name!r} must contain k/v"
                )
            for scale_name, record in scales.items():
                if not {
                    "dtype",
                    "shape",
                    "bytes_hex",
                }.issubset(record):
                    raise RuntimeError(
                        f"Let's Infer {layer_name} {scale_name} scale is incomplete"
                    )
        self._scale_basis = {
            name: scales for name, scales in sorted(scale_basis.items())
        }
        logger.info(
            "LetsInferPrefixConnector: registered immutable FP8 scale basis "
            "for %d attention layers",
            len(self._scale_basis),
        )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, LetsInferPrefixConnectorMetadata):
            return
        for plan in metadata.plans:
            if plan.kind == "restore":
                self._restore(plan)
            elif plan.kind == "restore_native":
                self._restore_native(plan)

    def capture_exact_hidden_states(
        self,
        hidden_states: Any,
        logits_indices: torch.Tensor,
        num_tokens_padded: int,
    ) -> None:
        """Snapshot the final prompt row before ``wait_for_save`` runs."""
        metadata = self._get_connector_metadata()
        plans = [
            plan
            for plan in getattr(metadata, "plans", ())
            if plan.kind == "capture" and plan.has_hidden
        ]
        if not plans:
            return
        if len(plans) != 1 or not isinstance(hidden_states, torch.Tensor):
            raise RuntimeError(
                "Let's Infer exact capture requires one tensor-output request"
            )
        if logits_indices.numel() != 1:
            raise RuntimeError(
                "Let's Infer exact capture does not support prompt logprobs"
            )
        hidden = hidden_states[logits_indices][0].detach().clone()
        self._capture_hidden[plans[0].req_id] = hidden

    def get_exact_hidden_states(
        self,
        logits_indices: torch.Tensor,
        num_tokens_padded: int,
    ) -> torch.Tensor | None:
        """Return a final-prompt capsule and skip the synthetic forward."""
        metadata = self._get_connector_metadata()
        plans = [
            plan
            for plan in getattr(metadata, "plans", ())
            if plan.skip_forward
        ]
        if not plans:
            return None
        if len(plans) != 1 or logits_indices.numel() != 1:
            raise RuntimeError(
                "Let's Infer exact restore requires one request and one logit row"
            )
        plan = plans[0]
        hidden = self._restore_hidden.pop(plan.req_id, None)
        if hidden is None:
            raise RuntimeError(
                f"Let's Infer exact hidden state missing for request {plan.req_id}"
            )
        output = hidden.new_zeros((num_tokens_padded, hidden.numel()))
        output.index_copy_(0, logits_indices.to(dtype=torch.long), hidden.reshape(1, -1))
        logger.info(
            "Let's Infer exact hidden RESTORED req=%s boundary=%d forward_skipped=1",
            plan.req_id,
            plan.boundary,
        )
        return output

    def _insert_worker_native(
        self, plan: _ReqPlan, hidden: torch.Tensor | None = None
    ) -> bool:
        if self._native_capacity_bytes == 0:
            return False
        key = self._native_key(plan.token_ids)
        if key in self._worker_native:
            existing = self._worker_native[key]
            if not plan.has_hidden or existing.has_hidden:
                self._worker_native.move_to_end(key)
                return True
            self._worker_native.pop(key)
            self._worker_native_bytes -= existing.record_bytes
        layout = self._plan_layout(plan)
        record_bytes = 0
        recurrent: dict[str, torch.Tensor] = {}
        for region in layout:
            tensor = self._region_tensor(region)
            tensor_block_ids = self._region_tensor_block_ids(region, tensor)
            region_bytes = (
                len(tensor_block_ids)
                * tensor[0].numel()
                * tensor.element_size()
            )
            record_bytes += region_bytes
            if region.state_index is not None:
                index = torch.tensor(
                    region.block_ids, dtype=torch.long, device=tensor.device
                )
                recurrent[region.name] = torch.index_select(tensor, 0, index)
        if plan.has_hidden:
            if hidden is None:
                raise RuntimeError("Let's Infer exact native entry is missing hidden state")
            record_bytes += hidden.numel() * hidden.element_size()
        if record_bytes > self._native_capacity_bytes:
            return False
        torch.cuda.synchronize()
        while (
            self._worker_native
            and self._worker_native_bytes + record_bytes
            > self._native_capacity_bytes
        ):
            _, victim = self._worker_native.popitem(last=False)
            self._worker_native_bytes -= victim.record_bytes
        resident_group_ids = self._resident_group_ids(plan)
        self._worker_native[key] = _WorkerNativeEntry(
            boundary=plan.boundary,
            token_ids=list(plan.token_ids),
            group_block_ids=tuple(tuple(ids) for ids in resident_group_ids),
            recurrent=recurrent,
            record_bytes=record_bytes,
            hidden=(hidden.detach().clone() if hidden is not None else None),
            has_hidden=plan.has_hidden,
        )
        self._worker_native_bytes += record_bytes
        logger.info(
            "Let's Infer native worker PROMOTED boundary=%d bytes=%d "
            "state_regions=%d entries=%d attention_leases=%s",
            plan.boundary,
            record_bytes,
            len(recurrent),
            len(self._worker_native),
            {
                index: list(resident_group_ids[index])
                for index in sorted(self._attention_group_indices)
            },
        )
        return True

    def _restore_native(self, plan: _ReqPlan) -> None:
        started = time.perf_counter()
        key = self._native_key(plan.token_ids)
        entry = self._worker_native.get(key)
        if (
            entry is None
            or entry.boundary != plan.boundary
            or entry.token_ids != plan.token_ids
        ):
            raise RuntimeError(
                f"Let's Infer native lease vanished before restore "
                f"(req={plan.req_id}, boundary={plan.boundary})"
            )
        self._worker_native.move_to_end(key)
        copied_bytes = 0
        attention_copy_bytes = 0
        shared_tokens = plan.boundary - 1 if plan.skip_forward else plan.boundary
        for group_index in self._attention_group_indices:
            spec = self._groups[group_index].kv_cache_spec
            request_ids = plan.group_block_ids[group_index]
            resident_ids = list(entry.group_block_ids[group_index])
            full_blocks = shared_tokens // spec.block_size
            if request_ids[:full_blocks] != resident_ids[:full_blocks]:
                raise RuntimeError(
                    "Let's Infer native attention lease mismatch "
                    f"group {group_index}: "
                    f"request={request_ids[:full_blocks]} "
                    f"resident={resident_ids[:full_blocks]}"
                )
            if shared_tokens % spec.block_size:
                if (
                    len(request_ids) <= full_blocks
                    or len(resident_ids) <= full_blocks
                    or request_ids[full_blocks] == resident_ids[full_blocks]
                ):
                    raise RuntimeError(
                        "Let's Infer partial native tail was not copy-on-write isolated"
                    )
                attention_copy_bytes += self._group_block_bytes[group_index]
            elif len(resident_ids) != full_blocks:
                raise RuntimeError(
                    "Let's Infer native attention lease length mismatch "
                    f"group {group_index}: expected={full_blocks} "
                    f"resident={len(resident_ids)}"
                )
        for region in self._plan_layout(plan):
            if region.state_index is None:
                continue
            source = entry.recurrent.get(region.name)
            if source is None:
                raise RuntimeError(
                    f"Let's Infer native lease missing recurrent region {region.name}"
                )
            tensor = self._region_tensor(region)
            tensor_block_ids = self._region_tensor_block_ids(region, tensor)
            index = torch.tensor(
                tensor_block_ids, dtype=torch.long, device=tensor.device
            )
            tensor.index_copy_(0, index, source)
            copied_bytes += source.numel() * source.element_size()
        if plan.skip_forward:
            if entry.hidden is None:
                raise RuntimeError("Let's Infer native exact entry has no hidden state")
            self._restore_hidden[plan.req_id] = entry.hidden
        torch.cuda.synchronize()
        logger.info(
            "Let's Infer prefix NATIVE RESTORED req=%s boundary=%d "
            "attention_copy_bytes=%d state_copy_bytes=%d total_ms=%.3f",
            plan.req_id,
            plan.boundary,
            attention_copy_bytes,
            copied_bytes,
            (time.perf_counter() - started) * 1000,
        )

    def _restore(self, plan: _ReqPlan) -> None:
        started = time.perf_counter()
        reader = self._store.longest_prefix(
            self._fingerprint, plan.token_ids, self._min_tokens
        )
        if reader is None or reader.token_count != plan.boundary:
            # The scheduler already marked these tokens computed; a
            # vanished record here is unrecoverable for this request, so
            # fail loudly rather than serve garbage.
            raise RuntimeError(
                f"Let's Infer prefix record vanished before restore "
                f"(req={plan.req_id}, boundary={plan.boundary})"
            )
        try:
            names = reader.region_names
            region_index = {name: i for i, name in enumerate(names)}
            meta_index = region_index.get("meta")
            if meta_index is None:
                raise RuntimeError("Let's Infer record missing meta region")
            meta_blob = json.loads(bytes(reader.read_region(meta_index)))
            if meta_blob.get("boundary") != plan.boundary:
                raise RuntimeError("Let's Infer prefix record boundary mismatch")
            if meta_blob.get("scale_basis") != self._scale_basis:
                raise RuntimeError("Let's Infer prefix record FP8 scale basis mismatch")
            layout = self._plan_layout(plan)
            expected_names = {region.name for region in layout} | {"meta"}
            if plan.has_hidden:
                expected_names.add("hidden")
            if set(names) != expected_names:
                raise RuntimeError("Let's Infer prefix record layout mismatch")
            restored_hidden: torch.Tensor | None = None
            if plan.has_hidden:
                hidden_info = meta_blob.get("hidden")
                if not isinstance(hidden_info, dict):
                    raise RuntimeError("Let's Infer exact record missing hidden metadata")
                dtype_name = hidden_info.get("dtype")
                dtype = {
                    "torch.bfloat16": torch.bfloat16,
                    "torch.float16": torch.float16,
                    "torch.float32": torch.float32,
                }.get(dtype_name)
                shape = hidden_info.get("shape")
                if dtype is None or not isinstance(shape, list) or not shape:
                    raise RuntimeError("Let's Infer exact hidden metadata is invalid")
                hidden_index = region_index["hidden"]
                expected_hidden_bytes = (
                    int(torch.tensor(shape).prod().item())
                    * torch.empty((), dtype=dtype).element_size()
                )
                if reader.region_byte_count(hidden_index) != expected_hidden_bytes:
                    raise RuntimeError("Let's Infer exact hidden byte count mismatch")
                hidden_bytes = torch.empty(
                    expected_hidden_bytes,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=True,
                )
                reader.read_region_into(hidden_index, hidden_bytes.numpy())
                device = self._region_tensor(layout[0]).device
                restored_hidden = (
                    hidden_bytes.view(dtype)
                    .reshape(tuple(int(value) for value in shape))
                    .to(device, non_blocking=True)
                )
            for region in layout:
                i = region_index.get(region.name)
                if i is None:
                    raise RuntimeError(
                        f"Let's Infer record missing region {region.name}"
                    )
                tensor = self._region_tensor(region)
                tensor_block_ids = self._region_tensor_block_ids(region, tensor)
                per_block_shape = tuple(tensor.shape[1:])
                expected_bytes = (
                    len(tensor_block_ids)
                    * tensor[0].numel()
                    * tensor.element_size()
                )
                if reader.region_byte_count(i) != expected_bytes:
                    raise RuntimeError(
                        f"Let's Infer region {region.name} expects {expected_bytes} "
                        f"bytes, record has {reader.region_byte_count(i)}"
                    )
                host_bytes = torch.empty(
                    expected_bytes,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=True,
                )
                reader.read_region_into(i, host_bytes.numpy())
                host = (
                    host_bytes
                    .view(tensor.dtype)
                    .reshape((len(tensor_block_ids), *per_block_shape))
                )
                index = torch.tensor(
                    tensor_block_ids, dtype=torch.long, device=tensor.device
                )
                tensor.index_copy_(
                    0, index, host.to(tensor.device, non_blocking=True)
                )
            torch.cuda.synchronize()
            self._store.touch(reader)
            if self._insert_worker_native(plan, restored_hidden):
                self._store.release_resident(reader)
            if plan.skip_forward:
                if restored_hidden is None:
                    raise RuntimeError("Let's Infer exact restore has no hidden state")
                self._restore_hidden[plan.req_id] = restored_hidden
            logger.info(
                "Let's Infer prefix RESTORED req=%s boundary=%d regions=%d "
                "total_ms=%.3f",
                plan.req_id,
                plan.boundary,
                len(names) - 1,
                (time.perf_counter() - started) * 1000,
            )
        finally:
            reader.close()

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        return  # capture happens once per step in wait_for_save

    def wait_for_save(self) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, LetsInferPrefixConnectorMetadata):
            return
        for plan in metadata.plans:
            if plan.kind == "capture":
                self._capture(plan)

    def _capture(self, plan: _ReqPlan) -> None:
        layout = self._plan_layout(plan)
        if not layout:
            return
        try:
            hidden = self._capture_hidden.get(plan.req_id)
            if plan.has_hidden and hidden is None:
                raise RuntimeError("Let's Infer exact capture has no hidden state")
            # Publish the hot native lease first: attention blocks stay in
            # vLLM's pool; mutable GDN state and the tiny final hidden row are
            # snapshotted into plugin-owned device tensors.
            native_promoted = self._insert_worker_native(plan, hidden)
            if self._requires_scale_basis and not self._scale_basis:
                raise RuntimeError("Let's Infer FP8 KV scale basis was not registered")
            meta: dict[str, Any] = {
                "boundary": plan.boundary,
                "abi": ABI_VERSION,
                "scale_basis": self._scale_basis,
            }
            if hidden is not None:
                meta["hidden"] = {
                    "dtype": str(hidden.dtype),
                    "shape": list(hidden.shape),
                }
            meta_blob = json.dumps(
                meta, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            regions: list[tuple[str, int]] = []
            for region in layout:
                tensor = self._region_tensor(region)
                tensor_block_ids = self._region_tensor_block_ids(region, tensor)
                if tensor.shape[0] == 0:
                    raise RuntimeError(
                        f"Let's Infer region {region.name} has no blocks"
                    )
                per_block_bytes = tensor[0].numel() * tensor.element_size()
                regions.append(
                    (region.name, len(tensor_block_ids) * per_block_bytes)
                )
            if hidden is not None:
                regions.append(
                    ("hidden", hidden.numel() * hidden.element_size())
                )
            regions.append(("meta", len(meta_blob)))
            writer = self._store.begin_capture(
                self._fingerprint, plan.token_ids, regions
            )
            if writer is None:
                return  # admission rejected: best-effort skip, never block decode
            try:
                torch.cuda.synchronize()
                for i, region in enumerate(layout):
                    tensor = self._region_tensor(region)
                    tensor_block_ids = self._region_tensor_block_ids(region, tensor)
                    index = torch.tensor(
                        tensor_block_ids, dtype=torch.long, device=tensor.device
                    )
                    host = torch.index_select(tensor, 0, index).cpu().contiguous()
                    writer.write_region_from(
                        i, host.view(torch.uint8).flatten().numpy()
                    )
                next_index = len(layout)
                if hidden is not None:
                    hidden_host = hidden.cpu().contiguous()
                    writer.write_region_from(
                        next_index,
                        hidden_host.view(torch.uint8).flatten().numpy(),
                    )
                    next_index += 1
                writer.write_region(next_index, meta_blob)
                writer.commit(promote_resident=not native_promoted)
                logger.info(
                    "Let's Infer prefix CAPTURED req=%s boundary=%d regions=%d "
                    "exact=%s",
                    plan.req_id,
                    plan.boundary,
                    len(layout),
                    plan.has_hidden,
                )
            except Exception:
                writer.cancel()
                raise
        finally:
            self._capture_hidden.pop(plan.req_id, None)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        return None, None
