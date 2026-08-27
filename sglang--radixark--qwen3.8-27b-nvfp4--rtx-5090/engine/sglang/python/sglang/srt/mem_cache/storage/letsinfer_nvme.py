# SPDX-License-Identifier: AGPL-3.0-only
"""Let's Infer durable NVMe backend for SGLang HiCache.

The engine-neutral PrefixStore owns record integrity, atomic commits, byte-LRU
capacity, TTL expiry, and direct NVMe reads.  This adapter groups one SGLang
storage batch into one record so a 64K prefix does not become tens of thousands
of tiny files.  Attention KV, draft KV, and hybrid Mamba state are isolated by
cryptographic compatibility fingerprints.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
from typing import Any, List, Optional

import torch

from letsinfer_prefix_store import PrefixStore
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)

logger = logging.getLogger(__name__)

_FORMAT = "letsinfer-sglang-hicache-batch-v1"
_MAX_BATCH_PAGES = 64
_META_REGION = "meta"


def _strict_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be boolean-like")


class LetsInferNVMeStorage(HiCacheStorage):
    """CRC-checked, restart-persistent HiCache storage on local NVMe."""

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        _kwargs: Optional[dict[str, Any]] = None,
    ):
        extra = storage_config.extra_config or {}
        root = os.environ.get("LETSINFER_CACHE_ROOT")
        if not root:
            raise ValueError("LETSINFER_CACHE_ROOT is required")

        capacity = int(extra.get("durable_capacity_bytes", 64 * 1024**3))
        ttl = int(extra.get("ttl_seconds", 7 * 24 * 60 * 60))
        resident = int(extra.get("resident_capacity_bytes", 0))
        direct = _strict_bool(extra.get("direct_reads", True), "direct_reads")
        if capacity <= 0 or ttl <= 0 or resident < 0:
            raise ValueError("invalid Let's Infer cache capacity or TTL")

        namespace = {
            "format": _FORMAT,
            "model": storage_config.model_name,
            "tp_rank": storage_config.tp_rank,
            "tp_size": storage_config.tp_size,
            "pp_rank": storage_config.pp_rank,
            "pp_size": storage_config.pp_size,
            "attn_cp_rank": storage_config.attn_cp_rank,
            "attn_cp_size": storage_config.attn_cp_size,
            "page_first": storage_config.is_page_first_layout,
            "split_heads": storage_config.should_split_heads,
        }
        self._namespace = (
            json.dumps(namespace, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        self._store = PrefixStore(
            os.path.join(root, "sglang-hicache-v1"),
            capacity,
            ttl,
            1,
            resident,
            direct,
        )
        self._registered_pools: dict[PoolName, Any] = {}
        logger.info(
            "Let's Infer NVMe HiCache enabled: root=%s capacity=%.1f GiB ttl=%ss",
            root,
            capacity / 1024**3,
            ttl,
        )

    @staticmethod
    def _key_digest(key: str) -> bytes:
        return hashlib.sha256(key.encode("utf-8")).digest()

    @classmethod
    def _key_tokens(cls, key: str) -> list[int]:
        return list(struct.unpack("<8I", cls._key_digest(key)))

    def _fingerprint(self, pool: str, key: str) -> bytes:
        digest = hashlib.sha256()
        digest.update(self._namespace)
        digest.update(pool.encode("ascii"))
        digest.update(b"\0")
        digest.update(key.encode("utf-8"))
        return digest.digest()

    @staticmethod
    def _page_buffer(page: torch.Tensor):
        if page.device.type != "cpu" or not page.is_contiguous():
            raise ValueError("Let's Infer HiCache pages must be contiguous CPU tensors")
        return page.detach().view(torch.uint8).numpy()

    @staticmethod
    def _region_names(count: int) -> list[str]:
        return [_META_REGION, *[f"p{i:03d}" for i in range(count)]]

    def _lookup_endpoint(self, pool: str, key: str):
        tokens = self._key_tokens(key)
        return self._store.longest_prefix(
            self._fingerprint(pool, key), tokens, len(tokens)
        )

    def _find_record(
        self,
        pool: str,
        full_keys: list[str],
        first_endpoint: int,
    ):
        for end in range(len(full_keys) - 1, first_endpoint - 1, -1):
            reader = self._lookup_endpoint(pool, full_keys[end])
            if reader is None:
                continue
            try:
                names = reader.region_names
                if not names or names[0] != _META_REGION:
                    continue
                metadata = bytes(reader.read_region(0))
                if not metadata or len(metadata) % 32:
                    continue
                count = len(metadata) // 32
                start = end + 1 - count
                if start < 0 or len(names) != count + 1:
                    continue
                if names != self._region_names(count):
                    continue
                expected = b"".join(
                    self._key_digest(key) for key in full_keys[start : end + 1]
                )
                if metadata != expected:
                    continue
                return reader, start, end + 1
            except Exception:
                logger.exception("Let's Infer prefix record validation failed")
            reader.close()
        return None

    def _write_batch(
        self,
        pool: str,
        keys: list[str],
        pages: list[torch.Tensor],
    ) -> list[bool]:
        if not keys or len(keys) != len(pages) or len(keys) > _MAX_BATCH_PAGES:
            return [False] * len(keys)
        existing = self._find_record(pool, keys, 0)
        if existing is not None:
            reader, _start, _end = existing
            self._store.touch(reader)
            reader.close()
            return [True] * len(keys)

        metadata = b"".join(self._key_digest(key) for key in keys)
        sizes = [len(metadata)]
        buffers = [metadata]
        for page in pages:
            buffer = self._page_buffer(page)
            sizes.append(int(buffer.nbytes))
            buffers.append(buffer)

        endpoint = keys[-1]
        writer = self._store.begin_capture(
            self._fingerprint(pool, endpoint),
            self._key_tokens(endpoint),
            list(zip(self._region_names(len(keys)), sizes)),
        )
        if writer is None:
            # Already-stored is success; every other admission rejection is a
            # safe cache miss and normal inference continues.
            existing = self._find_record(pool, keys, 0)
            if existing is None:
                return [False] * len(keys)
            reader, _start, _end = existing
            reader.close()
            return [True] * len(keys)
        try:
            writer.write_region(0, metadata)
            for index, buffer in enumerate(buffers[1:], 1):
                writer.write_region_from(index, buffer)
            # The HiCache write-through acknowledgement means durable and
            # immediately visible, so commit synchronously on its backup thread.
            writer.commit_sync()
            return [True] * len(keys)
        except Exception:
            writer.cancel()
            logger.exception("Let's Infer NVMe cache write failed")
            return [False] * len(keys)

    def _read_batch(
        self,
        pool: str,
        full_keys: list[str],
        prefix_count: int,
        pages: list[torch.Tensor],
    ) -> list[bool]:
        result = [False] * len(pages)
        found = self._find_record(pool, full_keys, prefix_count)
        if found is None:
            return result
        reader, record_start, record_end = found
        try:
            hit_count = min(len(pages), record_end - prefix_count)
            if hit_count <= 0 or record_start > prefix_count:
                return result
            region_offset = 1 + prefix_count - record_start
            for index in range(hit_count):
                reader.read_region_into(
                    region_offset + index, self._page_buffer(pages[index])
                )
                result[index] = True
            self._store.touch(reader)
            return result
        except Exception:
            logger.exception("Let's Infer NVMe cache read failed")
            return [False] * len(pages)
        finally:
            reader.close()

    def _hit_count(
        self, pool: str, keys: list[str], prefix_keys: Optional[list[str]]
    ) -> int:
        prefix = list(prefix_keys or [])
        found = self._find_record(pool, prefix + keys, len(prefix))
        if found is None:
            return 0
        reader, record_start, record_end = found
        reader.close()
        if record_start > len(prefix):
            return 0
        return max(0, min(len(keys), record_end - len(prefix)))

    def register_mem_pool_host(self, mem_pool_host):
        super().register_mem_pool_host(mem_pool_host)

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        super().register_mem_host_pool_v2(host_pool, host_pool_name)
        self._registered_pools[host_pool_name] = host_pool

    # Primary attention-KV zero-copy v1 path. Prefix keys make batch records
    # robust even when the radix tree splits the same prompt differently.
    def batch_exists(
        self,
        keys: List[str],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        prefix = extra_info.prefix_keys if extra_info else None
        return self._hit_count(str(PoolName.KV), list(keys), prefix)

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        page_size = int(self.mem_pool_host.page_size)
        pages = [
            self.mem_pool_host.get_data_page(int(host_indices[i * page_size].item()))
            for i in range(len(keys))
        ]
        prefix = list(extra_info.prefix_keys or []) if extra_info else []
        return self._read_batch(str(PoolName.KV), prefix + list(keys), len(prefix), pages)

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        page_size = int(self.mem_pool_host.page_size)
        pages = [
            self.mem_pool_host.get_data_page(int(host_indices[i * page_size].item()))
            for i in range(len(keys))
        ]
        return self._write_batch(str(PoolName.KV), list(keys), pages)

    # Hybrid Mamba/draft pools use the v2 path.
    def _pool_pages(self, transfer: PoolTransfer):
        pool = self._registered_pools.get(transfer.name)
        keys = list(transfer.keys or [])
        indices = transfer.host_indices
        if pool is None or indices is None:
            return pool, keys, []
        page_size = int(getattr(pool, "page_size", 1) or 1)
        if indices.numel() != len(keys) * page_size:
            return pool, keys, []
        pages = [
            pool.get_data_page(int(indices[i * page_size].item())) for i in range(len(keys))
        ]
        return pool, keys, pages

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        kv_pages = self.batch_exists(keys, extra_info)
        final_pages = kv_pages
        hit_count: dict[Any, int] = {PoolName.KV: kv_pages} if kv_pages else {}
        for transfer in pool_transfers or []:
            if final_pages <= 0:
                break
            pool_name = str(transfer.name)
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = self._hit_count(pool_name, list(keys[:final_pages]), None)
            else:
                boundary = 0
                trailing = max(1, len(transfer.keys or []))
                for prefix_len in range(final_pages, 0, -1):
                    start = max(0, prefix_len - trailing)
                    if (
                        self._hit_count(
                            pool_name, list(keys[start:prefix_len]), None
                        )
                        == prefix_len - start
                    ):
                        boundary = prefix_len
                        break
            if boundary:
                hit_count[transfer.name] = boundary
            final_pages = min(final_pages, boundary)
        return PoolTransferResult(final_pages, hit_count)

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        results = {}
        for transfer in transfers:
            _pool, keys, pages = self._pool_pages(transfer)
            results[transfer.name] = (
                self._read_batch(str(transfer.name), keys, 0, pages)
                if len(pages) == len(keys)
                else [False] * len(keys)
            )
        return results

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        results = {}
        for transfer in transfers:
            _pool, keys, pages = self._pool_pages(transfer)
            results[transfer.name] = (
                self._write_batch(str(transfer.name), keys, pages)
                if len(pages) == len(keys)
                else [False] * len(keys)
            )
        return results

    # Legacy generic surface, retained for SGLang management and fallback paths.
    def get(self, key: str, target_location=None, target_sizes=None):
        if target_location is None:
            return None
        result = self._read_batch("generic", [key], 0, [target_location])
        return target_location if result == [True] else None

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        targets = list(target_locations or [])
        if len(targets) != len(keys):
            return [None] * len(keys)
        result = self._read_batch("generic", list(keys), 0, targets)
        return [target if ok else None for target, ok in zip(targets, result)]

    def set(self, key: str, value=None, target_location=None, target_sizes=None):
        page = value if value is not None else target_location
        return bool(page is not None and self._write_batch("generic", [key], [page])[0])

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None):
        pages = list(values if values is not None else target_locations or [])
        return bool(keys) and all(self._write_batch("generic", list(keys), pages))

    def exists(self, key: str) -> bool:
        return self._hit_count("generic", [key], None) == 1

    def clear(self) -> bool:
        # Persistent data is intentionally not cleared by an engine reset.
        logger.warning("Let's Infer persistent cache clear requires lifecycle cleanup")
        return False

    def get_stats(self):
        return dict(self._store.statistics())

    def close(self) -> None:
        logger.info("Let's Infer NVMe HiCache statistics: %s", self.get_stats())
