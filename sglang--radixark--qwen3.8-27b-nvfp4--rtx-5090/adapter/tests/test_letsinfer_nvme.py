import gc
import os
import torch
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.storage.letsinfer_nvme import LetsInferNVMeStorage

class Pool:
    page_size = 1
    def __init__(self, pages):
        self.pages = [p.clone().contiguous() for p in pages]
    def get_data_page(self, index, flat=True):
        return self.pages[index]
    def get_dummy_flat_data_page(self):
        return torch.zeros_like(self.pages[0])
    def set_from_flat_data_page(self, index, data):
        self.pages[index].copy_(data)

cfg = HiCacheStorageConfig(
    tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
    attn_cp_rank=0, attn_cp_size=1, is_mla_model=False,
    enable_storage_metrics=False, is_page_first_layout=True,
    model_name="unit/qwen", extra_config={
        "durable_capacity_bytes": 64 * 1024 * 1024,
        "ttl_seconds": 3600,
        "resident_capacity_bytes": 0,
        "direct_reads": False,
    },
)
expected = [
    torch.arange(64, dtype=torch.uint8),
    torch.arange(64, dtype=torch.uint8).add(17),
]
keys = ["page-a", "page-b"]
info = HiCacheStorageExtraInfo(prefix_keys=["ancestor"])

first_pool = Pool(expected)
store = LetsInferNVMeStorage(cfg)
store.register_mem_pool_host(first_pool)
assert store.batch_set_v1(keys, torch.tensor([0, 1]), info) == [True, True]
stats = store.get_stats()
assert stats["committed_capture_count"] == 1
del store
gc.collect()

restored_pool = Pool([torch.zeros(64, dtype=torch.uint8) for _ in range(2)])
restored = LetsInferNVMeStorage(cfg)
restored.register_mem_pool_host(restored_pool)
assert restored.batch_exists(keys, info) == 2
assert restored.batch_get_v1(keys, torch.tensor([0, 1]), info) == [True, True]
assert all(torch.equal(a, b) for a, b in zip(expected, restored_pool.pages))

mamba_expected = torch.arange(96, dtype=torch.uint8).add(3)
mamba_pool = Pool([mamba_expected])
restored.register_mem_host_pool_v2(mamba_pool, PoolName.MAMBA)
write = PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([0]), keys=["mamba-end"])
assert restored.batch_set_v2([write])[PoolName.MAMBA] == [True]
mamba_pool.pages[0].zero_()
assert restored.batch_get_v2([write])[PoolName.MAMBA] == [True]
assert torch.equal(mamba_pool.pages[0], mamba_expected)
print("PASS restart_persistent=true primary_exact=true mamba_exact=true")
print(restored.get_stats())
