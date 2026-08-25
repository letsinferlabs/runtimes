# SPDX-License-Identifier: AGPL-3.0-only

from types import SimpleNamespace
from collections import OrderedDict

import torch

from letsinfer_prefix_connector.connector import (
    LetsInferPrefixConnector,
    _ReqPlan,
    _exact_capsules_enabled,
)
from vllm.v1.kv_cache_interface import MambaSpec


def bare_connector() -> LetsInferPrefixConnector:
    connector = object.__new__(LetsInferPrefixConnector)
    connector._min_tokens = 4096
    connector._max_tokens = 262144
    connector._attention_block_sizes = {}
    connector._req_tokens = {}
    connector._req_computed = {}
    connector._req_blocks = {}
    connector._req_captured = {}
    connector._pending_restore = {}
    connector._fingerprint = bytes([7]) * 32
    connector._requires_scale_basis = False
    connector._scale_basis = {}
    connector._native_capacity_bytes = 8 * (1024**3)
    connector._capture_tail_tokens = 2048
    connector._num_cache_blocks = 100
    connector._attention_group_indices = set()
    connector._group_block_bytes = []
    connector._gpu_block_pool = None
    connector._scheduler_native = OrderedDict()
    connector._scheduler_native_bytes = 0
    connector._worker_native = OrderedDict()
    connector._worker_native_bytes = 0
    connector._attention_rows_per_block = {}
    connector._exact_enabled = True
    connector._hidden_bytes = 0
    connector._capture_hidden = {}
    connector._restore_hidden = {}
    return connector


def test_speculative_exact_capsules_require_native_mtp_opt_in() -> None:
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        speculative_config=SimpleNamespace(method="mtp"),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
    )

    assert not _exact_capsules_enabled({}, config)
    assert _exact_capsules_enabled(
        {"exact_capsules_with_mtp": True}, config
    )
    config.speculative_config = SimpleNamespace(method="eagle")
    assert not _exact_capsules_enabled(
        {"exact_capsules_with_mtp": True}, config
    )
    config.speculative_config = None
    assert _exact_capsules_enabled({}, config)
    config.scheduler_config.max_num_seqs = 2
    assert not _exact_capsules_enabled({}, config)


def test_hybrid_layout_has_separate_recurrent_states() -> None:
    connector = bare_connector()
    attention = SimpleNamespace(block_size=2128)
    recurrent = MambaSpec(
        block_size=2128,
        shapes=((6, 3), (2, 4, 4)),
        dtypes=(torch.bfloat16, torch.float32),
    )
    connector._groups = [
        SimpleNamespace(kv_cache_spec=attention),
        SimpleNamespace(kv_cache_spec=recurrent),
    ]
    connector._layer_to_group = {"model.layers.3.attn": 0, "model.layers.0.gdn": 1}
    plan = _ReqPlan(
        req_id="r",
        kind="capture",
        boundary=63840,
        token_ids=list(range(63840)),
        group_block_ids=[list(range(30)), [91, 92, 93]],
    )

    layout = connector._plan_layout(plan)

    assert [(row.name, row.state_index) for row in layout] == [
        ("m1:000:0", 0),
        ("m1:000:1", 1),
        ("a0:001", None),
    ]
    assert layout[0].block_ids == [91, 92, 93]
    assert layout[2].block_ids == list(range(30))
    assert all(len(row.name.encode()) <= 12 for row in layout)


def test_heterogeneous_attention_groups_use_their_own_block_sizes() -> None:
    connector = bare_connector()
    connector._groups = [
        SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=size))
        for size in (4, 8, 64, 256)
    ]
    connector._layer_to_group = {
        f"model.layers.{index}.attn": index for index in range(4)
    }
    connector._attention_group_indices = set(range(4))
    connector._attention_block_sizes = {
        index: size for index, size in enumerate((4, 8, 64, 256))
    }
    plan = _ReqPlan(
        req_id="heterogeneous",
        kind="capture",
        boundary=10,
        token_ids=list(range(10)),
        group_block_ids=[
            [10, 11, 12],
            [20, 21],
            [30],
            [40],
        ],
    )

    layout = connector._plan_layout(plan)

    assert [row.block_ids for row in layout] == [
        [10, 11, 12],
        [20, 21],
        [30],
        [40],
    ]
    assert connector._resident_group_ids(plan) == [
        [10, 11, 12],
        [20, 21],
        [30],
        [40],
    ]


def test_capture_plan_skips_intermediate_boundary_outside_tail() -> None:
    connector = bare_connector()
    request = SimpleNamespace(
        req_id="r",
        prompt_token_ids=list(range(65536)),
        num_computed_tokens=0,
        block_ids=(list(range(15)), [90, 91, 92]),
    )
    cached = SimpleNamespace(
        req_ids=[], new_block_ids=[], num_computed_tokens=[]
    )
    aligned = SimpleNamespace(
        scheduled_new_reqs=[request],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={"r": 31920},
    )

    metadata = connector.build_connector_meta(aligned)

    assert metadata.plans == []


def test_capture_window_targets_only_the_declared_ttft_prompt() -> None:
    connector = bare_connector()
    connector._min_tokens = 60000
    connector._max_tokens = 61000
    connector._capture_tail_tokens = 1024

    outside = SimpleNamespace(
        req_id="long-64k",
        prompt_token_ids=list(range(62360)),
        num_computed_tokens=0,
        block_ids=(list(range(244)),),
    )
    inside = SimpleNamespace(
        req_id="ttft-64k",
        prompt_token_ids=list(range(60858)),
        num_computed_tokens=0,
        block_ids=(list(range(238)),),
    )
    cached = SimpleNamespace(
        req_ids=[], new_block_ids=[], num_computed_tokens=[]
    )

    outside_meta = connector.build_connector_meta(
        SimpleNamespace(
            scheduled_new_reqs=[outside],
            scheduled_cached_reqs=cached,
            num_scheduled_tokens={outside.req_id: 61440},
        )
    )
    inside_meta = connector.build_connector_meta(
        SimpleNamespace(
            scheduled_new_reqs=[inside],
            scheduled_cached_reqs=cached,
            num_scheduled_tokens={inside.req_id: 60416},
        )
    )

    assert outside_meta.plans == []
    assert [(plan.req_id, plan.boundary) for plan in inside_meta.plans] == [
        ("ttft-64k", 60416)
    ]


def test_capture_plan_uses_last_intermediate_scheduler_boundary() -> None:
    connector = bare_connector()
    request = SimpleNamespace(
        req_id="r",
        prompt_token_ids=list(range(65536)),
        num_computed_tokens=0,
        block_ids=(list(range(31)), [90, 91, 92]),
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[request],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], num_computed_tokens=[]
        ),
        num_scheduled_tokens={"r": 64000},
    )

    metadata = connector.build_connector_meta(output)

    assert [(row.kind, row.boundary) for row in metadata.plans] == [
        ("capture", 64000)
    ]


def test_unaligned_intermediate_boundary_is_captured() -> None:
    connector = bare_connector()
    connector._capture_tail_tokens = 65536
    request = SimpleNamespace(
        req_id="r",
        prompt_token_ids=list(range(65536)),
        num_computed_tokens=0,
        block_ids=(list(range(16)), [90, 91, 92]),
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[request],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], num_computed_tokens=[]
        ),
        num_scheduled_tokens={"r": 32768},
    )

    metadata = connector.build_connector_meta(output)

    assert len(metadata.plans) == 1
    plan = metadata.plans[0]
    assert (plan.kind, plan.boundary, plan.has_hidden) == (
        "capture",
        32768,
        False,
    )


def test_concurrent_requests_capture_their_actual_step_boundaries() -> None:
    connector = bare_connector()
    connector._exact_enabled = False
    connector._capture_tail_tokens = 65536
    positions = [15008, 13632, 12252, 10868]
    requests = [
        SimpleNamespace(
            req_id=f"r{index}",
            prompt_token_ids=list(range(index, index + 16384)),
            num_computed_tokens=0,
            block_ids=(list(range(8)), [90 + index]),
        )
        for index in range(4)
    ]
    output = SimpleNamespace(
        scheduled_new_reqs=requests,
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], num_computed_tokens=[]
        ),
        num_scheduled_tokens={
            request.req_id: position
            for request, position in zip(requests, positions, strict=True)
        },
    )

    metadata = connector.build_connector_meta(output)

    assert [plan.req_id for plan in metadata.plans] == [
        "r0",
        "r1",
        "r2",
        "r3",
    ]
    assert [plan.boundary for plan in metadata.plans] == positions
    assert all(not plan.has_hidden for plan in metadata.plans)


def test_exact_capture_plan_accepts_unaligned_full_prompt() -> None:
    connector = bare_connector()
    connector._min_tokens = 1024
    request = SimpleNamespace(
        req_id="r",
        prompt_token_ids=list(range(1024)),
        num_computed_tokens=0,
        block_ids=([17], [90]),
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[request],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], num_computed_tokens=[]
        ),
        num_scheduled_tokens={"r": 1024},
    )

    metadata = connector.build_connector_meta(output)

    assert len(metadata.plans) == 1
    plan = metadata.plans[0]
    assert (plan.kind, plan.boundary, plan.has_hidden, plan.skip_forward) == (
        "capture",
        1024,
        True,
        False,
    )


def test_prewarm_request_never_creates_a_new_record() -> None:
    connector = bare_connector()
    connector._min_tokens = 4
    request = SimpleNamespace(
        req_id="cmpl-letsinfer-prewarm-0000",
        prompt_token_ids=list(range(8)),
        num_computed_tokens=0,
        block_ids=([17], [90]),
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[request],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], num_computed_tokens=[]
        ),
        num_scheduled_tokens={request.req_id: 8},
    )

    metadata = connector.build_connector_meta(output)

    assert metadata.plans == []


def test_partial_layout_keeps_attention_tail_without_hidden_capsule() -> None:
    connector = bare_connector()
    connector._groups = [SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=8))]
    connector._layer_to_group = {"attn": 0}
    partial = _ReqPlan(
        req_id="r",
        kind="capture",
        boundary=10,
        token_ids=list(range(10)),
        group_block_ids=[[3, 4]],
    )

    layout = connector._plan_layout(partial)

    assert layout[0].block_ids == [3, 4]


def test_exact_native_hit_reports_one_synthetic_token() -> None:
    connector = bare_connector()
    connector._min_tokens = 4
    tokens = [10, 11, 12, 13]
    key = connector._native_key(tokens)
    connector._scheduler_native[key] = SimpleNamespace(
        boundary=4,
        token_ids=tokens,
        group_block_ids=((7,),),
        record_bytes=16,
        has_hidden=True,
    )
    connector._attention_group_indices = {0}
    request = SimpleNamespace(request_id="hit", prompt_token_ids=tokens)

    assert connector.get_num_new_matched_tokens(request, 0) == (3, False)
    assert connector.get_resident_block_ids(request, 3) == ([7],)
    pending = connector._pending_restore["hit"]
    assert pending.boundary == 4
    assert pending.computed_boundary == 3
    assert pending.skip_forward


def test_hidden_capsule_round_trip_without_model_forward() -> None:
    connector = bare_connector()
    hidden_states = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    capture = _ReqPlan(
        req_id="r",
        kind="capture",
        boundary=4,
        token_ids=[1, 2, 3, 4],
        group_block_ids=[],
        has_hidden=True,
    )
    connector._connector_metadata = SimpleNamespace(plans=[capture])

    connector.capture_exact_hidden_states(hidden_states, torch.tensor([3]), 6)

    assert torch.equal(connector._capture_hidden["r"], hidden_states[3])
    restore = _ReqPlan(
        req_id="r",
        kind="restore_native",
        boundary=4,
        token_ids=[1, 2, 3, 4],
        group_block_ids=[],
        has_hidden=True,
        skip_forward=True,
    )
    connector._connector_metadata = SimpleNamespace(plans=[restore])
    connector._restore_hidden["r"] = hidden_states[3].clone()

    # vLLM's live logits index tensor is int32; PyTorch index_copy requires
    # int64, so the connector owns this conversion.
    output = connector.get_exact_hidden_states(
        torch.tensor([0], dtype=torch.int32), 2
    )

    assert output is not None
    assert output.shape == (2, 4)
    assert torch.equal(output[0], hidden_states[3])
    assert torch.count_nonzero(output[1]) == 0
    assert "r" not in connector._restore_hidden


def test_region_tensor_selects_each_recurrent_component() -> None:
    connector = bare_connector()
    attention = torch.zeros((3, 2), dtype=torch.uint8)
    conv = torch.zeros((3, 2, 2), dtype=torch.bfloat16)
    state = torch.zeros((3, 2, 2, 2), dtype=torch.float32)
    connector._kv_caches = {"attn": attention, "gdn": [conv, state]}
    connector._groups = []
    connector._layer_to_group = {}

    assert connector._region_tensor(
        SimpleNamespace(layer_name="attn", state_index=None)
    ) is attention
    assert connector._region_tensor(
        SimpleNamespace(layer_name="gdn", state_index=0)
    ) is conv
    assert connector._region_tensor(
        SimpleNamespace(layer_name="gdn", state_index=1)
    ) is state


def test_attention_superpage_expands_to_all_physical_rows() -> None:
    connector = bare_connector()
    connector._num_cache_blocks = 4
    connector._groups = [SimpleNamespace(layer_names=["attn"])]
    connector._layer_to_group = {"attn": 0}
    tensor = torch.zeros((12, 2), dtype=torch.uint8)
    region = SimpleNamespace(
        layer_name="attn", state_index=None, block_ids=[1, 3]
    )

    assert connector._region_tensor_block_ids(region, tensor) == [3, 4, 5, 9, 10, 11]
    assert connector._attention_rows_per_block == {0: 3}


def test_recurrent_block_ids_are_not_expanded() -> None:
    connector = bare_connector()
    connector._num_cache_blocks = 4
    tensor = torch.zeros((4, 2), dtype=torch.float32)
    region = SimpleNamespace(
        layer_name="gdn", state_index=0, block_ids=[1, 3]
    )

    assert connector._region_tensor_block_ids(region, tensor) == [1, 3]


def test_native_scheduler_lease_adopts_only_attention_blocks() -> None:
    class Block:
        def __init__(self) -> None:
            self.ref_cnt = 1

    class Pool:
        def __init__(self) -> None:
            self.blocks = [Block() for _ in range(100)]

        def touch(self, blocks) -> None:
            for block in blocks:
                block.ref_cnt += 1

        def free_blocks(self, blocks) -> None:
            for block in blocks:
                block.ref_cnt -= 1

    connector = bare_connector()
    connector._groups = [
        SimpleNamespace(
            kv_cache_spec=MambaSpec(
                block_size=2128,
                shapes=((2,),),
                dtypes=(torch.float32,),
            )
        ),
        SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=2128)),
    ]
    connector._attention_group_indices = {1}
    connector._group_block_bytes = [64, 1024]
    connector._gpu_block_pool = Pool()
    tokens = list(range(63840))
    plan = _ReqPlan(
        req_id="capture",
        kind="capture",
        boundary=len(tokens),
        token_ids=tokens,
        group_block_ids=[[2], list(range(10, 41))],
    )

    assert connector._insert_scheduler_native(plan)
    assert all(
        connector._gpu_block_pool.blocks[i].ref_cnt == 2
        for i in range(10, 40)
    )
    assert connector._gpu_block_pool.blocks[40].ref_cnt == 1
    request = SimpleNamespace(
        request_id="hit", prompt_token_ids=tokens + [999]
    )
    assert connector.get_num_new_matched_tokens(request, 0) == (63840, False)
    resident = connector.get_resident_block_ids(request, 63840)
    assert resident == ([], list(range(10, 40)))


def test_native_worker_restore_copies_only_recurrent_state(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    connector = bare_connector()
    attention_spec = SimpleNamespace(block_size=4)
    recurrent_spec = MambaSpec(
        block_size=4,
        shapes=((2,), (2,)),
        dtypes=(torch.float32, torch.float32),
    )
    connector._groups = [
        SimpleNamespace(kv_cache_spec=recurrent_spec),
        SimpleNamespace(kv_cache_spec=attention_spec),
    ]
    connector._attention_group_indices = {1}
    connector._num_cache_blocks = 6
    connector._group_block_bytes = [16, 16]
    connector._layer_to_group = {"gdn": 0, "attn": 1}
    conv = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    state = torch.arange(20, 32, dtype=torch.float32).reshape(6, 2)
    attention = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    connector._kv_caches = {"gdn": [conv, state], "attn": attention}
    tokens = list(range(8))
    capture = _ReqPlan(
        req_id="capture",
        kind="capture",
        boundary=8,
        token_ids=tokens,
        group_block_ids=[[1], [2, 3]],
    )
    assert connector._insert_worker_native(capture)

    conv[4].fill_(-1)
    state[4].fill_(-2)
    attention_before = attention.clone()
    restore = _ReqPlan(
        req_id="restore",
        kind="restore_native",
        boundary=8,
        token_ids=tokens,
        # The request owns a private suffix block beyond the two adopted
        # prefix blocks. Native validation must not treat that suffix as part
        # of the resident lease.
        group_block_ids=[[4], [2, 3, 4]],
    )
    connector._restore_native(restore)

    assert torch.equal(conv[4], torch.tensor([2.0, 3.0]))
    assert torch.equal(state[4], torch.tensor([22.0, 23.0]))
    assert torch.equal(attention, attention_before)


def test_fp8_scale_basis_requires_exact_attention_layer_set() -> None:
    connector = bare_connector()
    connector._requires_scale_basis = True
    connector._groups = [SimpleNamespace(layer_names=["attn.0", "attn.1"])]
    connector._attention_group_indices = {0}
    record = {
        "dtype": "torch.float32",
        "shape": [],
        "bytes_hex": "0000803f",
    }

    connector.register_kv_scale_basis(
        {
            "attn.0": {"k": dict(record), "v": dict(record)},
            "attn.1": {"k": dict(record), "v": dict(record)},
        }
    )

    assert list(connector._scale_basis) == ["attn.0", "attn.1"]
