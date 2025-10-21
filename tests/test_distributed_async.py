"""Tests for asynchronous communication wrappers."""

from __future__ import annotations

from collections.abc import Iterable as ABCIterable

import pytest
import torch
import torch.distributed as dist

from mfcl.distributed import (
    PayloadCategory,
    all_gather,
    all_reduce,
    broadcast,
    reduce_scatter,
)
from mfcl.telemetry import comms_logger as comms
from mfcl.telemetry.comms_logger import (
    AsyncCollectiveHandle,
    close_comms_logger,
    configure_comms_logger,
)


class FakeWork:
    """Minimal stand-in for ``torch.distributed.Work``."""

    def __init__(self, finalize_cb) -> None:
        self._finalize_cb = finalize_cb
        self._completed = False

    def wait(self, *args, **kwargs):
        if not self._completed:
            self._finalize_cb()
            self._completed = True
        return self

    def is_completed(self) -> bool:
        return self._completed

    def synchronize(self):
        return self.wait()

    def result(self):
        return self.wait()

    def exception(self):
        return None


class FakeProcessGroup:
    """Fake process group that returns :class:`FakeWork` handles."""

    world_size = 2

    def allreduce(self, tensor: torch.Tensor, op: dist.ReduceOp) -> FakeWork:
        def finalize() -> None:
            tensor.mul_(self.world_size)

        return FakeWork(finalize)

    def allgather(self, tensor_list: list[torch.Tensor], tensor: torch.Tensor) -> FakeWork:
        def finalize() -> None:
            for bucket in tensor_list:
                bucket.copy_(tensor)

        return FakeWork(finalize)

    def reducescatter(
        self, output: torch.Tensor, input_list: ABCIterable[torch.Tensor], op: dist.ReduceOp
    ) -> FakeWork:
        def finalize() -> None:
            total = sum(input_list, torch.zeros_like(output))
            output.copy_(total)

        return FakeWork(finalize)

    def broadcast(self, tensor: torch.Tensor, src: int) -> FakeWork:  # noqa: ARG002 - parity
        def finalize() -> None:
            pass

        return FakeWork(finalize)


@pytest.fixture
def configured_logger(monkeypatch):
    """Provide an active communication logger with world-size overrides."""

    close_comms_logger()
    logger = configure_comms_logger(enabled=True, log_path=None, is_main=True)
    assert logger is not None
    monkeypatch.setattr(comms, "_safe_world_size", lambda: 2)
    logger.begin_step(epoch=0, step_index=0, global_step=0, timer=None)
    yield logger
    logger.end_step()
    close_comms_logger()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_async_collectives_emit_telemetry(monkeypatch, configured_logger):
    fake_pg = FakeProcessGroup()

    def fake_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if isinstance(group, FakeProcessGroup) and async_op:
            return group.allreduce(tensor, op=op)
        raise AssertionError("Unexpected call to all_reduce in test")

    def fake_all_gather(tensor_list, tensor, group=None, async_op=False):
        if isinstance(group, FakeProcessGroup) and async_op:
            return group.allgather(tensor_list, tensor)
        raise AssertionError("Unexpected call to all_gather in test")

    def fake_reduce_scatter(output, input_list, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if isinstance(group, FakeProcessGroup) and async_op:
            return group.reducescatter(output, input_list, op=op)
        raise AssertionError("Unexpected call to reduce_scatter in test")

    def fake_broadcast(tensor, src, group=None, async_op=False):
        if isinstance(group, FakeProcessGroup) and async_op:
            return group.broadcast(tensor, src)
        raise AssertionError("Unexpected call to broadcast in test")

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(dist, "all_gather", fake_all_gather)
    monkeypatch.setattr(dist, "reduce_scatter", fake_reduce_scatter)
    monkeypatch.setattr(dist, "broadcast", fake_broadcast)

    tensor = torch.ones(4, dtype=torch.float32)
    handle_reduce = all_reduce(
        tensor,
        group=fake_pg,
        async_op=True,
        category=PayloadCategory.OTHER,
    )
    assert isinstance(handle_reduce, AsyncCollectiveHandle)

    gather_list = [torch.zeros_like(tensor) for _ in range(fake_pg.world_size)]
    handle_gather = all_gather(
        gather_list,
        tensor,
        group=fake_pg,
        async_op=True,
        category=PayloadCategory.OTHER,
    )
    assert isinstance(handle_gather, AsyncCollectiveHandle)

    output = torch.zeros_like(tensor)
    input_list = [torch.ones_like(tensor), torch.full_like(tensor, 2.0)]
    handle_reduce_scatter = reduce_scatter(
        output,
        input_list,
        group=fake_pg,
        async_op=True,
        category=PayloadCategory.OTHER,
    )
    assert isinstance(handle_reduce_scatter, AsyncCollectiveHandle)

    broadcast_tensor = torch.zeros_like(tensor)
    handle_broadcast = broadcast(
        broadcast_tensor,
        src=0,
        group=fake_pg,
        async_op=True,
        category=PayloadCategory.OTHER,
    )
    assert isinstance(handle_broadcast, AsyncCollectiveHandle)

    for handle in (
        handle_reduce,
        handle_gather,
        handle_reduce_scatter,
        handle_broadcast,
    ):
        handle.wait()
        assert handle.is_completed()

    # Verify the fake collectives executed as expected.
    assert torch.allclose(tensor, torch.full_like(tensor, fake_pg.world_size))
    for bucket in gather_list:
        assert torch.allclose(bucket, tensor)
    assert torch.allclose(output, sum(input_list, torch.zeros_like(output)))
    assert torch.allclose(broadcast_tensor, torch.zeros_like(tensor))

    configured_logger.end_step()
    totals = configured_logger.pop_last_step_totals()
    assert totals == {"bytes_total": pytest.approx(64.0)}
