import torch

from mfcl.utils.queue import RingQueue


def test_ring_queue_wraparound_fifo_storage_order():
    queue = RingQueue(dim=2, size=4)
    queue.enqueue(torch.arange(0, 8, dtype=torch.float32).view(4, 2))
    queue.enqueue(torch.tensor([[8.0, 9.0], [10.0, 11.0]]))
    buf = queue.get()
    assert queue.full
    assert len(queue) == 4
    chronological = torch.cat((buf[queue.ptr :], buf[: queue.ptr]))
    expected = torch.tensor(
        [[4.0, 5.0], [6.0, 7.0], [8.0, 9.0], [10.0, 11.0]]
    )
    torch.testing.assert_close(chronological, expected)


def test_ring_queue_enqueue_empty_noop():
    queue = RingQueue(dim=3, size=3)
    queue.enqueue(torch.randn(2, 3))
    ptr_before = queue.ptr
    buf_before = queue.get().clone()
    queue.enqueue(torch.empty(0, 3))
    assert queue.ptr == ptr_before
    torch.testing.assert_close(queue.get(), buf_before)
