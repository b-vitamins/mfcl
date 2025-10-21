import torch

from mfcl.utils import dist as mdist


def test_dist_single_process_defaults():
    assert mdist.is_main_process()
    assert mdist.get_rank() == 0
    assert mdist.get_world_size() == 1
    # barrier should be no-op
    mdist.barrier()


def test_all_gather_tensor_scalar(monkeypatch):
    tensor = torch.tensor(3.0)

    monkeypatch.setattr(mdist, "is_dist", lambda: True)
    monkeypatch.setattr(mdist, "get_world_size", lambda: 2)

    def fake_all_gather(output_tensors, input_tensor):
        assert input_tensor.shape == torch.Size([1])
        for idx, out in enumerate(output_tensors):
            out.copy_(torch.full_like(out, float(idx)))

    monkeypatch.setattr(mdist.dist, "all_gather", fake_all_gather)

    gathered = mdist.all_gather_tensor(tensor)
    assert torch.equal(gathered, torch.tensor([0.0, 1.0]))
