import torch

from mfcl.metrics.knn import knn_predict


def test_knn_predict_handles_sparse_labels():
    features = torch.randn(2, 4)
    bank = torch.randn(3, 4)
    bank_labels = torch.tensor([0, 1000, 100000], dtype=torch.long)

    probs = knn_predict(features, bank, bank_labels, k=2, temperature=0.07)

    assert probs.shape == (2, 3)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)
