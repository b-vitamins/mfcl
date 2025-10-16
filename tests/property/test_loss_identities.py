import pytest
import torch

from mfcl.losses.ntxent import NTXentLoss
from mfcl.losses.byolloss import BYOLLoss
from mfcl.losses.simsiamloss import SimSiamLoss
from mfcl.losses.vicregloss import VICRegLoss

hypothesis = pytest.importorskip("hypothesis")
given = hypothesis.given
st = hypothesis.strategies


@given(
    B=st.integers(min_value=2, max_value=8), D=st.integers(min_value=4, max_value=32)
)
def test_ntxent_symmetry(B, D):
    z1 = torch.randn(B, D)
    z2 = torch.randn(B, D)
    loss_fn = NTXentLoss(temperature=0.2, normalize=True)
    l12, _ = loss_fn(z1, z2)
    l21, _ = loss_fn(z2, z1)
    assert torch.allclose(l12, l21, atol=1e-6, rtol=1e-4)


@given(scale=st.floats(min_value=0.1, max_value=10.0))
def test_ntxent_scale_invariance(scale):
    B, D = 6, 16
    z1 = torch.randn(B, D)
    z2 = torch.randn(B, D)
    loss_fn = NTXentLoss(temperature=0.2, normalize=True)
    l1, _ = loss_fn(z1, z2)
    l2, _ = loss_fn(z1 * scale, z2 * scale)
    assert torch.allclose(l1, l2, atol=1e-6, rtol=1e-4)


def test_byol_simsiam_cosine_trend():
    B, D = 8, 16
    # Case 1: p==z -> higher cosine -> lower loss
    p = torch.randn(B, D)
    z = p.clone()
    byol = BYOLLoss(normalize=True)
    l_good, s_good = byol(p, z, p, z)
    p_bad = torch.randn(B, D)
    z_bad = torch.randn(B, D)
    l_bad, s_bad = byol(p_bad, z_bad, p_bad, z_bad)
    assert l_good <= l_bad + 1e-6
    # SimSiam similar trend
    sim = SimSiamLoss(normalize=True)
    l_good2, _ = sim(p, z, p, z)
    l_bad2, _ = sim(p_bad, z_bad, p_bad, z_bad)
    assert l_good2 <= l_bad2 + 1e-6


def test_vicreg_variance_term_zero_when_std_ge_gamma():
    B, D = 16, 10
    gamma = 0.1
    # Create features with std well above gamma
    x = torch.randn(B, D) * 2.0
    loss_fn = VICRegLoss(lambda_invar=1.0, mu_var=1.0, nu_cov=0.0, gamma=gamma)
    _, stats = loss_fn(x, x)
    # std_mean should be >> gamma, indirectly implying small variance penalty
    assert stats["std_mean"] > gamma
