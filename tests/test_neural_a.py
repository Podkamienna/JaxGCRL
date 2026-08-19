"""Checks for Ben-style residual-MLP A (neural_A), no brax required."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def _load_networks():
    path = Path(__file__).resolve().parents[1] / "jaxgcrl" / "agents" / "crl" / "networks.py"
    spec = importlib.util.spec_from_file_location("crl_networks", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_networks = _load_networks()
NeuralANet = _networks.NeuralANet
GoalEncoderWithA = _networks.GoalEncoderWithA


def test_neural_a_net_is_identity_at_init():
    net = NeuralANet(repr_dim=8, hidden_size=16, depth=4)
    x = jnp.ones((5, 8))
    params = net.init(jax.random.PRNGKey(0), x)
    y = net.apply(params, x)
    np.testing.assert_allclose(np.asarray(y), np.asarray(x), atol=1e-6)


def test_goal_encoder_delta_zero_equals_psi():
    enc = GoalEncoderWithA(
        repr_dim=8,
        network_width=16,
        network_depth=2,
        use_A=True,
        neural_A=True,
        neural_A_depth=4,
        rotational_A=False,
        max_A_applications=4,
    )
    x = jnp.ones((3, 4))
    params = enc.init(jax.random.PRNGKey(1), x, jnp.zeros((3,), dtype=jnp.int32))
    phi, psi = enc.apply(params, x, jnp.zeros((3,), dtype=jnp.int32))
    np.testing.assert_allclose(np.asarray(phi), np.asarray(psi), atol=1e-5)


def test_perturbed_a_net_delta_two_is_double_apply():
    net = NeuralANet(repr_dim=8, hidden_size=16, depth=4)
    x = jax.random.normal(jax.random.PRNGKey(2), (4, 8))
    params = net.init(jax.random.PRNGKey(3), x)
    params = jax.tree_util.tree_map(lambda p: p + 0.05, params)
    y1 = net.apply(params, x)
    y2 = net.apply(params, y1)
    enc = GoalEncoderWithA(
        repr_dim=8,
        network_width=16,
        network_depth=2,
        use_A=True,
        neural_A=True,
        neural_A_depth=4,
        rotational_A=False,
        max_A_applications=4,
    )
    gx = jnp.ones((4, 4))
    gparams = enc.init(jax.random.PRNGKey(4), gx, jnp.ones((4,), dtype=jnp.int32))
    # Isolated A_net double-apply must differ from identity after perturbation.
    np.testing.assert_(float(jnp.mean((y2 - x) ** 2)) > 1e-6)
    np.testing.assert_(float(jnp.mean((y1 - x) ** 2)) > 1e-6)
    phi0, psi0 = enc.apply(gparams, gx, jnp.zeros((4,), dtype=jnp.int32))
    phi2, psi2 = enc.apply(gparams, gx, 2 * jnp.ones((4,), dtype=jnp.int32))
    np.testing.assert_allclose(np.asarray(psi0), np.asarray(psi2), atol=1e-5)
    np.testing.assert_allclose(np.asarray(phi0), np.asarray(psi0), atol=1e-5)
    # Zero-init last layer => identity even for delta=2 until A_net is trained.
    np.testing.assert_allclose(np.asarray(phi2), np.asarray(psi2), atol=1e-5)
