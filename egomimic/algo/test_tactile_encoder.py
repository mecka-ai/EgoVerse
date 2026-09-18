import torch

from egomimic.algo.tactile_encoder import TactileEncoder, sigmoid_anneal
from egomimic.models.tactile_nets import (
    compute_proxy_features,
    patchify,
    proxy_guided_cost_matrix,
    random_masking,
    unpatchify,
)

_PLATFORMS = {
    "human": {"input_shape": [8, 1, 8, 8], "patch_size": [2, 4, 4]},
    "robot": {"input_shape": [4, 1, 4, 4], "patch_size": [2, 2, 2]},
}


def _make_algo(**overrides):
    kwargs = dict(
        platforms=_PLATFORMS,
        d_model=32,
        d_latent=16,
        d_proj=8,
        proj_hidden_dim=16,
        num_encoder_layers=2,
        num_encoder_heads=4,
        decoder_num_layers=1,
        decoder_num_heads=2,
        mask_ratio=0.75,
        alpha=1.0,
        margin=0.05,
        warmup_steps=3,
        alignment_end_steps=6,
        lambda_max=2.0,
        device=torch.device("cpu"),
    )
    kwargs.update(overrides)
    return TactileEncoder(**kwargs)


def _make_raw_batch(batch_size=5, human_batch_size=None, robot_batch_size=None):
    return {
        "human": {"tactile": torch.rand(human_batch_size or batch_size, 8, 1, 8, 8)},
        "robot": {"tactile": torch.rand(robot_batch_size or batch_size, 4, 1, 4, 4)},
    }


# ---------------------------------------------------------------------------
# Pure tensor-math helpers (models/tactile_nets.py)
# ---------------------------------------------------------------------------


def test_patchify_unpatchify_roundtrip():
    x = torch.randn(2, 8, 1, 8, 8)
    patches, grid = patchify(x, patch_t=2, patch_h=4, patch_w=4)

    assert grid == (4, 2, 2)
    assert patches.shape == (2, 16, 2 * 4 * 4 * 1)

    recon = unpatchify(patches, grid, 2, 4, 4, 1)
    assert torch.allclose(recon, x)


def test_random_masking_ratio_and_consistency():
    tokens = torch.randn(3, 20, 16)
    visible, mask, ids_restore, ids_keep = random_masking(tokens, mask_ratio=0.75)

    assert visible.shape == (3, 5, 16)
    assert mask.shape == (3, 20)
    assert (mask.sum(dim=1) == 15).all()
    assert ids_restore.shape == (3, 20)
    for b in range(3):
        assert not mask[b, ids_keep[b]].any()


def test_compute_proxy_features():
    x = torch.zeros(1, 4, 1, 2, 2)
    x[0, 0] = 1.0  # total force 4
    x[0, 1] = 2.0  # total force 8
    x[0, 2] = 2.0
    x[0, 3] = 2.0

    phi = compute_proxy_features(x)

    # |dF/dt| across t: [4, 0, 0] -> mean 4/3; mean force = 7
    expected_relative_dF_dt = (4.0 / 3.0) / 7.0
    assert phi.shape == (1, 2)
    assert abs(phi[0, 0].item() - expected_relative_dF_dt) < 1e-4
    assert abs(phi[0, 1].item() - 1.0) < 1e-6  # always fully in contact


def test_proxy_guided_cost_matrix_matches_formula():
    z_h = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    z_r = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    phi_h = torch.tensor([[0.0], [1.0]])
    phi_r = torch.tensor([[0.0], [2.0]])

    C = proxy_guided_cost_matrix(z_h, z_r, phi_h, phi_r, alpha=2.0)

    expected = torch.tensor([[0.0, 9.0], [3.0, 4.0]])
    assert torch.allclose(C, expected)


def test_proxy_guided_cost_matrix_handles_unequal_batch_sizes():
    z_h = torch.randn(7, 4)
    z_r = torch.randn(3, 4)
    phi_h = torch.randn(7, 2)
    phi_r = torch.randn(3, 2)

    C = proxy_guided_cost_matrix(z_h, z_r, phi_h, phi_r, alpha=1.0)
    assert C.shape == (7, 3)


# ---------------------------------------------------------------------------
# sigmoid_anneal (Phase 3 schedule)
# ---------------------------------------------------------------------------


def test_sigmoid_anneal_hits_boundaries_exactly():
    assert sigmoid_anneal(0, 100, 200, 1.0) == 0.0
    assert sigmoid_anneal(100, 100, 200, 1.0) == 0.0  # inclusive: still stage 1
    assert sigmoid_anneal(200, 100, 200, 1.0) == 1.0
    assert sigmoid_anneal(300, 100, 200, 1.0) == 1.0


def test_sigmoid_anneal_is_monotonic_between_boundaries():
    values = [sigmoid_anneal(s, 100, 200, 1.0) for s in range(100, 201, 10)]
    assert all(a <= b + 1e-9 for a, b in zip(values, values[1:]))
    assert 0.0 < sigmoid_anneal(150, 100, 200, 1.0) < 1.0


# ---------------------------------------------------------------------------
# TactileEncoder (Algo interface + two-stage schedule integration)
# ---------------------------------------------------------------------------


def test_stage1_warmup_gates_ot_loss_to_exactly_zero():
    algo = _make_algo()
    raw_batch = _make_raw_batch()

    for step in range(1, algo.warmup_steps + 1):
        batch = algo.process_batch_for_training(raw_batch)
        predictions = algo.forward_training(batch)
        losses = algo.compute_losses(predictions, batch)
        log = algo.log_info({"losses": losses})

        assert algo.training_step == step
        assert losses["lambda_ot"].item() == 0.0
        assert losses["ot_loss"].item() == 0.0
        assert log["stage"] == 1.0
        assert torch.isfinite(losses["action_loss"])


def test_stage2_activates_annealed_ot_loss():
    algo = _make_algo()
    raw_batch = _make_raw_batch()
    batch = algo.process_batch_for_training(raw_batch)

    for _ in range(algo.warmup_steps):
        predictions = algo.forward_training(batch)
        losses = algo.compute_losses(predictions, batch)

    # First step past warmup: lambda_ot should be > 0 but not yet lambda_max.
    predictions = algo.forward_training(batch)
    losses = algo.compute_losses(predictions, batch)
    log = algo.log_info({"losses": losses})

    assert log["stage"] == 2.0
    assert 0.0 < losses["lambda_ot"].item() < algo.lambda_max
    assert torch.isfinite(losses["action_loss"])

    # Fast-forward well past alignment_end_steps: lambda_ot saturates at lambda_max.
    for _ in range(20):
        predictions = algo.forward_training(batch)
        losses = algo.compute_losses(predictions, batch)
    assert losses["lambda_ot"].item() == algo.lambda_max


def test_action_loss_key_present_for_model_wrapper():
    # pl_utils.pl_model.ModelWrapper.training_step hard-codes
    # `losses["action_loss"]` as the tensor it backpropagates through.
    algo = _make_algo()
    batch = algo.process_batch_for_training(_make_raw_batch())
    predictions = algo.forward_training(batch)
    losses = algo.compute_losses(predictions, batch)
    assert "action_loss" in losses
    assert losses["action_loss"].requires_grad


def test_projector_gets_no_gradient_during_warmup_but_does_in_stage2():
    # margin=0.0 here: with the default margin, the raw Sinkhorn distance
    # between two tiny, untrained projections can legitimately fall below it,
    # correctly zeroing the gated OT loss (and its gradient) even in stage 2.
    # That margin-gating behavior is exact pure-tensor-math (see
    # `test_proxy_guided_cost_matrix_matches_formula`); zeroing it out here
    # isolates what this test actually checks: that the OT branch is wired
    # to the projector's gradient at all once stage 2 begins.
    torch.manual_seed(0)
    algo = _make_algo(warmup_steps=1, alignment_end_steps=3, lambda_max=1.0, margin=0.0)
    model = algo.nets["policy"]
    batch = algo.process_batch_for_training(_make_raw_batch())

    predictions = algo.forward_training(batch)  # step 1: still warmup
    losses = algo.compute_losses(predictions, batch)
    losses["action_loss"].backward()
    grad = model["projector"].net[0].weight.grad
    assert grad is None or torch.all(grad == 0)
    model.zero_grad(set_to_none=True)

    for _ in range(2):  # steps 2-3: stage 2, OT active
        predictions = algo.forward_training(batch)
        losses = algo.compute_losses(predictions, batch)
    losses["action_loss"].backward()
    grad = model["projector"].net[0].weight.grad
    assert grad is not None and torch.any(grad != 0)


def test_unpaired_batch_sizes_between_platforms():
    algo = _make_algo(warmup_steps=0, alignment_end_steps=2, lambda_max=1.0)
    raw_batch = _make_raw_batch(human_batch_size=7, robot_batch_size=4)
    batch = algo.process_batch_for_training(raw_batch)
    predictions = algo.forward_training(batch)
    losses = algo.compute_losses(predictions, batch)
    assert torch.isfinite(losses["action_loss"])


def test_forward_eval_supports_single_platform_batch():
    algo = _make_algo()
    algo.nets.eval()
    eval_batch = {"human": {"tactile": torch.rand(5, 8, 1, 8, 8)}}

    predictions = algo.forward_eval(eval_batch)

    assert predictions["human_z"].shape == (
        5,
        algo.nets["policy"]["encoder"].bottleneck_proj.out_features,
    )
    assert "robot_z" not in predictions


def test_export_policy_backbone_is_frozen_and_matches_shapes():
    algo = _make_algo()
    backbone = algo.export_policy_backbone()

    for p in backbone.parameters():
        assert not p.requires_grad

    z, tokens = backbone("human", torch.rand(5, 8, 1, 8, 8))
    assert z.shape == (5, 16)
    assert tokens.shape[0] == 5

    backbone.train()
    assert not backbone.training  # stays frozen/eval regardless of .train()


def test_requires_exactly_two_platforms():
    import pytest

    with pytest.raises(ValueError):
        TactileEncoder(platforms={"human": _PLATFORMS["human"]})
