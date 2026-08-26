import math

import torch

from fastwam.models.wan22.fastwam_visual_action import (
    CHANNEL_LAYOUT_IO_INIT_DUPLICATE_SQRT2,
    CHANNEL_LAYOUT_IO_INIT_RGB_PRESERVE,
    LATENT_LAYOUT_RGB_RAYMAP_CHANNEL,
    LATENT_LAYOUT_RGB_THEN_RAYMAP,
    FastWAMVideoOnlyRaymap,
    _expand_video_dit_io_for_rgb_raymap_channels,
)
from fastwam.models.wan22.wan_video_dit import WanVideoDiT


def _layout_only_model(layout: str) -> FastWAMVideoOnlyRaymap:
    model = FastWAMVideoOnlyRaymap.__new__(FastWAMVideoOnlyRaymap)
    torch.nn.Module.__init__(model)
    model.latent_layout = layout
    model.vae_latent_channels = 16
    model.num_latent_frames_per_modality = 5
    return model


def test_legacy_time_layout_round_trip_is_unchanged():
    model = _layout_only_model(LATENT_LAYOUT_RGB_THEN_RAYMAP)
    rgb = torch.randn(2, 16, 5, 4, 6)
    raymap = torch.randn_like(rgb)

    joint = model._join_rgb_raymap_latents(rgb, raymap)
    restored_rgb, restored_raymap = model._split_rgb_raymap_latents(joint)

    assert joint.shape == (2, 16, 10, 4, 6)
    torch.testing.assert_close(restored_rgb, rgb)
    torch.testing.assert_close(restored_raymap, raymap)


def test_channel_layout_round_trip_and_clean_condition():
    model = _layout_only_model(LATENT_LAYOUT_RGB_RAYMAP_CHANNEL)
    rgb = torch.randn(2, 16, 5, 4, 6)
    raymap = torch.randn_like(rgb)

    joint = model._join_rgb_raymap_latents(rgb, raymap)
    restored_rgb, restored_raymap = model._split_rgb_raymap_latents(joint)

    assert joint.shape == (2, 32, 5, 4, 6)
    torch.testing.assert_close(restored_rgb, rgb)
    torch.testing.assert_close(restored_raymap, raymap)

    noisy = torch.zeros(1, 32, 5, 4, 6)
    rgb_condition = torch.ones(1, 16, 1, 4, 6)
    raymap_condition = torch.full((1, 16, 1, 4, 6), 2.0)
    model._set_clean_conditions(noisy, rgb_condition, raymap_condition)
    torch.testing.assert_close(noisy[:, :16, 0:1], rgb_condition)
    torch.testing.assert_close(noisy[:, 16:, 0:1], raymap_condition)
    assert torch.count_nonzero(noisy[:, :, 1:]) == 0


def test_pretrained_dit_io_expansion_preserves_patch_order():
    dit = WanVideoDiT(
        hidden_dim=32,
        in_dim=16,
        ffn_dim=64,
        out_dim=16,
        text_dim=32,
        freq_dim=16,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        attn_head_dim=16,
        num_layers=1,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="bidirectional",
    )
    patch_weight = dit.patch_embedding.weight.detach().clone()
    patch_bias = dit.patch_embedding.bias.detach().clone()
    head_weight = dit.head.head.weight.detach().clone()
    head_bias = dit.head.head.bias.detach().clone()

    _expand_video_dit_io_for_rgb_raymap_channels(
        dit,
        target_channels=32,
        init_mode=CHANNEL_LAYOUT_IO_INIT_DUPLICATE_SQRT2,
    )

    assert dit.patch_embedding.weight.shape == (32, 32, 1, 2, 2)
    torch.testing.assert_close(
        dit.patch_embedding.weight[:, :16], patch_weight / math.sqrt(2.0)
    )
    torch.testing.assert_close(
        dit.patch_embedding.weight[:, 16:], patch_weight / math.sqrt(2.0)
    )
    torch.testing.assert_close(dit.patch_embedding.bias, patch_bias)

    expected_head_weight = torch.cat(
        (head_weight.reshape(4, 16, 32), head_weight.reshape(4, 16, 32)),
        dim=1,
    ).reshape(128, 32)
    expected_head_bias = torch.cat(
        (head_bias.reshape(4, 16), head_bias.reshape(4, 16)), dim=1
    ).reshape(128)
    assert dit.head.head.weight.shape == (128, 32)
    torch.testing.assert_close(dit.head.head.weight, expected_head_weight)
    torch.testing.assert_close(dit.head.head.bias, expected_head_bias)


def test_rgb_preserving_io_expansion_starts_from_exact_pretrained_rgb_features():
    dit = WanVideoDiT(
        hidden_dim=32,
        in_dim=16,
        ffn_dim=64,
        out_dim=16,
        text_dim=32,
        freq_dim=16,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        attn_head_dim=16,
        num_layers=1,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="bidirectional",
    )
    original_patch = dit.patch_embedding
    patch_weight = original_patch.weight.detach().clone()
    patch_bias = original_patch.bias.detach().clone()
    head_weight = dit.head.head.weight.detach().clone()
    head_bias = dit.head.head.bias.detach().clone()
    rgb = torch.randn(2, 16, 3, 4, 6)
    raymap = torch.randn_like(rgb)
    expected_rgb_features = original_patch(rgb)

    _expand_video_dit_io_for_rgb_raymap_channels(
        dit,
        target_channels=32,
        init_mode=CHANNEL_LAYOUT_IO_INIT_RGB_PRESERVE,
    )

    assert dit.patch_embedding.weight.shape == (32, 32, 1, 2, 2)
    torch.testing.assert_close(dit.patch_embedding.weight[:, :16], patch_weight)
    assert torch.count_nonzero(dit.patch_embedding.weight[:, 16:]) == 0
    torch.testing.assert_close(dit.patch_embedding.bias, patch_bias)
    torch.testing.assert_close(
        dit.patch_embedding(torch.cat((rgb, raymap), dim=1)),
        expected_rgb_features,
    )

    expected_head_weight = torch.cat(
        (head_weight.reshape(4, 16, 32), head_weight.reshape(4, 16, 32)),
        dim=1,
    ).reshape(128, 32)
    expected_head_bias = torch.cat(
        (head_bias.reshape(4, 16), head_bias.reshape(4, 16)), dim=1
    ).reshape(128)
    torch.testing.assert_close(dit.head.head.weight, expected_head_weight)
    torch.testing.assert_close(dit.head.head.bias, expected_head_bias)

    # Zero is only the initialization: both channel halves remain ordinary
    # trainable parameters and receive gradients on the first optimization step.
    loss = dit.patch_embedding(torch.cat((rgb, raymap), dim=1)).square().mean()
    loss.backward()
    grad = dit.patch_embedding.weight.grad
    assert grad is not None
    assert torch.count_nonzero(grad[:, :16]) > 0
    assert torch.count_nonzero(grad[:, 16:]) > 0
