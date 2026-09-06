import torch

from fastwam.models.wan22.fastwam_visual_action import (
    FUTURE_RGB_MODE_TRAIN_ONLY_AUXILIARY,
    FastWAMVideoOnlyRaymap,
)
from fastwam.models.wan22.wan_video_dit import (
    FRAME_ROLE_RAYMAP_CONDITION,
    FRAME_ROLE_RAYMAP_FUTURE,
    FRAME_ROLE_RGB_CONDITION,
    FRAME_ROLE_RGB_FUTURE,
    WanVideoDiT,
)


def _tiny_dit() -> WanVideoDiT:
    return WanVideoDiT(
        hidden_dim=36,
        in_dim=16,
        ffn_dim=72,
        out_dim=16,
        text_dim=36,
        freq_dim=18,
        eps=1e-6,
        patch_size=(1, 1, 1),
        num_heads=2,
        attn_head_dim=18,
        num_layers=2,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="independent_rgb_aux_ray",
        use_gradient_checkpointing=False,
    ).eval()


def _training_roles() -> tuple[str, ...]:
    return (
        FRAME_ROLE_RGB_CONDITION,
        *(FRAME_ROLE_RGB_FUTURE for _ in range(4)),
        FRAME_ROLE_RAYMAP_CONDITION,
        *(FRAME_ROLE_RAYMAP_FUTURE for _ in range(4)),
    )


def _inference_roles() -> tuple[str, ...]:
    return (
        FRAME_ROLE_RGB_CONDITION,
        FRAME_ROLE_RAYMAP_CONDITION,
        *(FRAME_ROLE_RAYMAP_FUTURE for _ in range(4)),
    )


def test_independent_rgb_aux_ray_frame_mask_matches_declared_dependencies():
    dit = _tiny_dit()
    mask = dit.build_video_to_video_mask(
        video_seq_len=10,
        video_tokens_per_frame=1,
        device=torch.device("cpu"),
        condition_frame_indices=(0, 5),
        frame_roles=_training_roles(),
    )
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(mask, expected)


@torch.no_grad()
def test_contiguous_rope_default_matches_explicit_positions_and_adds_no_parameters():
    torch.manual_seed(3)
    dit = _tiny_dit()
    x = torch.randn(1, 16, 10, 2, 2)
    timestep = torch.tensor([0.41])
    context = torch.randn(1, 3, 36)
    context_mask = torch.ones(1, 3, dtype=torch.bool)
    kwargs = {
        "x": x,
        "timestep": timestep,
        "context": context,
        "context_mask": context_mask,
        "fuse_vae_embedding_in_latents": True,
        "condition_latent_indices": (0, 5),
        "latent_frame_roles": _training_roles(),
    }
    legacy_positions = dit(**kwargs)
    explicit_positions = dit(
        **kwargs,
        temporal_position_indices=tuple(range(10)),
    )
    torch.testing.assert_close(legacy_positions, explicit_positions)

    # The feature only changes attention/position inputs. It must not add a
    # learned embedding, projection, or any other checkpoint parameter.
    assert not any(
        "role" in name or "position" in name
        for name in dit.state_dict()
    )


@torch.no_grad()
def test_pruned_ray_inference_is_the_same_masked_subgraph_as_training():
    torch.manual_seed(7)
    dit = _tiny_dit()
    full = torch.randn(1, 16, 10, 2, 2)
    pruned = torch.cat((full[:, :, 0:1], full[:, :, 5:]), dim=2)
    timestep = torch.tensor([0.63])
    context = torch.randn(1, 3, 36)
    context_mask = torch.ones(1, 3, dtype=torch.bool)

    full_output = dit(
        x=full,
        timestep=timestep,
        context=context,
        context_mask=context_mask,
        fuse_vae_embedding_in_latents=True,
        condition_latent_indices=(0, 5),
        latent_frame_roles=_training_roles(),
        temporal_position_indices=tuple(range(10)),
    )
    pruned_output = dit(
        x=pruned,
        timestep=timestep,
        context=context,
        context_mask=context_mask,
        fuse_vae_embedding_in_latents=True,
        condition_latent_indices=(0, 1),
        latent_frame_roles=_inference_roles(),
        temporal_position_indices=(0, 5, 6, 7, 8, 9),
    )

    # The retained RGB0 and complete Raymap block form exactly the same
    # per-layer computation graph after future RGB tokens are removed.
    retained_full_output = torch.cat(
        (full_output[:, :, 0:1], full_output[:, :, 5:]), dim=2
    )
    torch.testing.assert_close(
        pruned_output,
        retained_full_output,
        rtol=1e-5,
        atol=1e-5,
    )


class _OneStepScheduler:
    def build_inference_schedule(self, **_):
        return torch.tensor([0.5]), torch.tensor([0.1])

    def step(self, prediction, delta, latents):
        return latents + prediction * delta


@torch.no_grad()
def test_train_only_rgb_inference_builds_six_frames_and_skips_rgb_decode():
    model = FastWAMVideoOnlyRaymap.__new__(FastWAMVideoOnlyRaymap)
    torch.nn.Module.__init__(model)
    model.future_rgb_mode = FUTURE_RGB_MODE_TRAIN_ONLY_AUXILIARY
    model.inference_predict_future_rgb = False
    model.num_pixel_frames = 17
    model.num_latent_frames_per_modality = 5
    model.vae_latent_channels = 16
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.proprio_encoder = None
    model.infer_scheduler = _OneStepScheduler()
    model.vae = type("Vae", (), {"upsampling_factor": 8})()
    model.rothko_decode_anchor_alpha = 0.0

    calls = {}

    def encode(video, tiled=False):
        del tiled
        return torch.ones(
            video.shape[0], 16, 1, video.shape[-2] // 8, video.shape[-1] // 8
        )

    def model_fn(
        latents,
        timestep,
        context,
        context_mask,
        **kwargs,
    ):
        del timestep, context, context_mask
        calls["latent_shape"] = tuple(latents.shape)
        calls.update(kwargs)
        return torch.zeros_like(latents)

    decoded_shapes = []

    def decode(latents, tiled=False):
        del tiled
        decoded_shapes.append(tuple(latents.shape))
        return torch.zeros(
            latents.shape[0],
            3,
            17,
            latents.shape[-2] * 8,
            latents.shape[-1] * 8,
        )

    model._encode_video_latents = encode
    model._model_fn = model_fn
    model._decode_video_tensor = decode

    result = model.infer(
        prompt=None,
        input_image=torch.zeros(1, 3, 16, 16),
        input_raymap=torch.zeros(1, 3, 16, 16),
        context=torch.zeros(1, 2, 36),
        context_mask=torch.ones(1, 2, dtype=torch.bool),
        num_inference_steps=1,
    )

    assert calls["latent_shape"] == (1, 16, 6, 2, 2)
    assert calls["condition_latent_indices"] == (0, 1)
    assert calls["temporal_position_indices"] == (0, 5, 6, 7, 8, 9)
    assert calls["latent_frame_roles"] == _inference_roles()
    assert decoded_shapes == [(1, 16, 5, 2, 2)]
    assert set(result) == {"raymap"}
    assert result["raymap"].shape == (1, 3, 17, 16, 16)


@torch.no_grad()
def test_train_only_rgb_can_restore_full_training_graph_at_inference():
    model = FastWAMVideoOnlyRaymap.__new__(FastWAMVideoOnlyRaymap)
    torch.nn.Module.__init__(model)
    model.future_rgb_mode = FUTURE_RGB_MODE_TRAIN_ONLY_AUXILIARY
    model.inference_predict_future_rgb = True
    model.num_pixel_frames = 17
    model.num_latent_frames_per_modality = 5
    model.vae_latent_channels = 16
    model.latent_layout = "rgb_then_raymap"
    model.condition_latent_indices = (0, 5)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.proprio_encoder = None
    model.infer_scheduler = _OneStepScheduler()
    model.vae = type("Vae", (), {"upsampling_factor": 8})()
    model.rothko_decode_anchor_alpha = 0.0

    model._encode_video_latents = lambda video, tiled=False: torch.ones(
        video.shape[0], 16, 1, video.shape[-2] // 8, video.shape[-1] // 8
    )
    calls = {}

    def model_fn(latents, timestep, context, context_mask, **kwargs):
        del timestep, context, context_mask
        calls["latent_shape"] = tuple(latents.shape)
        calls.update(kwargs)
        return torch.zeros_like(latents)

    model._model_fn = model_fn
    model._decode_video_tensor = lambda latents, tiled=False: torch.zeros(
        latents.shape[0], 3, 17, latents.shape[-2] * 8, latents.shape[-1] * 8
    )
    model._video_tensor_to_pil = lambda video: [None] * video.shape[1]

    result = model.infer(
        prompt=None,
        input_image=torch.zeros(1, 3, 16, 16),
        input_raymap=torch.zeros(1, 3, 16, 16),
        context=torch.zeros(1, 2, 36),
        context_mask=torch.ones(1, 2, dtype=torch.bool),
        num_inference_steps=1,
    )

    assert calls["latent_shape"] == (1, 16, 10, 2, 2)
    assert calls["condition_latent_indices"] == (0, 5)
    assert calls["latent_frame_roles"] is None
    assert calls["temporal_position_indices"] is None
    assert "video" in result


@torch.no_grad()
def test_full_graph_can_skip_unused_future_rgb_decode():
    model = FastWAMVideoOnlyRaymap.__new__(FastWAMVideoOnlyRaymap)
    torch.nn.Module.__init__(model)
    model.future_rgb_mode = FUTURE_RGB_MODE_TRAIN_ONLY_AUXILIARY
    model.inference_predict_future_rgb = True
    model.num_pixel_frames = 17
    model.num_latent_frames_per_modality = 5
    model.vae_latent_channels = 16
    model.latent_layout = "rgb_then_raymap"
    model.condition_latent_indices = (0, 5)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.proprio_encoder = None
    model.infer_scheduler = _OneStepScheduler()
    model.vae = type("Vae", (), {"upsampling_factor": 8})()
    model.rothko_decode_anchor_alpha = 0.0

    model._encode_video_latents = lambda video, tiled=False: torch.ones(
        video.shape[0], 16, 1, video.shape[-2] // 8, video.shape[-1] // 8
    )
    model._model_fn = lambda latents, *args, **kwargs: torch.zeros_like(latents)
    decoded_shapes = []

    def decode(latents, tiled=False):
        del tiled
        decoded_shapes.append(tuple(latents.shape))
        return torch.zeros(
            latents.shape[0], 3, 17, latents.shape[-2] * 8, latents.shape[-1] * 8
        )

    model._decode_video_tensor = decode

    result = model.infer(
        prompt=None,
        input_image=torch.zeros(1, 3, 16, 16),
        input_raymap=torch.zeros(1, 3, 16, 16),
        context=torch.zeros(1, 2, 36),
        context_mask=torch.ones(1, 2, dtype=torch.bool),
        num_inference_steps=1,
        decode_future_rgb=False,
    )

    # Raymap decode is required for control; future RGB decode is omitted.
    assert decoded_shapes == [(1, 16, 5, 2, 2)]
    assert set(result) == {"raymap"}
