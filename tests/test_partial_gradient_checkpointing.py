from unittest.mock import patch

import pytest
import torch
import torch.utils.checkpoint

from fastwam.models.wan22 import wan_video_dit as module


def tiny(**kwargs):
    return module.WanVideoDiT(
        hidden_dim=36, in_dim=16, ffn_dim=72, out_dim=16,
        text_dim=36, freq_dim=18, eps=1e-6, patch_size=(1, 1, 1),
        num_heads=2, attn_head_dim=18, num_layers=2,
        has_image_input=False, seperated_timestep=True, **kwargs,
    )


@pytest.mark.parametrize('enabled,layers,expected', [
    (False, None, 0), (True, None, 2), (True, [0], 1),
    (True, [], 0), (False, [0], 0),
])
def test_selection_preserves_output_gradients_and_state(enabled, layers, expected):
    torch.manual_seed(7)
    base = tiny()
    model = tiny(use_gradient_checkpointing=enabled, gradient_checkpointing_layers=layers)
    model.load_state_dict(base.state_dict(), strict=True)
    assert model.state_dict().keys() == base.state_dict().keys()
    inputs = dict(x=torch.randn(1, 16, 2, 2, 2), timestep=torch.tensor([0.4]),
                  context=torch.randn(1, 3, 36), fuse_vae_embedding_in_latents=True)
    reference = base(**inputs)
    reference.square().mean().backward()
    with patch.object(module, 'gradient_checkpoint_forward',
                      wraps=module.gradient_checkpoint_forward) as call:
        actual = model(**inputs)
        actual.square().mean().backward()
        assert call.call_count == expected
    torch.testing.assert_close(actual, reference)
    for a, b in zip(model.parameters(), base.parameters()):
        if b.grad is None:
            assert a.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad)


@pytest.mark.parametrize('layers', [[-1], [2], [0.5], [True]])
def test_rejects_invalid_indices(layers):
    with pytest.raises(ValueError, match='gradient_checkpointing_layers'):
        tiny(gradient_checkpointing_layers=layers)
