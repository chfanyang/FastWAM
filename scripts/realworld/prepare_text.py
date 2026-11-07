#!/usr/bin/env python3
"""GPU text precomputation. Run explicitly only when a GPU is available."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from fastwam.realworld.common import PROMPT_TEMPLATE, sha256_file, write_json_checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="data/realworld_artifacts/ray0_absolute_left_wrist_val2_g08_v2")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import torch
    from fastwam.models.wan22.helpers.loader import _load_registered_model
    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
    from fastwam.models.wan22.fastwam_visual_action import FastWAMVideoOnlyRaymap
    root = Path(args.artifacts)
    manifest = json.loads((root / "split.json").read_text())
    encoder_path = Path(args.checkpoints) / "DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
    tokenizer_path = Path(args.checkpoints) / "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
    instructions = sorted({r["instruction"] for r in manifest["episodes"]})
    identity = dict(encoder_sha256=sha256_file(encoder_path),
        tokenizer_sha256={str(p.relative_to(tokenizer_path)): sha256_file(p)
                          for p in sorted(tokenizer_path.rglob("*")) if p.is_file()},
        context_len=128, batch_size=1, dtype="bfloat16", prompt_template=PROMPT_TEMPLATE)
    output = root / "text"
    write_json_checked(output / "identity.json", identity)
    encoder = _load_registered_model(str(encoder_path), "wan_video_text_encoder",
        torch_dtype=torch.bfloat16, device=args.device).eval().requires_grad_(False)
    tokenizer = HuggingfaceTokenizer(name=str(tokenizer_path), seq_len=128, clean="whitespace")
    model = SimpleNamespace(text_encoder=encoder, tokenizer=tokenizer, device=args.device)
    files = {}
    with torch.inference_mode():
        for instruction in instructions:
            prompt = PROMPT_TEMPLATE.format(task=instruction)
            digest = hashlib.sha256(prompt.encode()).hexdigest()
            path = output / f"{digest}.pt"
            context, mask = FastWAMVideoOnlyRaymap.encode_prompt(model, prompt)
            payload = dict(context=context[0].cpu(), context_mask=mask[0].cpu().bool(),
                metadata=dict(identity, prompt=prompt))
            if path.exists():
                old = torch.load(path, map_location="cpu", weights_only=False)
                if old["metadata"] != payload["metadata"] or not all(
                    torch.equal(old[k], payload[k]) for k in ("context", "context_mask")):
                    raise ValueError(f"Text cache differs: {path}")
            else:
                tmp = path.with_suffix(".partial")
                torch.save(payload, tmp)
                tmp.replace(path)
            files[path.name] = sha256_file(path)
            print(instruction, flush=True)
    write_json_checked(output / "manifest.json", dict(identity=identity, files=files))
    (output / "_SUCCESS").write_text("complete\n")


if __name__ == "__main__":
    main()
