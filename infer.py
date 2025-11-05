import math
import torchdiffeq
from torch.utils.data._utils.collate import default_collate
from pathlib import Path
import matplotlib.pyplot as plt
import soundfile
from audio_flow.utils import CombinedModel, parse_yaml, logmel

import torch

from train import get_adaptor,get_base,get_data_transform,get_dataset
import typing as tp

import argparse

import torchaudio
import numpy as np

from audio_flow.data_transforms.transt2music import Trans2MusicVAE

OUTPUT_FOLDER = "exp_infer"
@torch.no_grad()
def _load_model_for_infer(configs: dict, ckpt_path: str, device: str) -> CombinedModel:
    """Build model exactly as in training and load EMA weights."""
    base = get_base(configs).to(device)
    adaptor = get_adaptor(configs).to(device)
    model = CombinedModel(base, adaptor).to(device)

    if ckpt_path is None or ckpt_path == "":
        ckpt_path = configs["train"].get("resume_ckpt_path", "")
    if not ckpt_path:
        raise ValueError("Please provide --ckpt or set train.resume_ckpt_path in the YAML.")

    # Load on the right device
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

import re
def _sanitize_filename(s: str) -> str:
    # keep CJK and letters/numbers; replace forbidden chars with '_'
    return re.sub(r'[\\/:*?"<>|\n\r\t]+', '_', s).strip()


@torch.no_grad()
def infer_once(
    config_path: str,
    ckpt_path: str | None,
    split: tp.Literal["train", "test"] = "test",
    idx: int = -1,
    num: int = 1,
    out_dir: str = "./results/infer",
    seed: int = 0,
    ode_method: str = "dopri5",
    rtol: float = 1e-4,
    atol: float = 1e-4,
    override_caption: str | None = None,
    override_label_id: int | None = None,
):
    """
    Generate audio with a trained checkpoint.
    If idx >= 0, generate for that single item from the dataset.
    Otherwise, take `num` evenly-spaced items from the split.
    """
    configs = parse_yaml(config_path)
    # device = configs["train"]["device"]

    device = "cuda"

    # Build data side
    data_transform:Trans2MusicVAE = get_data_transform(configs).to(device)
    dataset = get_dataset(configs, split=split, mode="test")



    # Build & load model
    model = _load_model_for_infer(configs, ckpt_path, device)

    # Output directory
    config_name = Path(config_path).stem
    ckpt_name = Path(ckpt_path).stem if ckpt_path else "resume"
    out_dir = Path(out_dir, config_name, ckpt_name, split)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic noise per item if desired
    g = torch.Generator(device=device).manual_seed(seed)
    


    AUDIO_PATH = "20250927-121041.wav"
 
    wav, sr = torchaudio.load(AUDIO_PATH)  # [C, L], float32 in [-1, 1] typically
    target_sr = 48000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        wav = resampler(wav)
        sr = target_sr

    # Crop/pad to model’s clip length
    clip_seconds = float(configs["clip_duration"])
    target_len = int(sr * clip_seconds)
    start_frame = 48000 * 45

    print(f"wav.shape = {wav.shape}")

    if wav.size(1) >= target_len:
        if clip_seconds == 30.0:
            wav = wav[:, :target_len]
            wav = wav[:, start_frame: start_frame + target_len]


        if clip_seconds == 10.0:
            start_frame = 48000 * 20
            wav = wav[:,start_frame :start_frame + target_len]

    else:
        wav = torch.nn.functional.pad(wav, (0, target_len - wav.size(1)))

    print(f"wav.shape = {wav.shape}")
    # labels = ['舒伯特（Franz Schubert）', '莫扎特（Wolfgang Amadeus Mozart）', '巴赫（Johann Sebastian Bach）', '拉赫玛尼诺夫（Sergei Rachmaninoff）', '巴赫（Johann Sebastian Bach）', '马勒（Gustav Mahler）', '马勒（Gustav Mahler）', '德沃夏克（Antonín Dvořák）']

    labels = ['舒伯特（Franz Schubert）', '莫扎特（Wolfgang Amadeus Mozart）', '巴赫（Johann Sebastian Bach）', '拉赫玛尼诺夫（Sergei Rachmaninoff）',"贝多芬（Ludwig van Beethoven）"]


    B = len(labels)
    batched_wav = wav.unsqueeze(0).repeat(B, 1, 1)  # [B, 2, L]

    # Build a "data" dict that matches your pipeline (like a collated batch)
    all_idx = []
    for label in labels:
        idx = dataset.lb_to_ix[label]
        all_idx.append(idx) 
    all_idx = torch.LongTensor(all_idx)
    data = {"wav": batched_wav, "label": labels,"latent":torch.rand(B,64,int(clip_seconds*25)),"target":all_idx}

    x_real, cond_dict = data_transform(data)  # x_real: [B, D, T], cond_dict contains ids/emb info
    # ---------- Flow ODE (noise -> sample) ----------
    g = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn_like(x_real).to(device)
    emb = model.adaptor(cond_dict)
    
    traj = torchdiffeq.odeint(
        func=lambda t, x: model.base(t, x, emb),
        y0=noise,
        t=torch.linspace(0, 1, 2, device=device),
        method=ode_method,
        rtol=rtol,
        atol=atol,
    )
    x_gen = traj[-1]  # [B, D, T]

    # ---------- Decode & save per-label ----------
    # Use the transform’s output sample rate (usually matches training SR)
    sr_out = getattr(data_transform, "sr", sr)

    gen_audio = data_transform.latent_to_audio(x_gen).data.cpu()  # [B, C, L]
    # (Optional) GT decode if you want it:
    gt_audio = data_transform.latent_to_audio(x_real).data.cpu()  # [B, C, L]
    stem = Path(AUDIO_PATH).stem


    out_root = f"{OUTPUT_FOLDER}_{_sanitize_filename(stem)}_gen"
    out_dir = Path(out_root) / Path(config_path).stem / Path(ckpt_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, label in enumerate(labels):
        # fname = f"{_sanitize_filename(label)}_{_sanitize_filename(stem)}_gen.wav"
        fname = f"{_sanitize_filename(label)}.wav"


        path = out_dir / fname
        # wav_np = gen_audio[i].T.astype(np.float32)  # [L, C]
        # sf.write(file=str(path), data=wav_np, samplerate=sr_out)


        torchaudio.save(str(path),gen_audio[i],sample_rate =  48000)
        print(f"[✓] Wrote {path}")

    # path = out_dir / "recon.wav"
    # torchaudio.save(str(path),gt_audio[0], sample_rate = 48000)


    path = out_dir / "real.wav"
    torchaudio.save(str(path),wav, sample_rate = 48000)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)

    # ---------- NEW: inference flags ----------
    parser.add_argument("--infer", action="store_true", help="Run inference instead of training.")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to EMA .pt checkpoint. If omitted, uses train.resume_ckpt_path in YAML.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--idx", type=int, default=-1, help="Single index from the dataset to condition on; -1 = auto pick evenly-spaced items.")
    parser.add_argument("--num", type=int, default=1, help="How many items to generate if --idx = -1.")
    parser.add_argument("--out_dir", type=str, default="./results/infer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ode_method", type=str, default="dopri5", choices=["dopri5", "rk4", "euler", "midpoint"])
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--caption", type=str, default=None, help="Optional: override caption in cond_dict if your transform supports it.")
    parser.add_argument("--label_id", type=int, default=None, help="Optional: override label id in cond_dict if your transform supports it.")
    # ------------------------------------------

    args = parser.parse_args()

    infer_once(
        config_path=args.config,
        ckpt_path=args.ckpt,
        split=args.split,
        idx=args.idx,
        num=args.num,
        out_dir=args.out_dir,
        seed=args.seed,
        ode_method=args.ode_method,
        rtol=args.rtol,
        atol=args.atol,
        override_caption=args.caption,
        override_label_id=args.label_id,
    )
