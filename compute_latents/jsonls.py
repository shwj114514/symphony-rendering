import argparse
import os
import time
from pathlib import Path

import h5py
import librosa
import numpy as np
import torch
import torch.multiprocessing as mp

from audio_flow.utils import forward_in_chunks
from audio_flow.vae.levo import LevoVAE
from tqdm import tqdm

from pathlib import Path
import typing as tp
import os
import torchaudio

def get_filelist(
    folder: tp.Union[str, os.PathLike],
    extensions: tp.Optional[tp.List[str]] = None
) -> tp.List[str]:
    if extensions is None:
        extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']
    extensions = set(ext.lower() for ext in extensions)
    
    path = Path(folder)
    if not path.is_dir():
        raise ValueError(f"The provided directory '{folder}' is not a valid directory.")
    filelist = [str(file) for file in path.rglob('*') if file.suffix.lower() in extensions]
    return filelist


def compute_vae_worker(rank, world_size, args,jsonl = [],save_audio_np = True):
    out_dir = args.out_dir
    aug_repeats = args.augmentation_repeats

    device = f"cuda:{rank}" if world_size > 1 else "cuda"
    vae = LevoVAE().to(device)
    clip_duration = 30.0
    clip_samples = int(clip_duration * vae.sr)

    # labels = sorted(os.listdir(Path(root, "genres")))
    # tasks = []
    # for label in labels:
    #     for path in sorted(Path(root, "genres", label).glob("*.au")):
    #         tasks.append((label, path))

    task_jsonl= jsonl[rank::world_size]

    for idx,  task_json in tqdm(enumerate(task_jsonl)):
        # audio, fs = librosa.load(path=path, sr=vae.sr, mono=False)
        audio_path = task_json["path"]
        wav,sr = torchaudio.load(audio_path)
        label = task_json["author"]
        target_sr = vae.sr
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            wav = resampler(wav)
            sr = target_sr
        
        audio = wav.numpy()
        if audio.shape[0] == 1:
            audio = np.repeat(audio[None, :], repeats=2, axis=0)

        if save_audio_np:
            wavnp_path = Path(out_dir,"wav_np", label, f"{Path(audio_path).stem}.npy")
            wavnp_path.parent.mkdir(parents=True, exist_ok=True)

            np.save(wavnp_path, audio)



        t1 = time.time()
        latents = forward_in_chunks(vae, audio, clip_samples)  # (d, t)
        print(f"latent = {latents.shape}")
        t = time.time() - t1

        out_path = Path(out_dir,"vae_latent", label, f"{Path(audio_path).stem}_vae.h5")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, "w") as hf:
            hf.create_dataset("latent", data=latents, dtype=np.float32)
            hf.attrs.create("fps", data=vae.fps, dtype=float)
        print(f"[GPU {rank}/{world_size}] {label}/{Path(audio_path).name} -> {out_path} ({t:.2f}s)")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--augmentation_repeats", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()

    import json
    flielist = []
    with open(args.jsonl_path, 'r') as f:
        for line in f:
            flielist.append(json.loads(line.strip()))

    if args.num_gpus <= 1:
        compute_vae_worker(rank=0, world_size=1, args=args,jsonl = flielist)
    else:
        world_size = min(args.num_gpus, torch.cuda.device_count())
        mp.spawn(compute_vae_worker, args=(world_size, args,flielist), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()