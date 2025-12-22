from pathlib import Path
import re

import torch
import torchaudio
import torchdiffeq
import yaml
from audio_flow.utils import CombinedModel
from train import get_adaptor, get_base, get_data_transform, get_dataset

CONFIG_PATH = "configs/transcript2muic_30s.yaml"

# CKPT_PATH = hf_hub_download(
#     repo_id="shwj/symphony-rendering",
#     filename="model_traced_slakh.pt",
# )
CKPT_PATH = "/lan/ifc/downloaded_datasets/ljh_tmp/step=150000_ema.pt"

DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

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

with open(CONFIG_PATH, "r") as fr:
    model_config = yaml.load(fr, Loader=yaml.FullLoader)
model = _load_model_for_infer(model_config, CKPT_PATH, DEVICE)
print(f"total params = {sum(p.numel() for p in model.parameters())/1024/1024} M")


from audio_flow.data_transforms.transt2music import Trans2MusicVAE
data_transform:Trans2MusicVAE = get_data_transform(model_config).to(DEVICE)
print(f"total params = {sum(p.numel() for p in model.parameters())/1024/1024} M")


AUDIO_PATH  = "Bach-Minuet-in-G-major-BWV-Anh-114-Piano_360p.wav"  # condition
OUTPUT_ROOT = "exp_infer_notebook"                # output directory prefix

LABEL_TO_INDEX = {
    "Johannes Brahms": 0,
    "Johann Sebastian Bach": 1,
    "Antonín Dvořák": 2,
    "Sergei Rachmaninoff": 3,
    "Sergei Prokofiev": 4,
    "Pyotr Ilyich Tchaikovsky": 5,
    "Joseph Haydn": 6,
    "Dmitri Shostakovich": 7,
    "Franz Schubert": 8,
    "Wolfgang Amadeus Mozart": 9,
    "Ludwig van Beethoven": 10,
    "Gustav Mahler": 11,
}

LABELS = [
    "Franz Schubert",
    "Wolfgang Amadeus Mozart",
    "Johann Sebastian Bach",
    "Sergei Rachmaninoff",
    "Ludwig van Beethoven",
]


wav, sr = torchaudio.load(AUDIO_PATH)  # [C, L], float32 in [-1, 1] typically
target_sr = 48000
if sr != target_sr:
    resampler = torchaudio.transforms.Resample(sr, target_sr)
    wav = resampler(wav)
    sr = target_sr



# Crop/pad to model’s clip length
clip_seconds = float(model_config["clip_duration"])
target_len = int(sr * clip_seconds)
start_frame = 0
start_frame = wav.shape[1]//2 - target_len//2 

print(f"wav.shape = {wav.shape} start_frame = {start_frame} target_len = {target_len}")

if wav.size(1) >= target_len:
    if clip_seconds == 30.0:
        # wav = wav[:, :target_len]
        wav = wav[:, start_frame: start_frame + target_len]


B = len(LABELS)
batched_wav_16k = wav.unsqueeze(0).repeat(B, 1, 1)  # [B, 2, L]
all_idx = []
for label in LABELS:
    idx = LABEL_TO_INDEX[label]
    all_idx.append(idx) 
all_idx = torch.LongTensor(all_idx)
data = {"wav": batched_wav_16k, "label": LABELS,"latent":torch.rand(B,64,int(clip_seconds*25)),"target":all_idx}

with torch.no_grad():
    x_real, cond_dict = data_transform(data)  # x_real: [B, D, T], cond_dict contains ids/emb info
    # ---------- Flow ODE (noise -> sample) ----------
    g = torch.Generator(device=DEVICE).manual_seed(42)
    noise = torch.randn_like(x_real).to(DEVICE)
    emb = model.adaptor(cond_dict)
    traj = torchdiffeq.odeint(
        func=lambda t, x: model.base(t, x, emb),
        y0=noise,
        t=torch.linspace(0, 1, 2, device=DEVICE),
        method="dopri5",
        rtol=1e-4,
        atol=1e-4,
    )
    x_gen = traj[-1]  # [B, D, T]
    gen_audio = data_transform.latent_to_audio(x_gen).data.cpu()  # [B, C, L]
    # (Optional) GT decode if you want it:
    gt_latent = data_transform.vae.encode(batched_wav_16k.to(DEVICE))
    gt_audio = data_transform.latent_to_audio(gt_latent).data.cpu()  # [B, C, L]
    stem = Path(AUDIO_PATH).stem

import torchaudio
torchaudio.save(f"tmp.wav", gt_audio[0], sr)

del data_transform, model
import gc
gc.collect()
torch.cuda.empty_cache()


from train_classification import AudioClassifier
MODEL_PATH = "/home/jiahelei/github/audio_flow/checkpoints/train_classification/transcript2muic_10s/step=40000_ema.pt"
print(f"Loading model checkpoint from: {MODEL_PATH}")
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

classification_model = AudioClassifier(num_classes=12).to(DEVICE)
classification_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
classification_model.eval()
print("Model loaded successfully.")

print(f"total params of AudioClassifier = {sum(p.numel() for p in classification_model.parameters())/1024/1024} M")



# batched_wav_16k = batched_wav_16k.to(DEVICE)

resampler_48k_to_16k = torchaudio.transforms.Resample(48000, 16000).to(DEVICE)
batched_wav_16k = resampler_48k_to_16k(gen_audio.to(DEVICE))

b,c, l = batched_wav_16k.shape
seg_len = 16000 * 30

if l < seg_len:
    pad = torch.zeros((b, c, seg_len), dtype=batched_wav_16k.dtype, device=batched_wav_16k.device)
    pad[:, :, :l] = batched_wav_16k
    chunk_16k = pad
else:
    start = (l - seg_len) // 2
    chunk_16k = batched_wav_16k[:, :, start:start + seg_len]

cnt_correct = 0
with torch.no_grad():
    for idx in range(chunk_16k.shape[0]):
        true_target = all_idx[idx].item()
        chunk_16k_single = chunk_16k[idx:idx+1, :, :]

        logits = classification_model(chunk_16k_single)
        prediction = torch.argmax(logits, dim=1).item()

        if prediction == true_target:
            cnt_correct += 1
print(f"Accuracy: {cnt_correct}/{B} = {cnt_correct/B}")
