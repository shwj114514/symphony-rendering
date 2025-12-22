from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import soundfile
import torch
import torch.nn as nn
import torch.optim as optim
import torchdiffeq
import wandb
from audio_flow.utils import CombinedModel, LinearWarmUp, parse_yaml,requires_grad, update_ema, logmel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from tqdm import tqdm


def train(args) -> None:
    r"""Train audio generation with flow matching."""

    # Arguments
    wandb_log = not args.no_log
    config_path = args.config
    filename = Path(__file__).stem
    
    # Configs
    configs = parse_yaml(config_path)
    device = configs["train"]["device"]
    ckpt_path = configs["train"]["resume_ckpt_path"]

    # Checkpoints directory
    config_name = Path(config_path).stem
    ckpts_dir = Path("./checkpoints", filename, config_name)
    Path(ckpts_dir).mkdir(parents=True, exist_ok=True)

    # Datasets
    train_dataset = get_dataset(configs, split="train", mode="train")
    print("train_dataset[0]",train_dataset[0])

    # Sampler
    train_sampler = get_sampler(configs, train_dataset)

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=configs["train"]["batch_size_per_device"], 
        sampler=train_sampler,
        num_workers=configs["train"]["num_workers"], 
        pin_memory=True,
    )

    # Data processor
    data_transform = get_data_transform(configs).to(device)

    # Flow matching data processor
    fm = ConditionalFlowMatcher(sigma=0.)

    # Model
    base = get_base(
        configs=configs, 
    ).to(device)

    adaptor = get_adaptor(
        configs=configs,
    ).to(device)

    model = CombinedModel(base, adaptor)

    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt, strict=True)

    # EMA (optional)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    ema.eval()  # EMA model should always be in eval mode

    # Optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(
        configs=configs, 
        params=model.parameters()
    )
    
    # Logger
    if wandb_log:
        wandb.init(project="audio_flow", name=f"{filename}_{config_name}")

    for step, data in enumerate(tqdm(train_dataloader)):

        # ------ 1. Data preparation ------
        # 1.1 Transform data into latent representations and conditions
        # [16, 64, 250]
        '''
            [8, 64, 750]
            data['label']   
                ['舒伯特（Franz Schubert）', '莫扎特（Wolfgang Amadeus Mozart）', '巴赫（Johann Sebastian Bach）', '拉赫玛尼诺夫（Sergei Rachmaninoff）', '巴赫（Johann Sebastian Bach）', '马勒（Gustav Mahler）', '马勒（Gustav Mahler）', '德沃夏克（Antonín Dvořák）']
            data['wav'].shape
                [8, 2, 1440000])
        '''
        x_real, cond_dict = data_transform(data)
        # 1.2 Noise
        noise = torch.randn_like(x_real)

        # 1.3 Get input and velocity
        t, xt, ut = fm.sample_location_and_conditional_flow(x0=noise, x1=x_real)

        # ------ 2. Training ------
        # 2.1 Forward
        model.train()
        emb_dict = model.adaptor(cond_dict)
        '''
            emb_dict.keys() ['ct', 'c'])

            noise [16, 64, 250]
            vt [16, 64, 250]
            cond_dict
            {'id': tensor([3, 8, 3, 0, 3, 0, 3, 8, 1, 2, 0, 3, 0, 1, 3, 1], device='cuda:0'), 'caption': ['disco', 'reggae', 'disco', 'blues', 'disco', 'blues', 'disco', 'reggae', 'classical', 'country', 'blues', 'disco', 'blues', 'classical', 'disco', 'classical']}

            emb_dict["c"] [16, 768]
        '''


        vt = model.base(t=t, x=xt, emb_dict=emb_dict)

        # 2.2 Loss
        loss = torch.mean((vt - ut) ** 2)

        # 2.3 Optimize
        optimizer.zero_grad()  # Reset all parameter.grad to 0
        loss.backward()  # Update all parameter.grad
        optimizer.step()  # Update all parameters based on all parameter.grad
        update_ema(ema, model, decay=0.999)

        # 2.4 Learning rate scheduler
        if scheduler:
            scheduler.step()

        if step % 100 == 0:
            print("train loss: {:.4f}".format(loss.item()))

        # ------ 3. Evaluation ------
        # 3.1 Evaluate
        if step % configs["train"]["test_every_n_steps"] == 0:

            for split in ["train", "test"]:
                validate(
                    configs=configs,
                    data_transform=data_transform,
                    model=ema,
                    split=split,
                    out_dir=Path("./results", filename, config_name, f"steps={step}_ema")
                )

            if wandb_log:
                wandb.log(
                    data={
                        "train_loss": loss.item()
                    },
                    step=step
                )
        
        # 3.2 Save model
        if step % configs["train"]["save_every_n_steps"] == 0:
           
            ckpt_path = Path(ckpts_dir, f"step={step}_ema.pt")
            torch.save(ema.state_dict(), ckpt_path)
            print(f"Save model to {ckpt_path}")

        if step == configs["train"]["training_steps"]:
            break

        step += 1
        

def get_dataset(
    configs: dict, 
    split: Literal["train", "test"],
    mode: Literal["train", "test"]
) -> Dataset:
    r"""Get datasets."""

    ds = f"{split}_datasets"

    for name in configs[ds].keys():

        if name == "GtzanVAE":
            from audio_flow.datasets.gtzan_vae import GtzanVAE
            return GtzanVAE(
                root=configs[ds][name]["root"],
                split=configs[ds][name]["split"],
                test_fold=0,
                duration=configs["clip_duration"]
            )

        elif name == "MUSDB18HqVAE":
            from audio_flow.datasets.musdb18hq_vae import MUSDB18HqVAE
            return MUSDB18HqVAE(
                root=configs[ds][name]["root"],
                split=configs[ds][name]["split"],
                duration=configs["clip_duration"]
            )
            
        elif name == "MUSDB18HqMono2StereoVAE":
            from audio_flow.datasets.musdb18hq_mono2stereo_vae import MUSDB18HqMono2StereoVAE
            return MUSDB18HqMono2StereoVAE(
                root=configs[ds][name]["root"],
                split=configs[ds][name]["split"],
                duration=configs["clip_duration"]
            )
            
        elif name == "MUSDB18HqLowres2HighresVAE":
            from audio_flow.datasets.musdb18hq_lowres2highres_vae import MUSDB18HqLowres2HighresVAE
            return MUSDB18HqLowres2HighresVAE(
                root=configs[ds][name]["root"],
                split=configs[ds][name]["split"],
                duration=configs["clip_duration"]
            )

        elif name == "MUSDB18HqDac2StereoVAE":
            from audio_flow.datasets.musdb18hq_dac2stereo_vae import MUSDB18HqDac2StereoVAE
            return MUSDB18HqDac2StereoVAE(
                root=configs[ds][name]["root"],
                split=configs[ds][name]["split"],
                duration=configs["clip_duration"]
            )

        elif name == "LJSpeechVAE":
            from audio_flow.datasets.ljspeech_vae import LJSpeechVAE
            return LJSpeechVAE(
                root=configs[ds][name]["root"],
                split=configs[ds][name]["split"],
                duration=configs["clip_duration"]
            )
        elif name == "SymphonyVAE":
            from audio_flow.datasets.symphony_jsonl_vae import SymphonyJsonlVAE
            return SymphonyJsonlVAE(
                root=configs[ds][name]["root"],
                split=configs[ds][name]["split"],
                duration=configs["clip_duration"]
            )

        else:
            raise ValueError(name)


def get_sampler(configs: dict, dataset: Dataset) -> Iterable:
    r"""Get sampler."""

    name = configs["sampler"]

    if name == "RepeatShuffleSampler":
        from audio_flow.samplers.sampler import RepeatShuffleSampler
        return RepeatShuffleSampler(dataset)

    else:
        raise ValueError(name)


def get_data_transform(configs: dict):
    r"""Transform data into latent representations and conditions."""

    name = configs["data_transform"]["name"]

    if name == "Label2MusicVAE":
        from audio_flow.data_transforms.label2music import Label2MusicVAE
        return Label2MusicVAE()

    elif name == "MSS":
        from audio_flow.data_transforms.mss import MSSVAE
        return MSSVAE(target_stem=configs["data_transform"]["target_stem"])

    elif name == "Mono2StereoVAE":
        from audio_flow.data_transforms.mono2stereo import Mono2StereoVAE
        return Mono2StereoVAE()

    elif name == "SuperResolutionVAE":
        from audio_flow.data_transforms.mono2stereo import SuperResolutionVAE
        return SuperResolutionVAE()

    elif name == "Dac2StereoVAE":
        from audio_flow.data_transforms.dac2stereo import Dac2StereoVAE
        return Dac2StereoVAE()

    elif name == "Text2SpeechVAE":
        from audio_flow.data_transforms.text2speech import Text2SpeechVAE
        return Text2SpeechVAE()
    elif name == "Trans2MusicVAE":
        from audio_flow.data_transforms.transt2music import Trans2MusicVAE
        return Trans2MusicVAE()
    else:
        raise ValueError(name)


def get_base(
    configs: dict, 
) -> nn.Module:
    r"""Initialize base model."""

    name = configs["base"]["name"]

    if name == "Transformer1D":
        from audio_flow.models.transformer1d import Transformer1D
        return Transformer1D(**configs["base"])

    else:
        raise ValueError(name)    


def get_adaptor(
    configs: dict, 
):
    r"""Initialize adaptor."""

    name = configs["adaptor"]["name"]
    if name == "OnehotEncoder":
        from audio_flow.adaptors.onehot import OnehotEncoder
        return OnehotEncoder(
            num_classes=configs["adaptor"]["num_classes"], 
            dim=configs["base"]["dim"]
        )

    elif name == "VAEEncoder":
        from audio_flow.adaptors.vae import VAEEncoder
        return VAEEncoder(
            in_channels=configs["adaptor"]["dim"], 
            dim=configs["base"]["dim"]
        )

    elif name == "LatentEncoder":
        from audio_flow.adaptors.latent import LatentEncoder
        return LatentEncoder(
            in_channels=configs["adaptor"]["dim"], 
            dim=configs["base"]["dim"]
        )

    elif name == "OnehotTransEncoder":
        from audio_flow.adaptors.label_online_trans import OnehotTransEncoder
        return OnehotTransEncoder(
            num_classes=configs["adaptor"]["num_classes"], 
            in_channels=configs["adaptor"]["in_channels"], 
            cond_length = int(configs["clip_duration"]*25),

            dim=configs["base"]["dim"]
        )
    elif name == "TransEncoder":
        from audio_flow.adaptors.label_online_trans import TransEncoder

        return TransEncoder(
            in_channels=configs["adaptor"]["in_channels"],
            dim=configs["base"]["dim"],
            cond_length = int(configs["clip_duration"]*25),
        )
    else:
        raise ValueError(name)    


def get_optimizer_and_scheduler(
    configs: dict, 
    params: list[torch.Tensor]
) -> tuple[optim.Optimizer, None | optim.lr_scheduler.LambdaLR]:
    r"""Get optimizer and scheduler."""

    lr = float(configs["train"]["lr"])
    warm_up_steps = configs["train"]["warm_up_steps"]
    optimizer_name = configs["train"]["optimizer"]

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(params=params, lr=lr)

    if warm_up_steps:
        lr_lambda = LinearWarmUp(warm_up_steps)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    return optimizer, scheduler


def validate(
    configs: dict,
    data_transform: object,
    model: nn.Module,
    split: Literal["train", "test"],
    out_dir: str
) -> float:
    r"""Validate the model on part of data."""

    device = next(model.parameters()).device
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_audios = configs["valid_audios"]
    sr = data_transform.sr

    dataset = get_dataset(configs, split=split, mode="test")
    print(f"valid dataset = {len(dataset )}")
    dataset[0]

    # Evaluate only part of data
    if valid_audios:
        skip_n = max(1, len(dataset) // valid_audios)
    else:
        skip_n = 1

    for i, idx in enumerate(range(0, len(dataset), skip_n)):
        # ------ 1. Data preparation ------
        # 1.1 Get Data
        data = dataset[idx]
        data = default_collate([data])
        
        # 1.2 Transform data into latent representations and conditions
        x_real, cond_dict = data_transform(data)

        # 1.3 Noise
        noise = torch.randn_like(x_real)

        # ------ 2. Forward with ODE ------
        # 2.1 Iteratively forward
        with torch.no_grad():
            model.eval()
            emb_dict = model.adaptor(cond_dict)
            traj = torchdiffeq.odeint(
                lambda t, x: model.base(t, x, emb_dict),
                y0=noise,
                t=torch.linspace(0, 1, 2, device=device),
                atol=1e-4,
                rtol=1e-4,
                method="dopri5",
            )

        x_gen = traj[-1]  # (b, d, t)

        # 2.2 Latent to audio
        if "ct" in cond_dict and cond_dict["ct"].shape == x_real.shape:
            in_audio = data_transform.latent_to_audio(cond_dict["ct"]).data.cpu().numpy()[0]  # (c, l)
        else:
            in_audio = None

        gen_audio = data_transform.latent_to_audio(x_gen).data.cpu().numpy()[0]  # (c, l)
        gt_audio = data_transform.latent_to_audio(x_real).data.cpu().numpy()[0]  # (c, l)

        # ------ 3. Plot and Visualization ------
        if in_audio is not None:
            in_logmel = logmel(in_audio, sr)
        gen_logmel = logmel(gen_audio, sr)
        gt_logmel = logmel(gt_audio, sr)

        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        vmin, vmax = -10, 5
        if in_audio is not None:
            axs[0].matshow(in_logmel.T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
        axs[1].matshow(gen_logmel.T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
        axs[2].matshow(gt_logmel.T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
        axs[0].set_title("Input (if there are)")
        axs[1].set_title("Generation")
        axs[2].set_title("Ground truth")
        axs[2].xaxis.tick_bottom()

        if "caption" in cond_dict:
            caption = "_{}".format(cond_dict["caption"][0])
        else:
            caption = ""

        out_path = Path(out_dir, f"{split}_{i}{caption}.png")
        plt.savefig(out_path)
        print(f"Write out to {out_path}")

        # 3.2 Save audio
        if in_audio is not None:
            out_path = Path(out_dir, f"{split}_{i}{caption}_in.wav")
            soundfile.write(file=out_path, data=in_audio.T, samplerate=sr)
            print(f"Write out to {out_path}")

        out_path = Path(out_dir, f"{split}_{i}{caption}_gen.wav")
        soundfile.write(file=out_path, data=gen_audio.T, samplerate=sr)
        print(f"Write out to {out_path}")

        out_path = Path(out_dir, f"{split}_{i}{caption}_gt.wav")
        soundfile.write(file=out_path, data=gt_audio.T, samplerate=sr)
        print(f"Write out to {out_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)