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
from audio_flow.utils import LinearWarmUp, parse_yaml,requires_grad, update_ema, logmel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
import typing as tp

BATCH_SIZE = 4

def calculate_model_params(model:nn.Module):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print( "Total params: {}M, trainable params: {}M, size of saving: {}M".format(
        pytorch_total_params/1024/1024,
        pytorch_train_params/1024/1024,
        pytorch_total_params*4/1024/1024)
    )
    return

class AudioClassifier(nn.Module):

    def __init__(self, num_classes: int, ssl_model_name: str ="ZhenYe234/hubert_base_general_audio",head_hidden = 1024):

        super().__init__()
        
        from transformers import AutoModel,AutoProcessor,Wav2Vec2FeatureExtractor
        self.hubert = AutoModel.from_pretrained(ssl_model_name)
        
        for param in self.hubert.parameters():
            param.requires_grad = False
            
        ssL_hidden_size = self.hubert.config.hidden_size
        
        # self.head = nn.Linear(ssL_hidden_size, num_classes)

        self.head = nn.Sequential(
            nn.LayerNorm(ssL_hidden_size),
            nn.Linear(ssL_hidden_size, head_hidden),
            nn.GELU(),
            nn.LayerNorm(head_hidden),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.shape[1] > 1:
            wav = torch.mean(wav, dim=1)  # -> [batch, samples]

        outputs = self.hubert(wav)
        hidden_states = outputs.last_hidden_state
        
        pooled_output = torch.mean(hidden_states, dim=1)
        logits = self.head(pooled_output)
        
        return logits

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
    train_dataset = get_dataset(configs, split="train")
    print("train_dataset[0]",train_dataset[0])
    print(f"len train_dataset = {len(train_dataset)}")

    # Sampler
    train_sampler = get_sampler(configs, train_dataset)

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler,
        num_workers=configs["train"]["num_workers"], 
        pin_memory=True,
    )

    model = AudioClassifier(num_classes= 12).to(device)
    calculate_model_params(model)
    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt, strict=True)


    optimizer, scheduler = get_optimizer_and_scheduler(
        configs=configs, 
        params=model.parameters()
    )
    
    # Logger
    if wandb_log:
        wandb.init(project="audio_flow", name=f"{filename}_{config_name}")
    criterion = nn.CrossEntropyLoss()

    for step, data in enumerate(tqdm(train_dataloader)):
        '''
            data["target"] tensor([10,  7,  8,  8,  7,  5,  7,  0])

            # 双声道音频
            data["wav"] [8, 2, 480000])
        '''

    
        # 2.2 Loss
        wav = data["wav"].to(device)
        labels = data["target"].to(device)
        
        logits = model(wav)

        # 4.2 计算损失
        loss = criterion(logits, labels)
        # 2.3 Optimize
        optimizer.zero_grad()  # Reset all parameter.grad to 0
        loss.backward()  # Update all parameter.grad
        optimizer.step()  # Update all parameters based on all parameter.grad

        # 2.4 Learning rate scheduler
        if scheduler:
            scheduler.step()

        if step % 20 == 0:
            print("train loss: {:.4f}".format(loss.item()))

        # ------ 3. Evaluation ------
        # 3.1 Evaluate
        if step % 500 == 0:

            for split in ["train", "test"]:
                validate(
                    configs=configs,
                    model=model,
                    split=split,
                )

            if wandb_log:
                wandb.log(
                    data={
                        "train_loss": loss.item()
                    },
                    step=step
                )
        
        # 3.2 Save model
        if step % 5000 == 0:
           
            ckpt_path = Path(ckpts_dir, f"step={step}_ema.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Save model to {ckpt_path}")

        if step == configs["train"]["training_steps"]:
            break

        step += 1
        

def get_dataset(
    configs: dict, 
    split: Literal["train", "test"]
) -> Dataset:
    r"""Get datasets."""

    ds = f"{split}_datasets"

    from audio_flow.datasets.label_wav import LabelWavDatset
    dataset =  LabelWavDatset(
        root = "./datasets/youtube2",
        split=split,
        duration=configs["clip_duration"],
        data_sr = 16000
    )

    print(f"dataset[0] = {dataset[0]}")
    return dataset


def get_sampler(configs: dict, dataset: Dataset) -> Iterable:
    r"""Get sampler."""

    name = configs["sampler"]

    if name == "RepeatShuffleSampler":
        from audio_flow.samplers.sampler import RepeatShuffleSampler
        return RepeatShuffleSampler(dataset)

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
    model: nn.Module,
    split: Literal["train", "test"]
) -> float:
    r"""Validate the model on part of data."""
    print(f"--- Running validation on '{split}' split ---")

    device = next(model.parameters()).device

    valid_audios = configs["valid_audios"]

    dataset = get_dataset(configs, split=split)
    print(f"valid dataset = {len(dataset )}")


    valid_dataset = get_dataset(configs, split=split)
    dataloader = DataLoader(valid_dataset, batch_size=4)
    

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0


    with torch.no_grad(): # 在验证阶段不计算梯度
        for data in tqdm(dataloader, desc=f"Validating on {split}"):
            wav = data["wav"].to(device)
            labels = data["target"].to(device)
            
            logits = model(wav)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(logits, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    
    print(f"Validation Results for '{split}':")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")
    
    model.train() # 恢复到训练模式
    return accuracy

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)