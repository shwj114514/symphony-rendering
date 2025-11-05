import json
import torch.nn as nn
import torch
from torch import Tensor
from huggingface_hub import hf_hub_download
import sys

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.autoencoders import AudioAutoencoder


class LevoVAE(nn.Module):

    def __init__(self):
        super().__init__()

        config_path = hf_hub_download(
            repo_id="tencent/SongGeneration", 
            filename="ckpt/vae/stable_audio_1920_vae.json"
        )

        model_path = hf_hub_download(
            repo_id="tencent/SongGeneration", 
            filename="ckpt/vae/autoencoder_music_1320k.ckpt"
        )
        
        with open(config_path, "r") as f:
            model_config = json.load(f)

        self.vae = create_model_from_config(model_config)
        state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
        self.vae.load_state_dict(state_dict)

        self.sr = model_config["sample_rate"]
        self.fps = 25
        
    def encode(self, audio: Tensor) -> Tensor:
        r"""

        Args:
            audio: (b, 2, l)
        """

        with torch.no_grad():
            self.vae.eval()
            latent = self.vae.encode_audio(audio)

        return latent

    def decode(self, latent):
        with torch.no_grad():
            self.vae.eval()
            audio = self.vae.decode_audio(latent)

        return audio

    def __call__(self, audio: Tensor) -> Tensor:
        return self.encode(audio)
    
if __name__ == "__main__":
    vae = LevoVAE()
    