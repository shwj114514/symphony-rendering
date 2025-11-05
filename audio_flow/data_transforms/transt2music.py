import torch
import torch.nn as nn
import torchaudio
from audidata.datasets import GTZAN
from torch import Tensor
from einops import rearrange

from audio_flow.vae.levo import LevoVAE
from huggingface_hub import hf_hub_download


# no grad model
class Trans2MusicVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.vae = LevoVAE()
        self.sr = self.vae.sr
        trans_model_path = hf_hub_download(
            repo_id="shwj/symphony-rendering", 
            filename="model_traced_slakh.pt"
        )
        self.trans_model = torch.jit.load(trans_model_path, map_location="cpu")


    def audio_to_latent(self, data: dict) -> tuple[Tensor, dict]:
        r"""Transform data into latent representations and conditions.

        b: batch_size
        c: channels_num
        l: audio_samples
        t: frames_num
        f: mel bins
        """
        
        device = next(self.parameters()).device
        self.trans_model = self.trans_model.to(device)

        # Mel spectrogram target
        latent = data["latent"].to(device)  # (b, d, t)
        # latent = rearrange(latent, 'b d t -> b t d')

        ids = data["target"].to(device)  # (b,)
        captions = data["label"]  # (b,)
        wav = data["wav"]  # (b,)
        if len(wav.shape) == 3:
            wav = wav.mean(1,keepdim= True)
        wav = wav.to(device)

        # resampler = torchaudio.transforms.Resample(48000, 16000)
        # wav = resampler(wav)

        wav = torchaudio.functional.resample(wav, orig_freq=48000, new_freq=16000)
        
        # with torch.no_grad():
            # output_dict = self.trans_model(wav)             # [16, 1, 160000]
            # frame_roll = output_dict["frame_roll"]          # [16, 1001, 128]

        with torch.no_grad():
            chunk_samples = 16000 * 10  # 10 seconds at 16kHz
            total_samples = wav.shape[-1]

            # Check if the audio is longer than the model's 10-second training window
            if total_samples > chunk_samples:
                # --- Code completion starts here ---
                frame_roll_chunks = []
                # Split the waveform into 10-second chunks along the length dimension
                chunks = torch.split(wav, chunk_samples, dim=-1)
                
                for chunk in chunks:
                    current_len = chunk.shape[-1]
                    # The last chunk might be shorter, so we pad it to 10 seconds
                    if current_len < chunk_samples:
                        pad_len = chunk_samples - current_len
                        chunk = torch.nn.functional.pad(chunk, (0, pad_len))
                    
                    # Process the 10-second chunk
                    # Assuming model output is a dict; for dummy model, we create one
                    # output_dict = self.trans_model(chunk)
                    # import pdb;pdb.set_trace()
                    output_dict = self.trans_model(chunk)

                    frame_roll_chunks.append(output_dict["frame_roll"])
                
                # Concatenate the results from all chunks along the time dimension (dim=1)
                frame_roll = torch.cat(frame_roll_chunks, dim=1)
                # --- Code completion ends here ---
            else:
                # Audio is 10s or shorter, so pad it to 10s and process
                pad_len = chunk_samples - total_samples
                padded_wav = torch.nn.functional.pad(wav, (0, pad_len))
                
                # Process the padded 10-second audio
                # output_dict = self.trans_model(padded_wav)
                output_dict = self.trans_model(padded_wav)

                frame_roll = output_dict["frame_roll"]


        # Condition
        cond_dict = {
            "id": ids,
            "caption": captions,
            "frame_roll":frame_roll
        }

        return latent, cond_dict

    def latent_to_audio(self, x: Tensor) -> Tensor:
        r"""Ues vocoder to convert mel spectrogram to audio.

        Args:
            x: (b, c, t, f)

        Outputs:
            y: (b, c, l)
        """
        x = self.vae.decode(x)
        return x

    def __call__(self, data: dict) -> tuple[Tensor, dict]:
        return self.audio_to_latent(data)