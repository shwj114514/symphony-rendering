r"""Code modified from: https://github.com/AudioFans/audidata/blob/main/audidata/datasets/gtzan.py"""
from __future__ import annotations

import os
import re
from pathlib import Path

import h5py
import random
import pickle
import librosa
import numpy as np
from audidata.io.audio import load
from audidata.io.crops import StartCrop
from audidata.transforms.audio import Mono
from audidata.transforms.onehot import OneHot
from audidata.utils import call
from torch.utils.data import Dataset
from typing_extensions import Literal

import torch
import typing as tp

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


class SymphonyJsonlVAE(Dataset):
    r"""
        vae_latent wav_np
    """


    def __init__(
        self, 
        root: str = None, 
        split: Literal["train", "test"] = "train",
        test_fold: int = 0,  # E.g., fold 0 is used for testing. Fold 1 - 9 are used for training.
        duration: float = 10.
    ) -> None:
    
        self.root = root
        self.split = split
        self.test_fold = test_fold
        self.duration = duration
        vae_latent_folder = os.path.join(root,"vae_latent")
        LABELS = sorted(os.listdir(vae_latent_folder))

        CLASSES_NUM = len(LABELS)
        LB_TO_IX = {lb: ix for ix, lb in enumerate(LABELS)}
        IX_TO_LB = {ix: lb for ix, lb in enumerate(LABELS)}


        
        self.labels = LABELS
        self.lb_to_ix = LB_TO_IX
        self.ix_to_lb = IX_TO_LB

        self.meta_dict = self.load_meta()
        # vae_list = get_filelist(vae_latent_folder,[".h5"])

    def __getitem__(self, index: int) -> dict:
        try:
            vae_latent_path = str(self.meta_dict["path"][index])

            label = self.meta_dict["label"][index] 

            full_data = {
                "dataset_name": "GtzanVAE",
                "path": vae_latent_path,
            }

            # Load audio data
            # latent_data = self.load_latent_data(path=path)
            # full_data.update(latent_data)

            with h5py.File(vae_latent_path, 'r') as hf:
                latent = hf["latent"][:]
                fps = hf.attrs["fps"]

            clip_frames = int(self.duration * fps)
            
            # 有些音频超过90
            bgn_frame = random.randint(0, latent.shape[-1] - clip_frames)


            bgn_frame = max(0, bgn_frame)    
            # (64, 250)
            latent = latent[:, bgn_frame : bgn_frame + clip_frames]  # (d, t)

            wav_stem = Path(vae_latent_path).stem
            if wav_stem.endswith("_vae"):
                wav_stem = wav_stem[:-4]
            wav_np_path = os.path.join(self.root,"wav_np",label,wav_stem+".npy")
            large_waveform_mmap = np.load(wav_np_path, mmap_mode='r')

            wav_start_frame = 1920 * bgn_frame
            wav_end_frame = 1920 * (bgn_frame + clip_frames)

            chunk_np = large_waveform_mmap[:, wav_start_frame:wav_end_frame]

            chunk_torch = torch.from_numpy(chunk_np.copy())

            target = self.lb_to_ix[label]

            full_data = {
                "label": label,
                "target": target , # shape: (classes_num,)
                "wav": chunk_torch,
                "latent": latent,
                "fps": fps
            }


            return full_data
        except Exception as e:
            print(f"e = {e} in getitem index = {index} vae_latent_path = {vae_latent_path}",flush=True)
            return self.__getitem__(random.randrange(self.__len__()))

    def __len__(self) -> int:
        return len(self.meta_dict["name"])

    def load_meta(self) -> dict:
        r"""Load metadata of the GTZAN dataset.
        """

        meta_dict = {
            "name": [],
            "path": [],
            "label": [],
        }

        out_dir = Path(self.root, "vae_latent")

        for genre in self.labels:
            # 'datasets/youtube2/vae_latent' + '勃拉姆斯（Johannes Brahms）'
            names = sorted(os.listdir(Path(out_dir, genre)))
            # train_names 3  test_names 10
            train_names, test_names = self.split_train_test(names)

            if self.split == "train":
                filtered_names:tp.List= train_names

            elif self.split == "test":
                filtered_names = test_names
            else:
                raise ValueError(self.split)

            for name in filtered_names:
                path = str(Path(out_dir, genre, name))
                meta_dict["name"].append(name)
                meta_dict["path"].append(path)
                meta_dict["label"].append(genre)
        return meta_dict

    # def split_train_test(self, names: list) -> tuple[list, list]:

    #     train_names = []
    #     test_names = []

    #     test_ids = range(self.test_fold * 10, (self.test_fold + 1) * 10)
    #     # E.g., if test_fold = 3, then test_ids = [30, 31, 32, ..., 39]

    #     for name in names:

    #         audio_id = int(re.search(r'\d+', name).group())
    #         # E.g., if name is "blues.00037.h5", then audio_id = 37

    #         if audio_id in test_ids:
    #             test_names.append(name)

    #         else:
    #             train_names.append(name)

    #     return train_names, test_names

    def split_train_test(self, names: list) -> tuple[list, list]:

        train_names = []
        test_names = []

        # E.g., if test_fold = 3, then test_ids = [30, 31, 32, ..., 39]
        test_names = random.choices(names,k = 1)
        train_names = names
        return train_names, test_names


    def load_latent_data(self, path: str) -> dict:

        with h5py.File(path, 'r') as hf:
            latent = hf["latent"][:]
            fps = hf.attrs["fps"]

        clip_frames = int(self.duration * fps)
        bgn_frame = random.randint(0, latent.shape[-1] - clip_frames)
        bgn_frame = max(0, bgn_frame)    
        latent = latent[:, bgn_frame : bgn_frame + clip_frames]  # (d, t)

        data = {
            "latent": latent,
            "fps": fps
        }

        return data



    def load_target_data(self, label: str) -> dict:

        target = self.lb_to_ix[label]

        data = {
            "label": label,
            "target": target  # shape: (classes_num,)
        }

        return data