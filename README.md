# symphony-rendering
## prepare
### env
Python ≥ 3.10, PyTorch ≥ 2.1 (CUDA 11.8 or 12.x recommended)

Recent versions of the official stable_audio_tools may cause:
```sh
ModuleNotFoundError: No module named 'torch.nn.attention.flex_attention'
```
To avoid this, install the friendly fork: 
```
pip install "git+https://github.com/Stability-AI/stable-audio-tools.git@84315cc06f91caad218a0209445a0470f277cc17"
```

### get jsonl
Organize raw audio files by composer/author (any folder names are fine). Example:
```
raw_audio/
├── author1/
│   ├── aaa.wav
│   └── bbb.mp3
├── author2/
│   ├── ccc.wav
│   └── ddd.mp4
```
Generate the dataset manifest:
```sh 
python get_filelist.py
```
This will generate: `configs/data/youtube.jsonl`
### extract VAE latents
Run the following script to extract VAE latents:
```sh
bash extract_vaelatents_symphony.sh
```
The resulting latents will be stored in the specified output directory.
## train
Start training with the 30-second configuration:
```sh
bash train_30s
```
## infer
```sh
bash infer.sh
```
## Acknowledgements
This codebase is largely adapted from [audio_flow](https://github.com/qiuqiangkong/audio_flow); most source files are modified from that project.

We also **use the pretrained VAE from [Levo / SongGeneration](https://github.com/tencent-ailab/songgeneration)**.

