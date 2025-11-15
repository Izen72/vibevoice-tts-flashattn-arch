# VibeVoice TTS – FlashAttention2 + CUDA Setup (Arch / Linux)

This repo is a **working VibeVoice TTS setup** using:

- Python **3.12**
- PyTorch **2.5.1 + cu121**
- NVIDIA CUDA toolkit (`nvcc` available, `CUDA_HOME=/opt/cuda`)
- **FlashAttention2** for faster, lower-memory attention
- A clean `uv`-managed virtualenv

Tested on:

- **OS:** Arch Linux  
- **GPU:** RTX 3070 Laptop, 8 GB VRAM  
- **RAM:** 64 GB  

If you follow this README, you should end up with:

- VibeVoice 1.5B running on CUDA
- `attn_implementation: flash_attention_2`
- A reproducible `.venv` you can nuke/rebuild whenever

---

## 1. Repo layout

This repo is structured like this:

```text
.
├── VibeVoice/                  # Upstream VibeVoice repo (checked into this tree)
├── requirements.txt            # Curated deps for Python 3.12
├── requirements_p_3.12.11.txt  # Frozen version of the requirements.txt
├── command                     # Author's cheat sheet / notes
├── stack_test.py               # (Optional) Test before running*
└── .venv/                      # Created by uv (ignored in git)

*or after; I ain't telling you what to do.
````

VibeVoice itself lives in `./VibeVoice`.
Everything else in this README happens from the **repo root**.

### 📌 requirements_p_3.12.11.txt

This file is an **environment snapshot** generated via "pip freeze > requirements_p_3.12.11.txt".

It is **not meant for installation**, but serves as a reference if packages upgrade in the future and break the build.

If you hit dependency issues, diffing this file against `requirements.txt` can help identify what changed.

---

## 2. Prerequisites

### 2.1 NVIDIA drivers + CUDA toolkit

You need:

* Working NVIDIA drivers (e.g. `nvidia-smi` works)
* CUDA **toolkit** installed (not just runtime), with `nvcc` available

Check:

```bash
nvcc --version
```

Example output (your numbers may differ, that’s fine):

```text
Cuda compilation tools, release 13.0, V13.0.xx
```

Set `CUDA_HOME` if it isn’t already:

```bash
# bash/zsh
export CUDA_HOME=/opt/cuda

# fish
set -x CUDA_HOME /opt/cuda
```

You can add that to your shell config (`~/.bashrc`, `~/.config/fish/config.fish`) so it’s always set.

---

### 2.2 `uv` and `~/.local/bin` on PATH

This project uses [`uv`](https://github.com/astral-sh/uv) to manage Python + venv + pip.

Install `uv` however you like (package, script, etc).
Make sure `~/.local/bin` is on your `PATH` so `uv` and other tools are usable:

```bash
# bash/zsh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# fish
echo 'set -x PATH $HOME/.local/bin $PATH' >> ~/.config/fish/config.fish
source ~/.config/fish/config.fish
```

You can also let uv patch your shell:

```bash
uv python update-shell
```

---
### 2.3 Example (note: Saba is a fish; a *certified* feesh)

```bash
# CUDA
set -x CUDA_HOME /opt/cuda
set -x PATH $CUDA_HOME/bin $PATH
set -x LD_LIBRARY_PATH $CUDA_HOME/lib64 $LD_LIBRARY_PATH

# Path
set -x PATH $HOME/.local/bin $PATH
```

---

## 3. Create a Python 3.12 virtualenv with uv

From the repo root:

```bash
cd /path/to/TTS  # this repo
```

### 3.1 Install Python 3.12 via uv (if not already present)

```bash
uv python install 3.12
```

### 3.2 Create the venv

```bash
uv venv --python 3.12 .venv
```

### 3.3 Activate the venv

```bash
# bash/zsh
source .venv/bin/activate

# fish
source .venv/bin/activate.fish
```

Confirm:

```bash
python -V
# Python 3.12.x

which python
# .../TTS/.venv/bin/python
```

---

## 4. Install Python dependencies

This repo ships with a **curated** `requirements.txt` already adjusted for:

* Python 3.12
* VibeVoice
* Gradio frontend
* Audio / ML stack

> ⚠️ **Don’t regenerate this file with `pip freeze > requirements.txt`**
> That would lock in environment-specific weirdness (like Python 3.13 backports) and break reproducibility.

From the repo root with the venv active:

```bash
uv pip install -r requirements.txt
```

If it succeeds, your base environment is ready (except for CUDA Torch + FlashAttention).

---

## 5. Install the CUDA PyTorch stack

Now we install **GPU-enabled** PyTorch, torchvision, and torchaudio for CUDA 12.1.

Still in the venv:

```bash
uv pip install \
  "torch==2.5.1+cu121" \
  "torchvision==0.20.1+cu121" \
  "torchaudio==2.5.1+cu121" \
  --index-url https://download.pytorch.org/whl/cu121
```

Verify by running stack_test.py:

```bash
python stack_test.py
```

You should see something like:

```text
torch: 2.5.1+cu121
cuda: 12.1
cuda available: True
device: NVIDIA GeForce RTX 3070 Laptop GPU
[and flash attention will error out - intended for now]
```

If `cuda_available` is `False`, fix your NVIDIA / CUDA install before continuing.

---

## 6. Install the local VibeVoice package

We now “install” the vendored VibeVoice repo into this venv as an editable package.

From repo root:

```bash
uv pip install -e ./VibeVoice
```

That’s the equivalent of `pip install -e .` but pointed at the inner `VibeVoice` folder.

---

## 7. Install FlashAttention2

Now that:

* Python 3.12 is set
* Torch 2.5.1+cu121 is installed
* CUDA toolkit is visible (`nvcc` works, `CUDA_HOME` set)

…we can install `flash-attn` and let Transformers use **FlashAttention2**.

```bash
mkdir -p ~/tmp   # avoids some tmpdir / cross-device rename issues
```

Then:

```bash
# bash/zsh
export TMPDIR="$HOME/tmp"

# fish
set -x TMPDIR $HOME/tmp
```

And finally:

```bash
CUDA_HOME=/opt/cuda TMPDIR=$TMPDIR \
  uv pip install flash-attn==2.8.3 --no-build-isolation
```

Now we test with stack_test_py again, and we should see:


```text
torch: 2.5.1+cu121
cuda: 12.1
cuda available: True
device: NVIDIA GeForce RTX 3070 Laptop GPU
flash_attn imported OK
```

---

## 8. Run the VibeVoice demo

From repo root (`TTS/`), with the venv active:

```bash
python VibeVoice/demo/gradio_demo.py --model_path microsoft/VibeVoice-1.5B
```

You should see logs similar to:

```text
APEX FusedRMSNorm not available, using native implementation
🎙️ Initializing VibeVoice Demo with Streaming Support...
Loading processor & model from microsoft/VibeVoice-1.5B
Using device: cuda, torch_dtype: torch.bfloat16, attn_implementation: flash_attention_2
...
Language model attention: flash_attention_2
...
🚀 Launching demo on port 7860
* Running on local URL:  http://127.0.0.1:7860
```

Key things to look for:

* `Using device: cuda`
* `attn_implementation: flash_attention_2`
* `Language model attention: flash_attention_2`
* No “flash_attn seems to be not installed” warnings
* No fallback message like `Falling back to attention implementation: sdpa`

Then open the Gradio UI in your browser:

```text
http://127.0.0.1:7860
```

Pick a voice, type some text, generate, and enjoy.

---

## 9. Common warnings (safe to ignore)

You may see:

```text
APEX FusedRMSNorm not available, using native implementation
The tokenizer class you load from this checkpoint is not the same type...
```

Both are fine:

* Apex fused RMSNorm is just an optional speed optimization.
* The tokenizer warning is from using Qwen’s tokenizer behind a VibeVoice wrapper; it still works.

As long as:

* The model loads
* Attention implementation is `flash_attention_2`
* The demo UI launches on port 7860

…you’re in business.

---

## 10. Troubleshooting notes

### 10.1 If FlashAttention2 fails to build

* Double-check:

  ```bash
  nvcc --version
  echo $CUDA_HOME
  ```

* Make sure you’re using:

  * Python 3.12 (not 3.13)
  * Torch 2.5.1+cu121 (as above)
  * CUDA toolkit installed (not just driver)

If things are really cursed, you can temporarily comment out `flash-attn` and force VibeVoice to use SDPA by setting `attn_implementation="sdpa"` in `VibeVoice/demo/gradio_demo.py`. But in this repo, FlashAttention2 is expected to work.

---

## 11. Dev notes

* Environment is intentionally built around **Python 3.12** because:

  * Python 3.13 removes some stdlib audio modules, forcing use of `*-lts` backports.
  * Many CUDA and audio libs still target 3.10–3.12 first.
* The `requirements.txt` is **hand-curated** to:

  * Keep key libraries pinned (transformers, diffusers, gradio, numpy, scipy, etc.).
  * Let non-critical utilities float to avoid “no such version” issues.
* `uv` is used instead of raw `pip` because:

  * It resolves dependencies more strictly.
  * It handles multiple Python versions cleanly.
  * It makes rebuilds painless.

---

If you follow this from top to bottom, you should end up exactly where this setup ended:
VibeVoice 1.5B, running on your GPU, with FlashAttention2, streaming, and just vibing. 🎧
