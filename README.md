## Learning Without Augmenting: Unsupervised Time Series Representation Learning via Frame Projections

(NeurIPS 2025, Official Code)

**Berken Utku Demirel** and [**Christian Holz**](https://www.christianholz.net)<br/>
[Sensing, Interaction & Perception Lab](https://siplab.org), ETH Zürich, Switzerland

---

> This work introduces an unsupervised representation learning method for time series that replaces handcrafted augmentations with projections into orthonormal and overcomplete frames (Fourier & Gabor). We perform instance discrimination across domains and use lightweight latent-space mappers to benefit from complementary geometries. Without increasing data diversity, our method achieves strong improvements across heart rate estimation, activity recognition, cardiovascular disease classification, step counting, and sleep staging.

---

### TL;DR

* No augmentations
* Fourier + Gabor views
* Instance discrimination across domains
* Latent mappers at inference (no extra encoders)
* Outperforms prior SSL methods on 9 datasets / 5 tasks

---

### Contents

* [Datasets](#datasets)
* [Environment](#environment)
* [Setup & Training](#setup--training)
* [Implementation](#implementation)

---

### Datasets

We evaluate across nine public datasets and five tasks:

* **Heart rate estimation:** IEEE SPC12, IEEE SPC22, DaLiA
* **Activity recognition:** HHAR, USC
* **Cardiovascular disease classification:** CPSC2018, Chapman
* **Step counting:** Clemson
* **Sleep staging:** Sleep-EDF

You can prepare the data using scripts from our previous works:
[https://github.com/eth-siplab/Unsupervised_Periodicity_Detection/tree/main/Heuristic_and_data_prep](https://github.com/eth-siplab/Unsupervised_Periodicity_Detection/tree/main/Heuristic_and_data_prep)
[https://github.com/eth-siplab/Finding_Order_in_Chaos](https://github.com/eth-siplab/Finding_Order_in_Chaos)

Or download the **preprocessed datasets** directly from Hugging Face:
[https://huggingface.co/datasets/eth-siplab/Learning-with-FrameProjections](https://huggingface.co/datasets/eth-siplab/Learning-with-FrameProjections)

---

### Environment

**Option A — Conda**

```bash
conda env create -f environment.yml
conda activate isoalign

# Install PyTorch (choose according to your setup)
# CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# or CPU
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Option B — Pip**

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements_minimal.txt
```

---

### Quickstart

**Pre-training + testing (our method)**

```bash
python main.py \
  --framework isoalign \
  --backbone resnet \
  --dataset ieee_small \
  --n_epoch 256 \
  --batch_size 1024 \
  --lr 1e-3 \
  --lr_cls 0.03 \
  --cuda 0 \
  --cases subject_large
```

**Supervised baseline**

```bash
python main_supervised_baseline.py \
  --dataset ieee_small \
  --backbone resnet \
  --block 8 \
  --lr 5e-4 \
  --n_epoch 999 \
  --cuda 0
```

---

### Implementation

* **Our method + ablations:** `frameworks.py` (lines **301–469**)
* **Losses (NT-Xent, CLIP, etc.):** `loss.py`

**Notes**

* Adjust `--dataset`, `--n_epoch`, and `--batch_size` based on GPU memory
* `--framework` supports baselines (e.g., `simclr`) and our method (`isoalign`)

