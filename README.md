# HAR-MultiModal-DL
### Deep Learning-Based Human Activity Recognition Using MMG and IMU Wearable Sensors

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Activity Classes & Transition Graph](#activity-classes--transition-graph)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models](#models)
- [Experiments](#experiments)
- [Training Strategies](#training-strategies)
- [Fusion Strategies](#fusion-strategies)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Project Overview

This project investigates the use of deep learning for **Human Activity Recognition (HAR)**
using two complementary wearable sensor modalities:

- **MMG** — Mechanomyography: captures mechanical muscle vibrations during contraction
- **IMU** — Inertial Measurement Unit: captures kinematic motion data (acceleration, angular velocity)

Data was collected from **10 volunteers** performing **7 activity classes**, including both
steady-state locomotion activities and transitional movements. The project benchmarks multiple
deep learning architectures across multiple modality conditions and training protocols, with
a particular focus on accurately predicting activity class at and around **transition points**
between activities.

### Key Research Questions
1. Which deep learning architecture best classifies steady-state and transitional activities?
2. Does MMG, IMU, or their fusion yield the highest classification performance?
3. Is early or late sensor fusion more effective for this task?
4. Do models generalize across subjects (LOSO) or are they subject-specific?
5. How accurately can models predict the post-transition class from pre-transition data?
6. Does CNN+LSTM or CNN+GRU better model MMG temporal dynamics?
7. Does a Vision Transformer or CNN+Transformer better capture activity representations?

---

## 🔄 Activity Classes & Transition Graph

The project models a **directed activity state graph** with 7 nodes representing
human locomotion states and transitions:
Sit ──► Sit-to-Stand ──► Stand ◄══► Walk │ ◄══► Stair Ascent Stand-to-Sit ◄───┘ ◄══► Stair Descent │ ▼ Sit


### Activity Classes

| ID | Class | Type |
|----|-------|------|
| 0 | Sit | Steady-state |
| 1 | Stand | Steady-state (hub) |
| 2 | Walk | Steady-state |
| 3 | Sit-to-Stand | Transitional |
| 4 | Stand-to-Sit | Transitional |
| 5 | Stair Ascent | Steady-state |
| 6 | Stair Descent | Steady-state |

> **Note:** `Stand` is the central hub state — all transitions pass through it.
> Walk, Stair Ascent, and Stair Descent can each transition bidirectionally with Stand.

---

## 📁 Dataset

### Sensor Modalities
| Modality | Description |
|----------|-------------|
| **MMG** | Mechanomyography — muscle mechanical vibration signals |
| **IMU** | Inertial Measurement Unit — acceleration and angular velocity |

### Data Format
Each `.npy` file contains data of one of two tensor shapes:
Shape A: (width, height, channels, samples) Shape B: (width, 1, channels, samples)

### File Naming Convention
Files follow one of two naming patterns:

**Standard files:** N0XX<MMG|IMU>_.npy

**Transition-point files:** N0XX<MMG|IMU><transition_descriptor>.npy

| Field | Description |
|-------|-------------|
| `N0XX` | Volunteer ID (e.g., N001 – N010) |
| `MMG\|IMU` | Sensor modality |
| `<class>` | One of the 7 activity class labels |
| `<transition_descriptor>` | One of up to 4 descriptors indicating position relative to a transition point |

> ⚠️ **Important Labeling Rule:** Files containing data captured *just before* a transition
> point are labeled as the **class after the transition** — not the class currently being
> performed. This enables predictive classification at transition boundaries.

### Metadata Tracking
A **Pandas DataFrame** is constructed at runtime to catalog all data files with the
following fields:

| Column | Description |
|--------|-------------|
| `file_path` | Absolute path to the `.npy` file |
| `volunteer_id` | Subject identifier (e.g., N001) |
| `modality` | MMG or IMU |
| `class_label` | Integer class ID (0–6) |
| `class_name` | Human-readable class name |
| `is_transition_file` | Boolean flag |
| `transition_descriptor` | Transition point descriptor string (if applicable) |
| `shape` | Tensor shape of the file |

---

## 🗂️ Project Structure
HAR_Project/ │ ├── data/ │ ├── raw/ │ │ ├── MMG/
│ │ └── IMU/
│ ├── processed/
│ │ ├── MMG/ │ │ ├── IMU/ │ │ └── fused/ │ └── splits/
│ ├── subject_dependent/ │ └── LOSO/ │ ├── configs/
│ ├── models/ │ ├── training/ │ └── experiments/ │ ├── mmg_only/ │ ├── imu_only/ │ └── fused/ │ ├── early/ │ └── late/ │ ├── src/ │ ├── data/
│ ├── models/
│ │ └── fusion/
│ ├── training/
│ ├── evaluation/
│ ├── experiments/
│ │ └── comparison/
│ └── utils/
│ ├── notebooks/
├── results/ │ ├── checkpoints/ │ ├── logs/ │ ├── metrics/ │ └── figures/ │ ├── tests/
├── scripts/
├── requirements.txt ├── setup.py └── README.md

> See the full annotated folder structure in [`STRUCTURE.md`](STRUCTURE.md)

---

## 🧠 Models

Five deep learning architectures are implemented and benchmarked:

### 1. CNN (Baseline)
- Convolutional feature extractor with a fully connected classification head
- Establishes the performance baseline for all comparisons

### 2. CNN + GRU
- CNN frontend for local feature extraction
- GRU backend for temporal sequence modeling
- Parameter-efficient recurrent architecture

### 3. CNN + LSTM
- CNN frontend for local feature extraction
- LSTM backend with cell state for long-range temporal dependency modeling
- Directly compared against CNN+GRU on the MMG modality

### 4. CNN + Transformer
- CNN frontend generates token sequences from feature maps
- Lightweight Transformer encoder applies multi-head self-attention over tokens
- Combines CNN's local inductive bias with global context modeling

### 5. Small Vision Transformer (ViT)
- Raw sensor windows are directly tokenized via patch/channel-wise embedding
- Pure self-attention from input — no CNN frontend
- Directly compared against CNN+Transformer across all modality conditions

---

## 🧪 Experiments

### Core Experiment Matrix

| # | Architecture | Modality | Fusion |
|---|---|---|---|
| 1 | CNN | MMG only | — |
| 2 | CNN | IMU only | — |
| 3 | CNN | MMG + IMU | Early |
| 4 | CNN | MMG + IMU | Late |
| 5 | CNN + GRU | MMG only | — |
| 6 | CNN + GRU | IMU only | — |
| 7 | CNN + GRU | MMG + IMU | Early |
| 8 | CNN + GRU | MMG + IMU | Late |
| 9 | CNN + LSTM | MMG only | — |
| 10 | CNN + LSTM | IMU only | — |
| 11 | CNN + LSTM | MMG + IMU | Early |
| 12 | CNN + LSTM | MMG + IMU | Late |
| 13 | CNN + Transformer | MMG only | — |
| 14 | CNN + Transformer | IMU only | — |
| 15 | CNN + Transformer | MMG + IMU | Early |
| 16 | CNN + Transformer | MMG + IMU | Late |
| 17 | Small ViT | MMG only | — |
| 18 | Small ViT | IMU only | — |
| 19 | Small ViT | MMG + IMU | Early |
| 20 | Small ViT | MMG + IMU | Late |

> Each of the 20 configurations is run under **both SD and LOSO** protocols → **40 total training runs**

### Dedicated Comparison Experiments

| Comparison | Architectures | Modality | Goal |
|---|---|---|---|
| RNN Cell Type | CNN+LSTM vs. CNN+GRU | MMG only | Isolate effect of recurrent cell on muscle signals |
| Tokenization Strategy | ViT vs. CNN+Transformer | MMG, IMU, Fused | Isolate value of CNN frontend vs. raw tokenization |

---

## 🏋️ Training Strategies

### Subject-Dependent (SD)
- Train and test on data from the **same volunteer**
- Represents the **upper bound** of achievable performance
- Reveals maximum model capacity per subject

### Leave-One-Subject-Out (LOSO)
- Train on **9 volunteers**, test on the **held-out 1**
- Repeated across all 10 volunteers (10-fold)
- **Gold standard generalization metric**
- The SD–LOSO performance gap reveals subject-specificity of learned representations

### Reproducible Optuna training

The six public functions in `train/` cover eight model variants: standalone IMU
and MMG CNNs plus CNN and GRU fusion models, each in single-window and windowed
forms. Every entry point now follows the same experiment lifecycle:

1. Reserve a grouped, stratified 10% validation split from the development data.
2. Optimize architecture and training parameters with Optuna using
	`0.5 * validation accuracy + 0.5 * validation macro-F1`.
3. Apply class-balanced cross-entropy weights computed from training labels only.
4. Persist the study and export its trial table and interactive visualizations.
5. Rebuild the winning architecture and refit it on all non-test development data
	for the winning trial's best epoch count.
6. Save a uniquely named checkpoint and a reproducibility manifest.

The test tensors are deliberately not evaluated by these training functions.
They remain sealed for the dedicated model-evaluation workflow.

Each model run is stored under `results/training/<run-id>/` and contains:

| Artifact | Purpose |
|----------|---------|
| `<run-id>.pt` | Final model, optimizer/scheduler state, history, and training config |
| `study.sqlite3` | Complete resumable Optuna study |
| `trials.csv` | Portable export of all Optuna trials |
| `manifest.json` | Model, data, split, dependency, seed, metric, and hash provenance |
| `training_history.json` | Final all-development-data refit history |
| `plots/*.html` | Interactive optimization history, importance, coordinate, and slice plots |

Training functions accept keyword overrides including `n_trials`, `timeout`,
`artifact_root`, `run_label`, and `resume_run_id`. Fusion entry points use
separate `cnn_resume_run_id` and `gru_resume_run_id` values because the CNN and
GRU searches are independent studies. Existing fixed checkpoint arguments are
retained as compatibility aliases; the uniquely named artifact checkpoint is
the authoritative model recorded in the manifest.

---

## 🔀 Fusion Strategies

### Early Fusion
- MMG and IMU tensors are **concatenated along the channel dimension** before the model
- Single unified model processes the combined input
- Allows cross-modal interaction at every layer

### Late Fusion
- **Separate encoders** process MMG and IMU independently
- Feature vectors are merged just before the classification head
- Each encoder can specialize to its modality's signal characteristics

---

## ⚙️ Installation

### Prerequisites
- Python >= 3.11
- CUDA-compatible GPU (recommended)

### Setup

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/HAR-MultiModal-DL.git
cd HAR-MultiModal-DL
```

**2. Create and activate a virtual environment**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**3. Install PyTorch and project dependencies**

For CPU-only use, install the project requirements normally:

```powershell
python -m pip install -r requirements.txt
```

For NVIDIA GPU use, first choose the CUDA-enabled command for your operating
system and driver from the [official PyTorch installer](https://pytorch.org/get-started/locally/),
run that command, and then install the remaining requirements. Do not assume
that installing a local CUDA Toolkit makes a `+cpu` PyTorch wheel GPU-capable;
PyTorch itself must have been installed with CUDA support.

Verify the active environment before starting a long experiment:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

If the PyTorch version ends in `+cpu` or `torch.version.cuda` is `None`, that
environment cannot use an NVIDIA GPU.

### Compute-device selection

Every public function in `train/` accepts a `device` keyword:

| Value | Behavior |
|-------|----------|
| `"auto"` | Default. Use CUDA when available; otherwise use CPU. |
| `"cuda"` | Require the current CUDA device; fail clearly if unavailable. |
| `"cuda:0"` | Require a particular zero-based NVIDIA GPU index. |
| `"cpu"` | Force CPU execution even when CUDA is available. |

Example:

```python
result = train_and_evaluate_imu_cnn(
	prepared,
	device="cuda",
)
```

The selected device is used consistently for Optuna trials, final refitting,
class weights, input batches, and fusion checkpoint loading. Each run records
the requested and resolved device, GPU name, CUDA version, and cuDNN version in
its `manifest.json`. An explicit CUDA request never silently falls back to CPU.