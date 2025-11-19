# Title will be here

<div align="center">
  <a href="https://scholar.google.com/citations?user=DOljaG8AAAAJ&hl=en" target="_blank">Filipp&nbsp;Nikitin<sup>1,2</sup></a> &emsp; <b>&middot;</b> &emsp;
  <a href="#" target="_blank">Dylan&nbsp;M.&nbsp;Anstine<sup>2,3</sup></a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://olexandrisayev.com/" target="_blank">Olexandr&nbsp;Isayev<sup>1,2,4*</sup></a>
  <br>
  <sup>1</sup>Ray and Stephanie Lane Computational Biology Department, Carnegie Mellon University, Pittsburgh, PA, USA
  <br>
  <sup>2</sup>Department of Chemistry, Carnegie Mellon University, Pittsburgh, PA, USA
  <br>
  <sup>3</sup>Department of Chemical Engineering and Materials Science, Michigan State University, East Lansing, MI, USA
  <br>
  <sup>4</sup>Department of Materials Science and Engineering, Carnegie Mellon University, Pittsburgh, PA, USA
  <br><br>
  <a href="#" target="_blank">📄&nbsp;Paper</a> &emsp; <b>&middot;</b> &emsp;
  <a href="#citation">📖&nbsp;Citation</a> &emsp; <b>&middot;</b> &emsp;
  <a href="#setup">⚙️&nbsp;Setup</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://github.com/isayevlab/TSMegaGen" target="_blank">🔗&nbsp;GitHub</a>
  <br><br>
  <span><sup>*</sup>Corresponding author: olexandr@olexandrisayev.com</span>
</div>

---

## Overview

<!-- <div align="center">
    <img width="700" alt="Macrocycles" src="assets/macrocycles.svg"/>
</div> -->

### Abstract


---

## Key Features


---

## Setup

This setup has been tested on Ubuntu 22.04, but can be used across multiple platforms as PyTorch, Pytorch Geometric, and RdKit are widely supported. Installation will usually take up to 20 minutes. 

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for training)
- [Conda](https://docs.conda.io/) or [Mamba](https://mamba.readthedocs.io/) (recommended)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/isayevlab/LoQI.git
cd LoQI

# Create and activate conda environment
conda create -n loqi python=3.10 -y
conda activate loqi

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Data Setup

The training and evaluation require the **ChEMBL3D** dataset. 

**Available with this release:**
- 
- 


---

## Usage

Make sure that `src` content is available in your `PYTHONPATH` (e.g., `export PYTHONPATH="./src:$PYTHONPATH"`) if package is not installed locally (pip install -e .). 

### Model Training

```bash
python scripts/train.py --config-name=loqi outdir=./outputs train.gpus=1 data.dataset_root="./chembl3d_data"

# Customize training parameters
python scripts/train.py --config-name=loqi \
    outdir=./outputs \
    train.gpus=2 \
    train.n_epochs=800 \
    train.seed=42 \
    data.batch_size=150 \
    optimizer.lr=0.0001
```

### Model Inference and Sampling

#### Transition States Generation

```bash
# Generate transitoin states for a single reaction
python scripts/sample_conformers.py \
    --config ./conf/loqi/loqi.yaml \
    --ckpt ./data/loqi.ckpt \
    --input "c1ccccc1" \
    --output ./outputs/benzene_conformers.sdf \
    --n_confs 10 \
    --batch_size 1


# Generate conformers with evaluation
python scripts/sample_conformers.py \
    --config ./conf/loqi/loqi.yaml \
    --ckpt ./data/loqi.ckpt \
    --input "CCO" \
    --output ./outputs/ethanol_conformers.sdf \
    --n_confs 100 \
    --batch_size 10
```

For reference, on an RTX 3090 GPU, inference for a typical ChEMBL molecule takes approximately 0.1 seconds per conformer when processed within a batch.   

#### Available Configurations

**Available Configs:**
- `loqi.yaml` - LoQI stereochemistry-aware conformer generation model
- `nextmol.yaml` - Alternative configuration for NextMol-style generation

### Training Configuration

You can easily override configuration parameters:

```bash
# Example with custom parameters
python scripts/train.py --config-name=loqi \
    outdir=./my_training \
    run_name=my_experiment \
    train.gpus=4 \
    train.n_epochs=500 \
    data.batch_size=64 \
    data.dataset_root="/path/to/chembl3d" \
    wandb_params.mode=online
```

---

## Citation


```bibtex
# citation is coming sson
@article{
}
```



You may also find useful our paper and model for low-energy conformer generation:

```bibtex
@article{nikitin2025scalable,
  title={Scalable Low-Energy Molecular Conformer Generation with Quantum Mechanical Accuracy},
  author={Nikitin, Filipp and Anstine, Dylan M and Zubatyuk, Roman and Paliwal, Saee Gopal and Isayev, Olexandr},
  year={2025}
}
```

This work builds upon the Megalodon architecture. If you use the underlying architecture, please also cite:

```bibtex
@article{reidenbach2025applications,
  title={Applications of Modular Co-Design for De Novo 3D Molecule Generation},
  author={Reidenbach, Danny and Nikitin, Filipp and Isayev, Olexandr and Paliwal, Saee},
  journal={arXiv preprint arXiv:2505.18392},
  year={2025}
}
```