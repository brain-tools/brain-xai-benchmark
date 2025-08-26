# Repository for: "Explainable AI Methods for Neuroimaging: Systematic Failures of Common Tools, the Need for Domain-Specific Validation, and a Proposal for Safe Application"

Read our preprint here: [arXiv:2508.02560](https://www.arxiv.org/abs/2508.02560)

## Upfront Note
This project was computationally very demanding. Many steps required multiple days of runtime on an HPC cluster with a high degree of parallelization. This made building an automated end-to-end pipeline impractical.

Instead, the repository consists of a set of scripts and notebooks that must be run in the correct order to reproduce the results. The pipeline is divided into numbered segments, described below.


## Key Scripts Overview

While the full pipeline must be executed to reproduce results, the following scripts are the **core entry points** for the main analyses:

- **1-idp_correction/mass_decorrelate.py**  
  Performs IDP correction for a given IDP across a range of principal components.

- **2-model_training/models/ResNet/main.py**  
  Defines the ResNet model architecture and training logic.

- **3-compute_explanations/get_expls.py**  
  Computes model explanations for a specified tasks and trained model checkpoint using selected XAI methods.  

- **4-xai_validation/cidp_based/compute_rma_scores.py**  
  Evaluates explanations against ground-truth targets using the RMA metric.  


## Repository Structure

```plaintext
brain-xai-benchmark/
├── 0-prep/                     # Preparation scripts
├── 1-idp_correction/           # IDP correction and visualization
├── 2-model_training/           # Model definitions, configs, and training scripts
├── 3-compute_explanations/     # XAI implementations and scripts to compute model explanations
├── 4-xai_validation/           # Validation metrics and evaluation scripts
└── 5-natural_image_comparison/ # Natural image benchmark
```

Each folder corresponds to one segment of the pipeline, and the following sections explain them in order:

## 0. Preparation

Scripts to compute basic elements required for IDP correction:
- `save_lin_reg_imgs.py` → saves linearly registered MRI images to disk
- `fit_masker.py` → fits brain masker  
- `prep_mass_decorrelate.py` → creates dataframe mapping IDPs to MRI image paths

## 1. IDP Correction

Transforms raw IDPs into localized/corrected IDPs (cIDPs):

- **mass_decorrelate.py** → performs correction for a given IDP across a range of principal components
- Config files:  
  - `config_cortical.csv`  
  - `config_subcortical.csv`  
  (contain IDP name strings, UKBB field IDs, keys for binary brain atlas)  

**Visualization notebooks:**  
- `visualize_cidp_across_npcs.ipynb` → effect of correction across PCs  
- `visualize_cidps_given_npcs.ipynb` → plots for multiple cIDPs before/after correction  
- Cutout experiment: `2-model_training/models/ResNet/main_cutout.py`

## 2. Model Training

Scritps for training models to predict cIDPs and other targets.

- Implemented with **PyTorch Lightning** (`LightningModule` classes in `/models`)  
- Configurations stored in `*.yaml` files under `/run_models`  
- Required inputs:  
  - `split.json` → train/val/test split  
  - `data_loader_table.csv` → target variable + paths to images/warp coefficients

Setup training files:

### Stage 1 – cIDPs
- `create_dataloader_table.ipynb` (from corrected cIDPs)  
- `create_basic_split.ipynb` (random 80/10/10 split)  
- `adapt_basic_split.ipynb` (removes subjects with missing cortical IDPs)  

### Stage 2 – Artificial Diseases
- `table_and_split_for_art_dis.ipynb` → combines cIDPs into binary disease targets

### Stage 3 – WMH Lesion Load
- `create_dataloader_table.ipynb` using UKBB lesion load field  
- `adapt_basic_split.ipynb` to drop subjects without T2 image from base split

### Stage 4 – Brain Age
- `create_dataloader_table.ipynb` for T1/T2 MRIs  
- Healthy train split from Siegel et al. [(2025)](https://pubmed.ncbi.nlm.nih.gov/40489428/)
- `adapt_t1_brain_age_split.ipynb` by removing subjects without T2 images

## 3. Compute Explanations

- **get_expls.py** → generates model explanations  
  - Requires: `split.json`, dataloader table, model checkpoint  
  - Uses Lightning `Trainer` class and model `test_step()` functions  
- XAI methods defined in: `xai_methods.py`  

## 4. XAI Validation

**Prerequisite**: inverse-warp target regions from nonlinear MNI space to participants' linear MNI space using `prewarp_atlas.py`

### Stage 1 & 2 (cIDPs + Artificial Diseases)  
- Scripts for computing RMA, TPR, and FRP in `cidp_based/`  
- Metrics definitions: `metrics.py`  
- Ground-truth target generation: `1-idp_correction/atlas_methods.py`

### Stage 3 – WMH Lesion Load  
- `lesions/evaluate_lesion_expls.py`  
- Warps lesion masks, calculates RMA scores, plots explanations  

### Stage 4 – Brain Aging Marker Overlap
- `brain_age_ranking/compute_aging_marker_overlap.py`  
- Inputs: precomputed brain age explanations + known makers of aging from Walhovd et al. (2011) in `aging_volume_table.csv`  

### Brain Age (MS Explanation Contrast)  

- `brain_age_disease_diff/disease_diff_effect_sizes.ipynb`  
- Requires precomputed explanations + matching file (patients vs. controls)

## 5. Natural Image Comparison

- `eval_expls_with_seg_masks.py`
  - computes explanations for pretrained ResNet on ImageNet test/validation images
  - Evaluates with RMA metric and segmentation masks from [ImageNet-S](https://github.com/LUSSeg/ImageNet-S) (Gao et al., 2021)
- `filter_rma_score_eval.ipynb`
  - creates result dataframe for heatmap plot (Fig. 4) by filtering previously computed RMA scores (e.g. by mask size and explanation cutoff), and by grouping scores into ImageNet supercategories

## Installation (GPU machine required)

### 1. Clone the repository

```bash
git clone https://github.com/brain-tools/brain-xai-benchmark.git
cd brain-xai-benchmark
```

### 2. Create a minimal conda environment
```bash
conda create -n brain-xai python=3.10
conda activate brain-xai
```

### 3. Install the GPU-dependent dataloader first
```bash
pip install git+https://github.com/brain-tools/brain-deform.git
```

### 4. Install the rest of the dependencies
```bash
conda env update --file environment.yaml
```

## Data Availability

- **UK Biobank (UKBB)**: Restricted-access dataset. You need UKBB approval to obtain IDPs and images.  
- **ImageNet-S**: Available from https://github.com/LUSSeg/ImageNet-S.  

## Citation

If you use this code, please cite:

```bibtex
@article{siegel2025explainable,
  title={Explainable AI Methods for Neuroimaging: Systematic Failures of Common Tools, the Need for Domain-Specific Validation, and a Proposal for Safe Application},
  author={Siegel, Nys Tjade and Cole, James H and Habes, Mohamad and Haufe, Stefan and Ritter, Kerstin and Schulz, Marc-Andr{\'e}},
  journal={arXiv preprint arXiv:2508.02560},
  year={2025}
}
```
