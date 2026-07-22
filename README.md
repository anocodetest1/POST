# Spatio-Temporal Adversarial Learning for Spacecraft Fault Detection and Isolation

This repository contains the official PyTorch implementation of the paper **Spatio-Temporal Adversarial Learning for Spacecraft Fault Detection and Isolation**.

## 💡 Highlights
 - We propose a reconstruction-based FDI framework leveraging the synergy between Temporal Anomaly Self-Attention (TASA) and Spatial Anomaly Graph Attention (SAGA) modules. 
 - We construct SMD+, a dataset featuring precise channel-wise annotations. The dataset can serve as a generic testbed for anomaly localization in MTS. 
 - Extensive experiments validate that POST not only outperforms leading methods by a substantial margin in diverse MTSAD tasks, but also establishes a new state-of-the-art (SOTA) for spacecraft FDI applications.

## Suggested citation

Please consider citing the arxiv version of our work:

```
@misc{zhang2026postpriorobservationadversariallearning,
      title={POST: Prior-Observation Adversarial Learning of Spatio-Temporal Associations for Multivariate Time Series Anomaly Detection}, 
      author={Suofei Zhang and Yaxuan Zheng and Haifeng Hu},
      year={2026},
      eprint={2605.18128},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.18128}, 
}

```

## Get Started

1. Prerequisites
 - Python >= 3.6
 - PyTorch >= 1.4.0
2. Data Preparation
   You can obtain the four standard benchmarks (SMD, MSL, SMAP, PSM) from this [Google Cloud](https://drive.google.com/drive/folders/1gisthCoE-RrKJ0j3KPV7xiibhHWT9qRm?usp=sharing). **All the datasets are well pre-processed**. For the SWaT dataset, you can apply for it by following its official tutorial.
   For our newly proposed SMD+ dataset, please download it from Hugging Face: [SMD+ Dataset on Hugging Face](https://huggingface.co/datasets/www0wwwjs1/SMDPlus).
   Place all downloaded data into the ./dataset/ directory.
3. Pre-trained Models
   The best-performing models on all benchmarks (SMD, MSL, SMAP, SWaT, PSM, and SMD+) are available at: [Pre-trained Checkpoints](https://huggingface.co/www0wwwjs1/POST).
   Download and extract the .tar.gz file, and place the pre-trained weights into the ./checkpoints/ folder. Ensure the files are named following the format: <dataset_name>_checkpoint.pth (e.g., SMD_checkpoint.pth).

## Evaluation

To evaluate the trained models on the benchmarks, run the following scripts.

(Note: To evaluate on the SMD+ dataset, we directly use the model trained on the standard SMD dataset. Simply rename or copy the SMD_checkpoint.pth file to ./checkpoints/SMD+_checkpoint.pth before running the evaluation).
```bash
# SMD
python main.py --anomaly_ratio 0.5 --batch_size 64 --mode test --dataset SMD --data_path dataset/SMD --input_c 38 --output_c 38
# MSL
python main.py --anomaly_ratio 1. --batch_size 64 --mode test --dataset MSL --data_path dataset/MSL --input_c 55 --output_c 55
# SMAP
python main.py --anomaly_ratio 1. --batch_size 64 --mode test --dataset SMAP --data_path dataset/SMAP --input_c 25 --output_c 25
# SWaT
python main.py --anomaly_ratio 1. --win_size 150 --batch_size 64 --mode test --dataset SWaT --data_path dataset/SWaT --input_c 51 --output_c 51
# PSM
python main.py --anomaly_ratio 1. --batch_size 64 --mode test --dataset PSM --data_path dataset/PSM --input_c 25 --output_c 25
# SMD+
python main_channel.py --anomaly_ratio 0.5 --batch_size 64 --mode test --dataset SMD+ --data_path dataset/SMD+ --input_c 38 --output_c 38
```

## Training

To train the POST model from scratch on the respective datasets, execute the following commands:
```bash
# SMD
python main.py --anomaly_ratio 0.5 --num_epochs 10 --batch_size 64 --mode train --dataset SMD --data_path dataset/SMD --input_c 38 --output_c 38
# MSL
python main.py --anomaly_ratio 1. --num_epochs 10 --batch_size 64 --mode train --dataset MSL --data_path dataset/MSL --input_c 55 --output_c 55
# SMAP
python main.py --anomaly_ratio 1. --num_epochs 10 --batch_size 64 --mode train --dataset SMAP --data_path dataset/SMAP --input_c 25 --output_c 25
# SWaT
python main.py --anomaly_ratio 1. --win_size 150 --num_epochs 10 --batch_size 64 --mode train --dataset SWaT --data_path dataset/SWaT --input_c 51 --output_c 51
# PSM
python main.py --anomaly_ratio 1. --num_epochs 10 --batch_size 64 --mode train --dataset PSM --data_path dataset/PSM --input_c 25 --output_c 25
```

## Acknowledgments

We deeply appreciate the authors of the following repository for their brilliant open-source contributions, which provided useful modules and data preprocessing pipelines for our code:

- Anomaly-Transformer: https://github.com/thuml/Anomaly-Transformer

