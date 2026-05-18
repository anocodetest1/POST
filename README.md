# POST: Prior–Observation Adversarial Learning of Spatio–Temporal Associations for Multivariate Time Series Anomaly Detection

 - We propose POST, a novel MTSAD model that unifies spatial and temporal anomaly modeling under a joint prior–observation adversarial learning framework. 
 - We construct SMD+, a dataset featuring precise channel-wise annotations. The dataset can serve as a generic testbed for anomaly localization in MTS. 
 - Extensive experiments demonstrate that POST consistently outperforms state-of-the-art (SOTA) baselines in standard anomaly detection tasks. Moreover, the optimization of spatial association discrepancy yields superior anomaly localization, significantly enhancing the diagnostic interpretability of final results.

### Suggested citation

Please consider citing our work:

```
@misc{dycl,
      title={Dynamic Contrastive Learning for Hierarchical Retrieval: A Case Study of Distance-Aware Cross-View Geo-Localization}, 
      author={Suofei Zhang and Xinxin Wang and Xiaofu Wu and Quan Zhou and Haifeng Hu},
      year={2025},
      eprint={2506.23077},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.23077}, 
}

```

## Get Started

1. Install Python 3.6, PyTorch >= 1.4.0. 
2. Download data. You can obtain four benchmarks from [Google Cloud](https://drive.google.com/drive/folders/1gisthCoE-RrKJ0j3KPV7xiibhHWT9qRm?usp=sharing). **All the datasets are well pre-processed**. For the SWaT dataset, you can apply for it by following its official tutorial.
3. To evaluate trained model on benchmarks. You can use the model with path and name as: ./checkpoints/*_checkpoint.pth, where * represents the name of dataset. Then the experiment can be reproduced as:
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
4. To train model, scripts can be executed as:
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

For evaluation on SMD+, the trained model on SMD is directly adopted with modified name as './checkpoints/SMD+_checkpoint.pth'.

## Datasets

The proposed dataset SMD+ available at: https://huggingface.co/datasets/www0wwwjs1/DA_Campus/blob/main/DA_Campus.tar.gz

## Trained model

The best performing models on all benchmarks (SMD, MSL, SMAP, SWaT, PSM and SM+) are available at:

You can download, extract and put them in the `checkpoints/` folder.


## Resources

Links to repo with useful features used for this code:

- Anomaly-Transformer: https://github.com/thuml/Anomaly-Transformer

