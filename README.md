
## NAS-FTI-FDet

**NAS-FTI-FDet** has been accepted for publication in the IEEE Transactions on Industrial Informatics 2024.

**NAS-FTI-FDet** is a time- and memory- efficient differentiable architecture method. It mainly focuses on searching a task-specific detection head for fault detection of freight train images. For a detailed description of technical details and experimental results, please refer to our paper:
<div align="center"><img decoding="async" src="Framework.jpg" width="75%"/> </div>

[Efficient Visual Fault Detection for Freight Train via Neural Architecture Search with Data Volume Robustness](https://arxiv.org/pdf/2405.17004)

This code is based on the implementation of [FAD](https://github.com/msight-tech/research-fad)

## Installation
The full installation instructions refer to [INSTALL.md](INSTALL.md).

## Usage
#### Search on your dataset
To run the following example code, you will search for the detection head with FCOS on your dataset:

    CUDA_VISIBLE_DEVICES=0 \
    python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=$((RANDOM + 10000)) \
        tools/search_net.py \
        --skip-test \
        --config-file configs/fad/search/fad-fcos_imprv_R_50_FPN_1x.yaml \
        --use-tensorboard \
        DATALOADER.NUM_WORKERS 2 \
        OUTPUT_DIR training_dir/search/fad-fcos_imprv_R_50_FPN_1x

#### Training with searched architectures 
To run the following example code, you will train FCOS with the searched architecture on your dataset:

    CUDA_VISIBLE_DEVICES=0 \
    python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=$((RANDOM + 10000)) \
        tools/train_net.py \
        --config-file configs/fad/augment/fad-fcos_imprv_R_50_FPN_1x.yaml \
        --genotype-file training_dir/search/fad-fcos_imprv_R_50_FPN_1x/genotype.log \
        DATALOADER.NUM_WORKERS 2 \
        OUTPUT_DIR training_dir/augment/fad-fcos_imprv_R_50_FPN_1x

#### Evaluation
To run the following example code, you will evaluate the trained model on your dataset:

    CUDA_VISIBLE_DEVICES=0 \
    python -m torch.distributed.launch \
        tools/test_net.py \
        --config-file configs/fad/augment/fad-fcos_imprv_R_50_FPN_1x.yaml \
        --genotype-file training_dir/search/fad-fcos_imprv_R_50_FPN_1x/genotype.log \
        MODEL.WEIGHT path_to_the_weights.pth \
        TEST.IMS_PER_BATCH 6

## Citations
BibTeX reference is shown in the following.
```
@article{zhang2024efficient,
  title={Efficient Visual Fault Detection for Freight Train via Neural Architecture Search with Data Volume Robustness},
  author={Zhang, Yang and Li, Mingying and Pan, Huilin and Liu, Moyun and Zhou, Yang},
  journal={arXiv preprint arXiv:2405.17004},
  year={2024}
}
```