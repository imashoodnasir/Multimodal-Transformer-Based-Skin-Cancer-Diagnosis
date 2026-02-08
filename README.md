# Cross-Modal Robustness Analysis of Transformer-Based Skin Cancer Models

Full PyTorch code for the proposed methodology:
- Multimodal transformer with metadata token
- Robustness-aware training (resolution variation, token dropping, metadata masking)
- Multi-axis robustness evaluation (domain, resolution, modality)

## Install
pip install -r requirements.txt

## Train (3 runs)
python train.py --dataset derm7pt --config configs/default.yaml
python train.py --dataset pad_ufes_20 --config configs/default.yaml

## Robustness evaluation
python eval_robustness.py --train_dataset derm7pt --eval_dataset pad_ufes_20 --split test --seed 0 --model_path ./runs/derm7pt/seed_0/model.pt --config configs/default.yaml
