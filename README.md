# Post Training Quantization for Fully Quantized Vision Transformer

This repository contains unofficial TensorFlow 2.x based implementation of [FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer](https://arxiv.org/abs/2111.13824) (IJCAI, 2022)

This is re-write of PyToch 1.7 based implementation available [here](https://github.com/megvii-research/FQ-ViT). Unlike the official code, this repository only implements post training quantization for DeiT and ViT.

## Environment
- Tensorflow 2.x
- tfimm
- timm
- datasets

If you want to run tensorrt quantization, use [tensorrt:21.03-py3 docker image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt/tags)(nvcr.io/nvidia/tensorrt:21.03-py3).

## Getting started


### 1. Download ImageNet

You can download ImageNet automatically if you use `huggingface-cli login` and run the code.

### 2. Post train quantization on the pre-trained DeiT

	python main.py             # without PTQ
    python main.py --ptf       # with quantized layernorm using Power-Of-Two-Factor (PTF)
    python main.py --lis       # with quantized softmax using Log-Int-Softmax (LIS)
    python main.py --ptf --lis # with quantized layernorm and softmax (PTF and LIS)

### Available Models

- deit_tiny/small/base/large_patch16_224 (DeiT)
- vit_tiny/samll/base/large_patch16_224 (ViT)

You can change quantized model with `--model_name` with 'q' prefix.

(e.g. `python main.py --ptf --lis --model_name qvit_large_patch16_224`)


## Results on ImageNet

|     Method     | W/A/Attn Bits | DeiT-T | DeiT-S | DeiT-B | ViT-B | ViT-L |
| :------------: | :-----------: | :----: | :----: | :----: | :---: | :---: |
| Full Precision |   32/32/32    | 72.21  | 79.85  | 81.85  | 84.53 | 85.81 |
|     MinMax     |     8/8/8     | 70.94  | 75.05  | 78.02  | 23.64 | 3.37  |
|      EMA       |     8/8/8     | 71.17  | 75.71  | 78.82  | 30.30 | 3.53  |
|   Percentile   |     8/8/8     | 71.47  | 76.57  | 78.37  | 46.69 | 5.85  |
|      OMSE      |     8/8/8     | 71.30  | 75.03  | 79.57  | 73.39 | 11.32 |
|    [TensorRT](https://github.com/jhss/TF_FQVIT/blob/main/tensorrt.py)    |     8/8/8     | 70.18  | 78.23  |  82.18 | 82.23 | 83.36 |
|      PyTorch (Official)      |     8/8/8     | 71.61  | 79.17  | 81.20  | 83.31 | 85.03 |
|      PyTorch (Official)     |   8/8/**4**   | 71.07  | 78.40  | 80.85  | 82.68 | 84.89 |
|      **TF2.0 (This Repository)**| 8/8/**4** |69.5 |77.91 | 80.36 |81.0 | 83.54|
