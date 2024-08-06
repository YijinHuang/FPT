# FPT: PETL for High-resolution Medical Image Classification

This is the official implementation of the papers:

> Y. Huang, P. Cheng, R. Tam, and X. Tang, "FPT+: A Parameter and Memory Efficient Transfer Learning Method for High-resolution Medical Image Classification", under review, 2024. [[arXiv](https://arxiv.org/abs/2408.02426)]

> Y. Huang, P. Cheng, R. Tam, and X. Tang, "Fine-grained Prompt Tuning: A Parameter and Memory Efficient Transfer Learning Method for High-resolution Medical Image Classification", MICCAI, 2024. (accepted) [[arXiv](https://arxiv.org/abs/2403.07576)]

We present Fine-grained Prompt Tuning (FPT) and FPT+ for medical image classification. FPT and FPT+ are parameter-efficient transfer learning (PETL) methods that significantly improve memory efficiency over existing PETL methods, especially in the high-resolution context commonly encountered in medical image analysis.

![Framework](./framework.png)



## Performance
Performance using a ViT-B backbone initialized with ImageNet-21K pre-trained weights:
| Method           | # Learnable Parameters | Memory Usage | Average AUC |
|------------------|:----------------------:|:------------:|:-----------:|
| Full fine-tuning | 100                    | 23128        | 88.82       |
| Prompt-tuning    | 0.17                   | 20582        | 83.68       |
| Adapter          | 2.03                   | 19360        | 84.14       |
| LoRA             | 0.68                   | 20970        | 85.94       |
| FPT              | 1.81                   | 1824         | 86.40       |
| FPT+             | 1.03                   | 736          | 87.12       |



## Installation
To install the dependencies, run:
```bash
git clone https://github.com/YijinHuang/FPT
conda create -n fpt python=3.8
conda activate fpt
pip install -r requirements.txt
```



## Dataset
Eight publicly accessible datasets are used in this work:
- **Messidor-2 (Fundus image)** [[Images](https://www.adcis.net/en/third-party/messidor2/)] [[Labels](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades)]
- **DDR (Fundus image)** [[Homepage](https://github.com/nkicsl/DDR-dataset)]
- **ISIC 2016 (Dermoscopic image)** [[Homepage](https://challenge.isic-archive.com/landing/2016/)]
- **ISIC 2018 (Dermoscopic image)** [[Homepage](https://challenge.isic-archive.com/landing/2018/)]
- **Mini-DDSM (Mammography)** [[Homepage](https://ardisdataset.github.io/MiniDDSM/?trk=public_profile_project-button)]
- **CMMD (Mammography)** [[Homepage](https://www.cancerimagingarchive.net/collection/cmmd/)]
- **COVID (Chest X-ray)** [[Homepage](https://www.kaggle.com/datasets/sid321axn/covid-cxr-image-dataset-research)]
- **CHNCXR (Chest X-ray)** [[Homepage](http://archive.nlm.nih.gov/repos/chestImages.php)]



## How to Use
We use the Messidor-2 dataset as an example in the instructions.

### 1. Build dataset
Organize the Messidor-2 dataset as follows:

```
messidor2_dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   ├── class2/
│   │   ├── image3.jpg
│   │   ├── image4.jpg
│   ├── class3/
│   ├── ...
├── val/
├── test/
```

Ensure the `val` and `test` directories have the same structure as `train`. Then, update the `data_path` value in `/configs/dataset/messidor2.yaml` with the path to the Messidor-2 dataset.

### 2. Preloading
Pre-store features from LPM by running:
```bash
python preload.py dataset=messidor2
```

### 3. Training
To train the model, run:
```bash
python main.py dataset=messidor2
```



## Train on Your Own Dataset
### 1. Build your dataset
Organize your dataset similarly to Messidor-2.

### 2. Create and update configurations
Update the configurations marked as '???' in `/configs/dataset/customized.yaml`.

### 3. Preloading
```bash
python preload.py dataset=customized
```
If the image resolution is very large and causes out-of-memory issues, decrease the batch size for preloading:
```bash
python preload.py dataset=customized ++train.batch_size=4
```
After preloading, FPT or FPT+ can still be run with a large batch size.

### 4. Training
To train the model, run:
```bash
python main.py dataset=customized
```



## Other Configurations
You can update the configurations in `/configs`. Hydra is employed to manage configurations. For advanced usage, please check the [Hydra documentation](https://hydra.cc/docs/intro/).

### 1. Run FPT
The default method is FPT+. To run FPT, update the command to:
```bash
python preload.py network=FPT
python main.py network=FPT
```

### 2. Pre-trained model
Most ViT-based models from [Hugging Face](https://huggingface.co/models) uploaded by google/facebook/timm can be directly employed. Default pre-trained weights is `google/vit-base-patch16-384`. To change the LPM, set the pre-trained path in `/configs/network/FPT+.yaml` or update the command to:
```bash
python main.py ++network.pretrained_path=google/vit-base-patch16-384
```
Validated pre-trained weights in this work:
- google/vit-base-patch16-384
- google/vit-large-patch16-384
- facebook/dino-vitb8
- facebook/dino-vitb16

### 3. Disable prelading
To disable preloading, set the 'preload_path' in `/configs/dataset/your_dataset.yaml` to 'null' or update the command to:
```bash
python main.py ++dataset.preload_path=null
```

### 4. Learning rate
To change the learning rate, set the 'learning_rate' in `/configs/dataset/your_dataset.yaml` or update the command to:
```bash
python main.py ++dataset.learning_rate=0.0001
```

### 5. Random seed
To control randomness, set the 'seed' to a non-negative integer in `/configs/config.yaml` or update the command to:
```bash
python main.py ++base.seed=0
```



## Citation
If you find this repository useful, please cite the papers:
```bibtex
@article{huang2024fptp,
  title={FPT+: A Parameter and Memory Efficient Transfer Learning Method for High-resolution Medical Image Classification},
  author={Huang, Yijin and Cheng, Pujin and Tam, Roger and Tang, Xiaoying},
  journal={arXiv preprint arXiv:2408.02426},
  year={2024}
}

@article{huang2024fpt,
  title={Fine-grained Prompt Tuning: A Parameter and Memory Efficient Transfer Learning Method for High-resolution Medical Image Classification},
  author={Huang, Yijin and Cheng, Pujin and Tam, Roger and Tang, Xiaoying},
  journal={arXiv preprint arXiv:2403.07576},
  year={2024}
}
```
