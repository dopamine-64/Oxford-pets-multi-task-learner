# Oxford-pets-multi-task-learner
Multi-task deep learning Digital Image Processing project

A single notebook (`main.ipynb`) that trains two models (U-Net and Attention U-Net) 
jointly to:
1. **Segment** the pet from the background (binary foreground/background mask)
2. **Classify** the breed (37 classes, cats and dogs) from the same shared encoder

## Dataset
[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — 7,400 images 
across 37 breeds, with pixel-level trimap annotations.

## Architecture
- Shared CNN encoder (4-level U-Net style, with a bottleneck)
- Segmentation decoder: standard U-Net decoder (skip connections via concatenation) 
  vs. Attention U-Net decoder (skip connections gated by learned attention)
- Classification head: lightweight conv head attached to the bottleneck features, 
  with dropout for regularization
- Joint loss: weighted combination of BCE + Dice (segmentation) and 
  class-weighted Cross-Entropy with label smoothing (classification)
- Data augmentation (random flip, rotation, brightness jitter, random crop) applied 
  to the training set to reduce overfitting

## Results

| Model      | Checkpoint  | Split | mIoU   | Dice   | PixelAcc | ClsAcc | Precision | Recall | F1     |
|------------|-------------|-------|--------|--------|----------|--------|-----------|--------|--------|
| U-Net      | unet_aug2   | Train | 0.8792 | 0.9321 | 0.9491   | 0.9279 | 0.9289    | 0.9278 | 0.9273 |
| U-Net      | unet_aug2   | Val   | 0.8479 | 0.9109 | 0.9342   | 0.7058 | 0.7126    | 0.7053 | 0.7037 |
| U-Net      | unet_aug2   | Test  | 0.8526 | 0.9150 | 0.9375   | 0.6754 | 0.6811    | 0.6752 | 0.6702 |
| Attn U-Net | attn_aug2   | Train | 0.8812 | 0.9333 | 0.9501   | 0.9204 | 0.9199    | 0.9201 | 0.9191 |
| Attn U-Net | attn_aug2   | Val   | 0.8473 | 0.9100 | 0.9345   | 0.6977 | 0.7006    | 0.6970 | 0.6935 |
| Attn U-Net | attn_aug2   | Test  | 0.8541 | 0.9157 | 0.9383   | 0.6961 | 0.7032    | 0.6959 | 0.6915 |

Both models perform near-identically on segmentation (0.85 test mIoU) and 
classification (0.68-0.70 test accuracy). Attention U-Net does not show a 
meaningful advantage over the base U-Net for this dataset, likely because pet 
subjects tend to occupy large, well-centered image regions, giving attention 
gates less to selectively focus on compared to tasks with small/scattered targets.

Data augmentation was the single largest improvement lever: baseline (no 
augmentation) test classification accuracy was 0.58, rising to 0.68-0.70 
after augmentation — a reduction in the train/val overfitting gap from 42 
percentage points to 22-25 points.

## Repository structure
- `main.ipynb` — complete pipeline: data download/preprocessing, model definitions 
  (U-Net, Attention U-Net, classifier head), training loop with checkpointing, 
  evaluation, and all required visualizations
  
## Drive structure
- `cache` — downloaded dataset (images.tar.gz and annotations.tar.gz)
- `checkpoints` — trained model weights

## Setup
Designed for Google Colab with a mounted Google Drive for checkpoint persistence 
and dataset caching. Run all cells top to bottom to reproduce from scratch or 
skip the training cells and load the checkpoints directly from this drive link: https://drive.google.com/drive/folders/1UcJcaB0Axg5zy697m3hxmyhOER7ivpj2?usp=sharing

## Notes
- Trimap boundary pixels are folded into the foreground class, reducing the 
  3-class trimap to a binary segmentation problem per the assignment spec.
- Train/val/test split is stratified by breed with a fixed seed for 
  full reproducibility.
- You can get the checkpoint files in checkpoints folder or you can run `main.ipynb` to get them.
