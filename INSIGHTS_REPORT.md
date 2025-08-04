# Insights Report – Ayna ML Assignment

## Overview
This project implements a UNet based model for polygon segmentation with color conditioning. The model predicts polygon shapes filled with a specified color, taking as input both the grayscale polygon image and a color name.

---

## Hyperparameters and Rationale

| Parameter       | Value | Rationale |
|-----------------|-------|-----------|
| Learning Rate    | 1e-4  | Stable training with Adam optimizer |
| Batch Size       | 8     | Balanced memory usage and performance |
| Epochs           | 20    | Early stopping prevents overfitting |
| Loss Function    | MSE   | Works well for pixel-wise prediction |
| Optimizer        | Adam  | Handles sparse gradients effectively |
| Scheduler        | ReduceLROnPlateau | Dynamically reduces LR when validation loss plateaus |

---

## Architecture and Conditioning

- **UNet Architecture**: Classic encoder-decoder with skip connections, enabling the model to capture both spatial and semantic information.
- **Color Conditioning**: Added as a 4th channel to input images, allowing the model to adapt polygon coloring based on a provided color vector.

---

## Training Dynamics

- Loss steadily decreased for both training and validation sets, indicating good generalization.
- Best model was saved based on lowest validation loss (0.81).
- Training metrics and qualitative results were logged with [Weights & Biases](https://wandb.ai/peekaaileen314-personal/ayna-ml-assignment).

---

## Typical Failure Modes

- Minor color bleeding near polygon boundaries.
- Model occasionally predicts slightly faded fills for less common colors.

---

## Key Learnings

1. **Conditioning Approach** – Using an additional channel for color input significantly improved model accuracy.
2. **Experiment Tracking** – WandB dashboards provided clear visibility into training dynamics and helped fine-tune hyperparameters.
3. **Dataset Quality** – Performance strongly depends on diversity of polygon shapes and color combinations.

---

## Deliverables

- **Code and Dataset**: [GitHub Repository](https://github.com/aileenpeeka/polygon-coloring-unet)
- **Trained Models**: [Google Drive Folder](https://drive.google.com/drive/folders/1l7JGHDUbOGKg-59sKNpKlNTePIZIV19z)
- **Experiment Logs**: [WandB Project](https://wandb.ai/peekaaileen314-personal/ayna-ml-assignment)

---

**Note**: This report summarizes the approach, choices and insights gained during development and training of the UNet model.
