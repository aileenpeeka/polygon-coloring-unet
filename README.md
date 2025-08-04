# Ayna ML Assignment - Polygon Segmentation with Color Conditioning

This project implements a UNet based deep learning model for polygon segmentation with color conditioning. The model takes an input image and a color name and generates an output where polygons are filled with the specified color.

## 🏗️ Architecture

### UNet Model
- **Input**: 4-channel image (3 RGB channels + 1 color condition channel)
- **Output**: 3-channel RGB segmentation mask
- **Architecture**: Classic UNet with skip connections
- **Parameters**: ~31M trainable parameters

### Color Conditioning
The model uses a 4th channel to condition the segmentation on specific colors:
- **Red**: RGB(1, 0, 0)
- **Blue**: RGB(0, 0, 1) 
- **Yellow**: RGB(1, 1, 0)
- **Green**: RGB(0, 1, 0)

## 📁 Project Structure

```
ayna_ml_assignment/
├── dataset/
│   ├── training/
│   │   ├── inputs/          # Training input images
│   │   ├── outputs/         # Training target images
│   │   └── data.json        # Training metadata
│   └── validation/
│       ├── inputs/          # Validation input images
│       ├── outputs/         # Validation target images
│       └── data.json        # Validation metadata
├── model/
│   └── unet.py             # UNet model implementation
├── train.py                # Training script with wandb integration
├── inference.ipynb         # Jupyter notebook for inference
├── utils.py                # Dataset loader and utilities
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation

Place your dataset in the following structure:
- `dataset/training/inputs/` - Training input images
- `dataset/training/outputs/` - Training target images  
- `dataset/training/data.json` - Training metadata
- `dataset/validation/inputs/` - Validation input images
- `dataset/validation/outputs/` - Validation target images
- `dataset/validation/data.json` - Validation metadata

The `data.json` file should contain:
```json
[
  {
    "input": "image1.png",
    "output": "image1.png", 
    "color": "red"
  }
]
```

### 3. Training

```bash
python train.py
```

The training script will:
- Initialize wandb logging
- Train for 20 epochs with early stopping
- Save the best model as `best_unet_model.pth`
- Generate training curves
- Log metrics to wandb

### 4. Inference

Open `inference.ipynb` in Jupyter:
```bash
jupyter notebook inference.ipynb
```

## 🔧 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 1e-4 | Adam optimizer learning rate |
| Batch Size | 8 | Training batch size |
| Epochs | 20 | Maximum training epochs |
| Loss Function | MSE | Mean squared error loss |
| Optimizer | Adam | Adam optimizer with weight decay |
| Scheduler | ReduceLROnPlateau | Learning rate reduction on plateau |

## 📊 Training Dynamics

### Loss Curves
The training script automatically generates and saves training curves showing:
- Training loss over epochs
- Validation loss over epochs
- Learning rate scheduling

### Wandb Integration
All training metrics are logged to wandb including:
- Train/validation loss
- Learning rate
- Model checkpoints
- Training curves

## 🎯 Model Performance

### Key Features
- **Color Conditioning**: The model learns to segment polygons based on color conditions
- **Skip Connections**: UNet architecture preserves spatial information
- **Batch Normalization**: Stabilizes training and improves convergence
- **Data Augmentation**: Ready for future augmentation techniques

### Expected Behavior
- Input: Image with polygons + color condition
- Output: Segmented image with polygons colored according to condition
- The model should learn to identify polygon boundaries and apply the specified color

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in `train.py`
   - Use CPU training by setting `device = "cpu"`

2. **Dataset Not Found**
   - Ensure dataset structure matches the expected format
   - Check file paths in `data.json`

3. **Model Not Loading**
   - Verify `best_unet_model.pth` exists
   - Check model architecture matches saved weights

### Debugging Tips
- Use the inference notebook to test individual predictions
- Check wandb logs for training dynamics
- Monitor GPU memory usage during training

## 📈 Future Improvements

1. **Data Augmentation**
   - Rotation, scaling, flipping
   - Color jittering
   - Random cropping

2. **Advanced Architectures**
   - Attention mechanisms
   - Residual connections
   - Multi-scale processing

3. **Loss Functions**
   - Dice loss for better segmentation
   - Focal loss for class imbalance
   - Combined losses

4. **Evaluation Metrics**
   - IoU (Intersection over Union)
   - Dice coefficient
   - Pixel accuracy

## 🔗 Model & Logs

- [📂 Download Trained Models](https://drive.google.com/drive/folders/1l7JGHDUbOGKg-59sKNpKlNTePIZIV19z)
- [📊 Training Metrics on wandb](https://wandb.ai/peekaaileen-vellore-institute-of-technology/ayna-ml-assignment)

