# Ayna ML Assignment - Polygon Segmentation with Color Conditioning

This project implements a UNet-based deep learning model for polygon segmentation with color conditioning. The model takes an input image and a color condition to generate a segmented output where polygons are colored according to the specified condition.

## Architecture

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

## Project Structure

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
│   └── unet.py              # UNet model implementation
├── train.py                 # Training script with wandb integration
├── inference.ipynb          # Jupyter notebook for inference
├── utils.py                 # Dataset loader and utilities
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Setup Instructions

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

## Hyperparameters

| Parameter      | Value               | Description                             |
|----------------|---------------------|-----------------------------------------|
| Learning Rate  | 1e-4                | Adam optimizer learning rate            |
| Batch Size     | 8                   | Training batch size                     |
| Epochs         | 20                  | Maximum training epochs                 |
| Loss Function  | MSE                 | Mean squared error loss                 |
| Optimizer      | Adam                | Adam optimizer with weight decay        |
| Scheduler      | ReduceLROnPlateau   | Learning rate reduction on plateau      |

## Training Dynamics

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

## Model Performance

### Key Features
- **Color Conditioning**: Segments polygons based on color conditions
- **Skip Connections**: Preserves spatial information
- **Batch Normalization**: Stabilizes training
- **Data Augmentation**: Ready for future enhancements

### Expected Behavior
- **Input**: Image with polygons + color condition
- **Output**: Segmented polygons filled with specified color

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in `train.py`
   - Use CPU training by setting `device = "cpu"`

2. **Dataset Not Found**
   - Verify dataset paths and structure match the specified format

3. **Model Not Loading**
   - Ensure `best_unet_model.pth` exists and matches the model architecture

### Debugging Tips
- Check individual predictions via inference notebook
- Review wandb logs for detailed metrics
- Monitor GPU memory during training

## Future Improvements

- Data Augmentation (rotation, scaling, flipping, color jittering)
- Advanced Architectures (attention, residual connections, multi-scale)
- Improved Loss Functions (Dice loss, Focal loss)
- Additional Metrics (IoU, Dice coefficient, pixel accuracy)

## Dataset Included

- **Note**: The validation dataset is included for easy inference testing.

## Model & Logs

- [📂 Download Trained Models](https://drive.google.com/drive/folders/1l7JGHDUbOGKg-59sKNpKlNTePIZIV19z)
- [📊 Training Metrics on wandb](https://wandb.ai/peekaaileen314-personal/ayna-ml-assignment)

## License

This project is part of the Ayna ML Assignment.

## Support

For questions or issues:

- Review the troubleshooting section above.
- Check wandb logs for detailed training metrics.
- Test with the inference notebook provided.
- Raise an issue with detailed error information if needed.

---

**Note**: Robust error handling and fallback mechanisms are included, making this implementation suitable for development and testing even without a complete dataset.
