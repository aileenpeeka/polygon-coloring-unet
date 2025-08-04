import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
import torchvision.transforms as transforms

COLOR_MAP = {
    "red": (1, 0, 0),
    "blue": (0, 0, 1),
    "yellow": (1, 1, 0),
    "green": (0, 1, 0),
}

class PolygonDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.inputs_dir = os.path.join(root_dir, "inputs")
        self.outputs_dir = os.path.join(root_dir, "outputs")
        self.transform = transform
        
        # Load data.json file
        json_path = os.path.join(root_dir, "data.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                self.data = json.load(f)
        else:
            # Fallback: create data from directory structure
            self.data = self._create_data_from_dirs()
    
    def _create_data_from_dirs(self):
        """Create data list from directory structure if data.json doesn't exist"""
        data = []
        if os.path.exists(self.inputs_dir):
            for filename in os.listdir(self.inputs_dir):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    # Try to infer color from filename or use default
                    color = "red"  # default color
                    if "blue" in filename.lower():
                        color = "blue"
                    elif "yellow" in filename.lower():
                        color = "yellow"
                    elif "green" in filename.lower():
                        color = "green"
                    
                    data.append({
                        "input": filename,
                        "output": filename,  # assuming same filename for output
                        "color": color
                    })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load input image
        input_path = os.path.join(self.inputs_dir, item["input"])
        if os.path.exists(input_path):
            img = Image.open(input_path).convert("RGB")
        else:
            # Create a dummy image if file doesn't exist
            img = Image.new('RGB', (256, 256), color='white')
        
        # Load target image
        output_path = os.path.join(self.outputs_dir, item["output"])
        if os.path.exists(output_path):
            target = Image.open(output_path).convert("RGB")
        else:
            # Create a dummy target if file doesn't exist
            target = Image.new('RGB', (256, 256), color='black')
        
        # Create color condition channel
        color_name = item["color"].lower()
        color_vec = COLOR_MAP.get(color_name, COLOR_MAP["red"])
        color_channel = np.ones((img.size[1], img.size[0], 1)) * np.array(color_vec).mean()
        
        # Combine image with color condition
        img_array = np.array(img) / 255.0
        img_cond = np.concatenate([img_array, color_channel], axis=2)
        
        # Apply transforms
        if self.transform:
            img_cond = self.transform(img_cond)
            target = self.transform(target)
        else:
            img_cond = transforms.ToTensor()(img_cond)
            target = transforms.ToTensor()(target)
        # Ensure float32 dtype
        img_cond = img_cond.float()
        target = target.float()
        return img_cond, target

def get_color_condition(color_name):
    """Get color condition tensor for a given color name"""
    color_vec = COLOR_MAP.get(color_name.lower(), COLOR_MAP["red"])
    return torch.tensor(color_vec, dtype=torch.float32) 