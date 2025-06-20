# Author: Arda Baran
# Description: YOLOv9e-based training script for detecting personal protective equipment (PPE) and construction-related items.
#              Includes advanced training configuration, resume support, data augmentation, and logging to Google Drive.
#              This script trains the model with the combined dataset.Two separate datasets whose are 'SH17' for ppe-related items and 'ACID'
#              for construction-related items are merged.There are 17026 images in this merged dataset. 13446 images are used for training 
#              and 3580 images are used for validation. There are 27 classes in this dataset and 17 classes belong to 'SH17' and the other 
#              10 classes belong to 'ACID'.All the class IDs adjusted to the combined datasets with helper scripts.
#              Dataset URLs:
#              'SH17' Dataset url: https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection
#              'ACID' Dataset url: https://universe.roboflow.com/test-blhxw/acid-dataset/dataset/1
import os
import yaml
from ultralytics import YOLO
import torch

def main():
    # CUDA settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Enables expandable memory segments to reduce CUDA out-of-memory errors on large models or batch sizes
    torch.backends.cudnn.benchmark = True # Enables cuDNN auto-tuner to find the best algorithms for hardware, improving performance

    device = 'cuda' if torch.cuda.is_available() else 'cpu' # Automatically selects GPU (cuda) if available, otherwise falls back to CPU
    print(f"Device using: {device.upper()}")
    if device == 'cuda': # If using GPU, prints memory allocation and total available memory for monitoring
        print(f"GPU Memory Status: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")
    
    #Training configuration and Explicit Augmentation Settings
    config = {
        'data': '/content/drive/MyDrive/PPE_and_Construction-Site_Detection_YOLOv9e/datasets/PPE_And_Construction_Datasets_Merged/dataset.yaml',# Path to the dataset configuration YAML file stored in Google Drive
        'epochs': 200, # Longer training duration
        'batch': 32,  #Increasing the batch size can provide more stable gradient calculations
        'imgsz': 640, # Input image size for training and validation; 640 is a common default for a good trade-off between speed and accuracy
        'device': device,# Device to use for training 'cuda' for GPU.
        'workers': 16, # Number of workers increased to speed up the data loading process.
        'optimizer': 'SGD', # SGD is preferred over AdamW because SGD provides better generalization when used with momentum and weight decay.
        'lr0': 0.01, # Initial learning rate a higher starting rate is selected based on the experimental results
        'momentum': 0.937, # High momentum value as in typical YOLO configurations.
        'weight_decay': 0.0005, # Appropriate weight decay to prevent overfitting.
        'cos_lr': True, # Cosine annealing learning rate scheduler is used because it shows successful results
        'patience': 50,  # Longer patience value for the learning rate reduction strategy.
        'augment': True, # Enables augmentations
        'mosaic': True,  # Enables Mosaic augmentation
        'flipud': 0.5,   # 50% chance for vertical flip
        'fliplr': 0.5,   # 50% chance for horizontal flip
        'name': 'ppe_detection_optimized',# Custom name for the training run; this will be used as the subfolder name inside the 'project' directory
        'exist_ok': True, # Overwrites if a folder with the same name already exists.
        'project': '/content/drive/MyDrive/PPE_and_Construction-Site_Detection_YOLOv9e/runs/train'# Directory where training results (models, metrics, logs) will be saved on Google Drive
    }

    # Constructs the directory path where the training run results (models, metrics, logs) will be stored
    run_dir = os.path.join(config['project'], config['name']) # Combines the 'project' directory and the 'name' of the run to form the path for the current training session
    # Constructs the directory path where the weights (trained models) will be stored within the run directory
    weights_dir = os.path.join(run_dir, 'weights') # Creates the 'weights' folder within the current run's directory
    # Constructs the path to the 'last.pt' file, which contains the weights of the last saved model
    last_weights = os.path.join(weights_dir, 'last.pt') # Path to the checkpoint file for resuming training
    # Constructs the path to save the training configuration in a YAML file (opt.yaml)
    opt_yaml_path = os.path.join(run_dir, 'opt.yaml') # Path to save the configuration options for the training run

    # training
    if os.path.exists(last_weights): # If a previous weight file exists, resumes training from the last checkpoint
        print("Previous weights found. Resuming training from the last checkpoint.")
        model = YOLO(last_weights).to(device)  # Load the previously trained model and send it to the appropriate device (GPU)
        results = model.train(resume=True)  # <<< RESUME
    else: # If no previous weights are found, starts a new training session
        print("Starting new model training.")
        model = YOLO('yolov9e.pt').to(device) # Loads the YOLOv9e model and move it to the GPU
        results = model.train(**config) # Starts training with the specified configuration

        # Creates the directory to save training options/config if it doesn't exist
        os.makedirs(run_dir, exist_ok=True)
        # Saves the training configuration to a YAML file (opt.yaml)
        with open(opt_yaml_path, 'w') as f:
            yaml.dump(config, f)
        print(f"Training configuration saved to {opt_yaml_path}.")

    # Prints mAP after validation
    metrics = model.val()
    print(f" mAP50-95: {metrics.get('box/map', 'N/A')} | mAP50: {metrics.get('box/map50', 'N/A')}")

if __name__ == '__main__':
    main()