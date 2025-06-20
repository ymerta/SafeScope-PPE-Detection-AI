# Author: Arda Baran
# Description:
# Applies YOLOv9e object detection on test images, saves prediction results with bounding boxes,
# and optimizes GPU memory usage by cleaning after each prediction cycle.
from ultralytics import YOLO
import torch
import os
import glob
from IPython.display import Image, display
import gc

# Loads the trained YOLO model
model_path = "/content/drive/MyDrive/PPE_and_Construction-Site_Detection_YOLOv9e/runs/train/ppe_detection_optimized/weights/best.pt"
model = YOLO(model_path)  # Loads the YOLO model from the specified path

# Defines the directory containing test images
test_images_dir = "/content/drive/MyDrive/test_images"

# Gathers all image file paths with supported extensions
image_paths = glob.glob(os.path.join(test_images_dir, "*.jpg")) + \
              glob.glob(os.path.join(test_images_dir, "*.png")) + \
              glob.glob(os.path.join(test_images_dir, "*.jpeg"))

# Defines the output directory to save prediction results
output_dir = "/content/drive/MyDrive/predicted_test_images"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Optimized prediction function for a single image
def predict_and_save(image_path):
    """
    Runs object detection on a single image and saves the prediction results.

    Args:
        image_path (str): Path to the input image.

    Returns:
        result (ultralytics.engine.results.Results or None): Detection results if successful, otherwise None.
    """
    try:
        result = model.predict(
            source=image_path,
            imgsz=640,         # Image resizing for inference (640x640)
            conf=0.25,         # Minimum confidence threshold for predictions
            device="cuda",     # Use GPU (cuda) for faster inference
            save=True,         # Save the output image with bounding boxes
            save_txt=False,    # Do not save YOLO text labels separately
            save_conf=False,   # Do not save prediction confidence values
            project=output_dir, # Save outputs under this directory
            name="",            # No subfolder, saves directly in project folder
            exist_ok=True       # Overwrite existing outputs without error
        )
        return result  # Return the prediction result object
    except Exception as e:
        print(f"Error: {str(e)}")  # Print error message if prediction fails
        return None

# Processes all test images one by one
for img_path in image_paths:
    print(f"Processing: {os.path.basename(img_path)}")  # Prints the name of the current image
    predict_and_save(img_path)  # Predicts and save results
    torch.cuda.empty_cache()  # Frees up CUDA memory after each prediction
    gc.collect()  # Triggers garbage collection for RAM cleanup

# Final output message
print("\nPredictions saved to:", output_dir)
