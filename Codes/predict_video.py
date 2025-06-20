# Author: Arda Baran
# Description: This script loads yolov9-e object detection model trained for personal protective equipment (PPE) and construction-related items.
# It processes all videos in the specified directory, predicts all classes in the model,then saves the results as avi file with boxes.
from ultralytics import YOLO
import torch
import os
import glob
import gc
import time

# CUDA settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"# Enables expandable memory segments to reduce CUDA out-of-memory errors on large models
torch.backends.cudnn.benchmark = True # Enables cuDNN auto-tuner to find the best algorithms for hardware, improving performance
device = "cuda" if torch.cuda.is_available() else "cpu"# Automatically selects GPU (cuda) if available, otherwise falls back to CPU

# Loads the trained YOLO model
model_path = "/content/drive/MyDrive/PPE_and_Construction-Site_Detection_YOLOv9e/runs/train/ppe_detection_optimized/weights/best.pt" #Model path
model = YOLO(model_path) # Loads the model from the specified path
model.model.to(device) # Moves the model to the specified device
model.eval() # Sets the model to evaluation mode and it is very important for inference because it disables dropout

# Video paths
video_dir = "/content/drive/MyDrive/test_videos" #the directory of where the test videos are stored.
output_dir = "/content/drive/MyDrive/predicted_test_videos_stream" #the directory of where the output results with json logs are saved
os.makedirs(output_dir, exist_ok=True) 
video_paths = glob.glob(os.path.join(video_dir, "*.mp4")) #gets all files with .mp4 extension from the video_dir

# Prediction parameters
predict_params = {
    'imgsz': 640, #default image size
    'conf': 0.25, #minimum confidence threshold for detections. 
    'device': device,# Device to use for inference 'cuda' for GPU.
    'stream': True, #memory efficient for processing videos.
    'save': True, #the output results save to the drive account with predicted boxes.
    'save_txt': False, # Disabled for clean results.
    'save_conf': False, # Disabled for clean results. 
    'exist_ok': True, # Overwrites the already existed directory.
    'project': output_dir, #output directory where the predicted videos are saved with boxes. 
    'name': '' # No sub-folder
}


def memory_cleanup():
                     """
  Memory clean-up function to prevent out of memory errors.

                     """  
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


def predict_video(video_path):
 """
    Processes a given video file, detects all classes per frame, and saves avi file with boxes.

    Args:
        video_path (str): Path to the input video file.

 """
    try:
        print(f" Processing: {os.path.basename(video_path)}")
        start = time.time()
        results = model.predict(source=video_path, **predict_params)
        
        # Processes frames in stream mode
        for r in results:
            _ = r  # Consumes the result to avoid memory buildup

        memory_cleanup()
        print(f" Completed: {time.time() - start:.2f}s")

    except Exception as e:
        print(f"Error: {str(e)}")
        memory_cleanup()

# Entry point of the script
if __name__ == "__main__":
    for video_path in video_paths: # Iterates over all video files found in the specified directory
        predict_video(video_path)

    print("\n All videos processed successfully using GPU.")
