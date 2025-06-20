# Author: Arda Baran
# Description: This script loads yolov9-e object detection model trained for personal protective equipment (PPE) and construction-related items.
# It processes all videos in the specified directory, detects Proximity violations such as if a person unsafe distance to the working heavy machine.
#  logs violations frame-by-frame, and exports the results into structured JSON files.
# The detection helps improve construction site worker safety by automatically identifying proximity violations between person and heavy machines.
from ultralytics import YOLO
import torch
import os
import glob
import gc
import time
import json
import cv2
from datetime import datetime

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
output_dir = "/content/drive/MyDrive/predicted_test_videos_logs"#the directory of where the output results with json logs are saved
os.makedirs(output_dir, exist_ok=True)
video_paths = glob.glob(os.path.join(video_dir, "*.mp4")) #gets all files with .mp4 extension from the video_dir.

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
    'project': output_dir, #output directory where the predicted videos are saved with boxes and json proximity logs. 
    'name': '' # No sub-folder
}

def memory_cleanup():
                     """
  Memory clean-up function to prevent out of memory errors.

                     """                       
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
	
# Heavy Machine List    
vehicle_classes = ["backhoe-loader", "cement-truck", "compactor", "dozer", "dump-truck", "excavator", "grader", "mobile-crane", "tower-crane", "wheel-loader"]
	
def detect_proximity_violation(result):
    """
    Detects proximity violations between persons and construction vehicles in a given detection result.

    This function processes the output of yolov9-e object detection model ,
    identifies detected persons and vehicles, and evaluates their spatial relationships
    to determine if any person is dangerously close to a vehicle (excluding drivers).
    If a dangerous proximity is found, it logs the violation details.

    Args:
        result (ultralytics.engine.results.Results): 
            The detection result from a single frame or image, containing bounding boxes, classes, and labels.

    Returns:
        dict or None:
            If violations are detected, returns a dictionary containing:
                - 'proximity_violations': List of detected violations, 
                  each containing person bounding box, vehicle bounding box, and vehicle label.
                - 'violation_count': Total number of violations detected.
            If no violations are found, returns None.

    Example:
        >>> violations = detect_proximity_violation(result)
        >>> if violations:
        >>>     print(violations['violation_count'])

    Notes:
        - Drivers (persons correctly positioned inside vehicles) are ignored.
        - Only specific vehicle classes are considered for proximity checks.
        - Bounding boxes are expected in (x1, y1, x2, y2) format.
    """
    names = result.names
    boxes = result.boxes
    class_ids = boxes.cls.tolist()
    bboxes = boxes.xyxy.tolist()

    # Lists to store detected persons and vehicles separately
    persons = []
    vehicles = []

    # 1. Separates detections into persons and vehicles
    for cid, bbox in zip(class_ids, bboxes):
        label = names[int(cid)] # Gets the label name for the detected class

        if label == "person": # If detected object is a person, adds to persons list
            persons.append(bbox)
        elif label in vehicle_classes: # If it's a vehicle, stores both bbox and its label 
            vehicles.append((bbox, label)) 

    violations = [] # List to store proximity violations

    # 2. Checks proximity between each person and each vehicle
    for person_bbox in persons:
        for vehicle_bbox, vehicle_label in vehicles:

            # Skips if the person is likely the driver of the vehicle
            if is_driver(person_bbox, vehicle_bbox):
                continue

            # If dangerously close to the vehicle, logs as a violation
            if is_dangerously_close(person_bbox, vehicle_bbox, vehicle_label):
                violations.append({
                    "person_bbox": person_bbox,
                    "vehicle_bbox": vehicle_bbox,
                    "vehicle_label": vehicle_label
                })

    # 3. Returns violation results
    if violations:
        return {
            "proximity_violations": violations,
            "violation_count": len(violations)
        }
    else:
        return None

	
	
def is_driver(person_bbox: list[float], vehicle_bbox: list[float]) -> bool:
        """
        Checks if a person is a driver based on position near a vehicle.

        Args:
            person_bbox (List[float]): Bounding box of person.
            vehicle_bbox (List[float]): Bounding box of vehicle.

        Returns:
            bool: True if the person is likely the driver, False otherwise.
        """
        # Extracts coordinates and dimensions of person and vehicle boxes
        person_bottom_y = person_bbox[3]
        person_top_y = person_bbox[1]
        person_left_x = person_bbox[0]
        person_right_x = person_bbox[2]
        person_width = person_bbox[2] - person_bbox[0]
        person_height = person_bbox[3] - person_bbox[1]

        vehicle_top_y = vehicle_bbox[1]
        vehicle_bottom_y = vehicle_bbox[3]
        vehicle_left_x = vehicle_bbox[0]
        vehicle_right_x = vehicle_bbox[2]
        vehicle_height = vehicle_bbox[3] - vehicle_bbox[1]

        # 1. Checks vertical bottom position: person's bottom should be above
        #    the vehicle's bottom by at least half the person's height
        if not (
            person_bottom_y < vehicle_bottom_y
            and vehicle_bottom_y - person_bottom_y >= person_height / 2
        ):
            return False

        # 2. Checks horizontal position: person's edges should not extend
        #    beyond half the width of the person from the vehicle's edges
        if not (
            person_left_x >= vehicle_left_x - person_width / 2
            and person_right_x <= vehicle_right_x + person_width / 2
        ):
            return False

        # 3. The person's top must be below the vehicle's top
        if not (person_top_y > vehicle_top_y):
            return False

        # 4. Person's height is less than or equal to half the vehicle's height
        if not (person_height <= vehicle_height / 2):
            return False

        return True


def is_dangerously_close(
        person_bbox: list[float],
        vehicle_bbox: list[float],
        label: str,
    ) -> bool:
        """
        Determines if a person is dangerously close to machinery or vehicles.

        Args:
            person_bbox (list[float]): Bounding box of person.
            vehicle_bbox (list[float]): Machine/vehicle box.
            label (str): Type of the second object ('machinery' or 'vehicle').

        Returns:
            bool: True if the person is dangerously close, False otherwise.
        """
        # Calculates dimensions of the person bounding box
        person_width = person_bbox[2] - person_bbox[0]
        person_height = person_bbox[3] - person_bbox[1]
        person_area = person_width * person_height

        # Calculates the area of the vehicle bounding box
        vehicle_area = (vehicle_bbox[2] - vehicle_bbox[0]) * \
            (vehicle_bbox[3] - vehicle_bbox[1])
        acceptable_ratio = 0.1 if label in vehicle_classes else 0.05

        # Checks if person area ratio is acceptable compared to vehicle area
        if person_area / vehicle_area > acceptable_ratio:
            return False

        # Defines danger distances
        danger_distance_horizontal = 5 * person_width
        danger_distance_vertical = 1.5 * person_height

        # Calculates min horizontal/vertical distance between person and vehicle
        horizontal_distance = min(
            abs(person_bbox[2] - vehicle_bbox[0]),
            abs(person_bbox[0] - vehicle_bbox[2]),
        )
        vertical_distance = min(
            abs(person_bbox[3] - vehicle_bbox[1]),
            abs(person_bbox[1] - vehicle_bbox[3]),
        )

        # Determines if the person is dangerously close
        return (
            horizontal_distance <= danger_distance_horizontal
            and vertical_distance <= danger_distance_vertical
        )	

def frame_to_timestamp(frame_idx, fps):
 """
    Converts a given frame index to a timestamp in the format HH:MM:SS,MMM.

    Args:
        frame_idx (int): The index of the frame (starting from 0).
        fps (float): Frames per second of the video.

    Returns:
        str: Timestamp string in the format 'HH:MM:SS,mmm' 
             where HH = hours, MM = minutes, SS = seconds, mmm = milliseconds.
    
    Example:
        If frame_idx = 75 and fps = 25, the output will be "00:00:03,000"
    """  
    total_seconds = frame_idx / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds * 1000) % 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
        
def save_violation_to_json_video(video_path: str, violations_log: list, json_dir: str):
    """
    Saves the proximity violations detected in a video into a JSON file with timestamp.

    Args:
        video_path (str): Path to the processed video.
        violations_log (list): List of violations with frame info.
        json_dir (str): Directory where JSON files are saved.
    """
    base_filename = os.path.basename(video_path)
    json_filename = os.path.splitext(base_filename)[0] + ".json"
    save_path = os.path.join(json_dir, json_filename)

    current_time = datetime.now().isoformat()

    data = {
        "timestamp": current_time,
        "video_name": base_filename,
        "total_violation_events": len(violations_log),
        "violations": violations_log
    }

    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f" Violation log saved: {save_path}")
        
def predict_video(video_path):
 """
    Processes a given video file, detects Proximity violations per frame, and saves the results into a JSON file and avi file with boxes.

    Args:
        video_path (str): Path to the input video file.

    This function:
        - Runs inference frame by frame on the video.
        - Detects if any Proximity violations occur.
        - Logs the violations with timestamps (start and end time).
        - Saves the logged violations into a JSON file.
    """
    try:
        print(f"Processing: {os.path.basename(video_path)}")
        start = time.time()
        results = model.predict(source=video_path, **predict_params)# Runs model prediction on the video
        
        fps = 30 # default frame rate of 30 FPS
        frame_idx = 0 # Initializes frame counter
        violations_log = [] # Lists to store detected violations per frame

        # Iterates through each prediction result per frame
        for r in results:
            violation_info = detect_proximity_violation(r)# Checks if there is a Proximity violation in the frame

            if violation_info:# if there is a violation 
                
                start_time = frame_to_timestamp(frame_idx, fps) # Converts current frame index to timestamp
                end_time = frame_to_timestamp(frame_idx + 5, fps) # Assumes violation lasts for 5 frames 

                # add violation to the log list
                violation_info_full = { # Prepares the violation record
                    "frame": frame_idx,
                    "start_time": start_time,
                    "end_time": end_time,
                    "violation_count": violation_info["violation_count"],
                    "proximity_violations": violation_info["proximity_violations"]
                }
                violations_log.append(violation_info_full)

            _ = r # Consumes the result to avoid memory buildup
            frame_idx += 1 # Moves to next frame

        
        if violations_log: # if proximity violation is detected ,saves json log.
            save_violation_to_json_video(video_path, violations_log, output_dir)

        memory_cleanup()  # Cleans up memory after processing
        print(f"Completed: {time.time() - start:.2f}s")

    except Exception as e:
        print(f" Error: {str(e)}")
        memory_cleanup()


# Entry point of the script
if __name__ == "__main__":
    for video_path in video_paths: # Iterates over all video files found in the specified directory
        predict_video(video_path) # Processes each video individually

    print("\n All videos processed successfully using GPU.") 		