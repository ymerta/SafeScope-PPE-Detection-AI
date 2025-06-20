# Author: Arda Baran
# Description: This script loads yolov9-e object detection model trained for personal protective equipment (PPE) and construction-related items.
# It processes all videos in the specified directory, detects PPE violations such as missing helmets, vests, suits, and other soft equipment.
# It distinguishes between critical and soft PPE violations, logs violations frame-by-frame, and exports the results into structured JSON files.
# The detection helps improve construction site worker safety by automatically identifying missing PPE in recorded videos.
from ultralytics import YOLO
import torch
import os
import glob
import gc
import time
import json


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
    'project': output_dir, #output directory where the predicted videos are saved with boxes and json logs. 
    'name': '' # No sub-folder
}


def memory_cleanup():
                     """
  Memory clean-up function to prevent out of memory errors.

                     """                       
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()



def detect_violation(result):
"""
This function detects if there is a PPE violation for construction workers. PPE violations are categorized into two types: 
Critical PPE Violations and Soft PPE Violations.

Critical PPE Violation Items:
- Helmet
- Safety Suit
- Safety Vest

Soft PPE Violation Items:
- Gloves
- Glasses
- Face Mask
- Face Guard
- Ear Muffs

For the safety of construction workers, it is mandatory that all workers wear their appropriate PPE equipment at all times. 
Missing critical PPE items can lead to severe injuries, permanent disabilities, or even fatalities. Research studies conducted 
in the field of occupational safety have demonstrated that the absence of critical PPE such as helmets, safety vests, and safety 
suits accounts for the majority of fatal accidents at construction sites.

Soft PPE, while not immediately life-saving in all scenarios, plays a crucial role in preventing long-term health issues, 
minor injuries, or complications such as respiratory problems, eye injuries, and hand lacerations.

Therefore, this function prioritizes detection of critical PPE violations over soft PPE violations. It first checks whether 
every detected worker is properly equipped with critical PPE. If any worker is found without essential protection, it flags a 
critical violation. If all critical PPE is worn but soft PPE is missing, it flags a soft violation instead.


Safety Vest and Safety Suit are considered together because both serve similar purposes in ensuring high visibility and protecting 
the upper body of workers at construction sites. Although they are different types of garments, they both play a crucial role in 
preventing accidents by making workers easily noticeable in busy, hazardous environments. As a result, the presence of either one is
typically sufficient to meet critical safety requirements related to visibility and torso protection.


Face Mask and Face Guard are considered together because both are designed to protect the worker's face from potential hazards such as 
dust, debris, chemical splashes, and physical impacts. While a face mask primarily protects the respiratory system by filtering airborne
particles, a face guard provides a physical barrier against larger projectiles or mechanical injuries. 
Since both contribute significantly to the protection of the facial area, the presence of either one is generally considered acceptable 
for meeting essential face protection requirements on construction sites.

"""
    names = result.names # Extracts the list of class names 
    boxes = result.boxes # Extracts the detected bounding boxes and associated class IDs
    class_ids = boxes.cls.tolist() # Converts the tensor of class IDs into a list

    counts = {}  # Initializes a dictionary to count how many times each class appears
    for cid in class_ids: # Iterates over all detected class IDs
        label = names[int(cid)]  # Finds the label (class name) corresponding to the class ID
        counts[label] = counts.get(label, 0) + 1 # Increments the count of this label in the dictionary

    person_count = counts.get("person", 0) # Gets the number of detected persons; if "person" is not found, default to 0

# Helper function to evaluate how many critical or soft PPE items are detected 
# It sums up the counts for each PPE class listed in the given ppe_list  
    def evaluate_ppe(ppe_list):
        return sum([counts.get(cls, 0) for cls in ppe_list])

    if person_count == 0:
        return None  # No violation possible if there are no persons

# Initializes flags to keep track of whether critical or soft violations occur
    critical = False
    soft = False

    helmet_missing = person_count > evaluate_ppe(["helmet"])# Checks if the number of helmets detected is less than the number of persons
    vest_or_suit_missing = person_count > evaluate_ppe(["safety-vest", "safety-suit"]) # Checks if the number of safety-vests OR safety-suits detected is less than the number of persons
    gloves_missing = person_count > evaluate_ppe(["gloves"]) # Checks if the number of gloves detected is less than the number of persons
    face_mask_or_guard_missing = person_count > evaluate_ppe(["face-mask", "face-guard"]) # Checks if the number of face-masks OR face-guards detected is less than the number of persons
    glasses_missing = person_count > evaluate_ppe(["glasses"]) # Checks if the number of glasses detected is less than the number of persons
    ear_mufs_missing = person_count > evaluate_ppe(["ear-mufs"]) # Check if the number of ear-mufs detected is less than the number of persons

    if helmet_missing or vest_or_suit_missing: # If any critical PPE (helmet, vest or suit) is missing, marks it as a critical violation
        critical = True

    if gloves_missing or face_mask_or_guard_missing or glasses_missing: # If any soft PPE (gloves, face protection, glasses) is missing, marks it as a soft violation
        soft = True	

    if critical: # If a critical violation is found, prepares a detailed report in json format 
        return {
            "violation_type": "CRITICAL",
            "details": {
                "persons_detected": person_count,
                "helmet_found": evaluate_ppe(["helmet"]),
                "no_helmet": person_count - evaluate_ppe(["helmet"]),
                "safety_vest_or_suit_found": evaluate_ppe(["safety-vest", "safety-suit"]),
                "no_safety_vest_or_suit": person_count - evaluate_ppe(["safety-vest", "safety-suit"]),
                "note": "Someone is Not Wearing The Most Critical PPE items. Take action immediately!"
            }
        }
    
    elif soft: # If a soft violation is found (but no critical), prepares a separate soft violation report in json format 
        return {
            "violation_type": "SOFT",
            "details": {
                "persons_detected": person_count,
                "gloves_found": evaluate_ppe(["gloves"]),
                "face_mask_or_guard_found": evaluate_ppe(["face-mask", "face-guard"]),
                "glasses_found": evaluate_ppe(["glasses"]),
                "note": "Minor PPE items are missing, please check!"
            }
        }
    
    else: # If neither critical nor soft violations are detected, returns None
        return None


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



def predict_video(video_path):
 """
    Processes a given video file, detects PPE violations per frame, and saves the results into a JSON file and avi file with boxes.

    Args:
        video_path (str): Path to the input video file.

    This function:
        - Runs inference frame by frame on the video.
        - Detects if any PPE violations occur.
        - Logs the violations with timestamps (start and end time).
        - Saves the logged violations into a JSON file.
    """

    try:
        print(f"Processing: {os.path.basename(video_path)}")
        start = time.time()
        results = model.predict(source=video_path, **predict_params)  # Runs model prediction on the video
        
        fps = 30 # default frame rate of 30 FPS
        frame_idx = 0 # Initializes frame counter
        violations_log = [] # Lists to store detected violations per frame

        for r in results:# Iterates through each prediction result per frame
            violation_info = detect_violation(r) # Checks if there is a PPE violation in the frame

            if violation_info: # if there is a violation 
                start_time = frame_to_timestamp(frame_idx, fps) # Converts current frame index to timestamp
                end_time = frame_to_timestamp(frame_idx + 5, fps) # Assumes violation lasts for 5 frames

                violation_info_full = { # Prepares the violation record
                    "frame": frame_idx,
                    "start_time": start_time,
                    "end_time": end_time,
                    "violation_type": violation_info["violation_type"],
                    "details": violation_info["details"]
                }

                violations_log.append(violation_info_full) # Adds violation record to the log

            _ = r # Consumes the result to avoid memory buildup
            frame_idx += 1 # Moves to next frame

        if violations_log: # After all frames are processed, saves violations if any were found
            video_name = os.path.basename(video_path)
            json_name = os.path.splitext(video_name)[0] + ".json"
            json_path = os.path.join(output_dir, json_name)

            with open(json_path, "w", encoding="utf-8") as f: # Writes the violations log into a JSON file
                json.dump(violations_log, f, indent=4, ensure_ascii=False)

            print(f"JSON file created: {json_path}")
        else: # If no violations detected
            print("No PPE violations detected, no JSON generated.")

        memory_cleanup() # Cleans up memory after processing
        print(f"Completed: {time.time() - start:.2f}s")

    except Exception as e:
        print(f"Error: {str(e)}")
        memory_cleanup()


# Entry point of the script
if __name__ == "__main__":
    for video_path in video_paths: # Iterates over all video files found in the specified directory
        predict_video(video_path) # Processes each video individually

    print("\n All videos processed successfully using GPU.")