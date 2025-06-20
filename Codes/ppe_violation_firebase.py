# Author: Arda Baran
# Description: PPE tespiti yapan ve ihlalleri Firestore'a kaydeden YOLOv9e tabanlı video işleyici.

from ultralytics import YOLO
import torch
import os
import glob
import gc
import time
import json
import firebase_admin
from firebase_admin import credentials, firestore

# CUDA settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model yükleme
model_path = "/content/drive/MyDrive/PPE_and_Construction-Site_Detection_YOLOv11/runs/train/ppe_detection_optimized/weights/best.pt"
model = YOLO(model_path)
model.model.to(device)
model.eval()

# Video dizinleri
video_dir = "/content/drive/MyDrive/firebase_test_video_ppe"
output_dir = "/content/drive/MyDrive/firebase_results"
os.makedirs(output_dir, exist_ok=True)
video_paths = glob.glob(os.path.join(video_dir, "*.mp4"))

# Prediction ayarları
predict_params = {
    'imgsz': 640,
    'conf': 0.25,
    'device': device,
    'stream': True,
    'save': True,
    'save_txt': False,
    'save_conf': False,
    'exist_ok': True,
    'project': output_dir,
    'name': ''
}

# Firebase başlat
def firebase_init():
    if not firebase_admin._apps:
        cred = credentials.Certificate("/content/drive/MyDrive/ppe-and-proximity-detection-firebase-adminsdk-fbsvc-4499327b5e.json")
        firebase_admin.initialize_app(cred)
    return firestore.client()

# Bellek temizleme
def memory_cleanup():
    """
    Bellek sızıntılarını önlemek için GPU belleğini temizler.
    """
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

# İhlal tespiti
def detect_violation(result):
    names = result.names
    boxes = result.boxes
    class_ids = boxes.cls.tolist()

    counts = {}
    for cid in class_ids:
        label = names[int(cid)]
        counts[label] = counts.get(label, 0) + 1

    person_count = counts.get("person", 0)

    def evaluate_ppe(ppe_list):
        return sum([counts.get(cls, 0) for cls in ppe_list])

    if person_count == 0:
        return None

    critical = False
    soft = False

    helmet_missing = person_count > evaluate_ppe(["helmet"])
    vest_or_suit_missing = person_count > evaluate_ppe(["safety-vest", "safety-suit"])
    gloves_missing = person_count > evaluate_ppe(["gloves"])
    face_mask_or_guard_missing = person_count > evaluate_ppe(["face-mask", "face-guard"])
    glasses_missing = person_count > evaluate_ppe(["glasses"])
    ear_mufs_missing = person_count > evaluate_ppe(["ear-mufs"])

    if helmet_missing or vest_or_suit_missing:
        critical = True
    if gloves_missing or face_mask_or_guard_missing or glasses_missing:
        soft = True

    if critical:
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

    elif soft:
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

    else:
        return None

# Frame'i timestamp'e çevirir
def frame_to_timestamp(frame_idx, fps):
    total_seconds = frame_idx / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds * 1000) % 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

# Firestore'a veri gönderme
def upload_violations_to_firestore(video_name, violations_log, db):
    doc_ref = db.collection("ppe_violations").document(video_name)
    try:
        doc_ref.set({
            "video_name": video_name,
            "violations": violations_log,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        print(f"Uploaded to Firestore: {video_name}")
    except Exception as e:
        print(f"Firestore error: {str(e)}")

# Video tahmini
def predict_video(video_path):
    try:
        print(f"Processing: {os.path.basename(video_path)}")
        start = time.time()
        results = model.predict(source=video_path, **predict_params)

        fps = 30
        frame_idx = 0
        violations_log = []

        for r in results:
            violation_info = detect_violation(r)

            if violation_info:
                start_time = frame_to_timestamp(frame_idx, fps)
                end_time = frame_to_timestamp(frame_idx + 5, fps)

                violation_info_full = {
                    "frame": frame_idx,
                    "start_time": start_time,
                    "end_time": end_time,
                    "violation_type": violation_info["violation_type"],
                    "details": violation_info["details"]
                }

                violations_log.append(violation_info_full)

            _ = r
            frame_idx += 1

        if violations_log:
            video_name = os.path.basename(video_path)
            json_name = os.path.splitext(video_name)[0] + ".json"
            json_path = os.path.join(output_dir, json_name)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(violations_log, f, indent=4, ensure_ascii=False)

            print(f"JSON file created: {json_path}")

            db = firebase_init()
            upload_violations_to_firestore(video_name, violations_log, db)
        else:
            print("No PPE violations detected, no JSON generated.")

        memory_cleanup()
        print(f"Completed: {time.time() - start:.2f}s")

    except Exception as e:
        print(f"Error: {str(e)}")
        memory_cleanup()

# Ana giriş
if __name__ == "__main__":
    for video_path in video_paths:
        predict_video(video_path)

    print("\nAll videos processed successfully using GPU.")
