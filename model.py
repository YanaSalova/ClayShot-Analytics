import cv2
import torch
import pandas as pd
import os

from ultralytics import YOLO
from utils import create_csv_record


class Model:
    def __init__(self, model_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path).to(device)

    def process_video(self, video_path, log_interval, result_dir):
        cap = cv2.VideoCapture(video_path)

        fps = round(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = int(fps * log_interval)
        frame_idx = 0

        csv_records = []
        video_filename = os.path.basename(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp_sec = frame_idx / fps

                with torch.no_grad():
                    results = self.model.predict(source=frame, conf=0.15, verbose=False)

                added_detection = False
                boxes = results[0].boxes
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()

                    target_id = f"{video_filename}_t{int(timestamp_sec*1000)}_#{i}"

                    csv_records.append(
                        create_csv_record(
                            video_filename, timestamp_sec, cls_id, xyxy, conf, target_id
                        )
                    )

                    added_detection = True

                if not added_detection:
                    csv_records.append(create_csv_record(video_filename, timestamp_sec))

            frame_idx += 1

        cap.release()

        df = pd.DataFrame(csv_records)
        csv_path = os.path.join(result_dir, os.path.basename(video_path) + ".csv")
        df.to_csv(csv_path, index=False)
