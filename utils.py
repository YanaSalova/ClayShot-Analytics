import os
import shutil


def create_empty_dir(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)


def create_csv_record(
    video_filename, timestamp_sec, cls_id=None, xyxy=None, conf=None, target_id=None
):
    if cls_id is None:
        target_color = "None"
        target_state = "None"
    else:
        color_map = {0: "Dark", 1: "Dark", 2: "Red", 3: "Red"}
        state_map = {0: "Broken", 1: "Entire", 2: "Broken", 3: "Entire"}
        target_color = color_map.get(cls_id)
        target_state = state_map.get(cls_id)

    conf = "None" if conf is None else round(conf, 3)
    target_id = "None" if target_id is None else target_id

    if xyxy is None:
        pos_x = pos_y = width = height = "None"
    else:
        x1, y1, x2, y2 = xyxy
        pos_x = round(x1, 2)
        pos_y = round(y1, 2)
        width = round(x2 - x1, 2)
        height = round(y2 - y1, 2)

    return {
        "video_file": video_filename,
        "timestamp_sec": round(timestamp_sec, 3),
        "target_id": target_id,
        "target_color": target_color,
        "target_state": target_state,
        "pos_x": pos_x,
        "pos_y": pos_y,
        "width": width,
        "height": height,
        "confidence": conf,
    }
