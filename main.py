import os
import argparse

from model import Model
from spark_application import SparkApplication
from utils import create_empty_dir
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from pathlib import Path


MODEL_PATH = "./models/train/weights/best.pt"
RESULT_DIR = "./results"
INTERVAL_SEC = 0.5
SPARK_APP_NAME = "LongestHitSeries"


def run_model(args):
    video, model_path, interval, result_dir = args
    model = Model(model_path)
    model.process_video(video, interval, result_dir)


def run_spark(appName, result_dir):
    sparkApp = SparkApplication(appName)
    sparkApp.analyze_csv(result_dir)
    sparkApp.stop()


def main():
    parser = argparse.ArgumentParser(description="Laboratory work 4")
    parser.add_argument("path", type=str, help="Path to folder with videos")

    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        print(f"Path {path} is not exists")
        return

    if not path.is_dir():
        print(f"Path {path} is not directory")
        return

    if not any(path.iterdir()):
        print(f"Folder {path} is empty")
        return

    video_files = glob(os.path.join(path, "*.MP4"))

    create_empty_dir(RESULT_DIR)

    args_list = [(video, MODEL_PATH, INTERVAL_SEC, RESULT_DIR) for video in video_files]
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(run_model, args_list)

    run_spark(SPARK_APP_NAME, RESULT_DIR)


if __name__ == "__main__":
    main()
