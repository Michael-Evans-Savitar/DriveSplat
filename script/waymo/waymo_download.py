import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import List


def download_file(filename: str, target_dir: str, source: str) -> None:
    result = subprocess.run(
        [
            "gsutil",
            "cp",
            "-n",
            f"{source}/{filename}.tfrecord",
            target_dir,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)


def download_files(file_names: List[str], target_dir: str, source: str) -> None:
    total_files = len(file_names)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(download_file, filename, target_dir, source)
            for filename in file_names
        ]

        for counter, future in enumerate(futures, start=1):
            try:
                future.result()
                print(f"[{counter}/{total_files}] Downloaded successfully!")
            except Exception as e:
                print(f"[{counter}/{total_files}] Failed to download. Error: {e}")


def read_segment_names(segment_file: str) -> List[str]:
    with open(segment_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def read_scene_ids(split_file: str) -> List[int]:
    with open(split_file, "r") as f:
        lines = f.readlines()[1:]
    return [int(line.strip().split(",")[0]) for line in lines if line.strip()]


if __name__ == "__main__":
    print("note: `gcloud auth login` is required before running this script")
    print("Downloading Waymo dataset from Google Cloud Storage...")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_dir",
        type=str,
        default="data/waymo/raw",
        help="Path to the target directory",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="gs://waymo_open_dataset_scene_flow/valid",
        help="Waymo GCS source, e.g. gs://waymo_open_dataset_scene_flow/train or valid",
    )
    parser.add_argument(
        "--segment_file",
        type=str,
        default="script/waymo/waymo_val_list.txt",
        help="Segment list matching --source",
    )
    parser.add_argument(
        "--scene_ids",
        type=int,
        nargs="+",
        default=None,
        help="Scene ids to download",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="CSV split file whose first column contains scene ids",
    )
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)
    total_list = read_segment_names(args.segment_file)

    if args.split_file is not None:
        scene_ids = read_scene_ids(args.split_file)
    elif args.scene_ids is not None:
        scene_ids = args.scene_ids
    else:
        raise ValueError("Either --scene_ids or --split_file must be provided.")

    missing = [scene_id for scene_id in scene_ids if scene_id < 0 or scene_id >= len(total_list)]
    if missing:
        raise ValueError(f"Scene ids out of range for {args.segment_file}: {missing}")

    file_names = [total_list[i] for i in scene_ids]
    download_files(file_names, args.target_dir, args.source)
