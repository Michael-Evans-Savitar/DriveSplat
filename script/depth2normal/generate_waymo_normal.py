import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Convert DriveSplat depth priors to normal priors.")
    parser.add_argument("--datadir", required=True, help="Processed scene directory containing a depth/ folder.")
    parser.add_argument("--max_depth", type=float, default=255.0)
    args = parser.parse_args()

    depth_dir = os.path.join(args.datadir, "depth")
    normal_dir = os.path.join(args.datadir, "normal")
    os.makedirs(normal_dir, exist_ok=True)

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "depth_to_normal_map.py")
    for filename in sorted(os.listdir(depth_dir)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        subprocess.run(
            [
                "python",
                script,
                "--input",
                os.path.join(depth_dir, filename),
                "--output",
                os.path.join(normal_dir, filename),
                "--max_depth",
                str(args.max_depth),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
