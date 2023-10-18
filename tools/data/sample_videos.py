import argparse
import concurrent.futures as cf
from pathlib import Path

import cv2
from tqdm import tqdm, trange


def sample_video(
    video_path: Path,
    extract_dir: Path,
    sampling_period: int = 6,
    jobs: int = 1,
) -> None:
    vid = cv2.VideoCapture(str(video_path))
    extract_dir.mkdir(exist_ok=True)
    n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    parallel_saver = cf.ThreadPoolExecutor(max_workers=jobs)

    for frame_idx in trange(n_frames, desc=video_path.name, position=1, leave=False):
        _, frame = vid.read()
        # print(frame_idx//period, frame_idx%period)
        if frame_idx % sampling_period == 0:
            parallel_saver.submit(cv2.imwrite, str(extract_dir / f"{frame_idx:09d}.png"), frame)
    vid.release()


def main(args):
    # find all the files that need to be processed
    if not args.recursive:
        video_dirs = [Path(args.video_dir).resolve()]
    else:
        video_dirs = [v_p.parent for v_p in Path(args.video_dir).rglob(f"*.{args.suffix}")]

    for directory in tqdm(video_dirs, desc="unpacking dataset", position=0):
        if args.video_name is None:
            # in case directory name and video file is same name
            video_name = f"{directory.name}.{args.suffix}"
        else:
            # in case all video file is same name
            video_name = f"{args.video_name}.{args.suffix}"

        # validate video path
        if not (directory.exists() and (directory / video_name).exists()):
            print(
                f"{directory} does not a video directory. \
                    please make sure video directory path is correct"
            )

        rgb_dir = directory / args.image_dirname

        if rgb_dir.exists() and not args.overwrite:
            print(f"{video_name} is already packed.")
            continue

        sample_video(directory / video_name, rgb_dir, args.sampiling_period, args.jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", help="path pointing to the video directory")
    parser.add_argument(
        "-n",
        "--sampiling-period",
        help="number of sampling frames from a video",
        default=6,
        type=int,
    )
    parser.add_argument(
        "-r",
        "--recursive",
        help="search recursively for video directories that have video_left.avi as a child",
        action="store_true",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        help="number of parallel works to use when saving images",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--overwrite",
        help="overwrite images",
        action="store_true",
    )
    parser.add_argument(
        "--video-name",
        help="video file name. It requires the same name for all videos.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--suffix",
        help="suffix of video without dot.",
        default="mp4",
        type=str,
    )
    parser.add_argument(
        "--image-dirname",
        help="directory name to save frame sampled.",
        default="rgb",
        type=str,
    )

    SystemExit(main(parser.parse_args()))
