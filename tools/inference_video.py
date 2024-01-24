import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from mmaction.apis import init_recognizer
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from segmental_score import (
    frame_accuracy,
    frame_precision_recall,
    segmental_f1score,
    segmental_precision_recall,
)
from tqdm import trange


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MMAction2 test (and eval) a video")
    parser.add_argument("config", type=Path, help="test config file path")
    parser.add_argument("checkpoint", type=Path, help="checkpoint file")
    parser.add_argument("video", type=Path, help="video file")
    parser.add_argument("annotation", type=Path, help="annotation file")
    parser.add_argument("--out", type=Path, help="output result file")
    parser.add_argument(
        "--out-items",
        "--out_items",
        choices=["label", "score"],
        nargs="+",
        default=["label"],
        help="class label or/and confidence score",
    )
    parser.add_argument("--num-input-frames", "--num_input_frames", type=int, default=32)
    parser.add_argument("--frame-intervals", "--frame_intervals", type=int, default=1)
    parser.add_argument("--num-overlap", "--num_overlap", type=int, default=0)
    parser.add_argument(
        "--out-of-bound-opt",
        "--out_of_bound_opt",
        type=str,
        default="repeat_last",
        help="repeat last frame of loop clip for last clip",
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.1, help="tolerance for segmental score"
    )
    parser.add_argument("--gpu-id", "--gpu_id", type=int, default=0)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    inferencer = WholeVideoInferencer(
        Path(args.config),
        Path(args.checkpoint),
        gpu_id=args.gpu_id,
    )

    video_path = args.video
    # NOTE: MPC-HCはframe indexが1から始まるが、opencvは0から始まるので1を引く
    start_frame = int(args.annotation.read_text().split("\n")[0].split(" ")[0]) - 1
    end_frame = int(args.annotation.read_text().split("\n")[-2].split(" ")[1]) - 1
    print(f"{video_path} is being predicted...")

    results = inferencer.predict_on_video(
        video_path,
        args.num_input_frames,
        args.frame_intervals,
        start_frame=start_frame,
        end_frame=end_frame,
        num_overlap=args.num_overlap,
    )

    preds = results["pred_labels"]
    confidences = results["pred_scores"]

    if args.out and "label" in args.out_items:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as fw:
            fw.write("\n".join(map(str, preds.tolist())))

    if args.out and "score" in args.out_items:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.with_suffix(".score").open("w") as fw:
            writer = csv.writer(fw)
            writer.writerows(confidences.tolist())

    annotation = convert_start_end2continues(args.annotation, inferencer.classes)
    frame_acc = frame_accuracy(preds, annotation[: len(preds)])
    frame_prec_rec = frame_precision_recall(preds, annotation[: len(preds)])
    segmental_f1 = segmental_f1score(
        preds, annotation[: len(preds)], n_classes=inferencer.num_classes, overlap=args.tolerance
    )
    segmental_prec_rec = segmental_precision_recall(
        preds, annotation[: len(preds)], n_classes=inferencer.num_classes, overlap=args.tolerance
    )
    scores = {
        "video": video_path.stem,
        "total": len(preds),
        "accuracy": frame_acc,
        "precision": frame_prec_rec[0],
        "recall": frame_prec_rec[1],
        f"segmental_precision@{int(args.tolerance*100)}": segmental_prec_rec[0],
        f"segmental_recall@{int(args.tolerance*100)}": segmental_prec_rec[1],
        f"segmental_f1_score@{int(args.tolerance*100)}": segmental_f1,
    }
    print(scores)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with (args.out.parent / f"scores_{video_path.stem}.json").open("w") as fw:
            json.dump(scores, fw, indent=4)


class WholeVideoInferencer:
    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        gpu_id: int = 0,
    ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu"
        self._init_recognizer()
        self.num_classes = self.model.cls_head.num_classes
        self.classes = self.model.cfg.get("classes")

    def _init_recognizer(self) -> None:
        self.model = init_recognizer(
            self.config_path,
            str(self.checkpoint_path),
            device=self.device,
        )
        cfg = self.model.cfg
        init_default_scope(cfg.get("default_scope", "mmaction"))
        # ignore first 3 pipelines (OpenCVInit, SampleFrames, OpenCVDecode)
        # start from Resize
        self.test_pipeline = Compose(cfg.test_pipeline[3:])

    def predict_on_video(
        self,
        video_path: Path,
        num_input_frames: int,
        frame_intervals: int,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        num_overlap: int = 0,
        out_of_bound_opt: str = "repeat_last",
    ) -> Dict[str, np.ndarray]:
        assert out_of_bound_opt in ["repeat_last", "loop"]

        video = cv2.VideoCapture(str(video_path))
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        if end_frame is None:
            end_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = end_frame - start_frame + 1

        segment_len = num_input_frames * frame_intervals
        num_overlap = min(num_overlap, num_input_frames - 1)  # clip maximum of num_overlap

        if num_overlap == 0:
            all_scores = []
            all_labels = []
        else:
            all_scores = np.full(
                (total_frames, self.num_classes, num_overlap), np.nan, dtype=np.float32
            )
            all_labels = np.full((total_frames, num_overlap), np.nan, dtype=np.float16)

        pbar = trange(total_frames)
        frames = []
        for i in pbar:
            pbar.set_description(f"{video_path.name}/frame_{i + start_frame:06d}")

            ret, frame = video.read()
            if not ret:
                break

            # skip frame
            if i % frame_intervals != 0:
                continue

            # stack frames for model input
            frames.append(frame)
            if len(frames) < num_input_frames:
                continue

            output = self.predict_on_segment(frames, segment_len)
            if num_overlap == 0:
                all_scores.extend(output["pred_score"])
                all_labels.extend(output["pred_label"])
            else:
                ring_idx = i % num_overlap
                all_scores[i : i + segment_len, :, ring_idx] = output["pred_score"]
                all_labels[i : i + segment_len, ring_idx] = output["pred_label"]

            # reset inputs
            if num_overlap == 0:
                frames = []
            else:
                frames = frames[-num_overlap:]

            pbar.set_postfix(
                pred_label=output["pred_label"][0],
                # limiting displayed decimal places
                pred_score=list(map(lambda x: float(f"{x:.2f}"), output["pred_score"][0])),
            )

        # predict on last clip
        if len(frames) > num_overlap:
            # repeat current frames for last clip
            last_frames = self._augment_last_clip(frames, num_input_frames, out_of_bound_opt)
            output = self.predict_on_segment(last_frames, total_frames - len(all_scores))
            if num_overlap == 0:
                all_scores.extend(output["pred_score"])
                all_labels.extend(output["pred_label"])
            else:
                ring_idx = i % num_overlap
                all_scores[i : i + segment_len, :, ring_idx] = output["pred_score"]
                all_labels[i : i + segment_len, ring_idx] = output["pred_label"]

        if num_overlap == 0:
            return {
                "pred_scores": np.array(all_scores),  # (N,C)
                "pred_labels": np.array(all_labels),  # (N,)
            }
        else:
            # Compute average of scores and mode of labels for each frame
            return {
                "pred_scores": np.nanmean(all_scores, axis=-1),  # (N,C,segment) -> (N,C)
                "pred_labels": np_nanmode(all_labels, axis=-1),  # (N,segment) -> (N,)
            }

    def predict_on_segment(
        self, imgs: List[np.ndarray], segment_len: int
    ) -> Dict[str, List[Union[int, float]]]:
        # to_tensor
        data = self._prepare_data(imgs)
        batch = pseudo_collate([self.test_pipeline(data)])
        # Forward the model
        with torch.no_grad():
            result = self.model.test_step(batch)[0]
        return {
            "pred_score": [result.pred_score.tolist()] * segment_len,  # (N,C)
            "pred_label": result.pred_label.tolist() * segment_len,  # (N,)
        }

    def _prepare_data(
        self, imgs: List[np.ndarray]
    ) -> Dict[str, Union[List[np.ndarray], Tuple[int, int], int]]:
        imgs = np.array(imgs)
        # The default channel order of OpenCV is BGR, thus we change it to RGB
        imgs = imgs[:, :, :, ::-1]
        data = {
            "imgs": list(imgs),
            "original_shape": imgs[0].shape[:2],
            "img_shape": imgs[0].shape[:2],
            "clip_len": len(imgs),
            "num_clips": 1,
        }
        return data

    def _augment_last_clip(
        self,
        frames: List[np.ndarray],
        num_input_frames: int,
        out_of_bound_opt: str,
    ):
        if out_of_bound_opt == "repeat_last":
            frames.extend([frames[-1]] * (num_input_frames - len(frames)))
        elif out_of_bound_opt == "loop":
            frames.extend(frames[: num_input_frames - len(frames)])
        return frames


def convert_start_end2continues(
    annotation_file: Path, classes: Optional[Sequence[str]]
) -> np.ndarray:
    annotation = []
    with annotation_file.open() as fr:
        reader = csv.reader(fr, delimiter=" ")
        for row in reader:
            start, end, classname, *_ = row
            for _ in range(int(end) - int(start)):
                if classes is not None:
                    label = classes.index(classname)
                else:
                    label = int(classname)
                annotation.append(label)
    return np.array(annotation)


def np_nanmode(arr: Sequence, axis: Optional[int] = None) -> np.ndarray:
    arr = np.array(arr)
    num_axises = arr.ndim
    assert axis is None or axis < num_axises, f"axis must be less than {num_axises}"
    if axis is None:
        arr = arr.flatten()
        axis = 0
    return np.apply_along_axis(lambda x: _mode(x), axis, arr)


def _mode(arr: np.ndarray) -> int:
    uniques, counts = np.unique(arr, return_counts=True)
    mask = ~np.isnan(uniques)
    counts = counts[mask]
    uniques = uniques[mask]
    if len(counts) == 0:
        return np.nan
    return uniques[np.argmax(counts)]


if __name__ == "__main__":
    main()
