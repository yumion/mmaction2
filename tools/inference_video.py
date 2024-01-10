import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from mmaction.apis import inference_recognizer, init_recognizer
from segmental_score import _accuracy, _f1k
from tqdm import tqdm


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
    print(f"{video_path} is being predicted...")

    results = inferencer.predict_on_video(
        video_path,
        args.num_input_frames,
        args.frame_intervals,
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
    segmental_f1 = _f1k(
        preds, annotation[: len(preds)], n_classes=inferencer.num_classes, overlap=args.tolerance
    )
    frame_acc = _accuracy(preds, annotation[: len(preds)])
    scores = {
        "video": video_path,
        "total": len(preds),
        f"segmental_f1_score@{int(args.tolerance*100)}": segmental_f1,
        "accuracy": frame_acc,
    }
    print(scores)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with (args.out.parent / f"scores_{video_path.stem}.json").open("w") as fw:
            json.dump(scores, fw, indent=4)


class WholeVideoInferencer:
    def __init__(self, config_path: Path, checkpoint_path: Path, gpu_id: int = 0):
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

    def predict_on_video(
        self,
        video_path: Path,
        num_input_frames: int,
        frame_intervals: int,
        num_overlap: int = 0,
    ) -> Dict[str, np.ndarray]:
        segment_len = num_input_frames * frame_intervals
        num_overlap = min(num_overlap, segment_len - 1)  # clip maximum of num_overlap
        inputs = self._preprocess_data(video_path, segment_len, num_overlap)
        if num_overlap == 0:
            all_scores = []
            all_labels = []
        else:
            all_scores = np.full(
                (self.total_frames, self.num_classes, num_overlap), np.nan, dtype=np.float32
            )
            all_labels = np.full((self.total_frames, num_overlap), np.nan, dtype=np.float16)

        pbar = tqdm(inputs)
        for i, (start_index, segment_len) in enumerate(pbar):
            pbar.set_description(f"{video_path.name} - {start_index:06d}")

            output = self.predict_on_segment(video_path, start_index, segment_len)
            if num_overlap == 0:
                all_scores.extend(output["pred_score"])
                all_labels.extend(output["pred_label"])
            else:
                _idx = i % num_overlap
                all_scores[start_index : start_index + segment_len, :, _idx] = output["pred_score"]
                all_labels[start_index : start_index + segment_len, _idx] = output["pred_label"]

            pbar.set_postfix(
                pred_label=output["pred_label"][0], pred_score=output["pred_score"][0]
            )

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

    def _preprocess_data(
        self,
        video_path: Path,
        segment_len: int,
        num_overlap: int,
    ) -> List[Tuple[int, int]]:
        self.total_frames = int(cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FRAME_COUNT))
        # calculate how many segments can be divided
        num_segments = self.total_frames // (segment_len - num_overlap)
        remainder = self.total_frames - num_segments * (segment_len - num_overlap)
        # (start_index, segment_len)
        # NOTE: change segment_len if remainder < segment_len in order not to beyond total_frames
        inputs = [
            (
                i * (segment_len - num_overlap),
                min(segment_len, self.total_frames - i * (segment_len - num_overlap)),
            )
            for i in range(num_segments)
        ]
        # add remainder as last segment
        if remainder > 0:
            inputs += [(num_segments * (segment_len - num_overlap), remainder)]
        return inputs

    def predict_on_segment(
        self, video_path: Path, start_index: int, segment_len: int
    ) -> Dict[str, List[Union[int, float]]]:
        data = dict(
            filename=str(video_path),
            label=-1,
            start_index=start_index,
            total_frames=segment_len,
            modality="RGB",
        )
        result = inference_recognizer(self.model, data)
        return {
            "pred_score": [result.pred_score.tolist()] * segment_len,  # (N,C)
            "pred_label": result.pred_label.tolist() * segment_len,  # (N,)
        }


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
