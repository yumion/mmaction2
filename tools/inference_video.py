import argparse
from pathlib import Path

import cv2
import numpy as np
from mmaction.apis import inference_recognizer, init_recognizer
from segmental_score import _f1k
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="MMAction2 test (and eval) a video")
    parser.add_argument("config", type=Path, help="test config file path")
    parser.add_argument("checkpoint", type=Path, help="checkpoint file")
    parser.add_argument("video", type=Path, help="video file")
    parser.add_argument("--num-input-frames", "--num_input_frames", type=int, default=32)
    parser.add_argument("--frame-intervals", "--frame_intervals", type=int, default=2)
    parser.add_argument("--num-overlap", "--num_overlap", type=int, default=0)
    parser.add_argument("--gpu-id", "--gpu_id", type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    video_path = args.video
    inferencer = WholeVideoInferencer(
        Path(args.config),
        Path(args.checkpoint),
        gpu_id=args.gpu_id,
    )

    results = inferencer.predict_on_video(
        video_path,
        args.num_input_frames,
        args.frame_intervals,
        num_overlap=args.num_overlap,
    )

    preds = results["pred_labels"]
    confidences = results["pred_scores"][:, :8]

    print(confidences.shape, preds.shape)


class WholeVideoInferencer:
    def __init__(self, config_path: Path, checkpoint_path: Path, gpu_id: int = 0):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu"
        self._init_recognizer()

    def _init_recognizer(self):
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
    ):
        results = {"pred_scores": [], "pred_labels": []}
        inputs = self._preprocess_data(video_path, num_input_frames, frame_intervals, num_overlap)
        pbar = tqdm(inputs)
        for i, (start_index, segment_len) in enumerate(pbar):
            pbar.set_description(f"{video_path.name} - {start_index:06d}")

            result = self.predict_on_segment(video_path, start_index, segment_len)
            results["pred_scores"].extend(result["pred_score"])
            results["pred_labels"].extend(result["pred_label"])

            pbar.set_postfix(
                pred_label=result["pred_label"][0], pred_score=result["pred_score"][0]
            )
            if i > 2:
                break

        results["pred_scores"] = np.array(results["pred_scores"])
        results["pred_labels"] = np.array(results["pred_labels"])
        return results

    def _preprocess_data(
        self,
        video_path: Path,
        num_input_frames: int,
        frame_intervals: int,
        num_overlap: int = 0,
    ):
        segment_len = num_input_frames * frame_intervals
        total_frames = int(cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FRAME_COUNT))
        num_subsegments = total_frames // segment_len
        remainder = total_frames - num_subsegments * segment_len

        inputs = [(i * segment_len, segment_len) for i in range(num_subsegments)]
        if remainder > 0:
            inputs += [(num_subsegments * segment_len, remainder)]
        return inputs

    def predict_on_segment(self, video_path: Path, start_index: int, segment_len: int):
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


if __name__ == "__main__":
    main()
