import logging
from glob import glob
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from mmengine.fileio import exists, list_from_file
from mmengine.logging import print_log

from .base import BaseActionDataset


@DATASETS.register_module()
class VideoWorkflowDataset(BaseActionDataset):
    """
    Example of a annotation file like action_continues.txt

    .. code-block:: txt
        start-frame end-frame class_id

            435098 435545 1
            435546 436502 2
            436503 437438 3
            437439 437804 1
            437805 438109 2

    Args:
        ann_file (str): Annotation file name.
        pipeline (list[dict | callable]): A sequence of data transforms.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(
        self,
        ann_file: str,
        classes: Sequence[str],
        pipeline: List[Union[ConfigType, Callable]],
        data_root: str,
        data_prefix: ConfigType = dict(video=""),
        video_name: Optional[str] = None,
        suffix: str = ".mp4",
        segment_len: Optional[int] = None,
        drop_last: bool = False,
        delimiter: str = " ",
        test_mode: bool = False,
        **kwargs,
    ):
        # common arguments
        self.classes = classes
        self.segment_len = segment_len
        self.drop_last = drop_last
        self.delimiter = delimiter
        # for video file
        self.video_name = video_name
        self.suffix = suffix if video_name is None else video_name.split(".")[-1]
        # ignore `_join_prefix()`
        super().__init__(
            ann_file,
            pipeline=pipeline,
            test_mode=test_mode,
            multi_class=False,
            num_classes=len(classes),
            modality="RGB",
            data_root=None,
            lazy_init=True,
            **kwargs,
        )
        # overwrite path
        self.ann_file = ann_file
        self.data_root = Path(data_root)
        self.data_prefix = data_prefix
        if not kwargs.get("lazy_init", False):
            self.full_init()

    def load_data_list(self) -> List[dict]:
        # search annotation file for each video
        ann_file_paths = glob(str(self.data_root / f"**/{self.ann_file}"), recursive=True)

        data_list = []
        for ann_file_path in ann_file_paths:
            data_list.extend(self._load_annotation(Path(ann_file_path)))
        print_log(
            f"Loaded {len(data_list)} clips from {self.ann_file}",
            logger="current",
            level=logging.INFO,
        )
        return data_list

    def _load_annotation(self, ann_file_path: Path) -> List[dict]:
        exists(ann_file_path)

        if self.video_name is not None:
            video_path = ann_file_path.parent / self.data_prefix["video"] / self.video_name
        else:
            video_path = (
                ann_file_path.parent / self.data_prefix["video"] / ann_file_path.parent.name
            ).with_suffix(self.suffix)

        return self._get_video_infos(ann_file_path, filename=video_path)

    def _get_video_infos(self, ann_file_path: Path, frame_dir: Path = None, filename: Path = None):
        assert (
            frame_dir is not None or filename is not None,
            "frame_dir or filename must be specified",
        )

        fin = list_from_file(ann_file_path)
        video_infos = []
        for line in fin:
            line_split = line.strip().split(self.delimiter)

            start_index = int(line_split[0])
            end_index = int(line_split[1])
            classname = line_split[2]
            label = self.classes.index(classname)

            total_frames = end_index - start_index
            segment_len = self.segment_len or total_frames
            num_subsegments = total_frames // segment_len
            remainder = total_frames - num_subsegments * segment_len
            for i in range(num_subsegments):
                video_info = {
                    "_start_index": start_index + i * segment_len,  # avoid overwrite start_index
                    "total_frames": segment_len,
                    "label": label,
                    "classname": classname,
                }
                if frame_dir is not None:
                    video_info["frame_dir"] = str(frame_dir)
                if filename is not None:
                    video_info["filename"] = str(filename)
                video_infos.append(video_info)
            if not self.drop_last:
                video_info = {
                    "_start_index": start_index + num_subsegments * segment_len,
                    "total_frames": remainder,
                    "label": label,
                    "classname": classname,
                }
                if frame_dir is not None:
                    video_info["frame_dir"] = str(frame_dir)
                if filename is not None:
                    video_info["filename"] = str(filename)
                video_infos.append(video_info)
        return video_infos

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)
        # overwrite start_index defined at init
        data_info["start_index"] = data_info["_start_index"]
        return data_info


@DATASETS.register_module()
class ImageWorkflowDataset(VideoWorkflowDataset):
    """
    Example of a annotation file like action_continues.txt

    .. code-block:: txt
        start-frame end-frame class_id

            435098 435545 1
            435546 436502 2
            436503 437438 3
            437439 437804 1
            437805 438109 2

    Args:
        ann_file (str): Annotation file name.
        pipeline (list[dict | callable]): A sequence of data transforms.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(
        self,
        ann_file: str,
        classes: Sequence[str],
        pipeline: List[Union[ConfigType, Callable]],
        data_root: str,
        data_prefix: ConfigType = dict(img=""),
        filename_tmpl: str = "{:09}.png",
        segment_len: Optional[int] = None,
        drop_last: bool = False,
        delimiter: str = " ",
        test_mode: bool = False,
        **kwargs,
    ):
        # for image files
        self.filename_tmpl = filename_tmpl
        super().__init__(
            ann_file=ann_file,
            classes=classes,
            pipeline=pipeline,
            data_root=data_root,
            data_prefix=data_prefix,
            segment_len=segment_len,
            drop_last=drop_last,
            delimiter=delimiter,
            test_mode=test_mode,
            **kwargs,
        )

    def _load_annotation(self, ann_file_path: Path) -> List[dict]:
        exists(ann_file_path)

        frame_dir = ann_file_path.parent / self.data_prefix["img"]
        return self._get_video_infos(ann_file_path, frame_dir=frame_dir)

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)
        data_info["filename_tmpl"] = self.filename_tmpl
        return data_info
