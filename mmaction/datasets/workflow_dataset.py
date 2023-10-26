from glob import glob
from pathlib import Path
from typing import Callable, List, Optional, Union

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from mmengine.fileio import exists, list_from_file

from .base import BaseActionDataset


@DATASETS.register_module()
class FrameWorkflowDataset(BaseActionDataset):
    """
    Example of a annotation file like action_discrete.txt

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
        pipeline: List[Union[ConfigType, Callable]],
        data_root: str,
        data_prefix: ConfigType = dict(img=""),
        filename_tmpl: str = "{:09}.png",
        num_classes: Optional[int] = None,
        test_mode: bool = False,
        delimiter: str = " ",
        **kwargs,
    ):
        self.delimiter = delimiter
        self.filename_tmpl = filename_tmpl
        # ignore `_join_prefix()`
        super().__init__(
            ann_file,
            pipeline=pipeline,
            test_mode=test_mode,
            multi_class=False,
            num_classes=num_classes,
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
        return data_list

    def _load_annotation(self, ann_file_path: Path) -> List[dict]:
        exists(ann_file_path)

        video_infos = []
        frame_dir = ann_file_path.parent / self.data_prefix["img"]
        fin = list_from_file(ann_file_path)
        for line in fin:
            line_split = line.strip().split(self.delimiter)

            start_index = int(line_split[0])
            end_index = int(line_split[1])
            label = int(line_split[2])

            video_info = {
                "frame_dir": str(frame_dir),
                "_start_index": start_index,  # avoid to overwrite
                "total_frames": end_index - start_index,
                "label": label,
            }
            video_infos.append(video_info)
        return video_infos

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)
        # overwrite start_index defined at init
        data_info["start_index"] = data_info["_start_index"]
        data_info["filename_tmpl"] = self.filename_tmpl
        return data_info


@DATASETS.register_module()
class VideoWorkflowDataset(FrameWorkflowDataset):
    """
    Example of a annotation file like action_discrete.txt

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
        pipeline: List[Union[ConfigType, Callable]],
        data_root: str,
        data_prefix: ConfigType = dict(video=""),
        video_name: Optional[str] = None,
        suffix: str = ".mp4",
        num_classes: Optional[int] = None,
        test_mode: bool = False,
        delimiter: str = " ",
        **kwargs,
    ):
        self.video_name = video_name
        self.suffix = suffix if video_name is None else video_name.split(".")[-1]
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            data_root=data_root,
            data_prefix=data_prefix,
            num_classes=num_classes,
            test_mode=test_mode,
            delimiter=delimiter,
            **kwargs,
        )

    def _load_annotation(self, ann_file_path: Path) -> List[dict]:
        exists(ann_file_path)

        video_infos = []
        if self.video_name is not None:
            video_path = ann_file_path.parent / self.data_prefix["video"] / self.video_name
        else:
            video_path = (
                ann_file_path.parent / self.data_prefix["video"] / ann_file_path.stem
            ).with_suffix(self.suffix)
        fin = list_from_file(ann_file_path)
        for line in fin:
            line_split = line.strip().split(self.delimiter)

            start_index = int(line_split[0])
            end_index = int(line_split[1])
            label = int(line_split[2])

            video_info = {
                "filename": str(video_path),
                "_start_index": start_index,  # avoid to overwrite
                "total_frames": end_index - start_index,
                "label": label,
            }
            video_infos.append(video_info)
        return video_infos

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super(FrameWorkflowDataset, self).get_data_info(idx)
        # overwrite start_index defined at init
        data_info["start_index"] = data_info["_start_index"]
        return data_info
