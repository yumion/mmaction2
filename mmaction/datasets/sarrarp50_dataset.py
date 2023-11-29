# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import os.path as osp
from glob import glob

from mmaction.registry import DATASETS
from mmengine.logging import print_log

from .base import BaseActionDataset


@DATASETS.register_module()
class SarRarp50Dataset(BaseActionDataset):
    """
    Example of a annotation file like action_discrete.txt

    .. code-block:: txt
        frame_ind class_id

        00000 0
        00006 1
        00012 2
        00018 3

    Args:
        ann_file (str): Annotation file name.
        pipeline (list[dict | callable]): A sequence of data transforms.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
            Default: 'RGB'.
        dara_type (str): Type of input data. Support 'frame', 'video'
        load_start_ind (int): Start frame index to load.
            If model predict frames after index(30), set load_start_ind 30
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(
        self,
        ann_file,
        pipeline,
        data_root,
        modality="RGB",
        data_type="frame",
        load_start_ind=0,
        **kwargs,
    ):
        assert (data_type == "frame") or (data_type == "video")
        self.data_type = data_type
        # if read annotation after id:30, set load_start_ind 30
        self.load_start_ind = load_start_ind
        # ignore `_join_prefix()`
        super().__init__(
            ann_file,
            pipeline=pipeline,
            multi_class=False,
            modality=modality,
            data_root=None,
            lazy_init=True,
            **kwargs,
        )
        # overwrite path
        self.ann_file = ann_file
        self.data_root = data_root
        if not kwargs.get("lazy_init", False):
            self.full_init()

    def load_data_list(self):
        data_list = []
        ann_file_paths = glob(osp.join(self.data_root, "*", self.ann_file))

        for ann_file_path in ann_file_paths:
            data_list.extend(self._load_annotation(ann_file_path))
        print_log(f"Loaded {len(data_list)} clips", logger="current", level=logging.INFO)
        return data_list

    def _load_annotation(self, ann_file_path):
        video_infos = []

        with open(ann_file_path, "r") as fin:
            for line in fin:
                line_split = line.strip().split(",")
                frame_ind = int(line_split[0])
                label = int(line_split[1])

                if frame_ind < self.load_start_ind:
                    continue

                video_name = osp.dirname(ann_file_path)
                if self.data_type == "video":
                    data_path = osp.join(self.data_root, video_name, "video_left.avi")
                    video_infos.append(dict(filename=data_path, frame_ind=frame_ind, label=label))

                elif self.data_type == "frame":
                    if self.modality == "RGB":
                        data_path = osp.join(self.data_root, video_name, "rgb")
                    elif self.modality == "Flow":
                        data_path = osp.join(self.data_root, video_name, "flow")

                    video_infos.append(dict(frame_dir=data_path, frame_ind=frame_ind, label=label))
        return video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results["modality"] = self.modality
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results["modality"] = self.modality
        return self.pipeline(results)
