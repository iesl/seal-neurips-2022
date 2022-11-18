from typing import Optional
import torch
from allennlp.nn.util import dist_reduce_sum
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric
import numpy as np
import time

@Metric.register("seg-iou")
class SegIoU(Metric):
    """
    Segmentation IoU.
    """

    supports_distributed = True

    def __init__(self) -> None:
        self.sum_seg_iou = 0.0
        self.total_count = 0.0
        # hard code: for visualizing predicted segmentation masks
        # self.predictions = []
        # self.random = str(time.time())[-7:]

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # hard code: for visualizing predicted segmentation masks
        # y_pred = y_pred.to("cuda")
        # y_true = y_true.to("cuda")
        assert len(y_pred.size()) in [4, 5] # (b, 1 or 2, h, w) or (b, 36, 1 or 2, h, w), no num_sample dim
        assert len(y_true.size()) == 4 # (b, 1, h, w)
        self.total_count += y_true.shape[0]

        if len(y_pred.size()) == 5: # average the 36 crops
            b = y_pred.size()[0]
            size = 32; crop_size = 24

            sum_36_crops = torch.zeros((b, y_pred.size()[-3], size, size)).to("cuda"); n = 0
            for h in [0, 2, 3, 4, 6, 8]:
                for w in [0, 2, 3, 4, 6, 8]:
                    sum_36_crops[:, :, h:h+crop_size, w:w+crop_size] = \
                        sum_36_crops[:, :, h:h+crop_size, w:w+crop_size] + y_pred[:, n, :, :, :]
                    n += 1

            scale_36_crops = torch.zeros((b, 1, size, size)).to("cuda")
            for h in [0, 2, 3, 4, 6, 8]:
                for w in [0, 2, 3, 4, 6, 8]:
                    scale_36_crops[:, :, h:h+crop_size, w:w+crop_size] = \
                        scale_36_crops[:, :, h:h+crop_size, w:w+crop_size] + 1

            y_pred = sum_36_crops / scale_36_crops

        if y_pred.size()[-3] > 1:
            y_pred = torch.argmax(y_pred, dim=-3)
            # hard code: for visualizing predicted segmentation masks
            # self.predictions.append(np.array(y_pred.clone().detach().cpu()))
        else:
            y_pred = y_pred > 0.5
        y_pred = y_pred.view(-1, y_pred.size()[-2] * y_pred.size()[-1]) # (b, h*w)
        y_true = y_true.view(-1, y_true.size()[-2] * y_true.size()[-1]) # (b, h*w)
        intersect = torch.sum(torch.min(y_pred, y_true), dim=-1)
        union = torch.sum(torch.max(y_pred, y_true), dim=-1)

        # epsilon = torch.full(union.size(), 10**-8).to(union.device)
        # self.sum_seg_iou += torch.sum(intersect / torch.max(epsilon, union))
        self.sum_seg_iou += torch.sum(intersect / union)

    def get_metric(self, reset: bool = False) -> float:
        """
        # Returns
        The accumulated segmentation iou.
        """
        if self.total_count > 0:
            seg_iou = float(self.sum_seg_iou) / float(self.total_count)
        else:
            seg_iou = 0.0

        # hard code: for visualizing predicted segmentation masks
        # np.save(
        #     f"/mnt/nfs/scratch1/username/SEAL/seal/predictions/weizmann_horse_seg_seal_dvn_{self.random}.npy",
        #     self.predictions
        # )

        if reset:
            self.reset()

        return seg_iou

    def reset(self):
        self.sum_seg_iou = 0.0
        self.total_count = 0.0
        # hard code: for visualizing predicted segmentation masks
        # self.predictions = []