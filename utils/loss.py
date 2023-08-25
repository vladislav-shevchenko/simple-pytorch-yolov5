import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import bbox_ciou


class YoloLoss(nn.Module):
    def __init__(self, anchors, n_classes, lambda_noobj=1, lambda_obj=1, lambda_box=1, lambda_cls=1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.anchors = anchors
        self.n_classes = n_classes

        self.lambda_cls = lambda_cls
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_box = lambda_box

    def forward(self, predictions, targets):
        device = predictions[0].device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        lnoobj = torch.zeros(1, device=device)
        for scale_idx, scale_prediction in enumerate(predictions):
            scale_target = targets[scale_idx].to(device)
            scale_anchors = self.anchors[scale_idx]
            obj = scale_target[..., 0] == 1
            noobj = scale_target[..., 0] == 0

            lnoobj += self.bce(
                (scale_prediction[..., 0:1][noobj]), (scale_target[..., 0:1][noobj]),
            )

            scale_anchors = scale_anchors.reshape(1, 3, 1, 1, 2)
            scale_prediction[..., 1:3] = F.sigmoid(scale_prediction[..., 1:3]) * 2 - 0.5
            scale_prediction[..., 3:5] = (F.sigmoid(scale_prediction[..., 3:5]) * 2) ** 2 * scale_anchors
            cious = bbox_ciou(scale_prediction[..., 1:5][obj], scale_target[..., 1:5][obj])
            lbox += torch.mean(1 - cious)

            cious = cious.detach().clamp(0)
            lobj += self.bce(scale_prediction[..., 0:1][obj], cious * scale_target[..., 0:1][obj])

            lcls += self.entropy(
                scale_prediction[..., 5:][obj],
                scale_target[..., 5][obj].type(torch.int64))

        loss = self.lambda_box * lbox + self.lambda_obj * lobj + self.lambda_noobj * lnoobj + self.lambda_cls * lcls

        return loss

