import torch
from torchvision import ops
from torch.nn import functional as F
from torchvision.ops import nms
import albumentations as A


def bboxes_to_target(bboxes, anchors, output_sizes, iou_threshold=0.2):
    n_anchors_per_scale = anchors.shape[1]
    anchors = anchors.reshape(anchors.shape[0] * anchors.shape[1], anchors.shape[2])
    targets = [torch.zeros((n_anchors_per_scale, size, size, 6)) for size in output_sizes]
    for box in bboxes:
            x, y, w, h, c = box
            intersection = torch.min(w, anchors[..., 0]) * torch.min(h, anchors[..., 1])
            union = (w * h + anchors[..., 0] * anchors[..., 1] - intersection)
            iou = intersection / union
            anchor_indices = iou.argsort(descending=True, dim=0)
            has_anchor = [False] * 3
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // n_anchors_per_scale
                scale_anchor_idx = anchor_idx % n_anchors_per_scale
                if iou[anchor_idx] > iou_threshold or not has_anchor[scale_idx]:
                    output_size = output_sizes[scale_idx]
                    row, col = int(output_size * y), int(output_size * x)
                    if targets[scale_idx][scale_anchor_idx, row, col, 0] == 0:
                        targets[scale_idx][scale_anchor_idx, row, col, 0] = 1
                        cell_x, cell_y = output_size * x - col, output_size * y - row
                        cell_w, cell_h = w * output_size, h * output_size
                        cell_box = torch.tensor([cell_x, cell_y, cell_w, cell_h])
                        targets[scale_idx][scale_anchor_idx, row, col, 1:5] = cell_box
                        targets[scale_idx][scale_anchor_idx, row, col, 5] = c
                        has_anchor[scale_idx] = True
    return targets

def pred_to_bboxes(model_out, scled_anchors, threshold, output_sizes):
    device = model_out[0].device
    bboxes = torch.Tensor().to(device)
    scores = torch.Tensor().to(device)
    labels = torch.Tensor().to(device)
    for scale_idx, output in enumerate(model_out):
        obj = F.sigmoid(output[..., 0]) > threshold
        for idx in torch.nonzero(obj):
            pred_anchor = scled_anchors[scale_idx, idx[1].item()]
            cell_box = output[idx[0].item(), idx[1].item(), idx[2].item(), idx[3].item(), :]
            cell_box[..., 1:3] = 2 * F.sigmoid(cell_box[..., 1:3]) - 0.5
            cell_box[..., 3:5] = (2 * F.sigmoid(cell_box[..., 3:5])) ** 2 * pred_anchor.to(device)
            x = ((cell_box[1] + idx[3].item()) / output_sizes[scale_idx]).item()
            y = ((cell_box[2] + idx[2].item()) / output_sizes[scale_idx]).item()
            width = (cell_box[3] / output_sizes[scale_idx]).item()
            height = (cell_box[4] / output_sizes[scale_idx]).item()
            bbox = ops.box_convert(torch.tensor((x, y, width, height)).unsqueeze(0), 'cxcywh', 'xyxy').to(device)
            bboxes = torch.cat((bboxes, bbox))
            scores = torch.cat((scores, F.sigmoid(cell_box[..., 0:1])))
            labels = torch.cat((labels, torch.argmax(F.sigmoid(cell_box[..., 5:])).unsqueeze(0)))
    return bboxes, scores, labels

def bboxes_to_image(bboxes, model_image_size, original_image):
    max_size = max(original_image.shape[:2])
    resize_factor = model_image_size / max_size
    x_padding = ((model_image_size - original_image.shape[1] * resize_factor) / 2) / resize_factor
    y_padding = ((model_image_size - original_image.shape[0] * resize_factor) / 2) / resize_factor
    bboxes[..., 0] = bboxes[..., 0] * max_size - x_padding
    bboxes[..., 1] = bboxes[..., 1] * max_size - y_padding
    bboxes[..., 2] = bboxes[..., 2] * max_size - x_padding
    bboxes[..., 3] = bboxes[..., 3] * max_size - y_padding
    return bboxes

def non_max_suppression(bboxes, scores, labels, iou_threshold=0.2):
    device = bboxes.device
    unique_labels = torch.unique(labels)
    bboxes_nms = torch.Tensor().to(device)
    scores_nms = torch.Tensor().to(device)
    labels_nms = torch.Tensor().to(device)
    for label in unique_labels:
        nms_idx = nms(bboxes[labels == label].clamp(0), scores[labels == label], iou_threshold)
        bboxes_nms = torch.cat([bboxes_nms, bboxes[labels == label][nms_idx]])
        scores_nms = torch.cat([scores_nms, scores[labels == label][nms_idx].unsqueeze(1)])
        labels_nms = torch.cat([labels_nms, labels[labels == label][nms_idx]])
    return bboxes_nms, scores_nms, labels_nms