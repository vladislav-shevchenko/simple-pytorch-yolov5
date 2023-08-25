from torchvision.utils import draw_bounding_boxes

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

from utils.bbox import pred_to_bboxes, non_max_suppression, bboxes_to_image
from model.model import *
import config
import glob
import os

if __name__ == '__main__':
    model = YOLOv5(torch.tensor(config.ANCHORS),
                   len(config.PASCAL_CLASSES),
                   depth_multiple=config.DEPTH_MULTIPLE,
                   width_multiple=config.WIDTH_MULTIPLE).to(config.DEVICE)
    checkpoint = torch.load(config.LOAD_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    images = glob.glob('examples/data/*')
    anchors = torch.tensor(config.ANCHORS) / config.IMAGE_SIZE
    scaled_anchors = anchors * torch.tensor(config.OUTPUT_SIZES).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=config.IMAGE_SIZE),
            A.PadIfNeeded(
                min_height=config.IMAGE_SIZE,
                min_width=config.IMAGE_SIZE,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
            ToTensorV2()
        ]
    )
    score_threshold = 0.8
    for i, image_path in enumerate(images):
        image = np.array(Image.open(image_path).convert('RGB'))
        transformed_image = test_transforms(image=image)['image'].to(config.DEVICE)
        outputs = model(transformed_image.unsqueeze(0))

        bboxes_nms, scores_nms, labels_nms = non_max_suppression(*pred_to_bboxes(outputs, scaled_anchors, score_threshold, config.OUTPUT_SIZES))
        bboxes_nms = bboxes_to_image(bboxes_nms, config.IMAGE_SIZE, image).type(torch.int)
        labels_nms = labels_nms.type(torch.int).tolist()
        labels = [f'{config.PASCAL_CLASSES[label_idx]} {scores_nms[i].item():.2f}' for i, label_idx in enumerate(labels_nms)]

        image_with_bboxes = draw_bounding_boxes(torch.tensor(image).permute(2, 0, 1),
                                  bboxes_nms,
                                  labels,
                                  colors="red",
                                  width=3).permute(1, 2, 0).numpy()
        plt.imshow(image_with_bboxes)
        plt.show()
        # plt.imsave(f'examples/results/{os.path.basename(image_path)}', image_with_bboxes)

