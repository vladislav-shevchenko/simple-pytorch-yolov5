import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import cv2

DEPTH_MULTIPLE = 0.33
WIDTH_MULTIPLE = 0.5

IMG_DIR = 'data/VOC2012/JPEGImages'
LABEl_PATH = 'data/VOC2012/Annotations'
IMAGE_SIZE = 640

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 25
LR = 0.001
LR_SCHEDULER_FACTOR = 0.3
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_THRESHOLD = 0.003
WEIGHT_DECAY = 1e-4
N_EPOCHS = 100
LAMBDA_NOOBJ=1
LAMBDA_OBJ=1
LAMBDA_BOX=1
LAMBDA_CLS=1
LOAD_MODEL = False
LOAD_PATH = 'checkpoint.pth.tar'
SAVE_MODEL = True
SAVE_PATH = 'checkpoint.pth.tar'
OUTPUT_SIZES = [IMAGE_SIZE // 8, IMAGE_SIZE // 16, IMAGE_SIZE // 32]


ANCHORS = (((10, 13), (16, 30), (33, 23)),
           ((30, 61), (62, 45), (59, 119)),
           ((116, 90), (156, 198), (373, 326)))

train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE),
            min_width=int(IMAGE_SIZE),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.Blur(p=0.1),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=[])
)

PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

