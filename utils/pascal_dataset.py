import glob
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
import torchvision.ops as ops

from utils.bbox import bboxes_to_target

class PascalDataset(Dataset):
    def __init__(self, img_dir, xml_dir, classes, anchors, transform=None, sizes=(80, 40, 20)):
        self.img_dir = img_dir
        self.classes = classes
        self.xml_files = glob.glob(xml_dir + f'/*.xml')
        self.images = glob.glob(f'{img_dir}/*')
        self.anchors = anchors
        self.transform = transform
        self.sizes = sizes

    def __len__(self):
        return len(self.xml_files)

    def __getitem__(self, index):
        xml_file = self.xml_files[index]
        image_name, bboxes, labels = self.read_xml(xml_file, self.classes)
        image_path = f'{self.img_dir}/{image_name}'
        image = Image.open(image_path)
        image = np.array(image)
        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes)
            image = transformed['image']
            bboxes = transformed['bboxes']

        bboxes = torch.tensor(bboxes)
        labels = torch.tensor(labels)
        bboxes = bboxes[..., :] / image.shape[1]
        bboxes = ops.box_convert(bboxes, 'xyxy', 'cxcywh')
        bboxes = torch.cat((bboxes, labels), 1)
        targets = bboxes_to_target(bboxes, self.anchors, self.sizes)
        return image, targets

    @staticmethod
    def read_xml(xml_file, classes):
        bboxes = []
        labels = []
        tree = ET.parse(xml_file)
        root = tree.getroot()
        image = root.find('filename').text
        for object in root.findall('object'):
            bbox = object.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
            label = object.find('name').text
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append([classes.index(label)])
        return image, np.array(bboxes), labels