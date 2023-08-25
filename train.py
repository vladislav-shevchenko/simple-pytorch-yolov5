from utils.pascal_dataset import *
from model.model import *
from utils.loss import *
import config

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm


if __name__ == '__main__':
    model = YOLOv5(torch.tensor(config.ANCHORS),
                   len(config.PASCAL_CLASSES),
                   depth_multiple=config.DEPTH_MULTIPLE,
                   width_multiple=config.WIDTH_MULTIPLE).to(config.DEVICE)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     'min',
                                                     config.LR_SCHEDULER_FACTOR,
                                                     config.LR_SCHEDULER_PATIENCE,
                                                     config.LR_SCHEDULER_THRESHOLD)
    if config.LOAD_MODEL:
        checkpoint = torch.load(config.LOAD_PATH, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    scaler = torch.cuda.amp.GradScaler()
    anchors = torch.tensor(config.ANCHORS) / config.IMAGE_SIZE
    dataset = PascalDataset(config.IMG_DIR,
                            config.LABEl_PATH,
                            classes=config.PASCAL_CLASSES,
                            sizes=config.OUTPUT_SIZES,
                            anchors = anchors,
                            transform=config.train_transforms, )
    train_loader = DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    scaled_anchors = (anchors * torch.tensor(config.OUTPUT_SIZES).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(
        config.DEVICE)
    criterion = YoloLoss(scaled_anchors,
                         len(config.PASCAL_CLASSES),
                         config.LAMBDA_NOOBJ,
                         config.LAMBDA_OBJ,
                         config.LAMBDA_BOX,
                         config.LAMBDA_CLS)
    for epoch in range(config.N_EPOCHS):
        loader_tqdm = tqdm(train_loader)
        losses = []
        for x, y in loader_tqdm:
            x = x.to(config.DEVICE)
            anchors = anchors.to(config.DEVICE)
            with torch.cuda.amp.autocast():
                output = model(x)
                loss = criterion(output, y)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            lr = optimizer.param_groups[0]["lr"]
            mean_loss = sum(losses) / len(losses)
            loader_tqdm.set_postfix(mean_epoch_loss=mean_loss, lr=lr)
        mean_loss = sum(losses) / len(losses)
        scheduler.step(mean_loss)
        if config.SAVE_MODEL:
            torch.save(
                {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
                },
                config.SAVE_PATH)
            print('CHECKPOINT SAVED')
