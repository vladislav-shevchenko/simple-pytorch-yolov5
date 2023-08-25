from model.modules import *


class Backbone(nn.Module):
    def __init__(self, depth_multiple=0.33, width_multiple=0.5):
        super().__init__()
        self.model = nn.ModuleList(
            [
                Conv(3, 32, 6, 2, 2, width_multiple=1),
                Conv(64, 128, 3, 2, width_multiple=0.5),
                C3(128, 128, 3, depth_multiple=0.33, width_multiple=0.5),
                Conv(128, 256, 3, 2, width_multiple=0.5),
                C3(256, 256, 6, depth_multiple=0.33, width_multiple=0.5),
                Conv(256, 512, 3, 2, width_multiple=0.5),
                C3(512, 512, 9, depth_multiple=0.33, width_multiple=0.5),
                Conv(512, 1024, 3, 2, width_multiple=0.5),
                C3(1024, 1024, 3, depth_multiple=0.33, width_multiple=0.5),
                SPP(1024, 1024, width_multiple=0.5)
            ]
        )

    def forward(self, x):
        output = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in (4, 6, 9):
                output.append(x)
        return output


class Neck(nn.Module):
    def __init__(self, depth_multiple=0.33, width_multiple=0.5):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.cv1 = Conv(1024, 512, 1, width_multiple=0.5)
        # Upsample
        # Concat
        self.c31 = C3(1024, 512, 3, False, depth_multiple=0.33, width_multiple=0.5)

        self.cv2 = Conv(512, 256, 1, width_multiple=0.5)
        # Upsample
        # Concat
        self.c32 = C3(512, 256, 3, False, depth_multiple=0.33, width_multiple=0.5)

        self.cv3 = Conv(256, 256, 3, 2, width_multiple=0.5)
        # Concat
        self.c33 = C3(512, 512, 3, False, depth_multiple=0.33, width_multiple=0.5)

        self.cv4 = Conv(512, 512, 3, 2, width_multiple=0.5)
        # Concat
        self.c34 = C3(1024, 1024, 3, False, depth_multiple=0.33, width_multiple=0.5)

    def forward(self, backbone_out):
        output = []
        cv1_out = self.cv1(backbone_out[2])
        x = self.up(cv1_out)
        x = torch.cat((x, backbone_out[1]), 1)
        x = self.c31(x)

        cv2_out = self.cv2(x)
        x = self.up(cv2_out)
        x = torch.cat((x, backbone_out[0]), 1)
        x = self.c32(x)
        output.append(x)

        x = self.cv3(x)
        x = torch.cat((x, cv2_out), 1)
        x = self.c33(x)
        output.append(x)

        x = self.cv4(x)
        x = torch.cat((x, cv1_out), 1)
        x = self.c34(x)
        output.append(x)

        return output


class Head(nn.Module):
    def __init__(self, anchors, n_classes, width_multiple=0.5, n_channels=(256, 512, 1024)):
        super().__init__()
        self.n_outputs = 5 + n_classes
        self.n_layers = anchors.shape[0]
        self.n_anchors = anchors.shape[1]
        self.conv = nn.ModuleList(nn.Conv2d(round(x * width_multiple), self.n_outputs * self.n_anchors, 1) for x in n_channels)

    def forward(self, neck_out):
        output = []
        for i in range(self.n_layers):
            x = self.conv[i](neck_out[i])
            x = x.view(x.shape[0], self.n_anchors, self.n_outputs, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)
            output.append(x)
        return output


class YOLOv5(nn.Module):
    def __init__(self, anchors, n_classes, depth_multiple=0.33, width_multiple=0.5):
        super().__init__()
        self.backbone = Backbone(depth_multiple, width_multiple)
        self.neck = Neck(depth_multiple, width_multiple)
        self.head = Head(anchors, n_classes, width_multiple)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x
