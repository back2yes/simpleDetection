from models.resnet import FPN
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from util.box_utils import to_var


class Heads(nn.Module):
    def __init__(self, in_channels, out_channels, final_activation=F.sigmoid):
        super(Heads, self).__init__()
        self.final_activation = final_activation
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.final_activation(x)
        return x


class NNSkeleton(nn.Module):
    def __init__(self, resModel):
        super(NNSkeleton, self).__init__()
        self.fpn = FPN(resModel)

        self.locHead_1 = Heads(3, 1)
        self.locHead0 = Heads(64, 1)
        self.locHead1 = Heads(64, 1)
        self.locHead2 = Heads(64, 1)
        self.locHead3 = Heads(128, 1)
        self.locHead4 = Heads(256, 1)
        self.locHead5 = Heads(512, 1)

        self.clsHead_1 = Heads(3, 1, final_activation=F.sigmoid)
        self.clsHead0 = Heads(64, 1, final_activation=F.sigmoid)
        self.clsHead1 = Heads(64, 1, final_activation=F.sigmoid)
        self.clsHead2 = Heads(64, 1, final_activation=F.sigmoid)
        self.clsHead3 = Heads(128, 1, final_activation=F.sigmoid)
        self.clsHead4 = Heads(256, 1, final_activation=F.sigmoid)
        self.clsHead5 = Heads(512, 1, final_activation=F.sigmoid)

    def forward(self, x):
        # blah
        h_1, h0, h1, h2, h3, h4, h5 = self.fpn(x)
        # locPreds_1 = self.locHead_1(h_1)  # 224x224
        # clsPreds_1 = self.clsHead_1(h_1)  # 224x224
        # locPreds0 = self.locHead0(h0)  # 112x112
        # clsPreds0 = self.clsHead0(h0)  # 112x112
        # locPreds1 = self.locHead1(h1)  # 56x56
        # clsPreds1 = self.clsHead1(h1)  # 56x56
        # locPreds2 = self.locHead2(h2)  # 56x56
        # clsPreds2 = self.clsHead2(h2)  # 56x56
        locPreds3 = self.locHead3(h3)  # 28x28
        clsPreds3 = self.clsHead3(h3)  # 28x28
        # locPreds4 = self.locHead4(h4)  # 14x14
        # clsPreds4 = self.clsHead4(h4)  # 14x14
        # locPreds5 = self.locHead5(h5)  # 7x7
        # clsPreds5 = self.clsHead5(h5)  # 7x7

        # retVals = (
        #     locPreds_1, clsPreds_1, locPreds0, clsPreds0, locPreds1, clsPreds1, locPreds2, clsPreds2, locPreds3,
        #     clsPreds3,
        #     locPreds4, clsPreds4, locPreds5, clsPreds5)
        # locPreds3.squeeze()
        # clsPreds3.squeeze()
        featHeight, featWidth = locPreds3.size()[-2:]
        imgHeight, imgWidth = x.size()[-2:]
        xgrid = torch.linspace(0.0, imgWidth - 1.0, featWidth).unsqueeze(0).expand((featHeight, featWidth))
        ygrid = torch.linspace(0.0, imgHeight - 1.0, featHeight).unsqueeze(1).expand((featHeight, featWidth))
        xgrid, ygrid = to_var(xgrid).detach(), to_var(ygrid).detach()
        # print(xgrid)
        # print(ygrid)

        # xPreds = locPreds3
        # print('xgrid, ', xgrid.size(), 'locPreds, ', locPreds3.size())
        xmin = xgrid - 0.5 * imgWidth * locPreds3.squeeze()
        ymin = ygrid - 0.5 * imgHeight * locPreds3.squeeze()
        xmax = xgrid + 0.5 * imgWidth * locPreds3.squeeze()
        ymax = ygrid + 0.5 * imgHeight * locPreds3.squeeze()

        bboxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1).view(-1, 4)
        scores = clsPreds3.view(-1)
        retVals = (bboxes, scores)

        # retVals = (xgrid, ygrid, locPreds3, clsPreds3)

        # for val in retVals:
        #     print(val.size())
        return retVals
