import torch
from models.resnet import FPN, resnet34
from models.skeleton import NNSkeleton
from torchvision.datasets import FakeData
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

print(torch.__version__)

ds = FakeData(image_size=(3, 512, 512), transform=transforms.ToTensor())
dl = DataLoader(ds, batch_size=16)

net = NNSkeleton(resnet34(True))
# net = FPN(resnet34(True))
net.cuda()


def to_var(x, is_cuda=True):
    return Variable(x) if not is_cuda else Variable(x).cuda()


for ii, (xs, ys) in enumerate(dl):
    # print(ii, xs.size(), ys.size())
    net(to_var(xs))
    # exit()
