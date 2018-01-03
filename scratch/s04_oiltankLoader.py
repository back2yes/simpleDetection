from data.oiltankLoader import VOCDetection
from torch.utils.data import Dataset, DataLoader

ds = VOCDetection('/home/x/data/VOCdevkit', image_sets=[('2007', 'train')])
dl = DataLoader(ds)

for epoch in range(100):
    for ii, (img, anno) in enumerate(dl, 0):
        # print(ii)
        print(anno.size())
        print(anno[:, :, -1])
