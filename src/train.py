import torch
from util.box_utils import nms, chamferDist, to_var, IoU, xentropy, draw_boxes
from data.oiltankLoader import VOCDetection as OilDataset
from torch.utils.data import DataLoader
from models.skeleton import NNSkeleton
from models.resnet import FPN, resnet34
from torch import optim
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import os

from torch import nn

""" Parameters here
"""
is_cuda = True

# dataroot = 'C:/Users/x/data/VOCdevkit'
dataroot = '/home/x/data/VOCdevkit'
num_epochs = 100
lam_loc = 0.01
lam_hgt = 1.0
lr = 1e-4
log_dir = 'logs/exp04_with_height'
writer = SummaryWriter(log_dir=log_dir)
save_path = 'saves/exp04_with_height'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
start_epoch = 0

ds = OilDataset(dataroot, image_sets=[('2007', 'trainval')])
dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2)

net = NNSkeleton(resnet34(True))
if is_cuda:
    net.cuda()

lambda2 = lambda epoch: 0.95 ** epoch
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = LambdaLR(optimizer, lr_lambda=lambda2)

# criterion = nn.
criterion_score = xentropy
criterion_height = nn.MSELoss()

try:
    net.load_state_dict(torch.load(save_path + '/model_{:03d}.pth'.format(start_epoch)))
except:
    pass
    # raise

for epoch in range(start_epoch, num_epochs):
    scheduler.step(epoch)
    try:
        os.remove(save_path + '/model_{:03d}.pth'.format(epoch - 5))
    except:
        pass
    torch.save(net.state_dict(), save_path + '/model_{:03d}.pth'.format(epoch))
    for ii, minibatch in enumerate(dl, 0):
        print(epoch, ii)
        global_step = ii + epoch * len(dl)

        xs, ts = to_var(minibatch[0]), to_var(minibatch[1])

        # print(xs)
        bboxes, lidHeights, scores = net(xs)
        # print(bboxes.size())
        # print(scores.size())
        keep, count = nms(bboxes.data, scores.data)
        # print(keep, count)
        kept_bboxes = bboxes[keep[:count]]
        kept_height = lidHeights[keep[:count]]
        kept_scores = scores[keep[:count]]
        # print(kept_bboxes)
        # print(kept_scores)

        # print(bboxes.size())
        # print(bboxes.size())
        gt_bboxes = ts[..., :4]
        gt_lidHeight = ts[..., 4]
        chamfer_loss, min_indexes = chamferDist(kept_bboxes[None, ...], gt_bboxes)
        # print(min_indexes)
        corresponding_gtbboxes = gt_bboxes.squeeze(0)[min_indexes.squeeze(0)]
        corresponding_gtheight = gt_lidHeight.squeeze(0)[min_indexes.squeeze(0)]
        # print(corresponding_gtbboxes)

        # print(kept_bboxes.size(), corresponding_gtbboxes.size())
        ious = IoU(kept_bboxes, corresponding_gtbboxes)
        print('ious: ', ious.data.cpu().numpy())
        # print(ious.size())
        # print(kept_scores.size())
        score_loss = criterion_score(kept_scores, ious)
        hgt_loss = criterion_height(kept_height, corresponding_gtheight)
        print('kept_scores: ', kept_scores.data.cpu().numpy())
        print(score_loss)
        print(chamfer_loss)
        print(hgt_loss)

        total_loss = lam_loc * chamfer_loss + score_loss + lam_hgt * hgt_loss
        # print(total_loss)
        writer.add_scalar('score_loss', score_loss.data[0], global_step)
        writer.add_scalar('chamfer_loss', chamfer_loss.data[0], global_step)
        writer.add_scalar('height_loss', hgt_loss.data[0], global_step)
        writer.add_scalar('total_loss', total_loss.data[0], global_step)

        # updating
        net.zero_grad()
        total_loss.backward()
        optimizer.step()
        # print(kept_bboxes.size())
        # bboxes

        if global_step % 100 == 0:
            draw_boxes(xs, kept_bboxes, kept_scores, pred_height=kept_height, gt_height=corresponding_gtheight,
                       save_path='outputs/{:06d}.png'.format(global_step))
