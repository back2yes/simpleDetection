import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import torch.nn as nn


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]  # xmin
    y1 = boxes[:, 1]  # ymin
    x2 = boxes[:, 2]  # xmax
    y2 = boxes[:, 3]  # ymax
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        remaining_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (remaining_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


def to_var(x, is_cuda=True):
    return Variable(x) if not is_cuda else Variable(x).cuda()


# by Joey Guo
def chamferDist(x, y):
    """new version when the lengths of x and y do not agree

    Args::
        input: x, of size (batchsize, num_of_points_x, dimensions)
        input: y, of size (batchsize, num_of_points_y, dimensions)
        return: the approx. avg emd

    Returns::
        The Chamfer distance between two sets.

    Example::
        x_npy = np.array([0.0, 1, 2])[None, :, None]
        y_npy = np.array([0.0, 1, 2, 3, 4])[None, :, None]
        # x = torch.randn(32, 1000, 1).cuda()
        # y = torch.randn(32, 1000, 1).cuda()
        x = torch.from_numpy(x_npy)
        y = torch.from_numpy(y_npy)
        print(batch_NN_loss(x, y))  # 0.3
    """
    # print(x.size())
    # print(y.size())
    bs, num_points_x, points_dim = x.size()
    bs, num_points_y, points_dim = y.size()
    xx = x.unsqueeze(2).expand(bs, num_points_x, num_points_y,
                               points_dim)  # x coord is fixed on dim 2 (start from 0)
    yy = y.unsqueeze(1).expand(bs, num_points_x, num_points_y,
                               points_dim)  # y coord is fixed on dim 1 (start from 0)

    D = xx - yy
    D = torch.sqrt(torch.sum(D * D, dim=-1))  # sum over space dimension

    # fix x to search on ys, so this is the min dist from each point in x to the set of y
    min_dist1, min_indexes = torch.min(D, dim=2)
    # min_dist1, _ = torch.min(D, dim=2)
    # fix y to search on xs, so this is the min dist from each point in y to the set of x
    min_dist2, _ = torch.min(D, dim=1)

    # actually it is only approx. avg_emd, or the Chamfer distance
    chamfer = 0.5 * (min_dist1.mean() + min_dist2.mean())
    # chamfer = (min_dist1.sum() + min_dist2.sum()) / (bs * (num_points_x + num_points_y))
    return chamfer, min_indexes
    # return chamfer


def IoU(predBBoxes, gtBBoxes):
    """
    :param predBBoxes: (xmin, ymin, xmax, ymax)
    :param gtBBoxes: (xmin, ymin, xmax, ymax)
    :return:
    """
    predAreas = (predBBoxes[:, 2] - predBBoxes[:, 0]) * (predBBoxes[:, 3] - predBBoxes[:, 1])
    gtAreas = (gtBBoxes[:, 2] - gtBBoxes[:, 0]) * (gtBBoxes[:, 3] - gtBBoxes[:, 1])

    # delta_x and delta_y between the preds and gts
    delta_x1 = predBBoxes[:, 2] - gtBBoxes[:, 0]
    delta_x2 = gtBBoxes[:, 2] - predBBoxes[:, 0]
    delta_x = torch.min(delta_x1, delta_x2).clamp(0.0, 100000000.0)

    delta_y1 = predBBoxes[:, 3] - gtBBoxes[:, 1]
    delta_y2 = gtBBoxes[:, 3] - predBBoxes[:, 1]
    delta_y = torch.min(delta_y1, delta_y2).clamp(0.0, 100000000.0)

    ovrAreas = delta_x * delta_y
    unionAreas = predAreas + gtAreas - ovrAreas

    iou_ratio = ovrAreas / unionAreas
    return iou_ratio


def xentropy(pred, gt):
    return -torch.mean((1 - gt) * torch.log(1 - pred) + gt * torch.log(pred))


def draw_boxes(img, bboxes, scores, ax=None, pred_height=None, gt_height=None, save_path='outputs/00001.png'):
    if ax is None:
        plt.figure(0)
        plt.clf()
        ax = plt.subplot(1, 1, 1)
        plt.imshow(img.data.cpu().numpy()[0].transpose((1, 2, 0)))

    for ii, bbox in enumerate(bboxes.squeeze().data.cpu().numpy()):
        score = scores.squeeze().data.cpu().numpy()[ii]
        if pred_height is not None:
            predH = pred_height.squeeze().data.cpu().numpy()[ii]
        if gt_height is not None:
            gtH = gt_height.squeeze().data.cpu().numpy()[ii]
        # print(score)
        rgba_color = (score, 0.5, (1 - score), 0.7)
        rectangle = mpatch.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False,
                                     color=rgba_color)
        ax.add_patch(rectangle)

        if score > 0.4:
            pred_height_text = ax.text(bbox[0], bbox[1], '{:.3f}'.format(predH), fontsize=8, color=rgba_color)
            gt_height_text = ax.text(bbox[0], bbox[1] + 16, '{:.3f}'.format(gtH), fontsize=8, color=(0.0, 1.0, 0.0, 0.6))
    # plt.xlim(0.0, 10.0)
    # plt.ylim(0.0, 10.0)
    plt.savefig(save_path)
