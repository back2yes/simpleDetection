from util.box_utils import nms
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

N = 100


def draw_boxes(ax, bboxes, inds=None):
    if inds is None:
        inds = np.arange(len(bboxes))
    for ii, bbox in enumerate(bboxes):
        rectangle = mpatch.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False,
                                     color=colors[inds[ii] % len(colors)])
        ax.add_patch(rectangle)
    plt.xlim(0.0, 10.0)
    plt.ylim(0.0, 10.0)


if __name__ == '__main__':

    colors = ['b', 'g', 'r', 'k', 'm']
    bboxes = np.zeros((N, 4), dtype='float32')
    bboxes[:, ::2] = np.sort(np.random.uniform(0.0, 10.0, (N, 2)), axis=-1)
    bboxes[:, 1::2] = np.sort(np.random.uniform(0.0, 10.0, (N, 2)), axis=-1)
    scores = np.random.uniform(0.0, 1.0, (N,)).astype('float32')

    ax = plt.subplot(1, 2, 1)
    draw_boxes(ax, bboxes)

    tkeep, tcount = nms(torch.FloatTensor(bboxes), torch.FloatTensor(scores), overlap=0.4)

    newbboxes = bboxes[tkeep[:tcount].numpy()]
    newkeep = tkeep.numpy()
    ax = plt.subplot(1, 2, 2)
    draw_boxes(ax, newbboxes, newkeep)
    plt.show()
    # print(tkeep, tcount)
