# encoding: utf-8


import torch
import numpy as np


def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids

def test_collate_fn(batch):
    batch_imgs, _, batch_cam_ids, batch_gt_labels, batch_img_paths = zip(*batch)
    batch_cam_ids = np.array(batch_cam_ids)
    batch_gt_labels = np.array(batch_gt_labels)
    batch_img_paths = np.array(batch_img_paths)
    return torch.stack(batch_imgs, dim=0), batch_cam_ids, batch_gt_labels, batch_img_paths
