# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline


def build_model(cfg, num_classes):
    # if cfg.MODEL.NAME == 'resnet50':
    #     model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE,cfg.MODEL.HOB,cfg.MODEL.HOB_NUM,
                     cfg.MODEL.INPUT_CHANNLS,cfg.MODEL.INTER_CHANNLS,cfg.MODEL.EMBEDINGS,cfg.MODEL.PARTS)

    return model