import os
import numpy as np
import cv2
from utils import get_test_datasets, get_transforms, get_txt_datasets
from datasets import STS_test_Dataset
from models import STS2DModel
from config import CFG
import torch
import torch.utils.data as data
from tqdm import tqdm
from collections import deque
import torch.nn as nn
from pathlib import Path


def TTA(x: torch.Tensor, model: nn.Module):
    # x.shape=(batch,c,h,w)
    inputs = [x, *[torch.rot90(x, k=i, dims=(-2, -1)) for i in range(1, 4)]]
    x = list(map(lambda x: model(x), inputs))
    x = [torch.rot90(x[i], k=-i, dims=(-2, -1)) for i in range(4)]
    x = torch.stack(x, dim=0)
    return x.mean(0)


class EnsembleModel:
    def __init__(self, use_tta=False):
        self.models = []
        self.use_tta = use_tta

    def __call__(self, x):
        if self.use_tta:
            outputs = [torch.sigmoid(TTA(x, model)).to('cpu').numpy()
                       for model in self.models]
        else:
            outputs = [torch.sigmoid(model(x)).to('cpu').numpy()
                       for model in self.models]
        avg_preds = np.mean(outputs, axis=0)
        return avg_preds

    def add_model(self, model):
        self.models.append(model)


def build_ensemble_model(models_checkpoints):
    model = EnsembleModel()
    model_names = models_checkpoints.keys()
    for model_name in model_names:
        checkpoints = models_checkpoints[model_name]
        print('loading model:{}'.format(model_name))
        for ckpt in checkpoints:
            print('loading checkpoint {}'.format(ckpt))
            state = torch.load(ckpt, map_location=CFG.device)
            _model = STS2DModel(CFG, model_name=model_name)
            _model.load_state_dict(state)
            _model.to(CFG.device)
            _model.eval()
            model.add_model(_model)

    return model


# test_sliced_images, test_xyxys, test_ori_sizes, names = get_test_datasets()
test_sliced_images, test_xyxys, test_ori_sizes, names = get_txt_datasets(CFG.test_txt)
test_ds = STS_test_Dataset(test_sliced_images, transform=get_transforms("test", cfg=CFG))
test_loader = data.DataLoader(test_ds,
                              batch_size=CFG.test_bs,
                              shuffle=False,
                              pin_memory=False,
                              num_workers=CFG.num_workers,
                              drop_last=False)


def test_step(test_loader, model, device, tta=False):
    sliced_images_preds = deque()
    if CFG.using_ensemble_models:
        for step, (images) in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = images.to(device)
            with torch.no_grad():
                y_pred = model(images)
                sliced_images_preds.extend(y_pred)

    else:
        model.to(device)
        model.eval()
        for step, (images) in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = images.to(device)
            with torch.no_grad():
                if tta:
                    y_pred = TTA(images, model)
                else:
                    y_pred = model(images)
                y_pred = torch.sigmoid(y_pred).to('cpu').numpy()
                sliced_images_preds.extend(y_pred)
    return sliced_images_preds


if __name__ == '__main__':
    if CFG.using_ensemble_models:
        model = build_ensemble_model(CFG.models_checkpoints)
    else:
        model = STS2DModel(CFG, model_name=CFG.model_name)
        model.load_state_dict(torch.load(CFG.checkpoint))
    print("test process:")
    test_sliced_preds = test_step(test_loader, model, CFG.device, tta=CFG.tta)
    print("generating each pred")
    for idx, name in tqdm(enumerate(names), total=(len(names))):
        per_xyxys = test_xyxys[idx]
        per_ori_size = test_ori_sizes[idx]
        pad0 = CFG.tile_size - per_ori_size[0] % CFG.tile_size
        pad1 = CFG.tile_size - per_ori_size[1] % CFG.tile_size

        mask_pred = np.zeros(per_ori_size)
        mask_count = np.zeros(per_ori_size)
        mask_pred = np.pad(mask_pred, [(0, pad0), (0, pad1)], constant_values=0)
        mask_count = np.pad(mask_count, [(0, pad0), (0, pad1)], constant_values=0)
        ori_h, ori_w = per_ori_size[-2], per_ori_size[-1]
        for i, (x1, y1, x2, y2) in enumerate(per_xyxys):
            mask_pred[y1:y2, x1:x2] += test_sliced_preds.popleft().squeeze()
            mask_count[y1:y2, x1:x2] += np.ones((CFG.test_tile_size, CFG.test_tile_size))
        mask_pred /= mask_count
        mask_pred = mask_pred[:ori_h, :ori_w]
        if CFG.thr:
            mask_pred = (mask_pred >= CFG.thr).astype(int)
            cv2.imwrite(CFG.test_preds_dir + f"{name}.png", mask_pred * CFG.show)
        else:
            np.save(CFG.test_preds_dir + f"{name}.npy", mask_pred)
    print("done!")
