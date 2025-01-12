import cv2
import torch
import numpy as np
from tqdm import tqdm
import wandb
import torch.nn as nn
from config import CFG
from torch import optim
from models import STS2DModel
import torch.utils.data as data
from datasets import STS_Dataset
from scheduler import get_scheduler
from torch.cuda.amp import GradScaler
from myscores.scores import scores_coef, record_scores
from utils import get_train_valid_datasets, get_transforms
from SegLoss.losses_pytorch.hausdorff import HausdorffDTLoss
from SegLoss.losses_pytorch.dice_loss import SoftDiceLoss, IoULoss

'''log'''
# wandb.login(key="99cf7d42bf6827ba650272cf79a1172a1f28db2c")
'''prepare dataloaders'''

datasets = get_train_valid_datasets()
train_sliced_images, train_sliced_masks = datasets["train_datasets"]
valid_sliced_images_list, valid_sliced_masks_list, valid_xyxys, valid_ori_sizes, valid_names = datasets[
    "valid_datasets"]

train_ds = STS_Dataset(train_sliced_images, train_sliced_masks, transform=get_transforms("train", CFG))
train_loader = data.DataLoader(train_ds,
                               batch_size=CFG.train_bs,
                               shuffle=True,
                               pin_memory=False,
                               num_workers=CFG.num_workers,
                               drop_last=True)

valid_dataloader_list = []

for sliced_images, sliced_masks in zip(valid_sliced_images_list, valid_sliced_masks_list):
    valid_ds = STS_Dataset(sliced_images, sliced_masks, transform=get_transforms("valid", CFG))
    valid_dataloader = data.DataLoader(valid_ds,
                                       batch_size=CFG.valid_bs,
                                       shuffle=False,
                                       pin_memory=False,
                                       num_workers=CFG.num_workers,
                                       drop_last=False)

    valid_dataloader_list.append(valid_dataloader)

'''
每组idex都是记录每张valid的信息
valid_dataloader_list
valid_xyxys
valid_ori_sizes
valid_names
'''

'''train step & valid step'''


def train_step(train_loader, model, criterion, optimizer, device, epoch):
    model.train()
    epoch_loss = 0
    scaler = GradScaler(enabled=CFG.use_amp)
    bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (image, label) in bar:
        optimizer.zero_grad()
        if CFG.model_name == "Segformer":
            outputs = model(image.to(device), label.to(device).squeeze().long())
            loss, logits = outputs.loss, outputs.logits
            optimizer.step()
        else:
            outputs = model(image.to(device))
            loss = criterion(outputs.to(torch.float), label.to(device).to(torch.float))
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), CFG.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        bar.set_postfix(loss=f'{loss.item():0.4f}', epoch=epoch + 1, gpu_mem=f'{mem:0.2f} GB',
                        lr=f'{optimizer.state_dict()["param_groups"][0]["lr"]:0.2e}')
        epoch_loss += loss.item()

    torch.cuda.empty_cache()
    return epoch_loss / len(train_loader)


'''once_valid_step 只是处理单一张valid图片'''


def once_valid_step(valid_loader, xyxys, ori_size, name, model, criterion, device, epoch):
    model.eval()

    mask_pred = np.zeros(ori_size)
    mask_count = np.zeros(ori_size)
    mask_gt = np.zeros(ori_size)
    ori_h, ori_w = ori_size[-2], ori_size[-1]
    pad0 = CFG.tile_size - ori_h % CFG.tile_size
    pad1 = CFG.tile_size - ori_w % CFG.tile_size

    mask_pred = np.pad(mask_pred, [(0, pad0), (0, pad1)], constant_values=0)
    mask_count = np.pad(mask_count, [(0, pad0), (0, pad1)], constant_values=1)
    mask_gt = np.pad(mask_gt, [(0, pad0), (0, pad1)], constant_values=0)

    epoch_loss = 0
    best_th = 0
    best_score = 0
    ori_score = best_score

    out_scores = dict()
    '''一下是对一张 320 × 640 的图片的所有分割片识别后再合到一起'''
    bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    for step, (image, label) in bar:
        image = image.to(device)
        label = label.to(device)
        with torch.no_grad():
            if CFG.model_name == "Segformer":
                outputs = model(image, label.squeeze().long())
                loss, logits = outputs.loss, outputs.logits
                upsampled_logits = nn.functional.interpolate(
                    logits,
                    size=image.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                y_pred = upsampled_logits.argmax(dim=1)[0]
            else:
                y_pred = model(image)
                loss = criterion(y_pred, label)
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        bar.set_postfix(loss=f'{loss.item():0.4f}', epoch=epoch + 1, gpu_mem=f'{mem:0.2f} GB')
        y_pred = torch.sigmoid(y_pred).to('cpu').numpy()
        label = label.to('cpu').numpy()
        start_idx = step * CFG.valid_bs
        end_idx = start_idx + CFG.valid_bs
        for i, (x1, y1, x2, y2) in enumerate(xyxys[start_idx:end_idx]):
            mask_pred[y1:y2, x1:x2] += y_pred[i].squeeze()
            mask_count[y1:y2, x1:x2] += np.ones((CFG.tile_size, CFG.tile_size))
            mask_gt[y1:y2, x1:x2] = label[i].squeeze(0)
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(valid_loader)  # save in pth.name
    assert np.any(mask_count) != 0
    mask_pred /= mask_count
    mask_pred = mask_pred[:ori_h, :ori_w]
    mask_gt = mask_gt[:ori_h, :ori_w]
    assert not np.isnan(np.any(mask_pred))
    # 找出最佳th
    for th in np.arange(3, 7, 0.5) / 10:
        scores = record_scores(torch.from_numpy(mask_gt).to(device), torch.from_numpy(mask_pred).to(device), thr=th)
        avg_score = scores["avg_score"]
        if avg_score > best_score:
            ori_score = best_score
            best_score = avg_score
            out_scores = scores
            best_th = th
            print("best_th={}, score up: {:2f}->{:2f}".format(best_th, ori_score, best_score))
    mask_pred = (mask_pred >= best_th).astype(int)
    if epoch % CFG.save_log_turn == 0:  # 每5轮存一次图片
        cv2.imwrite(CFG.log_samples_dir + f"{name}-epoch_{epoch + 1}.png", mask_pred * 255)
    if epoch == CFG.epochs - 1:  # 最后一轮显示对比原图
        cv2.imwrite(CFG.log_samples_dir + f"{name}-gt.png", mask_gt * 255)

    '''这里只是验证一张图片而已，所以暂时不在这里保存权重'''
    return avg_loss, best_th, out_scores


'''以下处理一个list的valid'''


def valid_step(valid_loader_list, valid_xyxys, valid_ori_sizes, valid_names, model, criterion, device, epoch):
    all_loss = []
    all_th = []
    all_avg_scores = []
    all_dice_scores = []
    all_iou_scores = []
    all_hausdorff_scores = []
    for valid_loader, xyxys, ori_size, name in zip(valid_loader_list, valid_xyxys, valid_ori_sizes, valid_names):
        avg_loss, best_th, best_scores = once_valid_step(valid_loader, xyxys, ori_size, name, model, criterion, device,
                                                        epoch)
        all_loss.append(avg_loss)
        all_th.append(best_th)
        all_avg_scores.append(best_scores["avg_score"].to("cpu").numpy().item())
        all_dice_scores.append(best_scores["dice_score"].to("cpu").numpy().item())
        all_iou_scores.append(best_scores["iou_score"].to("cpu").numpy().item())
        all_hausdorff_scores.append((best_scores["hausdorff_score"]))

    mean_loss = np.mean(np.array(all_loss))
    index = all_avg_scores.index(np.max(all_avg_scores))
    best_th = all_th[index]
    mean_score = np.mean(np.array(all_avg_scores))
    mean_dice_score = np.mean(np.array(all_dice_scores))
    mean_iou_score = np.mean(np.array(all_iou_scores))
    mean_hausdorff_score = np.mean(np.array(all_hausdorff_scores))

    torch.cuda.empty_cache()
    return mean_loss,best_th, mean_score, mean_dice_score, mean_iou_score, mean_hausdorff_score


class complex_criterion():
    def __init__(self):
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.iou_loss = IoULoss()
        self.hd_loss = HausdorffDTLoss()

    def __call__(self, pred, label):
        return self.bce_loss(pred, label) \
            # + 0.5 * self.hd_loss(pred, label)
            # + 0.5 * self.iou_loss(pred, label)



if __name__ == '__main__':

    # wandb.init(
    #     project='MuscularAtrophyModel',
    #     name=f"experiment_{CFG.exp}",
    #     config={
    #         "learning_rate": CFG.lr,
    #         "min_lr": CFG.min_lr,
    #         "weight_decay": CFG.weight_decay,
    #         "model": CFG.model_name,
    #         "tile_size": CFG.tile_size,
    #         "train_stride_step": CFG.tile_size // CFG.train_stride,
    #         "epochs": CFG.epochs,
    #         "train_bs": CFG.train_bs
    #     }
    # )

    model = STS2DModel(CFG, model_name=CFG.model_name)
    if CFG.pretrained:
        model.load_state_dict(torch.load(CFG.checkpoint))
    model = model.to(CFG.device)
    criterion = nn.BCEWithLogitsLoss()  # nn.BCEWithLogitsLoss() nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=CFG.lr,
                            betas=(0.9, 0.999),
                            weight_decay=CFG.weight_decay
                            )
    scheduler = get_scheduler(CFG, optimizer)
    start_epoch = CFG.start_epoch

    for epoch in range(start_epoch, CFG.epochs + start_epoch):
        print("train:")
        train_loss = train_step(train_loader, model, criterion, optimizer, CFG.device, epoch)
        print("valid:")
        valid_loss, thr, score, dice, iou, hausdorff = valid_step(valid_dataloader_list, valid_xyxys, valid_ori_sizes,
                                                               valid_names, model, criterion, CFG.device, epoch)
        # wandb.log({
        #     "train loss": train_loss,
        #     "valid loss": valid_loss,
        #     "average score": score,
        #     "dice score": dice,
        #     "iou score": iou,
        #     "hausdorff score": hausdorff
        # })
        save_path = CFG.log_checkpoint_dir + \
                    "{}-score_{:3f}-thr_{}-dice_{:4f}-iou_{:4f}-hdf_{:4f}-train_loss_{:3f}-valid_loss_{:3f}-epoch_{}.pth".format(
                        CFG.model_name, score, thr, dice, iou, hausdorff, train_loss, valid_loss, epoch
                    )
        if CFG.model_name == "Segformer":
            model.save_pretrained(save_path)
        else:
            torch.save(model.state_dict(), save_path)

    # wandb.finish()
