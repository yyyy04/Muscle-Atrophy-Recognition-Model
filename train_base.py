import wandb
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from config import CFG
from torch import optim
from models import STS2DModel
import torch.utils.data as data
from datasets import STS_Dataset
from scheduler import get_scheduler
from torch.cuda.amp import GradScaler
from myscores.scores import record_scores
from utils import get_train_vaild_base_datasets, get_transforms

'''log'''
wandb.login(key="99cf7d42bf6827ba650272cf79a1172a1f28db2c")

'''prepare dataloaders'''
train_images, train_masks, valid_images, valid_masks = get_train_vaild_base_datasets()

'''no aug'''

train_ds = STS_Dataset(train_images, train_masks, transform=get_transforms("train", CFG))
valid_ds = STS_Dataset(valid_images, valid_masks, transform=get_transforms('valid', CFG))
train_loader = data.DataLoader(
    train_ds,
    batch_size=CFG.train_bs,
    shuffle=True,
    pin_memory=False,
    num_workers=CFG.num_workers,
    drop_last=True
)
valid_loader = data.DataLoader(
    valid_ds,
    batch_size=CFG.valid_bs,
    shuffle=False,
    num_workers=CFG.num_workers,
    drop_last=False
)


def train_step(train_loader, model, criterion, optimizer, device, epoch):
    model.train()
    epoch_loss = 0
    scaler = GradScaler(enabled=CFG.use_amp)
    bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (image, label) in bar:
        optimizer.zero_grad()
        outputs = model(image.to(device))
        # cpu_output = outputs.to("cpu").detach().numpy()
        # cpu_label = label.to("cpu").detach().numpy()
        loss = criterion(outputs.squeeze().to(torch.float), label.to(device).to(torch.float))
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


def once_thr_valid(valid_loader, model, criterion, device, thr, epoch):
    model.eval()
    all_loss = 0
    all_avg_scores = 0
    all_dice_scores = 0
    all_iou_scores = 0
    all_hausdorff_scores = 0

    bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    for step, (image, label) in bar:
        image = image.to(device)
        label = label.to(device).to(torch.float)
        with torch.no_grad():
            y_pred = model(image).to(torch.float)
            loss = criterion(y_pred.squeeze(0), label)
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        bar.set_postfix(loss=f'{loss.item():0.4f}', thr=thr, epoch=epoch + 1, gpu_mem=f'{mem:0.2f} GB')
        y_pred = torch.sigmoid(y_pred)
        all_loss += loss.item()
        # 这里的score是一张一张地比较的，所以可以把valid_bs设为1，并且维度都squeeze掉

        scores = record_scores(label.squeeze(), y_pred.squeeze(), thr=thr)
        all_avg_scores += scores["avg_score"]
        all_dice_scores += scores["dice_score"]
        all_iou_scores += scores["iou_score"]
        all_hausdorff_scores += scores["hausdorff_score"]
    avg_loss = all_loss / len(valid_loader)
    avg_score = all_avg_scores / len(valid_loader)
    dice_score = all_dice_scores / len(valid_loader)
    iou_score = all_iou_scores / len(valid_loader)
    hausdorff_score = all_hausdorff_scores / len(valid_loader)

    return {
        "avg_loss": avg_loss,
        "avg_score": avg_score,
        "dice_score": dice_score,
        "iou_score": iou_score,
        "hausdorff_score": hausdorff_score
    }


def valid_step(valid_loader, model, criterion, device, epoch):
    best_thr = 0
    best_score = 0
    best_dice_score = 0
    best_iou_score = 0
    best_hausdorff_score = 0
    valid_loss = 0
    for thr in np.arange(3, 7, 0.5) / 10:
        result = once_thr_valid(valid_loader, model, criterion, device, thr, epoch)
        if best_score < result["avg_score"]:
            best_thr = thr
            valid_loss = result["avg_loss"]
            best_score = result["avg_score"]
            best_dice_score = result["dice_score"]
            best_iou_score = result["iou_score"]
            best_hausdorff_score = result["hausdorff_score"]

    torch.cuda.empty_cache()
    return {
        "best_thr": best_thr,
        "valid_loss": valid_loss,
        "best_score": best_score,
        "best_dice_score": best_dice_score,
        "best_iou_score": best_iou_score,
        "best_hausdorff_score": best_hausdorff_score
    }


if __name__ == '__main__':

    wandb.init(
        project='MuscularAtrophyModel',
        name=f"experiment_{CFG.exp}",
        config={
            "learning_rate": CFG.lr,
            "min_lr": CFG.min_lr,
            "weight_decay": CFG.weight_decay,
            "model": CFG.model_name,
            "size": CFG.size,
            "epochs": CFG.epochs,
            "train_bs": CFG.train_bs,
        }
    )

    model = STS2DModel(CFG, model_name=CFG.model_name)
    if CFG.pretrained:
        model.load_state_dict(torch.load(CFG.checkpoint))
    model = model.to(CFG.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=CFG.lr,
                            betas=(0.9, 0.999),
                            weight_decay=CFG.weight_decay,
                            )
    scheduler = get_scheduler(CFG, optimizer)
    start_epoch = CFG.start_epoch

    for epoch in range(start_epoch, CFG.epochs + start_epoch):
        print("train:")
        train_loss = train_step(train_loader, model, criterion, optimizer, CFG.device, epoch)
        print("valid:")
        results = valid_step(valid_loader, model, criterion, CFG.device, epoch)
        thr = results["best_thr"]
        valid_loss = results["valid_loss"]
        score = results["best_score"]
        dice = results["best_dice_score"]
        iou = results["best_iou_score"]
        hausdorff = results["best_hausdorff_score"]

        wandb.log({
            "train loss": train_loss,
            "valid loss": valid_loss,
            "average score": score,
            "dice score": dice,
            "iou score": iou,
            "hausdorff score": hausdorff
        })
        save_path = CFG.log_checkpoint_dir + \
            "{}-score_{:3f}-thr_{}-dice_{:4f}-iou_{:4f}-hdf_{:4f}-train_loss_{:3f}-valid_loss_{:3f}-epoch_{}.pth".format(
                CFG.model_name, score, thr, dice, iou, hausdorff, train_loss, valid_loss, epoch
            )
        torch.save(model.state_dict(), save_path)
