from infere import test_step as get_sliced_images_preds
from utils import get_txt_datasets, get_transforms
from myscores.scores import record_scores
from datasets import STS_test_Dataset
import torch
from tqdm import tqdm
from config import CFG
import torch.utils.data as data
from models import STS2DModel
import numpy as np
import cv2

print("start")
valid_sliced_images, valid_xyxys, valid_ori_sizes, names = get_txt_datasets(CFG.valid_txt)
valid_ds = STS_test_Dataset(valid_sliced_images, transform=get_transforms("valid", cfg=CFG))
valid_loader = data.DataLoader(valid_ds,
                               batch_size=CFG.test_bs,
                               shuffle=False,
                               pin_memory=False,
                               num_workers=CFG.num_workers,
                               drop_last=False)

if __name__ == '__main__':
    model = STS2DModel(CFG, model_name=CFG.model_name)
    model.load_state_dict(torch.load(CFG.checkpoint))
    print("valid process:")
    sliced_images_preds = get_sliced_images_preds(valid_loader, model, CFG.device, tta=CFG.tta)
    print("generating each pred")
    all_avg_scores = []
    all_dice_scores = []
    all_iou_scores = []
    all_hausdorff_scores = []
    for idx, name in tqdm(enumerate(names), total=(len(names))):
        per_xyxys = valid_xyxys[idx]
        per_ori_size = valid_ori_sizes[idx]
        ori_h, ori_w = per_ori_size[-2], per_ori_size[-1]

        pad0 = CFG.tile_size - ori_h % CFG.tile_size
        pad1 = CFG.tile_size - ori_w % CFG.tile_size

        mask_pred = np.zeros(per_ori_size)
        mask_pred = np.pad(mask_pred, [(0, pad0), (0, pad1)], constant_values=0)
        mask_count = np.zeros(mask_pred.shape)

        for i, (x1, y1, x2, y2) in enumerate(per_xyxys):
            mask_pred[y1:y2, x1:x2] += sliced_images_preds.popleft().squeeze()
            mask_count[y1:y2, x1:x2] += np.ones((CFG.test_tile_size, CFG.test_tile_size))
        mask_pred /= mask_count
        mask_pred = mask_pred[:ori_h, :ori_w]
        mask_gt_path = CFG.data_dirs + "/mask/" + name + CFG.msk_suffix
        mask_gt = cv2.imread(mask_gt_path, 0) / 255.
        if CFG.thr:
            mask_pred = (mask_pred >= CFG.thr).astype(int)
        scores = record_scores(torch.from_numpy(mask_gt).to(CFG.device), torch.from_numpy(mask_pred).to(CFG.device))
        all_avg_scores.append(scores["avg_score"].to("cpu").numpy().item())
        all_dice_scores.append(scores["dice_score"].to("cpu").numpy().item())
        all_iou_scores.append(scores["iou_score"].to("cpu").numpy().item())
        all_hausdorff_scores.append(scores["hausdorff_score"])

    mean_score = np.mean(np.array(all_avg_scores))
    mean_dice_score = np.mean(np.array(all_dice_scores))
    mean_iou_score = np.mean(np.array(all_iou_scores))
    mean_hausdorff_score = np.mean(np.array(all_hausdorff_scores))
    with open(f"results/record_tta_scores.txt", "a") as f:
        f.write(str(CFG.model_name)+":")
        f.write("avg-score:{} dice-score:{} iou-score:{} hausdorff-score:{}\n".format(
            mean_score, mean_dice_score, mean_iou_score, mean_hausdorff_score
        ))
