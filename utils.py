import albumentations as A
from config import CFG
from pathlib import Path
import random
import os
import numpy as np
import cv2
from tqdm import tqdm


def get_slice(stride, img_path, msk_path=None):
    images = []
    masks = []
    xyxys = []
    if msk_path:
        if CFG.model_name == "Segformer":
            image = cv2.imread(img_path, )
        else:
            image = cv2.imread(img_path, 0)
        if CFG.msk_suffix == ".npy":
            mask = np.load(msk_path)
        else:
            mask = cv2.imread(msk_path, 0).astype('float32') / 255.0
        ori_size = image.shape[:2]

        # assert image.shape == mask.shape
        pad0 = CFG.tile_size - image.shape[0] % CFG.tile_size
        pad1 = CFG.tile_size - image.shape[1] % CFG.tile_size

        if len(image.shape) == 3:
            image = np.pad(image, [(0, pad0), (0, pad1), (0, 0)], constant_values=0)
        else:
            image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)

        x1_list = list(range(0, image.shape[1] - CFG.tile_size + 1, stride))
        y1_list = list(range(0, image.shape[0] - CFG.tile_size + 1, stride))

        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + CFG.tile_size
                x2 = x1 + CFG.tile_size
                if CFG.mask_loation:
                    if np.all(mask[y1:y2, x1:x2]) == 0:
                        continue
                images.append(image[y1:y2, x1:x2, None])
                masks.append(mask[y1:y2, x1:x2, None])
                xyxys.append([x1, y1, x2, y2])
    else:
        if CFG.model_name == "Segformer":
            image = cv2.imread(img_path)
        else:
            image = cv2.imread(img_path, 0)
        ori_size = image.shape[:2]

        pad0 = CFG.test_tile_size - image.shape[0] % CFG.test_tile_size
        pad1 = CFG.test_tile_size - image.shape[1] % CFG.test_tile_size
        if len(image.shape) == 3:
            image = np.pad(image, [(0, pad0), (0, pad1), (0, 0)], constant_values=0)
        else:
            image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        x1_list = list(range(0, image.shape[1] - CFG.tile_size + 1, stride))
        y1_list = list(range(0, image.shape[0] - CFG.tile_size + 1, stride))

        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + CFG.test_tile_size
                x2 = x1 + CFG.test_tile_size
                images.append(image[y1:y2, x1:x2, None])
                xyxys.append([x1, y1, x2, y2])

    return images, masks, xyxys, ori_size


def get_sliced_img_msk(names_list, images_dir, mask_dir=None, tag="train"):
    sliced_images = []
    sliced_masks = []
    sliced_xyxys = []
    ori_sizes = []
    names = []

    if tag == "train":
        stride = CFG.train_stride
        print("prepare train datas")
    elif tag == "valid":
        stride = CFG.valid_stride
        print("prepare valid datas")
    elif tag == "test":
        stride = CFG.test_stride
        print("prepare test datas")
    else:
        print("the function 'get_sliced_img_msk' need to set the 'tag' as train, valid or test.")

    bar = tqdm(names_list, total=len(names_list))

    for name in bar:
        image = images_dir + os.sep + name + CFG.img_suffix
        if tag != "test":
            mask = mask_dir + os.sep + name + CFG.msk_suffix
            sliced_image_list, sliced_mask_list, xyxys, ori_size = get_slice(stride, image, mask)
        else:
            sliced_image_list, _, xyxys, ori_size = get_slice(stride, image)

        if tag == "train":
            sliced_images.extend(sliced_image_list)
            sliced_masks.extend(sliced_mask_list)

        if tag == "valid":
            sliced_images.append(sliced_image_list)
            sliced_masks.append(sliced_mask_list)
            sliced_xyxys.append(xyxys)
            ori_sizes.append(ori_size)
            names.append(name)

        if tag == "test":
            sliced_images.extend(sliced_image_list)
            sliced_xyxys.append(xyxys)
            ori_sizes.append(ori_size)
            names.append(name)

    return sliced_images, sliced_masks, sliced_xyxys, ori_sizes, names


def data_names_log(doc_txt: str, data_list: list):
    with open(doc_txt, "w") as f:
        f.write("\n".join(data_list))


def get_train_valid_datasets():
    images_dir = CFG.data_dirs + "/image"
    mask_dir = CFG.data_dirs + "/mask"
    data_path_list = list(iter(Path(images_dir).glob("*" + CFG.img_suffix)))
    data_names_list = [f.name.split(".")[0] for f in data_path_list]

    # train datas
    random.shuffle(data_names_list)
    test_numbers = int(CFG.rate_test * len(data_names_list))
    train_valid_names_list = data_names_list[:-test_numbers]
    test_names_list = data_names_list[-test_numbers:]

    valid_numbers = int(CFG.rate_valid * len(train_valid_names_list))
    train_names_list = data_names_list[:-valid_numbers]
    valid_names_list = data_names_list[-valid_numbers:]

    train_sliced_images, train_sliced_masks, _, _, _ = get_sliced_img_msk(train_names_list, images_dir, mask_dir,
                                                                          tag="train")
    valid_sliced_images_list, valid_sliced_masks_list, valid_xyxys, valid_ori_sizes, names = get_sliced_img_msk(
        valid_names_list, images_dir, mask_dir, tag="valid")
    # names = list(map(lambda n: n.split('.')[0], names))
    return {
        "train_datasets": (train_sliced_images, train_sliced_masks),
        "valid_datasets": (valid_sliced_images_list, valid_sliced_masks_list, valid_xyxys, valid_ori_sizes, names)
    }


def decode_images(images_list, flag="image"):
    res = []
    bar = tqdm(images_list, total=len(images_list))
    for img in bar:
        if flag == "image":
            img = cv2.imread(img, 0)
        elif flag == "mask":
            img = cv2.imread(img, 0) / 255
        res.append(img)
    return res


def get_train_vaild_base_datasets():
    images_dir = CFG.data_dirs + "/image"
    mask_dir = CFG.data_dirs + "/mask"
    data_path_list = list(iter(Path(images_dir).glob("*" + CFG.img_suffix)))
    data_names_list = [f.name.split(".")[0] for f in data_path_list]

    random.shuffle(data_names_list)
    test_numbers = int(CFG.rate_test * len(data_names_list))
    train_valid_names_list = data_names_list[:-test_numbers]
    test_names_list = data_names_list[-test_numbers:]

    valid_numbers = int(CFG.rate_valid * len(train_valid_names_list))
    train_names_list = train_valid_names_list[:-valid_numbers]
    valid_names_list = train_valid_names_list[-valid_numbers:]
    data_names_log("results/train.txt", train_names_list)
    data_names_log("results/valid.txt", valid_names_list)
    data_names_log("results/test.txt", test_names_list)

    train_images = list(iter(map(lambda x: images_dir + "/" + x + CFG.img_suffix, train_names_list)))
    train_masks = list(iter(map(lambda x: mask_dir + "/" + x + CFG.msk_suffix, train_names_list)))
    valid_images = list(iter(map(lambda x: images_dir + "/" + x + CFG.img_suffix, valid_names_list)))
    valid_masks = list(iter(map(lambda x: mask_dir + "/" + x + CFG.msk_suffix, valid_names_list)))

    print("prepare train_images:")
    train_images = decode_images(train_images, "image")
    print("prepare train_masks:")
    train_masks = decode_images(train_masks, "mask")
    print("prepare valid_images:")
    valid_images = decode_images(valid_images, "image")
    print("prepare valid_masks")
    valid_masks = decode_images(valid_masks, "mask")

    return train_images, train_masks, valid_images, valid_masks


def get_test_datasets():
    images_dir = CFG.test_data_dirs + "/image"
    images_path_list = list(iter(Path(images_dir).glob("*" + CFG.img_suffix)))
    images_names_list = [f.name.split(".")[0] for f in images_path_list]

    test_sliced_images, _, test_xyxys, test_ori_sizes, names = get_sliced_img_msk(
        images_names_list, images_dir, mask_dir=None, tag="test"
    )
    # names = list(map(lambda n: n.split('.')[0], names))
    return test_sliced_images, test_xyxys, test_ori_sizes, names


def get_txt_datasets(data_path):
    images_names_list = []
    with open(data_path, "r") as f:
        for line in f:
            images_names_list.append(line.strip())
    images_dir = CFG.data_dirs + "/image"
    images_names_list = [str(name).split(".")[0] for name in images_names_list]
    test_sliced_images, _, test_xyxys, test_ori_sizes, names = get_sliced_img_msk(
        images_names_list, images_dir, mask_dir=None, tag="test"
    )
    # names = list(map(lambda n: n.split('.')[0], names))
    return test_sliced_images, test_xyxys, test_ori_sizes, names


def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)
    elif data == 'test':
        aug = A.Compose(cfg.test_aug_list)
    elif data == 'base':
        aug = A.Compose(cfg.base_aug_list)
    return aug
