from PIL import Image
import numpy as np
import cv2
from pathlib import Path

def exchange_msk(msk_dir):
    msk_dir = Path(msk_dir)
    msk_list = msk_dir.glob("*.png")
    for name in msk_list:
        msk = np.asarray(Image.open(name))
        msk = msk * 255.
        cv2.imwrite(str(name), msk)

if __name__ == '__main__':
    exchange_msk("data/mask")
