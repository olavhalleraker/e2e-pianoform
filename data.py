import gin
import os
import cv2
import torch

import numpy as np

from loguru import logger
from rich.progress import track
from torch.utils.data import Dataset
from torchvision import transforms
from utils import check_and_retrieveVocabulary

@logger.catch
def batch_preparation_ctc(data):
    images = [sample[0] for sample in data]
    gt = [sample[1] for sample in data]
    L = [sample[2] for sample in data]
    T = [sample[3] for sample in data]

    max_image_width = max([img.shape[2] for img in images])
    max_image_height = max([img.shape[1] for img in images])

    X_train = torch.ones(size=[len(images), 1, max_image_height, max_image_width], dtype=torch.float32)

    for i, img in enumerate(images):
        c, h, w = img.size()
        X_train[i, :, :h, :w] = img
    
    max_length_seq = max([len(w) for w in gt])
    Y_train = torch.zeros(size=[len(gt),max_length_seq])
    for i, seq in enumerate(gt):
        Y_train[i, 0:len(seq)] = torch.from_numpy(np.asarray([char for char in seq]))

    return X_train, Y_train, L, T

# @logger.catch
# @gin.configurable
# def load_data(partition_file, resize_ratio, use_raw_krn=False, load_distorted=False, extension=".bekrn"):
#     X = []
#     Y = []
#     with open(partition_file) as partfile:
#         part_lines = partfile.read()
#         part_lines = part_lines.split("\n")
#         for file_path in track(part_lines, description="Loading..."):
#             if extension != ".bekrn":
#                 file_path = file_path.replace(".bekrn", extension)
#             krn = None
#             krnlines = []
#             file_path = f"{file_path}"
#             if os.path.isfile(file_path):
#                 with open(file_path) as krnfile:
#                     krn = krnfile.read()
#                     krn = krn.replace(" ", " <s> ")
#                     krn = krn.replace("·", " ")
#                     lines = krn.split("\n")
#                     for line in lines:
#                         line = line.replace("\t", " <t> ")
#                         line = line.split(" ")
#                         if len(line) > 1:
#                             line.append("<b>")
#                             krnlines.append(line)
#                     # for raw_line in lines:
#                     #     # -- insert tab spacer, split on spaces ---------------------------
#                     #     raw_line = raw_line.replace("\t", " <t> ")
#                     #     tokens = raw_line.split(" ")

#                     #     # -- STRIP leading '*' from any token (clefs, meter, key, etc.) ---
#                     #     tokens = [tok[1:] if tok.startswith('*') else tok
#                     #             for tok in tokens]

#                     #     # -- skip empty or 1-token lines, add bar token -------------------
#                     #     if len(tokens) > 1:
#                     #         tokens.append("<b>")
#                     #         krnlines.append(tokens)
#                     if os.path.exists(f"{file_path.split('.')[0]}.jpg"):
#                         if load_distorted:
#                             height = 256
#                             img = cv2.imread(f"{file_path.split('.')[0]}_distorted.jpg", 0)
#                             width = int(float(height * img.shape[1]) / img.shape[0])
#                             img =  cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
#                             if (height//8) * (width//16) > len(sum(krnlines, [])):
#                                 width = int(np.ceil(img.shape[1] * resize_ratio))
#                                 height = int(np.ceil(img.shape[0] * resize_ratio))
#                                 img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
#                                 img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
#                                 X.append(img)
#                                 Y.append(sum(krnlines, []))
#                         else:
#                             img = cv2.imread(f"{file_path.split('.')[0]}.jpg", 0)
#                             width = int(np.ceil(img.shape[1] * resize_ratio))
#                             height = int(np.ceil(img.shape[0] * resize_ratio))
#                             img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
#                             img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
#                             X.append(img)
#                             Y.append(sum(krnlines, []))

#     return X, Y
# ────────────────────────────────────────────────────────────────────
@logger.catch
@gin.configurable
def load_data(partition_file,
              resize_ratio: float = 1.0,
              use_raw_krn: bool = False,
              load_distorted: bool = False,
              extension: str = ".bekrn"):
    """
    Read a partition file, build two parallel lists:
        X … list[np.ndarray]  – each staff image (grayscale)
        Y … list[list[str]]  – token sequence for that staff
    All header lines (**ekern …) are skipped and every remaining token
    is normalised by removing a leading '*'.
    """
    X, Y = [], []

    with open(partition_file) as partfile:
        for file_path in track(partfile.read().splitlines(), description="Loading…"):

            if not file_path:
                continue
            if extension != ".bekrn":
                file_path = file_path.replace(".bekrn", extension)

            if not os.path.isfile(file_path):
                continue

            # ---- read & tokenise label --------------------------------------
            with open(file_path) as krnfile:
                tokens_per_line = []
                for raw in krnfile.read().replace("·", " ").split("\n"):

                    # 1) skip header lines  **ekern …      (* covers **ekern_1.0)
                    if raw.lstrip().startswith("**ekern"):
                        continue

                    # 2) replace tabs by explicit <t> spacer, split on spaces
                    raw = raw.replace("\t", " <t> ")
                    toks = raw.split()

                    # 3) remove leading '*' from musical directive tokens
                    toks = [tok[1:] if tok.startswith("*") else tok for tok in toks]

                    if toks:                      # keep non-empty lines
                        toks.append("<b>")        # bar delimiter
                        tokens_per_line.extend(toks)

            # ---- read image --------------------------------------------------
            img_path = f"{file_path.rsplit('.', 1)[0]}.jpg"
            if not os.path.exists(img_path):
                continue                          # skip if image missing

            img = cv2.imread(img_path, 0)         # grayscale
            h = int(np.ceil(img.shape[0] * resize_ratio))
            w = int(np.ceil(img.shape[1] * resize_ratio))
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            X.append(img)
            Y.append(tokens_per_line)

    return X, Y
# ────────────────────────────────────────────────────────────────────

class PoliphonicDataset(Dataset):
    def __init__(self, partition_file) -> None:
        self.x, self.y = load_data(partition_file)

        self.tensorTransform = transforms.ToTensor()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = self.tensorTransform(self.x[index])
        gt = torch.from_numpy(np.asarray([self.w2i[token] for token in self.y[index]]))
        
        return image, gt, (image.shape[2] // 8) * (image.shape[1] // 16), len(gt)

    def get_max_hw(self):
        m_width = np.max([img.shape[1] for img in self.x])
        m_height = np.max([img.shape[0] for img in self.x])

        return m_height, m_width
    
    def get_max_seqlen(self):
        return np.max([len(seq) for seq in self.y])

    def vocab_size(self):
        return len(self.w2i)

    def get_gt(self):
        return self.y
    
    def set_dictionaries(self, w2i, i2w):
        self.w2i = w2i
        self.i2w = i2w
        self.padding_token = w2i['<pad>']
    
    def get_dictionaries(self):
        return self.w2i, self.i2w
    
    def get_i2w(self):
        return self.i2w

@gin.configurable
def load_dataset(train_path=None, val_path=None, test_path=None, corpus_name=None):
    train_dataset = PoliphonicDataset(partition_file=train_path)
    val_dataset = PoliphonicDataset(partition_file=val_path)
    test_dataset = PoliphonicDataset(partition_file=test_path)

    w2i, i2w = check_and_retrieveVocabulary([train_dataset.get_gt(), val_dataset.get_gt(), test_dataset.get_gt()], "vocab/", f"{corpus_name}")

    train_dataset.set_dictionaries(w2i, i2w)
    val_dataset.set_dictionaries(w2i, i2w)
    test_dataset.set_dictionaries(w2i, i2w)

    return train_dataset, val_dataset, test_dataset

from pathlib import Path, PurePath
import random

def label_to_image(label_path: Path, variant: str | None = None) -> Path:
    """
    Map a .bekrn/.krn file to its image.

    variant:
        None   – if both clean & distorted exist, return one at random
        "clean"      – always pick <stem>.jpg
        "distorted"  – always pick <stem>_distorted.jpg
    """
    stem = label_path.with_suffix("")
    clean = stem.with_suffix(".jpg")
    dist  = stem.with_name(stem.name + "_distorted.jpg")

    if variant == "clean":
        return clean
    if variant == "distorted":
        return dist
    # variant None → choose whichever exists (or the only one that exists)
    if clean.exists() and dist.exists():
        return random.choice([clean, dist])
    return clean if clean.exists() else dist