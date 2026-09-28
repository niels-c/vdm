# Every transform below operates on `imgmap`, a list of PIL frames from one video clip (not a
# single image). `consistent=True` applies the same randomly-sampled transform to every frame
# in the clip, which is what you want for video: a shared crop/flip/jitter keeps the frames
# geometrically and photometrically aligned so the clip still depicts coherent motion.
# `consistent=False` instead re-samples the transform independently per frame.

import numbers
import random
from collections.abc import Iterable

from PIL import Image
from torchvision import transforms


class Scale:
    """Resize every frame of a clip.

    BILINEAR: build_transforms scales a 224 crop down to img_dim (64).
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgmap):
        # assert len(imgmap) > 1 # list of images
        img1 = imgmap[0]
        if isinstance(self.size, int):
            w, h = img1.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return imgmap
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return [i.resize((ow, oh), self.interpolation) for i in imgmap]
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return [i.resize((ow, oh), self.interpolation) for i in imgmap]
        else:
            return [i.resize(self.size, self.interpolation) for i in imgmap]


class CenterCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgmap):
        img1 = imgmap[0]
        w, h = img1.size
        th, tw = self.size
        x1 = round((w - tw) / 2.0)
        y1 = round((h - th) / 2.0)
        return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]


class RandomCrop:
    def __init__(self, size, consistent=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.consistent = consistent

    def __call__(self, imgmap):
        img1 = imgmap[0]
        w, h = img1.size
        if self.size is not None:
            th, tw = self.size
            if w == tw and h == th:
                return imgmap
            if self.consistent:
                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)
                return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]
            else:
                result = []
                for i in imgmap:
                    x1 = random.randint(0, w - tw)
                    y1 = random.randint(0, h - th)
                    result.append(i.crop((x1, y1, x1 + tw, y1 + th)))
                return result
        else:
            return imgmap


class RandomHorizontalFlip:
    def __init__(self, consistent=True, command=None):
        self.consistent = consistent
        if command == "left":
            self.threshold = 0
        elif command == "right":
            self.threshold = 1
        else:
            self.threshold = 0.5

    def __call__(self, imgmap):
        if self.consistent:
            if random.random() < self.threshold:
                return [i.transpose(Image.FLIP_LEFT_RIGHT) for i in imgmap]
            else:
                return imgmap
        else:
            result = []
            for i in imgmap:
                if random.random() < self.threshold:
                    result.append(i.transpose(Image.FLIP_LEFT_RIGHT))
                else:
                    result.append(i)
            assert len(result) == len(imgmap)
            return result


class ToTensor:
    def __call__(self, imgmap):
        totensor = transforms.ToTensor()
        return [totensor(i) for i in imgmap]


class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, imgmap):
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        return [normalize(i) for i in imgmap]


def build_transforms(dataset, img_dim):
    """(train_transform, val_transform) for one dataset, shared by models/cpc/train_cpc.py and
    train_action_recog.py.

    The val pipeline is the deterministic counterpart of the train one: identical crop geometry
    and Normalize, but no random flip/crop/jitter/gray. Validating under the training
    augmentation makes val loss vary with the augmentation draw rather than the model, which
    matters because val loss drives early stopping and best-checkpoint selection.
    """
    if dataset in ("UCF101",):
        # short size 256, crop to 224x224, then scale to img_dim x img_dim
        train = transforms.Compose(
            [
                RandomHorizontalFlip(consistent=True),
                RandomCrop(size=224, consistent=True),
                Scale(size=(img_dim, img_dim)),
                ToTensor(),
                Normalize(),
            ]
        )
        val = transforms.Compose(
            [
                CenterCrop(size=224),
                Scale(size=(img_dim, img_dim)),
                ToTensor(),
                Normalize(),
            ]
        )
    else:
        raise ValueError(f"no transform pipeline defined for dataset={dataset!r}")
    return train, val
