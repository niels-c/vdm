"""CIFAR-10 read straight from TensorFlow Datasets' .tfrecord files.

The CIFAR-10 release available here is in TFRecord format rather than the pickled-batches format
torchvision.datasets.CIFAR10 expects, and neither `tensorflow` nor `tensorflow_datasets` is
installed. TFRecord framing and tf.train.Example's protobuf schema are simple and have been
stable for years, so this decodes them directly, and CIFAR10TFRecords wraps the result in the
Dataset interface the probe's DataLoader takes.
"""

import io

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from config.model import DPC_SEQ_LEN
from RSM.utils.tfrecords import read_tfrecords


# --- Minimal tf.train.Example decoder ----------------------------------------
def _read_varint(buf, pos):
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _decode_protobuf_fields(buf):
    pos, n = 0, len(buf)
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        field_number, wire_type = tag >> 3, tag & 0x7
        if wire_type == 0:
            value, pos = _read_varint(buf, pos)
        elif wire_type == 2:
            length, pos = _read_varint(buf, pos)
            value = buf[pos : pos + length]
            pos += length
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type} (unexpected in a tf.train.Example)")
        yield field_number, wire_type, value


def _decode_packed_varints(buf):
    values, pos = [], 0
    while pos < len(buf):
        v, pos = _read_varint(buf, pos)
        values.append(v)
    return values


def _decode_tf_example(data):
    features_bytes = next(iter(_decode_protobuf_fields(data)))[2]
    feats = {}
    for _fn, _wt, entry_bytes in _decode_protobuf_fields(features_bytes):
        key, value_msg = None, None
        for pf, _pwt, pv in _decode_protobuf_fields(entry_bytes):
            if pf == 1:
                key = pv.decode("utf-8")
            elif pf == 2:
                value_msg = pv
        kind_field, _kwt, kind_value = next(iter(_decode_protobuf_fields(value_msg)))
        if kind_field == 1:  # bytes_list -> BytesList.value (field 1)
            feats[key] = next(iter(_decode_protobuf_fields(kind_value)))[2]
        elif kind_field == 3:  # int64_list -> Int64List.value (field 1, packed)
            inner = next(iter(_decode_protobuf_fields(kind_value)))[2]
            feats[key] = _decode_packed_varints(inner)
    return feats


class CIFAR10TFRecords(Dataset):
    """CIFAR-10 loaded from data/cifar10/3.0.2/*.tfrecord, reshaped into
    CPC block format: each static image becomes seq_len repeated frames,
    forming one block of shape (N=1, C, seq_len, H, W). N=1 because a
    single image, unlike a video, doesn't have multiple blocks to offer.
    """

    def __init__(self, tfrecord_path, transform=None, seq_len=DPC_SEQ_LEN):
        self.transform = transform
        self.seq_len = seq_len
        self.images, self.targets = [], []
        for record in read_tfrecords(tfrecord_path):
            feats = _decode_tf_example(record)
            self.images.append(np.array(Image.open(io.BytesIO(feats["image"]))))
            self.targets.append(feats["label"][0])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.images[index], self.targets[index]
        seq = [Image.fromarray(img) for _ in range(self.seq_len)]
        t_seq = self.transform(seq) if self.transform is not None else seq
        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0).view(1, self.seq_len, C, H, W).transpose(1, 2)  # (N=1, C, SL, H, W)
        return t_seq, target
