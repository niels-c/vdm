"""Minimal TFRecord framing reader, shared by RSM/downstream/cifar10.py (whose records hold
tf.train.Example-encoded CIFAR-10 images) and plotting/plot_training_curves.py (whose records hold
tensorboardX.proto.event_pb2.Event-encoded training curves). The two payload types are decoded
independently by their own callers; only the framing underneath, i.e., length-prefixed records with a
masked CRC32C before and after each one, is common, and is simple and stable enough to read
directly rather than depending on tensorflow just for it.
"""

import struct


def read_tfrecords(path):
    """Yields each record's raw payload bytes from the TFRecord file at `path`, in order. CRC32C
    checksums are read past but not verified: these are trusted local files, not a network or
    otherwise untrusted source.
    """
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if not header:
                return
            (length,) = struct.unpack("<Q", header)
            f.read(4)  # masked CRC32C of length, not verified
            data = f.read(length)
            f.read(4)  # masked CRC32C of data, not verified
            yield data
