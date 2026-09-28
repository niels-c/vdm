import functools
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import transforms
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from datagen.moving_animals.synthetic_sprites_dataset import SpriteVideoDataset
from config.paths import REPO_ROOT, UCF101_DIR
from config.seed import seed_dataloader_worker

repo_root = REPO_ROOT


def resolve_frame_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(repo_root, path))


def pil_loader(path):
    with open(path, "rb") as f, Image.open(f) as img:
        return img.convert("RGB")


def resolve_split_csv(dataset_dir, mode, which_split=None, test_stem="test_split"):
    """Map a dataset dir + mode to its CSV split path.

    `test_stem` differs by dataset in this repo's on-disk layout: UCF101/other datasets
    reuse their `test_split` file for both "val" and "test" mode.
    """
    if mode == "train":
        stem = "train_split"
    elif mode in ("val", "test"):
        stem = test_stem
    else:
        raise ValueError("wrong mode")
    suffix = f"{which_split:02d}.csv" if which_split is not None else ".csv"
    return os.path.join(dataset_dir, f"{stem}{suffix}")


class _ClipDataset3D(data.Dataset):
    """Shared logic for datasets backed by extracted-frame folders + CSV splits.

    A split CSV lists (frame_dir, video_length) rows; each item is a clip made
    of `num_seq` blocks of `seq_len` frames, spaced `downsample` frames apart.
    Subclasses just resolve the dataset-specific paths/naming and call super().
    """

    def __init__(
        self,
        split_csv,
        action_file,
        action_sep,
        frame_name_fmt,
        mode,
        transform,
        seq_len,
        num_seq,
        downsample,
        action_id_offset=0,
        val_frac=0.3,
        val_random_state=None,
        unit_test_sample=None,
        show_progress=False,
        return_label=False,
    ):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.return_label = return_label
        self.frame_name_fmt = frame_name_fmt

        video_info = pd.read_csv(split_csv, header=None)

        self.action_dict_encode = {}
        self.action_dict_decode = {}
        action_df = pd.read_csv(action_file, sep=action_sep, header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id) - action_id_offset
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # filter out too-short videos
        drop_idx = []
        rows = tqdm(video_info.iterrows(), total=len(video_info)) if show_progress else video_info.iterrows()
        for idx, row in rows:
            _vpath, vlen = row
            if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)
        self.video_info[0] = self.video_info[0].map(resolve_frame_path)

        if mode == "val":
            self.video_info = self.video_info.sample(frac=val_frac, random_state=val_random_state)
        if unit_test_sample is not None:
            self.video_info = self.video_info.sample(unit_test_sample, random_state=val_random_state)
        # shuffle not necessary because use RandomSampler

    def idx_sampler(self, vlen, vpath, index=None):
        """Pick a window of the video and lay out its frame indices as a (num_seq, seq_len) grid:
        num_seq consecutive blocks of seq_len frames each, with `downsample` frames between
        consecutive sampled frames within a block.

        In "val" mode the window is drawn from a generator seeded by `index`, so a clip's frames
        are the same on every epoch. Otherwise val loss would move with the window draw as well
        as the model, and it's val loss that drives early stopping. Train mode keeps the global
        np.random draw seeded per worker by config/seed.py's seed_dataloader_worker.
        """
        span = vlen - self.num_seq * self.seq_len * self.downsample
        if span <= 0:
            return None
        rng = np.random.default_rng(index) if self.mode == "val" else np.random
        start_idx = rng.choice(range(span), 1)
        seq_idx = np.expand_dims(np.arange(self.num_seq), -1) * self.downsample * self.seq_len + start_idx
        seq_idx_block = seq_idx + np.expand_dims(np.arange(self.seq_len), 0) * self.downsample
        return [seq_idx_block, vpath]

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        items = self.idx_sampler(vlen, vpath, index=index)
        if items is None:
            # __init__ already drops videos this short, so reaching here means video_info and the
            # seq_len/num_seq/downsample it was filtered against have gone out of sync
            raise ValueError(
                f"{vpath} has {vlen} frames, too few for num_seq={self.num_seq} x "
                f"seq_len={self.seq_len} x downsample={self.downsample}"
            )

        idx_block, vpath = items
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        seq = [pil_loader(os.path.join(vpath, self.frame_name_fmt.format(i + 1))) for i in idx_block]
        t_seq = self.transform(seq)  # apply same transform

        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)

        if self.return_label:
            # action name is normally two levels above the frame dir (.../ActionName/video_id/), but
            # falls back one level up for datasets laid out without the extra video-id subfolder
            try:
                vname = vpath.split("/")[-3]
                vid = self.encode_action(vname)
            except KeyError:
                vname = vpath.split("/")[-2]
                vid = self.encode_action(vname)

            label = torch.LongTensor([vid])
            return t_seq, label

        return t_seq

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        """give action name, return category. Falls back to a case-insensitive match: UCF101's
        own classInd.txt disagrees in case with its extracted frame folder name for one class
        ("HandStandPushups" vs "HandstandPushups" on disk), a known quirk of the dataset itself
        rather than this repo."""
        if action_name in self.action_dict_encode:
            return self.action_dict_encode[action_name]
        lower_name = action_name.lower()
        for name, act_id in self.action_dict_encode.items():
            if name.lower() == lower_name:
                return act_id
        raise KeyError(action_name)

    def decode_action(self, action_code):
        """give action code, return action name"""
        return self.action_dict_decode[action_code]


class UCF101_3d(_ClipDataset3D):
    def __init__(
        self,
        mode="train",
        transform=None,
        seq_len=10,
        num_seq=5,
        downsample=3,
        which_split=1,
        return_label=False,
    ):
        self.which_split = which_split
        dataset_dir = UCF101_DIR

        super().__init__(
            split_csv=resolve_split_csv(dataset_dir, mode, which_split=which_split),
            action_file=os.path.join(dataset_dir, "classInd.txt"),
            action_sep=" ",
            action_id_offset=1,  # class ids in classInd.txt are 1-indexed
            frame_name_fmt="image_{:05d}.jpg",
            mode=mode,
            transform=transform,
            seq_len=seq_len,
            num_seq=num_seq,
            downsample=downsample,
            val_random_state=666,
            return_label=return_label,
        )


########## Moving animals dataset 3D dataloader ##########
# Adapted from https://github.com/fmi-basel/recurrent-predictive-learning/tree/main
# Possibly: https://github.com/loeweX/Greedy_InfoMax
class DistributedProxySampler(torch.utils.data.distributed.DistributedSampler):
    """Sampler that restricts data loading to a subset of input sampler indices.

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    Note:
        Input sampler is assumed to be of constant size.

    Args:
        sampler (torch.utils.data.Sampler): Input data sampler.
        num_replicas (int, optional): Number of processes participating in
            distributed training.
        rank (int, optional): Rank of the current process within num_replicas.
    """

    def __init__(self, sampler, num_replicas=None, rank=None):
        super(DistributedProxySampler, self).__init__(sampler, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.sampler = sampler

    def __iter__(self):
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)
        indices = list(self.sampler)

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise RuntimeError("{} vs {}".format(len(indices), self.total_size))

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("{} vs {}".format(len(indices), self.num_samples))

        return iter(indices)


def create_validation_sampler(dataset_size, validation_split, shuffle_dataset=True, distributed=False):
    """Creates a sampler for train and validation sets.

    Args:
        dataset_size (int): The size of the dataset.
        validation_split (float): The proportion of the dataset to use for validation.
        shuffle_dataset (bool): Whether to shuffle the dataset before splitting.

    Returns:
        torch.utils.data.sampler.SubsetRandomSampler: A sampler for the train set.
        torch.utils.data.sampler.SubsetRandomSampler: A sampler for the validation set.
    """
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders:
    if distributed:
        train_sampler = DistributedProxySampler(torch.utils.data.sampler.SubsetRandomSampler(train_indices))
        valid_sampler = DistributedProxySampler(torch.utils.data.sampler.SubsetRandomSampler(val_indices))
    else:
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler


class ImageDataLoader:
    """Generic Image DataLoader class for vision datasets.

    Args:
        dataset: Dataset class to use.
        data_dir: Directory where the data is stored.
        grayscale: Whether to convert images to grayscale.
        num_workers: Number of workers for data loading.
        augmentations: Dictionary of augmentations to apply.
        distributed: Whether to use distributed sampling.
        **kwargs: Additional arguments for the dataset class.
    """

    def __init__(
        self,
        dataset,
        data_dir,
        grayscale=False,
        num_workers=16,
        augmentations=None,
        distributed=False,
        worker_seed_base=None,
        **kwargs,
    ):
        self.dataset = dataset
        self.data_dir = data_dir
        self.grayscale = grayscale
        self.num_workers = num_workers
        self.augmentations = augmentations or {}
        self.sampler = DistributedSampler if distributed else None
        # per-rank base seed for DataLoader worker processes (e.g. args.seed + rank * 10000 under
        # DDP); None leaves workers on PyTorch's default fork-inherited RNG state, matching this
        # class's pre-existing (unseeded) behavior
        self.worker_seed_base = worker_seed_base

        self._train_subset = "train+unlabeled"
        self._train_supervised_subset = "train"
        self._val_supervised_subset = "val"
        self._val_subset = "val"
        self._test_subset = "test"
        self._dataset_kwargs = kwargs

    def get_transforms(self, eval=False):
        """Get the image transformations based on augmentations.

        Args:
            eval: Whether the transforms are for evaluation (True) or training (False).

        Returns:
            transforms_list: Composed list of transformations.
        """
        transforms_list = []

        # Random cropping for training
        if self.augmentations.get("randcrop", None) and not eval:
            transforms_list.append(transforms.RandomCrop(self.augmentations["randcrop"]))

        # Center cropping for evaluation
        if self.augmentations.get("randcrop", None) and eval:
            transforms_list.append(transforms.CenterCrop(self.augmentations["randcrop"]))

        # Grayscale conversion (can also be controlled by dataset)
        # and normalization if specified
        if self.augmentations.get("grayscale", None):
            transforms_list.append(transforms.Grayscale())
            transforms_list.append(transforms.ToTensor())
            transforms_list.append(
                transforms.Normalize(mean=self.augmentations["bw_mean"], std=self.augmentations["bw_std"])
            )
        elif self.augmentations.get("mean", None) and self.augmentations.get("std", None):
            transforms_list.append(transforms.ToTensor())
            transforms_list.append(transforms.Normalize(mean=self.augmentations["mean"], std=self.augmentations["std"]))
        else:
            transforms_list.append(transforms.ToTensor())

        transforms_list = transforms.Compose(transforms_list)
        return transforms_list

    def _make_datasets(self, splits):
        """Create datasets for the given splits.

        Args:
            splits: List of dataset splits to create.

        Returns:
            Dictionary of datasets for each split.
        """
        datasets = {}
        for split in splits:
            datasets[split] = self.dataset(
                self.data_dir,
                split=split,
                download=True,
                transform=self.get_transforms(eval=not split.startswith("train")),
                **self._dataset_kwargs,
            )
        return datasets

    def __get_loader(self, split, batch_size, shuffle, sampler=None):
        """Create a data loader for the given split.

        Args:
            split: Dataset split to create loader for.
            batch_size: Batch size for the data loader.
            shuffle: Whether to shuffle the data.
            sampler: Sampler to use for the data loader.

        Returns:
            DataLoader for the given split and the sampler used.
        """
        dataset = self.datasets[split]
        if self.sampler is not None:
            if sampler is None or (sampler is not None and not isinstance(sampler, DistributedSampler)):
                print("Warning: sampler overriden by distributed sampler")
                sampler = self.sampler(dataset)

        worker_init = (
            functools.partial(seed_dataloader_worker, base_seed=self.worker_seed_base)
            if self.worker_seed_base is not None
            else None
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init,
        ), sampler

    def get_train(self, batch_size, sampler=None):
        """Get dataloader for training data.

        Args:
            batch_size: Batch size for the data loader.
            sampler: Sampler to use for the data loader.

        Returns:
            DataLoader for the training split and the sampler used.
        """
        return self.__get_loader(self._train_subset, batch_size=batch_size, shuffle=True, sampler=sampler)

    def get_validation(self, batch_size, sampler=None):
        """Get dataloader for validation data.

        Args:
            batch_size: Batch size for the data loader.
            sampler: Sampler to use for the data loader.

        Returns:
            DataLoader for the validation split and the sampler used.
        """
        return self.__get_loader(self._val_subset, batch_size=batch_size, shuffle=False, sampler=sampler)

    def get_supervised_train(self, batch_size, sampler=None):
        """Get dataloader for supervised training data. (not used)

        Args:
            batch_size: Batch size for the data loader.
            sampler: Sampler to use for the data loader.

        Returns:
            DataLoader for the supervised training split and the sampler used.
        """
        return self.__get_loader(self._train_supervised_subset, batch_size=batch_size, shuffle=True, sampler=sampler)

    def get_supervised_validation(self, batch_size, sampler=None):
        """Get dataloader for supervised validation data. (not used)

        Args:
            batch_size: Batch size for the data loader.
            sampler: Sampler to use for the data loader.

        Returns:
            DataLoader for the supervised validation split and the sampler used.
        """
        return self.__get_loader(self._val_supervised_subset, batch_size=batch_size, shuffle=False, sampler=sampler)

    def get_test(self, batch_size, sampler=None):
        """Get dataloader for test data.

        Args:
            batch_size: Batch size for the data loader.
            sampler: Sampler to use for the data loader.

        Returns:
            DataLoader for the test split and the sampler used.
        """
        return self.__get_loader(self._test_subset, batch_size=batch_size, shuffle=False, sampler=sampler)


class SpriteClipDataset(data.Dataset):
    """Adapts SpriteVideoDataset's flat [T, C, H, W] clip + multi-attribute label tuple to the
    block-structured (num_blocks, C, block_size, H, W) input and single-class-index label that
    DPC_RNN/ActionRecogNet (models/cpc/model_3d.py) share with UCF101_3d.

    `block_size` means generated frames per block. `frame_stride` samples those generated frames
    before returning the model input, so a 15-frame block with stride 3 becomes a 5-frame block.
    The wrapped SpriteVideoDataset is asked to generate num_blocks * block_size frames total,
    and __getitem__ reshapes them into blocks the same way UCF101_3d.__getitem__ does:
    .view(num_blocks, block_size, C, H, W).transpose(1, 2), then samples each block.
    """

    def __init__(self, data_dir, block_size=32, num_blocks=1, frame_stride=1, return_label=False, **kwargs):
        if block_size <= 0 or frame_stride <= 0 or block_size % frame_stride != 0:
            raise ValueError("block_size must be a positive multiple of frame_stride")
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.frame_stride = frame_stride
        self.return_label = return_label
        self.inner = SpriteVideoDataset(data_dir, seq_len=num_blocks * block_size, **kwargs)
        self.mean = self.inner.mean
        self.std = self.inner.std

        # sprite_idx (which animal is in the clip) doubles as the action-recognition class id;
        # names are synthetic since SpriteVideoDataset doesn't keep sprite filenames after loading
        self.action_dict_encode = {f"sprite_{i}": i for i in range(len(self.inner.sprites))}
        self.action_dict_decode = {i: f"sprite_{i}" for i in range(len(self.inner.sprites))}

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index):
        video, (seq_labels, *_rest) = self.inner[index]
        _T, C, H, W = video.shape
        video = video.reshape(self.num_blocks, self.block_size, C, H, W).transpose(1, 2)
        video = video[:, :, :: self.frame_stride]

        if self.return_label:
            return video, torch.LongTensor([seq_labels[0].item()])
        return video


# the test split's trajectory seed sits this far above the train split's, so the two share no clips
SPRITE_TEST_SEED_OFFSET = 1000


def sprite_split_seed(seed, split):
    """Trajectory seed of `split` in a SpriteVideoDataLoader seeded with `seed`. The modulo keeps
    the offset test seed inside RandomState's uint32 range, since set_seed's seed_data can sit at
    its top."""
    return seed if split == "train" else (seed + SPRITE_TEST_SEED_OFFSET) % 2**32


class SpriteVideoDataLoader(ImageDataLoader):
    """
    DataLoader for the SpriteVideoDataLoader dataset that generates synthetic videos
    of moving sprites with various trajectories and transformations.

    Args:
        data_dir: Directory where the data is stored.
        num_workers: Number of workers for data loading.
        augmentations: Dictionary of augmentations to apply.
        distributed: Whether to use distributed sampling.
        val_size: Proportion of training data to use for validation.
        output_size: Size of output frames (width, height).
        block_size: Number of generated frames per block.
        num_blocks: Number of blocks per sample.
        frame_stride: Sample every Nth generated frame for the model input.
        device: Device used while generating dataset samples. Keep this on CPU for DataLoader
            workers; the training loop moves batches to the model device.
        frame_stride: Sample every Nth generated frame before returning a block.
        num_sequences: Number of trajectories to generate.
        num_train_sequences: Number of sequences to generate for the train split.
        num_test_sequences: Number of sequences to generate for the test split.
        seed: Trajectory seed of the train split; the test split draws with
            sprite_split_seed(seed, "test"). The trainers pass config.seed.set_seed's seed_data.
        background: Background color (grayscale level) for the video frames.
        max_sprites: Maximum number of sprites (i.e., image categories).
        exclude_latent_regions: Whether to exclude some latent regions from the sampling space during training.
        discretize_latents: Whether to use discretized latent variables to generate the videos.
        noise_type: Type of noise to add to frames (gaussian or salt_pepper). Default: None.
        noise_level: Level of noise to add to frames (float between 0 and 1).
        frozen_noise: Whether to use fixed noise patterns (across frames in each sequence).
        noise_on_top: Whether to add noise on top of the frame or only to the background.
        grid_enabled: Whether to use a grid overlay.
        frozen_grid: Whether to use a fixed grid pattern (across frames in each sequence).
        sprite_imgs: Name of the sprite images to use (should be a folder name with images in data_dir).
        grayscale: Whether to convert frames to grayscale.
        occlude_n_frames: Number of frames to occlude (e.g., frame-dropout) in each sequence.
    """

    def __init__(
        self,
        data_dir,
        num_workers=16,
        augmentations=None,
        distributed=False,
        worker_seed_base=None,
        val_size=0.1,
        output_size=(64, 64),
        block_size=15,
        num_blocks=8,
        frame_stride=3,
        device="cpu",
        return_label=False,
        num_sequences=10000,
        num_train_sequences=None,
        num_test_sequences=None,
        background=0.5,
        max_sprites=16,
        exclude_latent_regions=False,
        discretize_latents=False,
        noise_type=None,
        noise_level=0.1,
        frozen_noise=False,
        noise_on_top=False,
        grid_enabled=False,
        frozen_grid=False,
        sprite_imgs="sprites",
        grayscale=False,
        occlude_n_frames=0,
        *,
        seed,
    ):
        # Initialize with grayscale=False since SpriteVideoDataset generates RGB videos
        super(SpriteVideoDataLoader, self).__init__(
            dataset=SpriteClipDataset,
            data_dir=data_dir,
            grayscale=False,
            num_workers=num_workers,
            augmentations=augmentations,
            distributed=distributed,
            worker_seed_base=worker_seed_base,
            # Parameters specific to SpriteClipDataset / SpriteVideoDataset
            output_size=output_size,
            block_size=block_size,
            num_blocks=num_blocks,
            frame_stride=frame_stride,
            device=device,
            return_label=return_label,
            num_sequences=num_sequences,
            num_train_sequences=num_train_sequences,
            num_test_sequences=num_test_sequences,
            background=background,
            sprite_img_dir=sprite_imgs,
            discretize_latents=discretize_latents,
            max_sprites=max_sprites,
            noise_type=noise_type,
            noise_intensity=noise_level,
            freeze_noise=frozen_noise,
            noise_on_top=noise_on_top,
            grid_enabled=grid_enabled,
            freeze_grid=frozen_grid,
            occlude_n_frames=occlude_n_frames,
        )

        # Define subsets for training and testing
        self._train_subset = "train"
        self._val_subset = "train"
        self._test_subset = "test"

        self._train_supervised_subset = "train"
        self._val_supervised_subset = "train"

        self.exclude_latent_regions = exclude_latent_regions

        self.grayscale = grayscale
        self.seed = seed

        # Create datasets
        self.datasets = self._make_datasets([self._train_subset, self._test_subset])

        # Create train and validation samplers from the training dataset
        self.train_sampler, self.val_sampler = create_validation_sampler(
            len(self.datasets[self._train_subset]), val_size, distributed=distributed
        )

    def _make_datasets(self, splits):
        """Create datasets for the given splits.

        Args:
            splits: List of dataset splits to create.

        Returns:
            Dictionary of datasets for each split.
        """
        datasets = {}
        if "train" in splits:
            datasets["train"] = self.dataset(
                self.data_dir,
                split="train",
                download=True,
                transform=self.get_transforms(eval=False),
                seed=sprite_split_seed(self.seed, "train"),
                exclude_latent_regions=self.exclude_latent_regions,
                grayscale=self.grayscale,
                **self._dataset_kwargs,
            )
            mean, std = datasets["train"].mean, datasets["train"].std
        for split in splits:
            if split == "train":
                continue
            datasets[split] = self.dataset(
                self.data_dir,
                split=split,
                download=True,
                transform=self.get_transforms(eval=not split.startswith("train")),
                seed=sprite_split_seed(self.seed, split),
                mean=mean,
                std=std,
                exclude_latent_regions=False,
                grayscale=self.grayscale,
                **self._dataset_kwargs,
            )
        return datasets

    def get_train(self, batch_size):
        """Get dataloader for training data"""
        return super().get_train(batch_size, self.train_sampler)

    def get_validation(self, batch_size):
        """Get dataloader for validation data"""
        return super().get_validation(batch_size, self.val_sampler)

    def get_test(self, batch_size):
        """Get dataloader for test data"""
        # unlike train/val, the test split is its own dataset (no index split needed), so a plain
        # DistributedSampler shards it directly instead of going through create_validation_sampler
        test_sampler = DistributedSampler(self.datasets[self._test_subset], shuffle=False) if self.sampler else None
        return super().get_test(batch_size, test_sampler)
