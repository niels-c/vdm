"""Supervised action-recognition training entry point, producing checkpoints compatible with
RSM/deep_models.py's get_activations_ActionRecog (data/checkpoints/act_recog_p1.pth.tar /
act_recog_p2.pth.tar). Sibling of train_cpc.py (self-supervised DPC/CPC pretraining): shares its
TrainingConfig/CLI, DDP setup, dataloading, and checkpointing scaffolding via
models/cpc/trainer_common.py, but trains ActionRecogNet (models/cpc/model_3d.py) with
cross-entropy on action-class labels instead of DPC_RNN's contrastive future-block-prediction
objective.

Run from the repository root, e.g.:
    python models/cpc/train_action_recog.py --override net=monkeynet_p1
        --override batch_size=256 --override lr=5e-4 --override epochs=300
"""

import argparse
import functools
import os
import time

import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from torch import nn
from tqdm import tqdm

from config.model import action_recog_checkpoint_stem, monkeynet_num_paths
from config.seed import set_seed
from config.training import parse_path_channels
from models.cpc import trainer_common
from models.cpc.model_3d import ActionRecogNet
from models.utils.training_utils import (
    AverageMeter,
    all_reduce_meters,
    calc_topk_accuracy,
    denorm,
    load_checkpoint,
    zero_bn_momentum,
)

plt.switch_backend("agg")
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument(
    "--override",
    action="append",
    default=[],
    metavar="key=value",
    help="override a TrainingConfig field, repeatable, e.g. --override lr=3e-4",
)


def main():
    trainer_common.main(parser, run)


def run(rank, world_size, args):
    """Run training."""
    path_channels = parse_path_channels(args.path_channels, monkeynet_num_paths(args.net))
    seed_data = set_seed(args.seed)

    cuda, amp_enabled = trainer_common.init_ddp_and_device(rank, world_size, args)

    ### load data first (train_loader.dataset.action_dict_encode gives num_classes) ###
    train_loader, val_loader = trainer_common.load_data(
        args, rank, world_size, cuda, return_label=True, seed_data=seed_data
    )
    num_classes = len(train_loader.dataset.action_dict_encode)
    print(f"num_classes={num_classes}")

    ### action-recognition model ###
    model = ActionRecogNet(
        sample_size=args.img_dim,
        network=args.net,
        num_classes=num_classes,
        path_channels=path_channels,
        use_grad_checkpoint=args.use_grad_checkpoint,
    )

    model = trainer_common.wrap_model_for_training(model, world_size, rank, cuda)
    criterion = nn.CrossEntropyLoss()

    trainer_common.log_trainable_params(model)
    optimizer = trainer_common.build_optimizer(model, args)

    ### restart training ###
    # NOTE: unlike train_cpc.py, action-recog checkpoints are saved unwrapped (wrapped=False), so
    # resume loads via model.module rather than the DataParallel/DDP-wrapped `model`.
    iteration, best_acc, best_val_loss, epochs_no_improve, scheduler_state = trainer_common.restore_resume_checkpoint(
        model, optimizer, args, wrapped=False
    )
    scheduler = trainer_common.build_scheduler(optimizer, args, scheduler_state)

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print(f"=> loading pretrained checkpoint '{args.pretrain}'")
            checkpoint = load_checkpoint(args.pretrain)
            # both checkpoint families are unwrapped and key their VisualNet weights "backbone.*"
            # (models/cpc/model_3d.py: DPC_RNN.backbone / ActionRecogNet.backbone), so a CPC
            # checkpoint's backbone loads straight into this model's own backbone attribute.
            backbone_prefix = "backbone."
            backbone_state = {
                k[len(backbone_prefix) :]: v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith(backbone_prefix)
            }
            if not backbone_state:
                raise ValueError(
                    f"checkpoint at '{args.pretrain}' has no '{backbone_prefix}*' keys; "
                    "expected a cpc_p1/p2.pth.tar-style DPC_RNN checkpoint"
                )
            model.module.backbone.load_state_dict(backbone_state)
            print(f"=> loaded pretrained backbone weights from '{args.pretrain}'")
        else:
            print(f"=> no checkpoint found at '{args.pretrain}'")

    # pred_step=0, include_pred_step_segment=False: this trainer predicts no future blocks, so
    # every one of args.num_seq blocks is context (S = num_seq) and there's no T worth naming.
    img_path, model_path = trainer_common.set_path(args, rank, pred_step=0, include_pred_step_segment=False)
    writer_train, writer_val = trainer_common.setup_run_logging(args, rank, img_path, pred_step=0)
    ctx = trainer_common.StepContext(args, cuda, amp_enabled, criterion, world_size, writer_train, denorm())

    ### main loop ###
    # action-recog checkpoints are saved unwrapped (wrapped_state_dict=False), matching the
    # `wrapped=False` used above to resume one. log_grad_norms=False: unlike train_cpc.py, this
    # trainer has never logged per-layer gradient magnitude.
    trainer_common.run_training_loop(
        args,
        rank,
        world_size,
        cuda,
        train_loader,
        val_loader,
        model,
        optimizer,
        functools.partial(train, ctx),
        functools.partial(validate, ctx),
        model_path,
        writer_train,
        writer_val,
        wrapped_state_dict=False,
        log_grad_norms=False,
        iteration=iteration,
        best_acc=best_acc,
        best_val_loss=best_val_loss,
        epochs_no_improve=epochs_no_improve,
        scheduler=scheduler,
        checkpoint_stem_fn=action_recog_checkpoint_stem,
    )


def train(ctx, data_loader, model, optimizer, epoch, iteration):
    """One training epoch. `iteration` is the optimizer step count so far; the returned
    (loss, acc, [top1, top3, top5], iteration) carries it forward by one per batch."""
    args = ctx.args
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.train()

    for idx, (input_seq, label) in enumerate(data_loader):
        tic = time.time()
        input_seq = input_seq.to(ctx.cuda)
        B, N = input_seq.size(0), input_seq.size(1)
        # each block within a clip is trained as an independent sample carrying the clip's
        # action label, matching RSM/downstream/'s block-level linear-probe convention.
        target = label.to(ctx.cuda).repeat(1, N).view(-1)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ctx.amp_enabled):
            if (iteration == 0) or (iteration == args.print_freq):
                vis_input = input_seq[0:2] if B > 2 else input_seq
                ctx.writer_train.add_image(
                    "input_seq",
                    ctx.de_normalize(
                        vutils.make_grid(
                            vis_input.transpose(2, 3).contiguous().view(-1, 3, args.img_dim, args.img_dim),
                            nrow=args.num_seq * args.seq_len,
                        )
                    ),
                    iteration,
                )
                del vis_input

            output = model(input_seq)
            loss = ctx.criterion(output, target)
        top1, top3, top5 = calc_topk_accuracy(output, target, (1, 3, 5))

        accuracy_list[0].update(top1.item(), B * N)
        accuracy_list[1].update(top3.item(), B * N)
        accuracy_list[2].update(top5.item(), B * N)

        losses.update(loss.item(), B * N)
        accuracy.update(top1.item(), B * N)

        del input_seq, output

        optimizer.zero_grad()
        if args.use_grad_checkpoint:
            with zero_bn_momentum(model):
                loss.backward()
        else:
            loss.backward()

        optimizer.step()
        iteration += 1

        del loss

        if idx % args.print_freq == 0:
            print(
                f"Epoch: [{epoch}][{idx}/{len(data_loader)}]\t"
                f"Loss {losses.val:.6f} ({losses.avg:.4f})\t"
                f"Acc: top1 {top1:.4f}; top3 {top3:.4f}; top5 {top5:.4f} time in s:{time.time() - tic:.2f}\t"
            )
            ctx.writer_train.add_scalar("local/loss", losses.val, iteration)
            ctx.writer_train.add_scalar("local/accuracy", accuracy.val, iteration)

    if ctx.world_size > 1:
        # reduce so every rank reports (and stops on) the same whole-dataset numbers
        all_reduce_meters([losses, accuracy, *accuracy_list], ctx.cuda)
    return losses.avg, accuracy.avg, [i.avg for i in accuracy_list], iteration


def validate(ctx, data_loader, model, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.eval()

    with torch.no_grad():
        for _idx, (input_seq, label) in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_seq = input_seq.to(ctx.cuda)
            B, N = input_seq.size(0), input_seq.size(1)
            target = label.to(ctx.cuda).repeat(1, N).view(-1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ctx.amp_enabled):
                output = model(input_seq)
                loss = ctx.criterion(output, target)
            top1, top3, top5 = calc_topk_accuracy(output, target, (1, 3, 5))

            losses.update(loss.item(), B * N)
            accuracy.update(top1.item(), B * N)
            accuracy_list[0].update(top1.item(), B * N)
            accuracy_list[1].update(top3.item(), B * N)
            accuracy_list[2].update(top5.item(), B * N)

    if ctx.world_size > 1:
        # before the print, so every rank reports the whole validation set rather than its shard.
        # This is also what makes epochs_no_improve identical across ranks
        all_reduce_meters([losses, accuracy, *accuracy_list], ctx.cuda)
    print(
        f"Validation: [{epoch}/{ctx.args.epochs}] Loss {losses.avg:.4f}\tAcc: top1 {accuracy_list[0].avg:.4f}; top3 {accuracy_list[1].avg:.4f}; top5 {accuracy_list[2].avg:.4f} \t"
    )
    return losses.avg, accuracy.avg, [i.avg for i in accuracy_list]


if __name__ == "__main__":
    main()
