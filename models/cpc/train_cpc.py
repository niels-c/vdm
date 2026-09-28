import argparse
import functools
import os
import time

import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from torch import nn
from tqdm import tqdm

from config.model import checkpoint_stem, monkeynet_num_paths
from config.seed import set_seed
from config.training import parse_path_channels
from models.backbones.resnet_2d3d import neq_load_customized
from models.cpc import trainer_common
from models.cpc.model_3d import DPC_RNN
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
    """One training process's full lifecycle. Called directly for world_size==1 (today's
    single-process path, unchanged), or once per rank via torch.multiprocessing.spawn for
    world_size>1 (DDP). Must be a module-level function, not a closure; mp.spawn pickles a
    reference to it rather than the function itself."""
    path_channels = parse_path_channels(args.path_channels, monkeynet_num_paths(args.net))
    seed_data = set_seed(args.seed)

    cuda, amp_enabled = trainer_common.init_ddp_and_device(rank, world_size, args)

    ### dpc model ###
    if args.model == "dpc-rnn":
        model = DPC_RNN(
            sample_size=args.img_dim,
            seq_len=args.seq_len,
            network=args.net,
            pred_step=args.pred_step,
            path_channels=path_channels,
            use_grad_checkpoint=args.use_grad_checkpoint,
            world_size=world_size,
            rank=rank,
        )
    else:
        raise NotImplementedError("wrong model!")

    model = trainer_common.wrap_model_for_training(model, world_size, rank, cuda)
    criterion = nn.CrossEntropyLoss()

    trainer_common.log_trainable_params(model)
    optimizer = trainer_common.build_optimizer(model, args)

    ### restart training ###
    # CPC checkpoints are saved unwrapped (wrapped=False), matching action-recog's convention, so
    # resume loads into model.module rather than the DataParallel/DDP-wrapped `model`.
    iteration, best_acc, best_val_loss, epochs_no_improve, scheduler_state = trainer_common.restore_resume_checkpoint(
        model, optimizer, args, wrapped=False
    )
    scheduler = trainer_common.build_scheduler(optimizer, args, scheduler_state)

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print(f"=> loading pretrained checkpoint '{args.pretrain}'")
            checkpoint = load_checkpoint(args.pretrain)
            # unwrapped checkpoint, so match keys against model.module (not the DataParallel/DDP
            # -wrapped `model`, whose own state_dict keys would still carry a "module." prefix)
            neq_load_customized(model.module, checkpoint["state_dict"])
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(args.pretrain, checkpoint["epoch"]))
        else:
            print(f"=> no checkpoint found at '{args.pretrain}'")

    ### load data ###
    train_loader, val_loader = trainer_common.load_data(
        args, rank, world_size, cuda, return_label=False, seed_data=seed_data
    )

    # setup tools
    img_path, model_path = trainer_common.set_path(args, rank)
    writer_train, writer_val = trainer_common.setup_run_logging(args, rank, img_path)
    ctx = trainer_common.StepContext(args, cuda, amp_enabled, criterion, world_size, writer_train, denorm())

    ### main loop ###
    # CPC checkpoints are saved unwrapped (wrapped_state_dict=False), matching the `wrapped=False`
    # used above to resume one, and action-recog's convention. log_grad_norms=True: per-layer
    # gradient magnitude has always been logged for this trainer (unlike train_action_recog.py).
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
        log_grad_norms=True,
        iteration=iteration,
        best_acc=best_acc,
        best_val_loss=best_val_loss,
        epochs_no_improve=epochs_no_improve,
        scheduler=scheduler,
        checkpoint_stem_fn=checkpoint_stem,
    )


def process_output(mask):
    """task mask as input, compute the target for contrastive loss"""
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu

    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    # keep only the "positive" entries (the true prediction-target pairs); this boolean mask is later
    # collapsed with argmax to get a single target class index per row for cross-entropy
    (B, NP, SQ, B2, NS, _) = mask.size()  # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)


def train(ctx, data_loader, model, optimizer, epoch, iteration):
    """One training epoch. `iteration` is the optimizer step count so far; the returned
    (loss, acc, [top1, top3, top5], iteration) carries it forward by one per batch."""
    args = ctx.args
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.train()

    for idx, input_seq in enumerate(data_loader):
        tic = time.time()
        input_seq = input_seq.to(ctx.cuda, non_blocking=True)
        B = input_seq.size(0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ctx.amp_enabled):
            [score_, mask_] = model(input_seq)
            # visualize
            if (iteration == 0) or (iteration == args.print_freq):
                if B > 2:
                    input_seq = input_seq[0:2, :]
                ctx.writer_train.add_image(
                    "input_seq",
                    ctx.de_normalize(
                        vutils.make_grid(
                            input_seq.transpose(2, 3).contiguous().view(-1, 3, args.img_dim, args.img_dim),
                            nrow=args.num_seq * args.seq_len,
                        )
                    ),
                    iteration,
                )
            del input_seq

            if idx == 0:
                target_, (_, B2, NS, NP, SQ) = process_output(mask_)

            # score is a 6d tensor: [B, P, SQ, B, N, SQ]
            # flatten each (prediction, candidate) pair into a row so the contrastive task becomes a
            # standard classification problem: for each predicted block, pick the true block's index
            # out of all B2*NS*SQ candidates via softmax cross-entropy
            score_flattened = score_.reshape(B * NP * SQ, B2 * NS * SQ)
            # int8, not float32: target_ is boolean-derived (exactly one 1 per row) and argmax
            # only needs a relative ordering, so this avoids a 4x-larger transient copy of a
            # tensor that's already the single biggest allocation in the whole training step
            target_flattened = target_.reshape(B * NP * SQ, B2 * NS * SQ).to(torch.int8)
            target_flattened = target_flattened.argmax(dim=1)
            target_flattened = target_flattened.to(device=score_flattened.device, dtype=torch.long)

            loss = ctx.criterion(score_flattened, target_flattened)
        top1, top3, top5 = calc_topk_accuracy(score_flattened, target_flattened, (1, 3, 5))

        accuracy_list[0].update(top1.item(), B)
        accuracy_list[1].update(top3.item(), B)
        accuracy_list[2].update(top5.item(), B)

        losses.update(loss.item(), B)
        accuracy.update(top1.item(), B)

        del score_

        optimizer.zero_grad()
        if args.use_grad_checkpoint:
            # torch.utils.checkpoint recomputes the backbone forward inside backward(); without
            # this, its BatchNorm layers would update running_mean/running_var a second time for
            # the same batch
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

    # best-checkpoint selection read
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
        for idx, input_seq in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_seq = input_seq.to(ctx.cuda)
            B = input_seq.size(0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ctx.amp_enabled):
                [score_, mask_] = model(input_seq)
                del input_seq

                if idx == 0:
                    target_, (_, B2, NS, NP, SQ) = process_output(mask_)

                # [B, P, SQ, B, N, SQ]
                score_flattened = score_.reshape(B * NP * SQ, B2 * NS * SQ)
                target_flattened = target_.reshape(B * NP * SQ, B2 * NS * SQ).to(torch.int8)
                target_flattened = target_flattened.argmax(dim=1)
                target_flattened = target_flattened.to(device=score_flattened.device, dtype=torch.long)

                loss = ctx.criterion(score_flattened, target_flattened)
            top1, top3, top5 = calc_topk_accuracy(score_flattened, target_flattened, (1, 3, 5))

            losses.update(loss.item(), B)
            accuracy.update(top1.item(), B)

            accuracy_list[0].update(top1.item(), B)
            accuracy_list[1].update(top3.item(), B)
            accuracy_list[2].update(top5.item(), B)

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
