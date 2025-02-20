import time

import torch
import torch.distributed
from tqdm import tqdm, trange

from tools import builder
from utils.average_value import AverageMeter
from utils.logger import print_log


def train_iteration(startIter, nBatches, epoch_loss, losses, args, config, train_dataset, 
                    base_model, optimizer, scaler, epoch_num, writer, logger, wandber=None):
    _, train_sampler, training_data_loader = builder.dataloader_builder(args, train_dataset)
    # accumulate the loss, making sure the loss is stable at smaller batch_size(e.g. 2)
    # this operation makes the batch size equals to accum_steps times of the raw value
    accum_steps = int(config.accum_steps)
    subcacheset_time = AverageMeter()
    time_stamp = time.time()
    # loop each subcache
    for iteration, (query, query_pc, positives, positives_pc, negatives, negatives_pcs, negCounts, _) in \
            enumerate(tqdm(training_data_loader, position=2, leave=False, desc='Train Iter'.rjust(10)), startIter):
        # number of negative samples both two modalities
        nNeg = torch.sum(negCounts)
        # glob 2d data(tensor)
        input_tuplet_img = torch.cat([query, positives, negatives], dim=1) # B, (1+1+nNeg), C, H, W
        # glob 3d data(tensor)
        input_tuplet_pc = torch.cat((query_pc, positives_pc, negatives_pcs), dim=1) # B, (1+1+nNeg), N, C

        with torch.cuda.amp.autocast():
            # feed sample into models
            samples = dict(nNeg=train_dataset.nNeg, image=input_tuplet_img.to(args.local_rank), point_cloud=input_tuplet_pc.to(args.local_rank))
            # fetch the corresponding losses
            loss_query, loss_2dto2d, loss_2dto3d, loss_3dto2d, loss_3dto3d = base_model(samples)

        # sum loss
        _loss = loss_query + 0.1 * loss_2dto2d + loss_2dto3d + loss_3dto2d + 0.1 * loss_3dto3d
        _loss = _loss / (accum_steps if accum_steps > 1 else 1)
        # accumulate scaled gradients
        scaler.scale(_loss.sum()).backward()
        # clean the gradient when accumulated some batches, avoiding the smaller batch_size
        if (iteration + 1) % accum_steps == 0 or (iteration + 1) == len(training_data_loader):
            if config.optimizer.type != 'sam':
                # gradient clip
                scaler.unscale_(optimizer)
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.max_grad_norm)
                # optimizer.step
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.first_step(zero_grad=True)
                # sum loss
                _loss = loss_query + 0.1 * loss_2dto2d + loss_2dto3d + loss_3dto2d + 0.1 * loss_3dto3d
                _loss.backward()
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.max_grad_norm)
                optimizer.second_step(zero_grad=True)
            # # zero out gradients so we can accumulate new ones over batches again
            optimizer.zero_grad()
        # release memory
        del input_tuplet_img, input_tuplet_pc
        del query, query_pc, positives, positives_pc, negatives, negatives_pcs
        losses.update([loss_query.sum().item() * (accum_steps if accum_steps > 1 else 1), 
                    loss_2dto2d.sum().item() * (accum_steps if accum_steps > 1 else 1), 
                    loss_2dto3d.sum().item() * (accum_steps if accum_steps > 1 else 1), 
                    loss_3dto2d.sum().item() * (accum_steps if accum_steps > 1 else 1),
                    loss_3dto3d.sum().item() * (accum_steps if accum_steps > 1 else 1)])
        # loss in current batch
        batch_loss = _loss.sum().item() * (accum_steps if accum_steps > 1 else 1)
        epoch_loss.update(batch_loss)
        if writer is not None:
            # weighted loss in this batch
            writer.add_scalar('Train/Batch loss', batch_loss, ((epoch_num - 1) * nBatches) + iteration)
            # the total negatives in this batch
            writer.add_scalar('Train/The number of negative samples', nNeg, ((epoch_num - 1) * nBatches) + iteration)
            # learning rate
            writer.add_scalar('Train/Learning rate', optimizer.param_groups[0]['lr'], ((epoch_num - 1) * nBatches) + iteration)
            # separated loss
            for idx, l in enumerate(losses.val()):
                writer.add_scalar('Train/Loss_%d' % (idx + 1), l, ((epoch_num - 1) * nBatches) + iteration)
        # log metrics to wandb
        if wandber is not None:
            ret = {"Batch loss": batch_loss,
                    "The number of negative samples": nNeg,
                    "Learning rate": optimizer.param_groups[0]['lr']}
            ret_atd = dict(zip(['Loss_%d' % (idx + 1) for idx, l in enumerate(losses.val())], [float(l) for l in losses.val()]))
            ret.update(ret_atd)
            wandber.log(ret)
        # Sub cacheset time check
        subcacheset_time.update(time.time() - time_stamp)
        time_stamp = time.time()
        if iteration % 200 == 0:
            print_log('[Epoch %d/%d][Batch %d/%d] SubCacheSetTime = %.3f (s) Losses = %s lr = %.6f' %
                        (epoch_num, config.max_epoch, iteration + 1, nBatches, subcacheset_time.val(), 
                        ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger=logger)
        
        del _loss

    # start iteration in whole epoch, increase at batch_size step
    startIter += len(training_data_loader)
    del training_data_loader
    

def train_epoch(args, config, train_dataset, base_model, optimizer, scaler, epoch_num, writer, logger, wandber=None):
    # time stamp
    epoch_start_time = time.time()
    # initialize loss
    epoch_loss = AverageMeter()
    losses = AverageMeter(['Loss_query', 'Loss_2D_2D', 'Loss_2D_3D', 'Loss_3D_2D', 'Loss_3D_3D'])
    startIter = 1
    # update subcache dataset for partial mining, each epoch will be dividec again and traverse each cached subset
    train_dataset.new_epoch()
    # the number of total batches during in this epoch, each batch will contain batch_size samples
    nBatches = (len(train_dataset.qIdx) + int(config.total_bs) - 1) // int(config.total_bs)
    print_log('Number of total batches: %i' % nBatches, logger=logger)
    print_log('Number of triplets: %i' % (nBatches * config.total_bs), logger=logger)
    # zero out gradients so we can accumulate new ones over batches
    base_model.zero_grad()
    # forward neural networks, open BN and Droupout module
    base_model.train()
    if train_dataset.is_mining():
        # train_dataset.nCacheSubset is number of the subsets in one single epoch
        # each batch will be optimized during each cached subset, while the subset data are randomly selected
        for subIter in trange(train_dataset.nCacheSubset, desc='Training subcached set refreshing'.rjust(15), position=1):
            global_feature_dim = config.model.global_feat_dim
            #print_log("====> Building Cache <====", logger=logger)
            # generate triplets for training from current subset, here absolutely positive and negative tuplets are sampled
            train_dataset.update_subcache(model=None, outputdim=global_feature_dim, margin=0.1)
            train_iteration(startIter, nBatches, epoch_loss, losses, args, config, train_dataset, 
                        base_model, optimizer, scaler, epoch_num, writer, logger, wandber)
    else:
        train_iteration(startIter, nBatches, epoch_loss, losses, args, config, train_dataset, 
                        base_model, optimizer, scaler, epoch_num, writer, logger, wandber)

    torch.cuda.empty_cache()
    epoch_end_time = time.time()
    # average loss in current epoch with whole batch iterations
    avg_loss = epoch_loss.avg()
    if writer is not None:
        writer.add_scalar('Train/AvgLoss', avg_loss, epoch_num)
    if wandber is not None:    
        wandber.log({"Average loss": avg_loss})
    print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Average Losses = %s' % (epoch_num,  epoch_end_time - epoch_start_time, avg_loss), logger=logger)
