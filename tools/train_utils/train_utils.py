import glob
import os
import datetime
import torch
import time

from pcseg.utils import common_utils, commu_utils, caption_utils
from pcseg.config import cfg
from pcseg.models import load_data_to_gpu
from tools.eval_utils import eval_utils
from .optimization import adjust_lr

import matplotlib.pyplot as plt
import numpy as np
import pickle

def train_one_epoch(args, model, optimizer, train_loader, lr_scheduler, accumulated_iter, rank,
                    total_it_each_epoch, dataloader_iter, cur_epoch, tb_log=None, logger=None,
                    caption_items=None, text_encoder=None):
    if hasattr(cfg.DATA_CONFIG, 'base_class_idx'):
        num_train_class = len(train_loader.dataset.base_class_idx)
        train_loader.dataset.set_class_mode('base')
    else:
        num_train_class = len(train_loader.dataset.valid_class_idx)

    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()

    # evaluation meters
    intersection_meter = common_utils.AverageMeter()
    union_meter = common_utils.AverageMeter()
    target_meter = common_utils.AverageMeter()
    output_meter = common_utils.AverageMeter()
    binary_intersection_meter = common_utils.AverageMeter()
    binary_target_meter = common_utils.AverageMeter()

    n_caption_meter = common_utils.AverageMeter()

    end = time.time()

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')
        batch['epoch'] = cur_epoch

        data_timer = time.time()
        cur_data_time = data_timer - end

        # lr_scheduler.step(accumulated_iter)
        adjust_lr(cfg.OPTIMIZATION, optimizer, lr_scheduler, args.epochs, total_it_each_epoch, cur_epoch, cur_it, accumulated_iter)
        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        load_data_to_gpu(batch)
        
        #######################################################
        # forward text encoder to extract captions embeddings #
        #######################################################
        if cfg.MODEL.get('CAPTION_HEAD', False):
            batch = caption_utils.get_caption_batch(
                cfg.DATA_CONFIG.CAPTION_INFO, cfg.TEXT_ENCODER, batch, text_encoder
            )
            n_caption_scene = batch['num_caption']
            n_caption_scene = commu_utils.average_reduce_value(n_caption_scene)
            n_caption_meter.update(n_caption_scene)
            caption_disp_dict = {
                'n_captions': f'{n_caption_meter.val:.1f}({n_caption_meter.avg:.1f})'
            }

        #######################
        # forward vision part #
        #######################
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            ret_dict, tb_dict, disp_dict = model(batch)

        loss = ret_dict['loss'].mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        temp_param = next(model.parameters())
        if torch.isnan(temp_param.sum()):
            raise ValueError("The model parameters contain NaN.")

        accumulated_iter += 1
        
        model.eval()
        with torch.no_grad():
            ret_dict = model(batch)
        # print( "ret_dict after nograd--> ", batch['voxel_features'].shape)
        
        # with torch.no_grad():
        #     print('ret_dict1')
        #     print( "ret_dict1--> ", batch['voxel_features'].shape)

        #     ret_dict1 = model(batch)

        # model.eval()
        # with torch.no_grad():
        #     print('ret_dict2')
        #     print( "ret_dict2--> ", batch['voxel_features'].shape)

        #     ret_dict2 = model(batch)
        # np.savetxt("/home/daniel/spatial_understanding/benchmarks/PLA/deleteme/train1.txt", ret_dict1['seg_preds'].cpu().numpy())
        # np.savetxt("/home/daniel/spatial_understanding/benchmarks/PLA/deleteme/train2.txt", ret_dict2['seg_preds'].cpu().numpy())

        # # print("Train state")
        # # print({k: v.clone().cpu() for k, v in model.state_dict().items()})
        # with open("/home/daniel/spatial_understanding/benchmarks/PLA/deleteme/train_batch.pkl", "wb") as f:
        #     pickle.dump(batch, f)

        # # print("LA KEEEEEY", batch.keys())
        # # suu = batch['points_xyz'].cpu().numpy()
        # # plt.scatter(suu[:, 0], suu[:, 1])
        # # plt.gca().set_aspect('equal')
        # # plt.savefig("/home/daniel/spatial_understanding/benchmarks/PLA/deleteme/train.png")
        # print("savin train")
        # torch.save(model.state_dict(), "/home/daniel/spatial_understanding/benchmarks/PLA/deleteme/train_state.pth")
        # np.savetxt("/home/daniel/spatial_understanding/benchmarks/PLA/deleteme/train_xyz.txt", batch['points_xyz'].cpu().numpy())
        # np.savetxt("/home/daniel/spatial_understanding/benchmarks/PLA/deleteme/train.txt", ret_dict['seg_preds'].cpu().numpy())
        # np.savetxt("/home/daniel/spatial_understanding/benchmarks/PLA/deleteme/train_labels.txt", ret_dict['seg_labels'].cpu().numpy())
        # Save XYZ and labels in image to visualize in 2D
        # plt.figure(figsize=(15, 15))
        # plt.scatter(batch['points_xyz'][:, 0].cpu().numpy(), batch['points_xyz'][:, 1].cpu().numpy(), c=ret_dict['seg_labels'].cpu().numpy(), s=1)
        # plt.colorbar()
        # plt.gca().set_aspect('equal')
        # plt.savefig(f"/home/daniel/spatial_understanding/benchmarks/PLA/.vscode/deleteme/train_labels_{accumulated_iter}.png")
        
        # record evaluation metrics for segmentation
        intersection_meter, union_meter, target_meter, output_meter, accuracy = common_utils.update_meter(
            intersection_meter, union_meter, target_meter, output_meter, ret_dict['seg_preds'],
            ret_dict['seg_labels'], num_train_class
        )

        # record evaluation metrics for binary head
        if cfg.MODEL.get('BINARY_HEAD', None) and ret_dict['binary_preds'] is not None:
            binary_preds = ret_dict['binary_preds']
            binary_intersection_meter, binary_target_meter = common_utils.update_binary_acc_meter(
                binary_intersection_meter, binary_target_meter, binary_preds, ret_dict['seg_labels'],
                cfg.DATA_CONFIG.novel_class_idx, num_train_class
            )

        cur_batch_time = time.time() - end
        end = time.time()

        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        torch.cuda.empty_cache()

        # log to console and tensorboard
        if rank == 0:
            data_time.update(avg_data_time)
            batch_time.update(avg_batch_time)

            if (cur_it + 1) % args.print_freq == 0 or cur_it == (total_it_each_epoch - 1):
                remain_iter = total_it_each_epoch * (args.epochs - cur_epoch) - cur_it - 1
                remain_time = remain_iter * batch_time.avg
                remain_time = str(datetime.timedelta(seconds=int(remain_time)))

                disp_str = ', '.join([f'{key}={val:.2f}' for key, val in disp_dict.items() if key != 'lr'])
                if cfg.MODEL.get('CAPTION_HEAD', False):
                    caption_disp_str = ', '.join([f'{key}={val}' for key, val in caption_disp_dict.items()])
                else:
                    caption_disp_str = ''

                log_str = f'Epoch [{cur_epoch + 1}/{args.epochs}][{cur_it + 1}/{total_it_each_epoch}] '\
                          f'LR: {cur_lr:.2g}, ETA: {remain_time}, Data: {data_time.val:.2f} ({data_time.avg:.2f}), '\
                          f'Iter: {batch_time.val:.2f} ({batch_time.avg:.2f}), ' \
                          f'Accuracy: {accuracy:.2f}, ' \
                          f'{disp_str}, {caption_disp_str}'
                logger.info(log_str)

                if tb_log is not None:
                    tb_log.add_scalar('train/loss', loss, accumulated_iter)
                    tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                    for key, val in tb_dict.items():
                        tb_log.add_scalar('train/' + key, val, accumulated_iter)

    mIoU, mPre, mAcc, allPre, allAcc, iou_class, _, _, _ = common_utils.calc_metrics(
        intersection_meter, union_meter, target_meter, output_meter
    )
    logger.info('Train result at epoch [{}/{}]: mIoU/mPre/mAcc/allPre/allAcc \
            {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(cur_epoch + 1, args.epochs, mIoU, mPre, mAcc, allPre, allAcc))

    if cfg.MODEL.get('BINARY_HEAD', None):
        binary_mAcc, binary_allAcc, _ = common_utils.calc_binary_acc(
            binary_intersection_meter, binary_target_meter
        )
        logger.info('Train result at epoch [{}/{}]: binary_mAcc/binary_allAcc {:.4f}/{:.4f}.'.format(
            cur_epoch + 1, args.epochs, binary_mAcc, binary_allAcc
        ))

    return accumulated_iter


def train_model(args, model, optimizer, train_loader, val_loader, lr_scheduler, optim_cfg, start_epoch, start_iter, 
                rank, tb_log, ckpt_save_dir, task, train_sampler=None, caption_items=None, lr_warmup_scheduler=None, 
                logger=None, best_metric=None, best_epoch=None, dist_train=None, text_encoder=None, arnold=None):
    accumulated_iter = start_iter
    total_it_each_epoch = len(train_loader)
    if args.merge_all_iters_to_one_epoch:
        assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
        train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=args.epochs)
        total_it_each_epoch = len(train_loader) // max(args.epochs, 1)

    dataloader_iter = iter(train_loader)

    if args.validate_start:
        eval_utils.eval_one_epoch(
            cfg, args, model, val_loader, start_epoch, logger, dist_train, tb_log, best_metric, best_epoch, task
        )

    for cur_epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(cur_epoch)

        # train one epoch
        if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
            cur_scheduler = lr_warmup_scheduler
        else:
            cur_scheduler = lr_scheduler
        accumulated_iter = train_one_epoch(
            args, model, optimizer, train_loader,
            lr_scheduler=cur_scheduler,
            accumulated_iter=accumulated_iter,
            rank=rank, tb_log=tb_log,
            total_it_each_epoch=total_it_each_epoch,
            dataloader_iter=dataloader_iter,
            cur_epoch=cur_epoch,
            logger=logger,
            caption_items=caption_items,
            text_encoder=text_encoder
        )

        # save trained model
        trained_epoch = cur_epoch + 1

        if not args.occupy:
            save_trained_model(
                model, optimizer, accumulated_iter, trained_epoch, args.epochs, args.ckpt_save_interval, rank,
                ckpt_save_dir, args.max_ckpt_save_num, best_metric, best_epoch, logger, arnold=arnold
            )

        if trained_epoch == args.epochs or trained_epoch % args.eval_freq == 0:
            best_metric, best_epoch = eval_utils.eval_one_epoch(
                cfg, args, model, val_loader, trained_epoch, logger, dist_train, tb_log, best_metric, best_epoch, task
            )

            # record best metric
            if rank == 0:
                best_metric_record_list = glob.glob(str(ckpt_save_dir / '*.txt'))
                for best_metric_record in best_metric_record_list:
                    os.remove(best_metric_record)
                (ckpt_save_dir / f'Best_metric_{best_metric:.4f}_in_epoch_{best_epoch}.txt').touch()


def save_trained_model(model, optimizer, accumulated_iter, trained_epoch, total_epochs, ckpt_save_interval, rank,
                       ckpt_save_dir, max_ckpt_save_num, best_metric, best_epoch, logger, arnold=None):
    if rank == 0 and (trained_epoch == total_epochs or trained_epoch % ckpt_save_interval == 0):
        ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
        ckpt_list.sort(key=os.path.getmtime)

        if ckpt_list.__len__() >= max_ckpt_save_num:
            for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                os.remove(ckpt_list[cur_file_idx])

        ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
        try:
            import pcseg
            version = 'pcseg+' + pcseg.__version__
        except:
            version = 'none'

        logger.info(f"Checkpoint is saved to : {ckpt_save_dir}. Version: {version}")

        save_checkpoint(
            checkpoint_state(model, optimizer, trained_epoch, accumulated_iter, best_metric, best_epoch), filename=ckpt_name,
        )
        arnold.save_ckpt(str(ckpt_name) + '.pth', last_epoch=True)


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None, best_metric=None, best_epoch=None):
    # optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcseg
        version = 'pcseg+' + pcseg.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state,
            'version': version, 'best_metric': best_metric, 'best_epoch': best_epoch}


def save_checkpoint(state, filename='checkpoint'):
    filename = '{}.pth'.format(filename)
    if torch.__version__ >= '1.4':
        torch.save(state, filename, _use_new_zipfile_serialization=False)
    else:
        torch.save(state, filename)
