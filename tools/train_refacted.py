import gc
import itertools

import torch
from tqdm.autonotebook import tqdm

from datasets.build import build_dataset_from_cfg
from tools import builder
from tools.validate import validate
from utils.logger import get_logger, print_log
from utils.misc import count_parameters


class Trainer:
    def __init__(self, args, config, base_model, train_dataset, val_dataset, 
                 logger, writer, val_writer):
        self.args = args
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.base_model = base_model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.optimizer = None # 2d optimizer
        self.optimizer3D = None # available, if baseline model is used
        self.scheduler = None # 2d scheduler
        self.scheduler3D = None  # available, if baseline model is used

        self.writer = writer
        self.val_writer = val_writer
        self.logger = logger

        self.init_mode()

    def init_mode(self):
        """set architecture mode
        """
        self.mode = "none"
        self.collate_fn = None
        if self.config.model.mode == "vanilla":
            self.collate_fn = self.train_dataset.collate_fn
        else:
            self.mode = "pairwise"

    def train(self):
        best_score = 0
        best_score_5 = 0
        if self.args.resume:
            start_epoch, best_score = builder.resume_model(self.base_model, self.args, logger=self.logger)
        else:
            print_log("Training from scratch ...", logger=self.logger)
            start_epoch = 0
        self.base_model = self.base_model.to(self.device)
        # separate the lr for different modality network for baseline model
        if self.config.model.baseline and self.config.model.mode == "vanilla":
            # here the param_groups should be list embedded (double list-loop)
            param_groups = [[{'params': self.base_model.image_encoder.parameters(), 
                                'lr': self.config.optimizer.lr_2d}],
                            [{'params': self.base_model.pc_encoder.parameters(), 
                                'lr': self.config.optimizer.lr_3d, 
                                'initial_lr': self.config.optimizer.lr_3d}],
                            [{"params": itertools.chain(self.base_model.vision_proj.parameters(), 
                                                    self.base_model.pc_proj.parameters()), 
                                "lr": self.config.optimizer.lr_agg, 
                                "weight_decay": self.config.optimizer.lr_agg}],
                        ]
            self.optimizer, self.optimizer3D, self.scheduler, self.scheduler3D = builder.build_opti_sche(self.base_model, self.config, param_groups)
        else:
            # single optimizer is called, thus single loop list is enough
            param_groups = [{'params': self.base_model.image_encoder.parameters(), 
                                'lr': self.config.optimizer.lr_2d},
                            {'params': self.base_model.pc_encoder.parameters(), 
                                'lr': self.config.optimizer.lr_3d, 
                                'initial_lr': self.config.optimizer.lr_3d},
                            {"params": itertools.chain(self.base_model.vision_proj.parameters(), 
                                                    self.base_model.pc_proj.parameters()), 
                                "lr": self.config.optimizer.lr_agg, 
                                "weight_decay": self.config.optimizer.lr_agg},
                        ]
            self.optimizer, self.scheduler = builder.build_opti_sche(self.base_model, self.config, param_groups=param_groups)
        # resume optimizer
        if self.args.resume:
            if self.optimizer3D is not None:
                builder.resume_optimizer_refactor(self.optimizer, self.optimizer3D, self.args, logger=self.logger)
            else:
                builder.resume_optimizer(self.optimizer, self.args, logger=self.logger)

        accum_steps = int(self.config.accum_steps)
        accum_steps = (accum_steps if accum_steps > 1 else 1)
        self.base_model.zero_grad()

        for epoch in range(start_epoch + 1, self.config.max_epoch):
            if self.writer is not None:
                self.writer.add_scalar('train/lr', self.optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
                if self.config.model.baseline and self.config.model.mode == "vanilla":
                    self.writer.add_scalar('train/lr_3d', self.optimizer3D.state_dict()['param_groups'][0]['lr'], global_step=epoch)
                else:
                    self.writer.add_scalar('train/lr_3d', self.optimizer.state_dict()['param_groups'][1]['lr'], global_step=epoch)
                    self.writer.add_scalar('train/lr_aggregators', self.optimizer.state_dict()['param_groups'][2]['lr'], global_step=epoch)
            # train loop
            self.base_model.train()
            # update triplet each epoch based on random positive and negatives selection
            if self.config.model.mode == "vanilla":
                self.train_dataset.new_epoch()
            _, _, train_loader = builder.dataloader_builder(self.args, self.train_dataset, mode=self.mode, collate_fn=self.collate_fn)
            loss_each_epoch = 0
            used_num = 0
            if self.config.model.mode == "contrast":
                tqdm_object = tqdm(train_loader, total=len(train_loader), leave=False, desc='Train Iter'.rjust(10), colour="blue")
                for iteration, batch in enumerate(tqdm_object):
                    used_num += 1
                    samples = dict(image=batch['query_img'].to(self.device), 
                                    point_cloud=batch['query_pc'].to(self.device))
                    # unify the losses
                    _loss = self.base_model(samples)
                    # sum() works for multi-loss components
                    if torch.isnan(_loss.sum()):
                        print_log('Something went wrong in loss calculation!!!', logger=self.logger)
                        raise NotImplementedError(f'Sorry, gradient explored!!!')
                    # calculate the gradients
                    _loss = _loss / accum_steps
                    _loss.sum().backward()
                    if (iteration + 1) % accum_steps == 0 or (iteration + 1) == len(train_loader):
                        # optimize based on gradients
                        self.optimizer.step()
                        # clean gradients
                        self.optimizer.zero_grad()
                        self.base_model.zero_grad()

                    loss_each_epoch += _loss.sum().item() * accum_steps
                    tqdm_object.set_postfix(train_loss=loss_each_epoch/used_num)
            else:
                for iteration, (query, query_pc, positives, positives_pc, negatives, negatives_pcs, negCounts, _) in \
                    enumerate(tqdm(train_loader, position=2, leave=False, desc='Train Iter'.rjust(10))):
                    used_num += 1
                    # glob 2d data(tensor)
                    input_tuplet_img = torch.cat([query, positives, negatives], dim=1) # B, (1+1+nNeg), C, H, W
                    # glob 3d data(tensor)
                    if self.base_model.proxy:
                        input_tuplet_pc = torch.cat([query_pc, positives_pc, negatives_pcs], dim=1) # B, (1+1+nNeg), C, H, W
                    else:
                        input_tuplet_pc = torch.cat((query_pc, positives_pc, negatives_pcs), dim=1) # B, (1+1+nNeg), N, C
                    samples = dict(nNeg=self.train_dataset.nNeg, 
                                image=input_tuplet_img.to(self.device), 
                                point_cloud=input_tuplet_pc.to(self.device))
                    # unify the losses
                    _loss = self.base_model(samples)
                    # sum() works for multi-loss components
                    if torch.isnan(_loss.sum()):
                        print_log('Something wrong in loss calculation!!!', logger=self.logger)
                        raise NotImplementedError(f'Sorry, gradient explored!!!')
                    # calculate the gradients
                    _loss = _loss / accum_steps
                    _loss.sum().backward()
                    if (iteration + 1) % accum_steps == 0 or (iteration + 1) == len(train_loader):
                        self.optimizer.step() # optimize based on gradients
                        self.optimizer.zero_grad() # clean gradients
                        if self.config.model.baseline and self.config.model.mode == "vanilla":
                            self.optimizer3D.step() # optimize based on gradients
                            self.optimizer3D.zero_grad() # clean gradients
                        if not self.config.model.baseline:
                            self.base_model.zero_grad() # if param_groups is set, block this line!!!

                    loss_each_epoch += _loss.sum().item() * accum_steps
            # update the scheduler after each epoch
            if self.config.scheduler.type == "ReduceLROnPlateau":
                self.scheduler.step(metrics=(loss_each_epoch/used_num))
            elif self.config.scheduler.type == "CosLR":
                self.scheduler.step(epoch=epoch)
            else:
                self.scheduler.step()
            if self.config.model.baseline and self.config.model.mode == "vanilla":
                self.scheduler3D.step()
            print_log("Epoch {} average loss {}".format(epoch, loss_each_epoch / used_num), logger=self.logger)
            # train log writer
            if self.writer is not None:
                self.writer.add_scalar("train/avg_loss", loss_each_epoch / used_num, global_step=epoch)
            recalls = {}
            if (epoch % int(self.args.val_freq)) == 0 and epoch != 0:
                recalls = validate(self.base_model, self.val_dataset, self.args, self.config, 
                                self.logger, self.val_writer, wandber=None, epoch_num=epoch)
                if recalls[1] > best_score:
                    best_score = recalls[1] # only best_recall_at_1 is stored
                    best_score_5 = recalls[5]
                    builder.save_checkpoint_refactor(self.args, self.base_model, best_metrics=best_score, optimizer=self.optimizer, 
                                                     optimizer3D=None if self.optimizer3D is None else self.optimizer3D, 
                                                     epoch=epoch, metrics=recalls, logger=self.logger)
                else:
                    if (abs(recalls[1] - best_score) < 0.0001) and recalls[5] > best_score_5:
                        best_score = recalls[1] # only best_recall_at_1 is stored
                        best_score_5 = recalls[5]
                        builder.save_checkpoint_refactor(self.args, self.base_model, best_metrics=best_score, optimizer=self.optimizer, 
                                                     optimizer3D=None if self.optimizer3D is None else self.optimizer3D, 
                                                     epoch=epoch, metrics=recalls, logger=self.logger)
            # for training resume
            builder.save_checkpoint_refactor(self.args, self.base_model, best_metrics=best_score, optimizer=self.optimizer, 
                                             optimizer3D=None if self.optimizer3D is None else self.optimizer3D, 
                                             epoch=epoch, metrics=recalls, logger=self.logger, prefix='ckpt-last')
            message = '[Current Epoch] 2D->3D %s' % ['Recall@%i = %.4f' % (k, v/100) for k,v in recalls.items()]
            print_log(message, logger=self.logger)
            # clean dummy memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def train_refacted(args, config, train_writer=None, val_writer=None, wandber=None):
    '''
        Train the model with a simplified but still compact pipeline
    '''
    logger = get_logger(args.log_name)
    logger.info("Loading training datasets ...")
    # build model
    model = builder.model_builder(config.model)
    # print(model)
    print_log(f'Model {config.model.NAME} created, baseline? {config.model.baseline}, proxy? {config.model.proxy}, param count: {count_parameters(model)}', logger=logger)
    if not config.model.proxy:
        config.dataset.train._base_.point_cloud_proxy = "points"
        config.dataset.val._base_.point_cloud_proxy = "points"
        print_log(f'Make sure that the proxy is not used', logger=logger)
    # build dataset
    train_dataset = build_dataset_from_cfg(config.dataset.train._base_, config.dataset.train.others)
    print_log(train_dataset.get_dataset_info(), logger=logger)
    val_dataset = build_dataset_from_cfg(config.dataset.val._base_, config.dataset.val.others)
    print_log(val_dataset.get_dataset_info(), logger=logger)
    if model.proxy and (train_dataset.point_cloud_proxy == ("points" or "none")):
        raise NotImplementedError(f'Sorry, The model setting and dataset setting is not compatiable!')
    # set-up trainer
    trainer = Trainer(args, config, model, train_dataset, val_dataset, logger, writer=train_writer, val_writer=val_writer)
    trainer.train()