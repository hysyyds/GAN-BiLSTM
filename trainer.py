#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.loader import get_loader
import opt
from models.lossFun import FocalLoss
import utils.metrics as metrics
import pytorch_warmup as warmup



class Trainer():
    def __init__(self, model):
        self.save_dir = opt.save_dir
        self.data_dir = opt.data_dir
        self.batch_size = opt.batch_size
        self.number_workers= opt.number_workers
        self.time_tri= opt.time_tri
        self.useGAN= opt.useGAN
        self.device = opt.device
        self.model_name=opt.NAME
        self.accumulation_step=opt.accumulation_step
        self.max_epoch=opt.epochs
        self.filename=opt.filename

        os.makedirs(self.save_dir, exist_ok=True)
        self.train_loader, self.valid_loader, self.test_loader,_ = get_loader(self.data_dir,self.time_tri,self.batch_size,num_workers=self.number_workers)

        self.num_train = self.train_loader.dataset.tensors[0].size(0)
        self.num_valid = self.valid_loader.dataset.tensors[0].size(0)
        self.num_test  = self.test_loader.dataset.tensors[0].size(0)

        print('Find %d train numbers, %d validation numbers, %d test numbers' %(self.num_train, self.num_valid,self.num_test))
        print('batch size %d' %(self.batch_size))

        self.model = model.to(self.device)

        if opt.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),lr=opt.LEARNING_RATE,momentum=0.9)
        elif opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=opt.LEARNING_RATE,
                weight_decay=opt.WEIGHT_DECAY
            )
        elif opt.optimizer=='adamw':
            self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=opt.LEARNING_RATE,betas=(0.9, 0.999),weight_decay=opt.WEIGHT_DECAY)
        else:
            raise NotImplementedError
        if opt.lrsc=="warmup":
            self.lrsc=torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader) * self.max_epoch)
        else:
            self.lrsc=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        if opt.loss=="Focal":
            self.loss_fn=FocalLoss(logits=True,alpha=opt.ALPHA,gamma=opt.GAMMA)
        elif opt.loss=="CE":
            self.loss_fn=nn.CrossEntropyLoss()
        else:
            raise NotImplementedError
        self.start_epoch = 0
        self.best_loss = 1e10
        self.best_score = -1
        if opt.resume==True:
            if os.path.isfile(opt.resume_path):
                self.resume(opt.resume_path, load_optimizer=True)
            else:
                print("Checkpoint not found")

    def resume(self, path, load_optimizer=True):
        print("Resuming from {}".format(path))
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.best_score = checkpoint['best_score']
        self.model.load_state_dict(checkpoint['state_dict'])
        if "optimizer" in checkpoint.keys() and load_optimizer:
            print("Loading optimizer state dict")
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save_checkpoint(self, epoch, save_optimizer=True, suffix=""):
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "best_loss": self.best_loss,
            "best_score": self.best_score
        }
        if save_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        save_path = self.save_dir +'weights/'+ self.model_name + "_" + suffix + ".pth"
        os.makedirs(self.save_dir +'weights/', exist_ok=True)
        torch.save(checkpoint, save_path)
        # print("Save model checkpoint at {}".format(save_path))

    def train(self, epoch):
        # start = time.strftime("%H:%M:%S")
        # lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        # print("Starting epoch: %d | phase: train | ⏰: %s | Learning rate: %f" %(epoch, start, lr))
        self.model.train()
        self.optimizer.zero_grad()
        # warmup
        # num_steps = len(dataloader) * epochs
        # warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
        y_predict=torch.tensor(()).to(self.device)
        y_true=torch.tensor(()).to(self.device)
        # tbar = tqdm(self.train_loader, desc="\r")
        total_losses = 0
        for batch, (X, y) in enumerate(self.train_loader):
            onehot_target=torch.eye(2)[y.long().cpu(), :].to(self.device)
            pred = self.model(X)
            y_predict=torch.cat([y_predict,pred.argmax(1)],dim=0)
            y_true=torch.cat([y_true,y],dim=0)
            loss = self.loss_fn(pred.reshape(-1,1,2), onehot_target)
            # 梯度累加
            total_losses += float(loss)
            loss /= self.accumulation_step
            loss.backward()
            if (batch + 1) % self.accumulation_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            # tbar.set_description("Train loss: %.5f" % (total_losses / (batch + 1)))
            # lr_scheduler.step(lr_scheduler.last_epoch+1)
            # warmup_scheduler.dampen()

        met=metrics.scores(y_true.cpu().numpy(),y_predict.cpu().numpy())
        return loss.item(),met

    def valid(self,dataloader,epoch):
        self.model.eval()
        # lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        # start = time.strftime("%H:%M:%S")
        # print("Starting epoch: %d | phase: valid | ⏰: %s " % (epoch, start))
        test_loss = 0
        y_predict=torch.tensor(()).to(self.device)
        y_true=torch.tensor(()).to(self.device)
        num_batches = len(dataloader)
        # tbar = tqdm(dataloader, desc="\r")
        for batch, (X, y) in enumerate(dataloader):
            with torch.no_grad():
                onehot_target=torch.eye(2)[y.long().cpu(), :].to(self.device)
                pred = self.model(X)
                y_predict=torch.cat([y_predict,pred.argmax(1)],dim=0)
                y_true=torch.cat([y_true,y],dim=0)
                test_loss += self.loss_fn(pred.reshape(-1,1,2), onehot_target).item()
        test_loss /= num_batches
        met=metrics.scores(y_true.cpu().numpy(),y_predict.cpu().numpy())
        return test_loss,met

    def start_train(self):
        pbar = tqdm(total=self.max_epoch-self.start_epoch)
        lastTmet={}
        lastmet={}
        if opt.lrsc=="warmup":
            warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)
        for epoch in range(self.start_epoch, self.max_epoch):
            tloss,lastTmet=self.train(epoch)
            loss, lastmet = self.valid(self.valid_loader, epoch)
            #控制
            if opt.lrsc=="warmup":
                with warmup_scheduler.dampening():
                    self.lrsc.step()
            else:
                self.lrsc.step(loss)
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_score = lastTmet['accuracy']
            if epoch >= self.max_epoch // 2 and epoch % 10 == 0:
                self.save_checkpoint(epoch,save_optimizer=True,suffix="epoch" + str(epoch)+self.filename)
            self.save_checkpoint(epoch, save_optimizer=True, suffix="last"+self.filename)
            pbar.set_postfix({'val accuary:':lastmet['accuracy']})   
            pbar.update()
        test_lost,test_met=self.valid(self.test_loader,0)
        with open(self.save_dir + str(self.filename) + '_test_evaluate.txt', 'a+', encoding='utf8') as fw:
            fw.write("-----------模型预测评估-----------\n")
            fw.write("Trainscore:\n")
            fw.write(str(lastTmet))
            fw.write("\n")
            fw.write("Validscore:\n")
            fw.write(str(lastmet))
            fw.write("\n")
            fw.write("Testscore:\n")
            fw.write(str(test_met))
            fw.write("\n")
            fw.close()
        print("训练集上:")
        print(lastTmet)
        print("验证集上:")
        print(lastmet)
        print("测试集上:")
        print(test_met)