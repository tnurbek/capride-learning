import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
 
import os
import time
import shutil

from tqdm import tqdm
from utils import accuracy, AverageMeter
from models import resnet_model 
from tensorboard_logger import configure, log_value
from loss import ApproximateKLLoss 


class Trainer(object):
    
    def __init__(self, config, data_loader):
        self.config = config

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader[0].dataset) 
            print('[trainer.py] P0D0:', self.num_train)
            self.num_valid = len(self.valid_loader.dataset)
            self.total_train = sum([len(self.train_loader[j].dataset) for j in range(config.model_num)])
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
        self.num_classes = config.num_classes

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr
        self.weight_decay = config.weight_decay
        self.nesterov = config.nesterov
        self.gamma = config.gamma
        # misc params
        self.use_gpu = config.use_gpu
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.model_name = config.save_name
        self.model_num = config.model_num
        
        self.models = []
        self.optimizers = []
        self.schedulers = []
        
        self.loss_kl = ApproximateKLLoss()
        self.loss_ce = nn.CrossEntropyLoss()
        self.best_valid_accs = [0.] * self.model_num
        
        self.alpha = 200  # hyperparameter to balance between KL loss and CE loss [sum of multipliers of KL loss] 
        self.alphas =  [{i: self.alpha / (self.model_num - 1) for i in list(set(list(range(self.model_num))) - set([j, ]))} for j in range(self.model_num)]
        self.T = 5.0  # temperature hyperparameter 

        # tensorboard logging
        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + self.model_name
            print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

        for i in range(self.model_num):
            
            # models
            model = resnet_model()
            if self.use_gpu:
                model.cuda()
            self.models.append(model)  # ResNet18

            # initialize optimizer and scheduler
            optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum,
                                  weight_decay=self.weight_decay, nesterov=self.nesterov)
            self.optimizers.append(optimizer)

            # set learning rate decay
            scheduler = optim.lr_scheduler.MultiStepLR(self.optimizers[i], milestones=[50, 75], gamma=self.gamma)
            self.schedulers.append(scheduler)

        print('[*] Number of parameters of one model: {:,}'.format(
            sum([p.data.nelement() for p in self.models[0].parameters()])))

        self.num_train = self.num_train if config.is_train else self.num_test


    def train(self):
        
        # to load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

        print("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid)
        )

        for epoch in range(self.start_epoch, self.epochs):

            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch + 1, self.epochs, self.optimizers[0].param_groups[0]['lr'], )
            )

            # train for one epoch locally using only CE loss
            train_losses, train_accs = self.train_local(epoch)

            # evaluate on validation set
            valid_losses, valid_accs = self.validate_local(epoch)

            for scheduler in self.schedulers:
                scheduler.step()

            for i in range(self.model_num):
                is_best = valid_accs[i].avg > self.best_valid_accs[i]
                msg1 = "model_{:d}: train loss: {:.3f} - train acc: {:.3f} "
                msg2 = "- val loss: {:.3f} - val acc: {:.3f}"
                if is_best:
                    msg2 += " [*]"
                msg = msg1 + msg2
                print(msg.format(i + 1, train_losses[i].avg, train_accs[i].avg, valid_losses[i].avg, valid_accs[i].avg))

                self.best_valid_accs[i] = max(valid_accs[i].avg, self.best_valid_accs[i])
                self.save_checkpoint(i, {'epoch': epoch + 1, 'model_state': self.models[i].state_dict(),
                                         'optim_state': self.optimizers[i].state_dict(),
                                         'best_valid_acc': self.best_valid_accs[i], }, is_best)

        # validation recording
        print("Validation after Local Training: ")
        self.test_evaluation()

        if self.start_epoch < self.epochs: self.start_epoch = self.epochs

        # additional section for CaPriDe part
        for epoch in range(self.start_epoch, int(self.epochs + 3 * self.epochs)):

            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch + 1, self.epochs * 4, self.optimizers[0].param_groups[0]['lr'], )
            )
            
            if 'ind' in self.model_name: # to train locally
                train_losses, train_accs = self.train_local(epoch)
                valid_losses, valid_accs = self.validate_local(epoch)

            elif 'capride' in self.model_name: # collaborative learning
                train_losses, train_accs = self.train_capride(epoch)
                valid_losses, valid_accs = self.validate_capride(epoch)

            for scheduler in self.schedulers:
                scheduler.step()

            for i in range(self.model_num):
                is_best = valid_accs[i].avg > self.best_valid_accs[i]
                msg1 = "model_{:d}: train loss: {:.3f} - train acc: {:.3f} "
                msg2 = "- val loss: {:.3f} - val acc: {:.3f}"
                if is_best:
                    msg2 += " [*]"
                msg = msg1 + msg2
                print(msg.format(i + 1, train_losses[i].avg, train_accs[i].avg, valid_losses[i].avg, valid_accs[i].avg))

                self.best_valid_accs[i] = max(valid_accs[i].avg, self.best_valid_accs[i])
                self.save_checkpoint(i, {'epoch': epoch + 1, 'model_state': self.models[i].state_dict(),
                                         'optim_state': self.optimizers[i].state_dict(),
                                         'best_valid_acc': self.best_valid_accs[i], }, is_best)

        print("Validation after CaPriDe Training: ")
        self.test_evaluation()


    def test(self):
        
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # whether to load the best checkpoint 
        self.load_checkpoint(best=self.best)
        
        for i in range(self.model_num):
            self.models[i].eval()
            for _, (images, labels) in enumerate(self.test_loader):
                if self.use_gpu:
                    images, labels = images.cuda(), labels.cuda()
                images, labels = Variable(images), Variable(labels)

                # forward pass
                outputs = self.models[i](images)
                loss = self.loss_ce(outputs, labels)  # cross entropy

                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
                losses.update(loss.item(), images.size()[0])
                top1.update(prec1.item(), images.size()[0])
                top5.update(prec5.item(), images.size()[0])
            
            print(
                '[*] Test loss: {:.3f}, top1_acc: {:.3f}%, top5_acc: {:.3f}%'.format(
                    losses.avg, top1.avg, top5.avg)
            )

            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()


    def save_checkpoint(self, i, state, is_best):
        
        filename = self.model_name + str(i + 1) + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.model_name + str(i + 1) + '_model_best.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )


    def load_checkpoint(self, best=False):

        print("[*] Loading model from {}".format(self.ckpt_dir))
        
        for i in range(self.model_num):
            model_id = i + 1
            filename = self.model_name + f'{model_id}_ckpt.pth.tar'
            if best:
                filename = self.model_name + f'{model_id}_model_best.pth.tar'
            ckpt_path = os.path.join(self.ckpt_dir, filename)
            ckpt = torch.load(ckpt_path)
            
            # load variables from checkpoint
            self.start_epoch = ckpt['epoch']
            self.best_valid_acc = ckpt['best_valid_acc']
            self.models[i].load_state_dict(ckpt['model_state'])
            self.optimizers[i].load_state_dict(ckpt['optim_state'])


        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt['epoch'], ckpt['best_valid_acc'])
            )
        else:
            print(
                "[*] Loaded {} checkpoint @ epoch {}".format(
                    filename, ckpt['epoch'])
            )


    def test_evaluation(self):
        """
        To evaluate models on validation set
        """
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for i in range(self.model_num): 
            self.models[i].eval()
            for _, (images, labels) in enumerate(self.valid_loader): 
                if self.use_gpu:
                    images, labels = images.cuda(), labels.cuda()
                images, labels = Variable(images), Variable(labels)

                # forward pass
                outputs = self.models[i](images)

                loss = self.loss_ce(outputs, labels)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
                losses.update(loss.item(), images.size()[0])
                top1.update(prec1.item(), images.size()[0])
                top5.update(prec5.item(), images.size()[0])

            print(
                '[*] P{}. Validation loss: {:.3f}, top1_acc: {:.3f}%, top5_acc: {:.3f}%'.format(
                    i, losses.avg, top1.avg, top5.avg)
            )
            
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

    
    def train_local(self, epoch):
        """
        To train models locally [no collaboration]. 
        Args:
            epoch (int): current epoch

        Returns:
            losses: AverageMeter class object 
            accs: AverageMeter class object 
        """
        batch_time = AverageMeter()
        losses = []
        accs = []
        
        for i in range(self.model_num):
            self.models[i].train()
            losses.append(AverageMeter())
            accs.append(AverageMeter())

        tic = time.time()
        with tqdm(total=self.total_train) as pbar: 

            for i in range(self.model_num):
                for k, (images, labels) in enumerate(self.train_loader[i]):

                    if self.use_gpu:
                        images, labels = images.cuda(), labels.cuda()
                    images, labels = Variable(images), Variable(labels)

                    # forward pass
                    output = self.models[i](images)
                    loss = self.loss_ce(output, labels)

                    # measure accuracy and record loss
                    prec = accuracy(output.data, labels.data, topk=(1,))[0]
                    losses[i].update(loss.item(), images.size()[0])
                    accs[i].update(prec.item(), images.size()[0])

                    # compute gradients and update SGD
                    self.optimizers[i].zero_grad()
                    loss.backward()                    
                    self.optimizers[i].step()

                    # newly added
                    torch.cuda.empty_cache()

                    # measure elapsed time
                    toc = time.time()
                    batch_time.update(toc - tic)

                    pbar.set_description(
                        (
                            "{:.1f}s - model{}_loss: {:.3f} - model{}_acc: {:.3f}".format(
                                (toc - tic), i + 1, losses[i].avg, i + 1, accs[i].avg
                            )
                        )
                    )
                    self.batch_size = images.shape[0]
                    pbar.update(self.batch_size)

                    if self.use_tensorboard:
                        iteration = epoch * len(self.train_loader[i]) + k
                        log_value('train_loss_%d' % (i + 1), losses[i].avg, iteration)
                        log_value('train_acc_%d' % (i + 1), accs[i].avg, iteration)

        return losses, accs

    # validation of individually trained models
    def validate_local(self, epoch):
        """
        To validate models locally [no collaboration]. 
        Args:
            epoch (int): current epoch

        Returns:
            losses: AverageMeter class object 
            accs: AverageMeter class object 
        """
        losses = []
        accs = []

        for i in range(self.model_num):
            self.models[i].eval()
            losses.append(AverageMeter())
            accs.append(AverageMeter())

        for i, (images, labels) in enumerate(self.valid_loader):
            if self.use_gpu:
                images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)

            # forward pass
            outputs = []
            for model in self.models:
                outputs.append(model(images))

            for i in range(self.model_num):
                loss = self.loss_ce(outputs[i], labels)

                # measure accuracy and record loss
                prec = accuracy(outputs[i].data, labels.data, topk=(1,))[0]
                losses[i].update(loss.item(), images.size()[0])
                accs[i].update(prec.item(), images.size()[0])

                # newly added
                torch.cuda.empty_cache()

        # log to tensorboard for every epoch
        if self.use_tensorboard:
            for i in range(self.model_num):
                log_value('valid_loss_%d' % (i + 1), losses[i].avg, epoch + 1)
                log_value('valid_acc_%d' % (i + 1), accs[i].avg, epoch + 1)

        return losses, accs

    # train CaPriDe 
    def train_capride(self, epoch):
        """
        To train models collaboratively [CaPriDe Learning]. 
        Args:
            epoch (int): current epoch

        Returns:
            losses: AverageMeter class object 
            accs: AverageMeter class object 
        """
        batch_time = AverageMeter()
        losses = []
        kl_losses = []
        accs = []

        for i in range(self.model_num):
            self.models[i].train()
            losses.append(AverageMeter())
            accs.append(AverageMeter())
            kl_losses.append(AverageMeter())
        
        tic = time.time()
        with tqdm(total=self.total_train) as pbar: 

            for i in range(self.model_num):
                for k, (images, labels) in enumerate(self.train_loader[i]):
                    if self.use_gpu:
                        images, labels = images.cuda(), labels.cuda()
                    images, labels = Variable(images), Variable(labels)

                    # forward pass
                    outputs = []
                    for model in self.models:
                        outputs.append(model(images))

                    ce_loss = self.loss_ce(outputs[i], labels)

                    kl_loss = 0
                    for j in range(self.model_num):
                        if i != j:
                            current_loss = self.loss_kl(outputs[i], outputs[j].detach())
                            kl_loss += self.alphas[i][j] * current_loss

                    kl_loss = kl_loss / (self.model_num - 1)
                    loss = ce_loss + kl_loss

                    # measure accuracy and record loss
                    prec = accuracy(outputs[i].data, labels.data, topk=(1,))[0]
                    losses[i].update(loss.item(), images.size()[0])
                    accs[i].update(prec.item(), images.size()[0])
                    kl_losses[i].update(kl_loss, images.size()[0])

                    # compute gradients and update SGD
                    self.optimizers[i].zero_grad()
                    loss.backward()
                    self.optimizers[i].step()

                    # newly added
                    torch.cuda.empty_cache()

                    # measure elapsed time
                    toc = time.time()
                    batch_time.update(toc - tic)

                    pbar.set_description(
                        (
                            "{:.1f}s - model{}_loss: {:.3f} - model{}_acc: {:.3f} - klloss: {:.4f}".format(
                                (toc - tic), i + 1, losses[i].avg, i + 1, accs[i].avg, kl_losses[i].avg
                            )
                        )
                    )

                    self.batch_size = images.shape[0]
                    pbar.update(self.batch_size)

                    if self.use_tensorboard:
                        iteration = epoch * len(self.train_loader[i]) + k
                        log_value('train_loss_%d' % (i + 1), losses[i].avg, iteration)
                        log_value('train_acc_%d' % (i + 1), accs[i].avg, iteration)
                        log_value('train_kl_loss_%d' % (i + 1), kl_losses[i].avg, iteration)

        return losses, accs

    # validate CaPriDe training
    def validate_capride(self, epoch):
        """
        To evaluate models on validation set.  
        Args:
            epoch (int): current epoch

        Returns:
            losses: AverageMeter class object 
            accs: AverageMeter class object 
        """
        losses = []
        kl_losses = []
        accs = []

        for i in range(self.model_num):
            self.models[i].eval()
            losses.append(AverageMeter())
            accs.append(AverageMeter())
            kl_losses.append(AverageMeter())

        for _, (images, labels) in enumerate(self.valid_loader):
            if self.use_gpu:
                images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)

            # forward pass
            outputs = []
            for model in self.models:
                outputs.append(model(images))

            for i in range(self.model_num):
                ce_loss = self.loss_ce(outputs[i], labels)

                kl_loss = 0
                for j in range(self.model_num):
                    if i != j:
                        current_loss = self.loss_kl(outputs[i], outputs[j].detach())
                        kl_loss += self.alphas[i][j] * current_loss

                kl_loss =  kl_loss / (self.model_num - 1)
                loss = ce_loss + kl_loss 

                # measure accuracy and record loss
                prec = accuracy(outputs[i].data, labels.data, topk=(1,))[0]
                losses[i].update(loss.item(), images.size()[0])
                accs[i].update(prec.item(), images.size()[0])
                kl_losses[i].update(kl_loss.item(), images.size()[0])

                # newly added
                torch.cuda.empty_cache()

        # log to tensorboard for every epoch
        if self.use_tensorboard:
            for i in range(self.model_num):
                log_value('valid_loss_%d' % (i + 1), losses[i].avg, epoch + 1)
                log_value('valid_acc_%d' % (i + 1), accs[i].avg, epoch + 1)
                log_value('valid_kl_loss_%d' % (i + 1), kl_losses[i].avg, epoch + 1)

        return losses, accs
