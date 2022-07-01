# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import os
import shutil
import time
import torch.nn.parallel
from options import Options
# from gelsight_feature_loader import MyFeatureDataset
from xela_dataloader1 import MyDataset

import sys
sys.path.append('../fu_vision_flex_project/utils/progress/progress/')
import bar
from bar import Bar
sys.path.append('../fu_vision_flex_project/utils/')
import logger,misc
from logger import Logger,savefig
from misc import AverageMeter,ACC


#from utils import Bar, Logger, AverageMeter, savefig, ACC
from models.models import *    
import cv2
from network_utils import init_weights_xavier
from sklearn.metrics import accuracy_score,recall_score
import torch.nn.functional as F
from torch.utils import data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt





def main():
    opt = Options().parse()
    start_epoch = opt.start_epoch  # start from epoch 0 or last checkpoint epoch
    opt.phase = 'train'
    transform_v = transforms.Compose([transforms.Resize([opt.cropWidth, opt.cropHeight]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transform_f = transforms.Compose([transforms.Resize([4, 4]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    log=6 
    trainset = MyDataset('../fu_vision_flex_project/graspingdata',6,3,transform_v,transform_f,log,flag = opt.phase)
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=0
    )
    opt.phase = 'test'
    validset = MyDataset('../fu_vision_flex_project/graspingdata',6,3,transform_v,transform_f,log,flag = opt.phase)
    val_loader = torch.utils.data.DataLoader(
        dataset=validset,
        batch_size=opt.batchSize,
        shuffle=False,
         num_workers=0
    )


    # Model
    if opt.model_arch == 'C3D_vision_C1D_flex':
        model = C3D_vision_C1D_flex(drop_p_v=0.2,visual_dim=1024,  fc_hidden_1=128,fc_hidden_f = 20,num_classes=2)#visual_dim=4096
   
    if opt.use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.to( torch.device('cpu') )
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Loss and optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)

    title = opt.name
    if opt.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(opt.resume), 'Error: no checkpoint directory found!'
        opt.checkpoint = os.path.dirname(opt.resume)
        checkpoint = torch.load(opt.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(opt.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(opt.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Valid PSNR.'])

    if opt.evaluate:
        print('\nEvaluation only')
        val_loss, val_psnr = valid(val_loader, model, start_epoch, opt.use_cuda)
        print(' Test Loss:  %.8f, Test PSNR:  %.2f' % (val_loss, val_psnr))
        return

    # Train and val

    best_acc = 0
    train_acc_list=[]
    train_loss_list=[]
    test_acc_list=[]
    test_loss_list=[]
    for epoch in range(start_epoch, opt.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, opt.epochs, opt.lr))

        train_loss,train_acc = train(train_loader, model,  optimizer, epoch, opt.use_cuda)
        test_loss, test_acc= valid(val_loader, model,  epoch, opt.use_cuda)

        # append logger file
        logger.append([opt.lr, train_loss, test_loss, test_acc])
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=opt.checkpoint)
        print('Best acc:')
        print(best_acc)

    logger.close()
    np.save(
        'XELA_results/train/train_acc_' + opt.model_arch + str(opt.batchSize) + '_' + str(opt.lr) + '.npy',
        train_acc_list)
    np.save('XELA_results/train/train_loss_' + opt.model_arch + str(opt.batchSize) + '_' + str(opt.lr) + '.npy', train_loss_list)
    np.save(
        'XELA_results/test/test_acc_' + opt.model_arch + str(opt.batchSize) + '_' + str(opt.lr) + '.npy',
        test_acc_list)
    np.save(
        'XELA_results/test/test_loss_' + opt.model_arch + str(opt.batchSize) + '_' + str(opt.lr) + '.npy',
        test_loss_list)
    '''
    plt.plot(train_loss_list)
    plt.plot(train_acc_list)
    plt.plot(test_loss_list)
    plt.plot(test_acc_list)
    '''
    plt.plot(train_loss_list,label='train_loss')
    plt.plot(train_acc_list,label='train_acc')
    plt.plot(test_loss_list,label='test_loss')
    plt.plot(test_acc_list,label='test_acc')
    plt.legend()  #给图像加上图例
    
    plt.show()
  




def train(trainloader, model, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()
    psnr_input = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (x_visual,x_flex, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        x_visual,x_flex, targets = torch.autograd.Variable(x_visual), torch.autograd.Variable(x_flex),torch.autograd.Variable(targets)
        if use_cuda:
            x_visual=x_visual.cuda()
            x_flex=x_flex.cuda()
            targets = targets.cuda(non_blocking=True)
  
        x_flex = x_flex.to(torch.float32)
       
        x_flex = x_flex.unsqueeze(1)
        # compute output
        outputs = model(x_visual,x_flex)
        loss = F.cross_entropy(outputs, targets, reduction='mean') 
        y_pred = torch.max(outputs, 1)[1]  
        acc =  accuracy_score(y_pred.cpu().data.numpy(), targets.cpu().data.numpy())  #准确率      
       
        
        # measure the result
        losses.update(loss.item(), x_visual.size(0))
        avg_acc.update(acc, x_visual.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress | PSNR: {psnr: .4f} | PSNR(input): {psnr_in: .4f}
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}| ACC(input): {acc: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    acc=avg_acc.avg,
                    )
        bar.next()
    bar.finish()
    return losses.avg,avg_acc.avg


def valid(testloader, model, epoch, use_cuda):
    # switch to train mode
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()
    psnr_input = AverageMeter()
    end = time.time()

    y_targets_list = []
    y_pred_list = []
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, ( x_visual,x_flex, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        x_visual, x_flex, targets =  torch.autograd.Variable(x_visual),torch.autograd.Variable(x_flex), torch.autograd.Variable(targets)
        if use_cuda:
            x_visual = x_visual.cuda()
            x_flex = x_flex.cuda()
            targets = targets.cuda(non_blocking=True)

        x_flex = x_flex.to(torch.float32)
       
        x_flex = x_flex.unsqueeze(1)
        # compute output
        outputs = model(x_visual,x_flex)  
        loss = F.cross_entropy(outputs, targets)
        y_pred = torch.max(outputs, 1)[1]  # y_pred != output
        acc =  accuracy_score(y_pred.cpu().data.numpy(), targets.cpu().data.numpy())

        # measure the result
        losses.update(loss.item(), x_visual.size(0))
        avg_acc.update(acc, x_visual.size(0))

        pred = y_pred.cpu().numpy()       
        Y_pred=pred.tolist() 
        y_pred_list.append(Y_pred)        
        strNums=[str(Y_pred_i) for Y_pred_i in y_pred_list] 
        Ypred=",".join(strNums)

        target = targets.cpu().numpy()
        Targets=target.tolist() 
        y_targets_list.append(Targets)
        strNums=[str(Targets_i) for Targets_i in y_targets_list]
        Ytargets=",".join(strNums)
              
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress | PSNR: {psnr: .4f} | PSNR(input): {psnr_in: .4f}
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}| ACC(input): {acc: .4f}'.format(
            batch=batch_idx + 1,
            size=len(testloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=avg_acc.avg,
            # psnr_in=psnr_input.avg
        )
        bar.next()
    bar.finish()

    fileName='E:/github/fu_Experimental_results/'+ 'C3D_vision_C1D_flex' + '-' +'y_pred.txt'
    fw = open(fileName, 'w') 
    fw.write(Ypred)               
    fw.write("\n") 
    fw.close
    
    filename='E:/github/fu_Experimental_results/'+ 'C3D_vision_C1D_flex' + '-' +'y_targets.txt'
    FW = open(filename, 'w') 
    FW.write(Ytargets)
    FW.write("\n") 
    FW.close
    
    return (losses.avg, avg_acc.avg)
   

def adjust_learning_rate(optimizer, epoch, opt):
    if epoch % opt.schedule ==0 and epoch !=0 :
        opt.lr *= opt.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.lr


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


if __name__ == "__main__":
    main()

