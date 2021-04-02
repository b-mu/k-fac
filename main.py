'''Train CIFAR10/CIFAR100 with PyTorch.'''
import argparse
import os
from optimizers import (KFACOptimizer, EKFACOptimizer)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.network_utils import get_network
from utils.data_utils import get_dataloader
from torchsummary import summary

from backpack import backpack, extend
from backpack.extensions import FisherBlock, MNGD
import math
import time
import copy
import matplotlib.pylab as plt

from torch import einsum, matmul, eye
from torch.linalg import inv
# for REPRODUCIBILITY
torch.manual_seed(0)

# fetch args
parser = argparse.ArgumentParser()

parser.add_argument('--network', default='vgg16_bn', type=str)
parser.add_argument('--depth', default=19, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)

# densenet
parser.add_argument('--growthRate', default=12, type=int)
parser.add_argument('--compressionRate', default=2, type=int)

# wrn, densenet
parser.add_argument('--widen_factor', default=1, type=int)
parser.add_argument('--dropRate', default=0.0, type=float)


parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--resume', '-r', action='store_true')
parser.add_argument('--load_path', default='', type=str)
parser.add_argument('--log_dir', default='runs/pretrain', type=str)


parser.add_argument('--optimizer', default='kfac', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--milestone', default=None, type=str)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--stat_decay', default=0.95, type=float)
parser.add_argument('--damping', default=1e-3, type=float)
parser.add_argument('--kl_clip', default=1e-2, type=float)
parser.add_argument('--weight_decay', default=3e-3, type=float)
parser.add_argument('--TCov', default=10, type=int)
parser.add_argument('--TScal', default=10, type=int)
parser.add_argument('--TInv', default=100, type=int)

parser.add_argument('--eps', default=0.25, type=float)
parser.add_argument('--boost', default=1.001, type=float)
parser.add_argument('--drop', default=0.99, type=float)
parser.add_argument('--adaptive', default='false', type=str)
parser.add_argument('--alpha', default=0.1, type=float)
parser.add_argument('--taw', default=0.01, type=float)
parser.add_argument('--freq', default=10, type=int)
parser.add_argument('--low_rank', default='false', type=str)
parser.add_argument('--gamma', default=0.95, type=float)


parser.add_argument('--prefix', default=None, type=str)
args = parser.parse_args()

# init model
nc = {
    'cifar10': 10,
    'cifar100': 100,
    'mnist':10
}
num_classes = nc[args.dataset]
net = get_network(args.network,
                  depth=args.depth,
                  num_classes=num_classes,
                  growthRate=args.growthRate,
                  compressionRate=args.compressionRate,
                  widen_factor=args.widen_factor,
                  dropRate=args.dropRate)



net = net.to(args.device)




if args.dataset == 'mnist':
    summary(net, ( 1, 28, 28))
elif args.dataset == 'cifar10':
    summary(net, ( 3, 32, 32))
elif args.dataset == 'cifar100':
    summary(net, ( 3, 32, 32))

# init dataloader
trainloader, testloader = get_dataloader(dataset=args.dataset,
                                         train_batch_size=args.batch_size,
                                         test_batch_size=256)

# init optimizer and lr scheduler
optim_name = args.optimizer.lower()
tag = optim_name
if optim_name == 'sgd':
    optimizer = optim.SGD(net.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
elif optim_name == 'kfac':
    optimizer = KFACOptimizer(net,
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              stat_decay=args.stat_decay,
                              damping=args.damping,
                              kl_clip=args.kl_clip,
                              weight_decay=args.weight_decay,
                              TCov=args.TCov,
                              TInv=args.TInv)
elif optim_name == 'ekfac':
    optimizer = EKFACOptimizer(net,
                               lr=args.learning_rate,
                               momentum=args.momentum,
                               stat_decay=args.stat_decay,
                               damping=args.damping,
                               kl_clip=args.kl_clip,
                               weight_decay=args.weight_decay,
                               TCov=args.TCov,
                               TScal=args.TScal,
                               TInv=args.TInv)
elif optim_name == 'ngd':
    print('NGD optimizer selected.')
    optimizer = optim.SGD(net.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum)

    buf = {}
    if args.momentum != 0:
        for name, param in net.named_parameters():
                # print('initializing momentum buffer')
                buf[name] = torch.zeros_like(param.data).to(args.device) 

else:
    raise NotImplementedError

if args.milestone is None:
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(args.epoch*0.5), int(args.epoch*0.75)], gamma=0.1)
else:
    milestone = [int(_) for _ in args.milestone.split(',')]
    lr_scheduler = MultiStepLR(optimizer, milestones=milestone, gamma=0.1)

# init criterion
criterion = nn.CrossEntropyLoss()
criterion_none = nn.CrossEntropyLoss(reduction='none')

if optim_name == 'ngd':
    extend(net)
    extend(criterion)
    extend(criterion_none)
    # print(net.state_dict())


# parameters for damping update
# damping = args.damping





damping = args.damping
epsilon = args.eps
boost = args.boost
drop = args.drop
alpha_LM = args.alpha
taw = args.taw


start_epoch = 0
best_acc = 0
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.load_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.load_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('==> Loaded checkpoint at epoch: %d, acc: %.2f%%' % (start_epoch, best_acc))

# init summary writter

log_dir = os.path.join(args.log_dir, args.dataset, args.network, args.optimizer,
                       'lr%.3f_wd%.4f_damping%.4f' %
                       (args.learning_rate, args.weight_decay, args.damping))
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

loss_prev = 0.
taylor_appx_prev = 0.

non_descent = 0
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global alpha_LM
    global loss_prev
    global taylor_appx_prev
    global non_descent
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    print('\nKFAC damping: %f' % damping)
    print('\nNGD damping: %f' % (alpha_LM + taw))
    st_time = time.time()

    # 
    desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (tag, lr_scheduler.get_last_lr()[0], 0, 0, correct, total))

    writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], epoch)

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:

        if optim_name in ['kfac', 'ekfac', 'sgd'] :
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            if optim_name in ['kfac', 'ekfac'] and optimizer.steps % optimizer.TCov == 0:
                # compute true fisher
                optimizer.acc_stats = True
                with torch.no_grad():
                    sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.cpu().data, dim=1),1).squeeze().to(args.device)
                loss_sample = criterion(outputs, sampled_y)
                loss_sample.backward(retain_graph=True)
                optimizer.acc_stats = False
                optimizer.zero_grad()  # clear the gradient for computing true-fisher.
            loss.backward()
            optimizer.step()


        elif optim_name == 'ngd':
            if batch_idx % args.freq == 0:
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                optimizer.zero_grad()
                # net.set_require_grad(True)

                outputs = net(inputs)
                damp = alpha_LM + taw
                loss = criterion(outputs, targets)
                loss.backward(retain_graph=True)

                # storing original gradient for later use
                grad_org = []
                # grad_dict = {}
                for name, param in net.named_parameters():
                    grad_org.append(param.grad.reshape(1, -1))
                #     grad_dict[name] = param.grad.clone()
                grad_org = torch.cat(grad_org, 1)

                ###### now we have to compute the true fisher
                with torch.no_grad():
                # gg = torch.nn.functional.softmax(outputs, dim=1)
                    sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs, dim=1),1).squeeze().to(args.device)
                
                update_list, loss = optimal_JJT(outputs, sampled_y, args.batch_size, damping=damp, alpha=0.95, low_rank=args.low_rank, gamma=args.gamma)
                # optimizer.zero_grad()
                # update_list, loss = optimal_JJT_fused(outputs, sampled_y, args.batch_size, damping=damp)

                optimizer.zero_grad()
   
                # last part of SMW formula
                grad_new = []
                for name, param in net.named_parameters():
                    param.grad.copy_(update_list[name])
                    grad_new.append(param.grad.reshape(1, -1))
                grad_new = torch.cat(grad_new, 1)   
                # grad_new = grad_org

            else:
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                optimizer.zero_grad()
                # net.set_require_grad(True)

                outputs = net(inputs)
                damp = alpha_LM + taw
                loss = criterion(outputs, targets)
                loss.backward(retain_graph=True)

                # storing original gradient for later use
                grad_org = []
                # grad_dict = {}
                for name, param in net.named_parameters():
                    grad_org.append(param.grad.reshape(1, -1))
                #     grad_dict[name] = param.grad.clone()
                grad_org = torch.cat(grad_org, 1)

                ###### now we have to compute the true fisher
                # with torch.no_grad():
                # gg = torch.nn.functional.softmax(outputs, dim=1)
                    # sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs, dim=1),1).squeeze().to(args.device)
                
                for m in net.features.children():
                    if hasattr(m, "NGD_inv"):                    
                        grad = m.weight.grad
                        if isinstance(m, nn.Linear):
                            I = m.I
                            G = m.G
                            n = I.shape[0]
                            NGD_inv = m.NGD_inv
                            grad_prod = einsum("ni,oi->no", (I, grad))
                            grad_prod = einsum("no,no->n", (grad_prod, G))
                            v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()
                            gv = einsum("n,no->no", (v, G))
                            gv = einsum("no,ni->oi", (gv, I))
                            gv = gv / n
                            update = (grad - gv)/damp
                            m.weight.grad.copy_(update)
                        elif isinstance(m, nn.Conv2d):
                            if hasattr(m, "AX"):

                                if args.low_rank.lower() == 'true':
                                    # print('low_rank coomputations')
                                    ###### testing low rank structure
                                    U = m.U
                                    S = m.S
                                    V = m.V
                                    NGD_inv = m.NGD_inv
                                    n = NGD_inv.shape[0]

                                    grad_reshape = grad.reshape(grad.shape[0], -1)
                                    grad_prod = V @ grad_reshape.t().reshape(-1, 1)
                                    grad_prod = torch.diag(S) @ grad_prod
                                    grad_prod = U @ grad_prod
                                    grad_prod = grad_prod.squeeze()
                                    v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()
                                    gv= U.t() @ v.unsqueeze(1)
                                    gv = torch.diag(S) @ gv
                                    gv = V.t() @ gv
                                    gv = gv.reshape(grad_reshape.shape[1], grad_reshape.shape[0]).t()
                                    gv = gv.view_as(grad)
                                    gv = gv / n
                                    update = (grad - gv)/damp
                                    m.weight.grad.copy_(update)
                                else:
                                    AX = m.AX
                                    NGD_inv = m.NGD_inv
                                    n = AX.shape[0]

                                    grad_reshape = grad.reshape(grad.shape[0], -1)
                                    grad_prod = einsum("nkm,mk->n", (AX, grad_reshape))
                                    v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()
                                    gv = einsum("nkm,n->mk", (AX, v))
                                    gv = gv.view_as(grad)
                                    gv = gv / n
                                    update = (grad - gv)/damp
                                    m.weight.grad.copy_(update)
                            elif hasattr(m, "I"):
                                I = m.I
                                G = m.G
                                n = I.shape[0]
                                NGD_inv = m.NGD_inv
                                grad_reshape = grad.reshape(grad.shape[0], -1)
                                x1 = einsum("nkl,mk->nml", (I, grad_reshape))
                                grad_prod = einsum("nml,nml->n", (x1, G))
                                v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()
                                gv = einsum("n,nml->nml", (v, G))
                                gv = einsum("nml,nkl->mk", (gv, I))
                                gv = gv.view_as(grad)
                                gv = gv / n
                                update = (grad - gv)/damp
                                m.weight.grad.copy_(update)
                        
                        


                # update_list, loss = optimal_JJT(outputs, sampled_y, args.batch_size, damping=damp)
                # optimizer.zero_grad()
                # update_list, loss = optimal_JJT_fused(outputs, sampled_y, args.batch_size, damping=damp)

   
                # last part of SMW formula
                grad_new = []
                for name, param in net.named_parameters():
                    grad_new.append(param.grad.reshape(1, -1))
                grad_new = torch.cat(grad_new, 1)   
                # grad_new = grad_org

                # print('freq')


            ##### do kl clip
            lr = lr_scheduler.get_last_lr()[0]
            vg_sum = 0
            vg_sum += (grad_new * grad_org ).sum()
            vg_sum = vg_sum * (lr ** 2)
            nu = min(1.0, math.sqrt(args.kl_clip / vg_sum))
            for name, param in net.named_parameters():
                param.grad.mul_(nu)

            optimizer.step()

            
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (tag, lr_scheduler.get_last_lr()[0], train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)
        # if batch_idx % 10 == 0:
        #   print(time.time() - st_time, loss.item(),  100. * correct / total)
    writer.add_scalar('train/loss', train_loss/(batch_idx + 1), epoch)
    writer.add_scalar('train/acc', 100. * correct / total, epoch)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (tag,lr_scheduler.get_lr()[0], test_loss/(0+1), 0, correct, total))

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (tag, lr_scheduler.get_lr()[0], test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

    # Save checkpoint.
    acc = 100.*correct/total

    writer.add_scalar('test/loss', test_loss / (batch_idx + 1), epoch)
    writer.add_scalar('test/acc', 100. * correct / total, epoch)

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'loss': test_loss,
            'args': args
        }

        torch.save(state, '%s/%s_%s_%s%s_best.t7' % (log_dir,
                                                     args.optimizer,
                                                     args.dataset,
                                                     args.network,
                                                     args.depth))
        best_acc = acc

def optimal_JJT(outputs, targets, batch_size, damping=1.0, alpha=0.95, low_rank='false', gamma=0.95):
    jac_list = 0
    vjp = 0
    update_list = {}
    with backpack(FisherBlock(damping, alpha, low_rank, gamma)):
        loss = criterion(outputs, targets)
        loss.backward(retain_graph=True)
    for name, param in net.named_parameters():
        fisher_vals = param.fisher_block
        update_list[name] = fisher_vals[2]
    return update_list, loss

def optimal_JJT_fused(outputs, targets, batch_size, damping=1.0):
    jac_list = 0
    vjp = 0
    update_list = {}
    loss = 0
    with backpack(MNGD()):
        loss = criterion(outputs, targets)
        loss.backward(retain_graph=True)
    # for name, param in net.named_parameters():
        # fisher_vals = param.fisher_block
        # update_list[name] = fisher_vals[2]
    return update_list, loss
    

def memory_cleanup(module):
    """Remove I/O stored by backpack during the forward pass.

    Deletes the attributes created by `hook_store_io` and `hook_store_shapes`.
    """
    if hasattr(module, "output"):
        delattr(module, "output")
    if hasattr(module, "output_shape"):
        delattr(module, "output_shape")
    i = 0
    while hasattr(module, "input{}".format(i)):
        delattr(module, "input{}".format(i))
        i += 1
    i = 0
    while hasattr(module, "input{}_shape".format(i)):
        delattr(module, "input{}_shape".format(i))
        i += 1


def main():
    for epoch in range(start_epoch, args.epoch):
        train(epoch)
        test(epoch)
        lr_scheduler.step()

    return best_acc


if __name__ == '__main__':
    main()


