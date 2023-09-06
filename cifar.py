"""
    PyTorch training code for
    "Paying More Attention to Attention: Improving the Performance of
                Convolutional Neural Networks via Attention Transfer"
    https://arxiv.org/abs/1612.03928
    
    This file includes:
     * CIFAR ResNet and Wide ResNet training code which exactly reproduces
       https://github.com/szagoruyko/wide-residual-networks
     * Activation-based attention transfer
     * Knowledge distillation implementation

    2017 Sergey Zagoruyko
"""

import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import SGD
import torchvision.transforms as T
from torchvision import datasets
import torch.nn.functional as F
import torchnet as tnt
from torchnet.engine import Engine
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import utils

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Wide Residual Networks')      # 封装WRN
# Model options
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--width', default=1, type=float)                       # 默认模型深度为16，宽度为1，在训练教师模型时输入指定参数
parser.add_argument('--dataset', default='CIFAR10', type=str)               # 数据集默认为CIFAR10
parser.add_argument('--dataroot', default='.', type=str)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--nthread', default=2, type=int)                       # 多线程读取数据
parser.add_argument('--teacher_id', default='', type=str)

# Training options
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--epoch_step', default='[60,120,160]', type=str,       # 运行到[60,120,160]轮时减小lr
                    help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)            # lr逐渐减小
parser.add_argument('--resume', default='', type=str)                       # 从模型训练中断处恢复
parser.add_argument('--randomcrop_pad', default=4, type=float)
parser.add_argument('--temperature', default=4, type=float)
parser.add_argument('--alpha', default=0, type=float)       # 训练学生模型时Loss = L*(1-alpha) + alpha*Loss_KD + beta*Loss_AT
parser.add_argument('--beta', default=0, type=float)                        # a和b分别控制知识蒸馏和注意力迁移是否加入损失函数计算

# Device options
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--save', default='', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int,
                    help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')


# 下载数据集并预处理
def create_dataset(opt, train):
    transform = T.Compose([                                   # 对图像依次进行以下操作
        T.ToTensor(),                                         # PIL.Image或ndarray从(H x W x C)形状转换为(N x C x H x W)的tensor
        T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,  # 标准化
                    np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    if train:                                       # 训练时额外进行以下操作
        transform = T.Compose([
            T.Pad(4, padding_mode='reflect'),       # 以反射的方式填充4个元素
            T.RandomHorizontalFlip(),               # 随机水平翻转
            T.RandomCrop(32),                       # 随机剪裁为 32x32
            transform
        ])
    return getattr(datasets, opt.dataset)(opt.dataroot, train=train, download=True, transform=transform)  # 下载数据集并预处理
# 返回torchvision.datasets.CIFAR10(opt.dataroot, train=train, download=True, transform=transform)


# 构建WRN
def resnet(depth, width, num_classes):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'         # 确保模型深度为6n+4，每个残差块6层(BN+relu+cov+BN+relu+cov)
    n = (depth - 4) // 6                                        # 残差块数量
    widths = [int(v * width) for v in (16, 32, 64)]             # 网络每层的filter数

    def gen_block_params(ni, no):                                   # 初始化残差块参数
        return {
            'conv0': utils.conv_params(ni, no, 3),
            'conv1': utils.conv_params(no, no, 3),
            'bn0': utils.bnparams(ni),
            'bn1': utils.bnparams(no),
            'convdim': utils.conv_params(ni, no, 1) if ni != no else None,
        }

    def gen_group_params(ni, no, count):                            # 利用gen_block_params来初始化每组的残差块，每组有count个残差块
        return {'block%d' % i: gen_block_params(ni if i == 0 else no, no)
                for i in range(count)}

    flat_params = utils.cast(utils.flatten({                        # 初始化模型的所有参数，
        'conv0': utils.conv_params(3, 16, 3),
        'group0': gen_group_params(16, widths[0], n),
        'group1': gen_group_params(widths[0], widths[1], n),
        'group2': gen_group_params(widths[1], widths[2], n),
        'bn': utils.bnparams(widths[2]),
        'fc': utils.linear_params(widths[2], num_classes),
    }))

    utils.set_requires_grad_except_bn_(flat_params)                 # 设置非方差和标准差结尾的参数为可求导

    def block(x, params, base, mode, stride):                       # 构建残差块
        o1 = F.relu(utils.batch_norm(x, params, base + '.bn0', mode), inplace=True)     # BN后接relu
        y = F.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)             # 卷积
        o2 = F.relu(utils.batch_norm(y, params, base + '.bn1', mode), inplace=True)     # BN后接relu
        z = F.conv2d(o2, params[base + '.conv1'], stride=1, padding=1)                  # 卷积
        if base + '.convdim' in params:
            return z + F.conv2d(o1, params[base + '.convdim'], stride=stride)           # 残差结构
        else:
            return z + x

    def group(o, params, base, mode, stride):                       # 利用block函数构建残差组
        for i in range(n):
            o = block(o, params, f'{base}.block{i}', mode, stride if i == 0 else 1)
        return o

    def f(input, params, mode, base=''):                            # 构建整个网络 卷积>第一组>第二组>第三组>relu>ap>扁平化>fc
        x = F.conv2d(input, params[f'{base}conv0'], padding=1)
        g0 = group(x, params, f'{base}group0', mode, 1)
        g1 = group(g0, params, f'{base}group1', mode, 2)
        g2 = group(g1, params, f'{base}group2', mode, 2)
        o = F.relu(utils.batch_norm(g2, params, f'{base}bn', mode))
        o = F.avg_pool2d(o, 8, 1, 0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params[f'{base}fc.weight'], params[f'{base}fc.bias'])
        return o, (g0, g1, g2)
    # 返回输出值和各组输出

    return f, flat_params
# 返回模型构建函数f和模型的所有参数flat_params


# 进行模型训练，若训练的是学生模型，则进行KD或AT
def main():
    opt = parser.parse_args()
    print('parsed options:', vars(opt))
    epoch_step = json.loads(opt.epoch_step)
    num_classes = 10 if opt.dataset == 'CIFAR10' else 100

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id             # 选择主显卡

    def create_iterator(mode):              # 下载数据集并进行预处理，设置批大小并将数据加载到GPU上，mode=true为训练集
        return DataLoader(create_dataset(opt, mode), opt.batch_size, shuffle=mode,
                          num_workers=opt.nthread, pin_memory=torch.cuda.is_available())

    train_loader = create_iterator(True)
    test_loader = create_iterator(False)

    # deal with student first
    f_s, params_s = resnet(opt.depth, opt.width, num_classes)                       # f_s存放模型构建函数，params_s存放初始参数

    # deal with teacher
    if opt.teacher_id:                  # 如果构建的是学生模型，则要读取教师模型的相关参数
        with open(os.path.join('logs', opt.teacher_id, 'log.txt'), 'r') as ff:
            line = ff.readline()
            r = line.find('json_stats')
            info = json.loads(line[r + 12:])
        f_t = resnet(info['depth'], info['width'], num_classes)[0]                  # 存放教师模型的构建函数
        model_data = torch.load(os.path.join('logs', opt.teacher_id, 'model.pt7'))
        params_t = model_data['params']                                             # 存放教师模型的参数(已训练)

        # merge teacher and student params
        params = {'student.' + k: v for k, v in params_s.items()}                   # 将学生模型的参数和教师模型的参数放到一起
        for k, v in params_t.items():                                               # 将教师模型的参数设为不可求导
            params['teacher.' + k] = v.detach().requires_grad_(False)

        def f(inputs, params, mode):                                        # at_loss的计算
            y_s, g_s = f_s(inputs, params, mode, 'student.')                    # y_s用于进行KD，计算交叉熵;g_s为每个残差块组的输出，用于计算激活图
            with torch.no_grad():                                               # 教师模型的参数不需要求梯度
                y_t, g_t = f_t(inputs, params, False, 'teacher.')               # 教师模型的输出值和各组残差块的输出
            return y_s, y_t, [utils.at_loss(x, y) for x, y in zip(g_s, g_t)]    # 返回教师和学生模型的输出和两者各组残差块的 at_loss
    else:                               # 如果构建的是教师模型，则将模型构建函数和初始参数赋给f和params
        f, params = f_s, params_s

    def create_optimizer(opt, lr):              # 设置优化器
        print('creating optimizer with lr = ', lr)
        return SGD((v for v in params.values() if v.requires_grad),     # 需要更新的参数
                   lr,
                   momentum=0.9,
                   weight_decay=opt.weight_decay)   # https://blog.csdn.net/weixin_39228381/article/details/108310520

    optimizer = create_optimizer(opt, opt.lr)

    epoch = 0
    if opt.resume != '':        # resume初始为''  用于在重新开始训练时加载模型训练中断处参数
        state_dict = torch.load(opt.resume)
        epoch = state_dict['epoch']
        params_tensors = state_dict['params']
        for k, v in params.items():
            v.data.copy_(params_tensors[k])
        optimizer.load_state_dict(state_dict['optimizer'])      # 加载模型参数

    print('\nParameters:')
    utils.print_tensor_dict(params)

    n_parameters = sum(p.numel() for p in list(params_s.values()))      # 计算模型有多少参数
    print('\nTotal number of parameters:', n_parameters)

    # torchnet.meter 提供计算并记录相关数据的方法
    meter_loss = tnt.meter.AverageValueMeter()                      # 进行均值和方差计算，用于计算平均损失
    classacc = tnt.meter.ClassErrorMeter(accuracy=True)             # 用于计算分类误差
    timer_train = tnt.meter.TimeMeter('s')                          # 用于计算数据处理时间
    timer_test = tnt.meter.TimeMeter('s')
    meters_at = [tnt.meter.AverageValueMeter() for i in range(3)]   # 计算at_loss

    if not os.path.exists(opt.save):        # 模型存储路径
        os.mkdir(opt.save)

    # 计算损失函数
    def h(sample):
        inputs = utils.cast(sample[0], opt.dtype).detach()  # 输入转换为float，并设为不计算梯度
        targets = utils.cast(sample[1], 'long')             # 输出转换为long
        if opt.teacher_id != '':                            # 如果训练的是学生模型
            y_s, y_t, loss_groups = utils.data_parallel(f, inputs, params, sample[2], range(opt.ngpu))  # 当训练学生模型时，f已在main中重构
            loss_groups = [v.sum() for v in loss_groups]                                    # loss_groups为各组at_loss之和
            [m.add(v.item()) for m, v in zip(meters_at, loss_groups)]                       # 在meters_at中记录loss_groups
            return utils.distillation(y_s, y_t, targets, opt.temperature, opt.alpha) + opt.beta * sum(loss_groups), y_s
        else:                                               # 如果训练的是教师模型，则用标准交叉熵
            y = utils.data_parallel(f, inputs, params, sample[2], range(opt.ngpu))[0]
            return F.cross_entropy(y, targets), y
    # 返回损失损失函数和学生模型的输出
    # 如果训练的是教师模型，损失函数为交叉熵；如果训练学生模型，损失函数为L*(1-alpha) + alpha*Loss_KD + beta*Loss_AT

    # 将参数存储在log.txt中，模型存储在model.pt7中
    def log(t, state):
        torch.save(dict(params={k: v.data for k, v in params.items()},
                        optimizer=state['optimizer'].state_dict(),
                        epoch=t['epoch']),
                   os.path.join(opt.save, 'model.pt7'))
        z = vars(opt).copy(); z.update(t)
        logname = os.path.join(opt.save, 'log.txt')
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(z) + '\n')      # .dumps(z)将python对象编码成Json字符串
        print(z)

    # 采样时进行以下操作
    def on_sample(state):
        #print(state)
        state['sample'].append(state['train'])      # 'sample'下的列表中添加state['train']中的内容

    # 前向时进行以下操作
    def on_forward(state):
        classacc.add(state['output'].data, state['sample'][1])      # 计算acc并存入classacc这个meter中
        meter_loss.add(state['loss'].item())                        # 使用.item()取出的数据精度更高

    # 训练开始时进行以下操作
    def on_start(state):
        state['epoch'] = epoch          # epoch初始化为0

    # 新的epoch开始时进行以下操作
    def on_start_epoch(state):
        classacc.reset()                # 重置meter为默认设置nan
        meter_loss.reset()
        timer_train.reset()
        [meter.reset() for meter in meters_at]
        state['iterator'] = tqdm(train_loader)

        epoch = state['epoch'] + 1
        if epoch in epoch_step:             # 运行到[60,120,160]轮时减小lr
            lr = state['optimizer'].param_groups[0]['lr']
            state['optimizer'] = create_optimizer(opt, lr * opt.lr_decay_ratio)

    # 当前epoch结束时进行以下操作
    def on_end_epoch(state):
        train_loss = meter_loss.mean        # 取出各参数后重置meter
        train_acc = classacc.value()
        train_time = timer_train.value()
        meter_loss.reset()
        classacc.reset()
        timer_test.reset()

        engine.test(h, test_loader)

        test_acc = classacc.value()[0]
        print(log({
            "train_loss": train_loss,
            "train_acc": train_acc[0],
            "test_loss": meter_loss.mean,
            "test_acc": test_acc,
            "epoch": state['epoch'],
            "num_classes": num_classes,
            "n_parameters": n_parameters,
            "train_time": train_time,
            "test_time": timer_test.value(),
            "at_losses": [m.value() for m in meters_at],
           }, state))
        print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' % (opt.save, state['epoch'], opt.epochs, test_acc))

    engine = Engine()
    # 执行到以下步骤时，执行自定义的操作
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    # 训练模型
    engine.train(h, train_loader, opt.epochs, optimizer)


if __name__ == '__main__':
    main()
