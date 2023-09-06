from nested_dict import nested_dict
from functools import partial
import torch
from torch.nn.init import kaiming_normal_
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
import torch.nn.functional as F


# 知识蒸馏
def distillation(y, teacher_scores, labels, T, alpha):
    p = F.log_softmax(y/T, dim=1)                                       # 学生模型soft targets
    q = F.softmax(teacher_scores/T, dim=1)                              # 教师模型soft targets
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / y.shape[0]     # kl散度(相对熵)Loss_KD(相对熵=交叉熵-信息熵，论文中写的是S和T的交叉熵)
    l_ce = F.cross_entropy(y, labels)                                   # 学生soft targets和hard targets的交叉熵
    return l_kl * alpha + l_ce * (1. - alpha)


def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
# .size(0)求张量第一维度大小  .mean(1)对张量第二维度求均值  .view(x.size(0),-1)相当于将计算好的注意力图各元素值排为1列
# 利用论文中的第二种计算方法求注意力图，先对各激活值作平方操作，再对张量在通道维度上求平均（论文中是求和）得到(B, H ,W)的张量，并将注意力图排为一列(B, H*W)
# 论文中提到对注意力图归一化后再计算损失能使学生模型更好地训练


def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()
# 求at_loss，是一批数据中各份at_loss的平均值


# 转换参数类型为dtype类型
def cast(params, dtype='float'):
    if isinstance(params, dict):                                                # 如果参数是字典形式
        return {k: cast(v, dtype) for k,v in params.items()}
    else:
        return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()


def conv_params(ni, no, k=1):
    return kaiming_normal_(torch.Tensor(no, ni, k, k))


def linear_params(ni, no):
    return {'weight': kaiming_normal_(torch.Tensor(no, ni)), 'bias': torch.zeros(no)}


def bnparams(n):                                    # BN初始化
    return {'weight': torch.rand(n),                # 权重
            'bias': torch.zeros(n),                 # 偏置
            'running_mean': torch.zeros(n),         # 均值
            'running_var': torch.ones(n)}           # 方差


# 多线程处理数据，设置nthread为1则直接返回f(input, params, mode)
def data_parallel(f, input, params, mode, device_ids, output_device=None):
    device_ids = list(device_ids)
    if output_device is None:
        output_device = device_ids[0]

    if len(device_ids) == 1:
        return f(input, params, mode)

    params_all = Broadcast.apply(device_ids, *params.values())
    params_replicas = [{k: params_all[i + j*len(params)] for i, k in enumerate(params.keys())}
                       for j in range(len(device_ids))]

    replicas = [partial(f, params=p, mode=mode)
                for p in params_replicas]
    inputs = scatter([input], device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)


def flatten(params):
    return {'.'.join(k): v for k, v in nested_dict(params).items_flat() if v is not None}


def batch_norm(x, params, base, mode):
    return F.batch_norm(x, weight=params[base + '.weight'],
                        bias=params[base + '.bias'],
                        running_mean=params[base + '.running_mean'],
                        running_var=params[base + '.running_var'],
                        training=mode)


def print_tensor_dict(params):
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.shape)).ljust(23), torch.typename(v), v.requires_grad)


def set_requires_grad_except_bn_(params):
    for k, v in params.items():
        if not k.endswith('running_mean') and not k.endswith('running_var'):
            v.requires_grad = True
