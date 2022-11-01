
from torch import nn
from torch.utils import data
import copy

import numpy as np





class FedHook:
    def __init__(self, net):
        self.model = net
        self.grad_backward_in = []
        self.grad_backward_out = []
        self.feature_forward_in = []
        self.feature_forward_out = []
        self.modules = list(net.named_children())
        for name, module in self.modules:
            if isinstance(module, nn.Linear):
                # self.hook_forward = module.register_forward_hook(self.forward_hook)  # 钩子 1：前向输出的特征图
                self.hook_backward = module.register_backward_hook(
                    self.backward_hook)  # 钩子 2：反向传播的梯度 #考虑到全连接层对分类精度的重要性，注册hook到三个全连接层

    def backward_hook(self, module, gin, gout):
        # print(module)
        # print(gout)
        # print(gin)
        gin1 = gin[0].detach()  #
        gout1 = gout[0].detach()  # bug，不能直接存储梯度
        self.grad_backward_out.append(gout1)
        self.grad_backward_in.append(gin1)  # FedRep因为前几轮只有头部层，即最后一层训练。因此上面几层都没有梯度，但是有特征图。
        # 在卷积层，对 bias 的梯度为整个 batch 的数据在 bias 上的梯度之和：grad_input = (对feature的导数，对权重 W 的导数，对 bias 的导数)
        # 在全连接层，对 bias 的梯度是分开的，bach 中每条数据，对应一个 bias 的梯度：grad_input = ((data1 对 bias 的导数，data2 对 bias 的导数 ...)，对 feature 的导数，对 W 的导数)

    def forward_hook(self, module, fin, fout):
        self.feature_forward_in.append(fin)  # 输入为x
        self.feature_forward_out.append(fout)

    def remove_hook(self):
        # self.hook_forward.remove()
        self.hook_backward.remove()

        # # 获取 module，这里只针对 alexnet，如果是别的，则需修改
        # modules = list(self.model.named_children())
        # # 遍历所有 module, 注册 forward hook 和 backward hook
        # for name, module in modules:

        # # 对第1层卷积层注册 hook
        # first_layer = modules[0][1]
        # first_layer.register_backward_hook(first_layer_hook_fn)
        # self.model.fc3.register_forward_hook(forward_hook)
        # self.model.fc3.register_backward_hook(backward_hook)

    def grad_bp(self, input, target_class, args, loss_fun, optimizer):
        model = self.model
        labels = target_class
        output = model(input)  # 20张图像,10类 20*10
        loss = loss_fun(output, labels)
        # optimizer.zero_grad()
        loss.backward()  # backward之后反向传播才有梯度
        optimizer.step()
        self.remove_hook()
        # torch.cuda.empty_cache()
        # print(hash(FedHook))
        return loss, self.grad_backward_out, self.grad_backward_in


#     def visualize(self, input_image, target_class):
#         # 获取输出，之前注册的 forward hook 开始起作用
#         model_output = self.model(input_image)
#         self.model.zero_grad()
#         pred_class = model_output.argmax().item()
#
#         # 生成目标类 one-hot 向量，作为反向传播的起点
#         grad_target_map = torch.zeros(model_output.shape,
#                                       dtype=torch.float)
#         if target_class is not None:
#             grad_target_map[0][target_class] = 1
#         else:
#             grad_target_map[0][pred_class] = 1
#
#         # 反向传播，之前注册的 backward hook 开始起作用
#         model_output.backward(grad_target_map)
#         # 得到 target class 对输入图片的梯度，转换成图片格式
#         result = self.image_reconstruction.data[0].permute(1, 2, 0)
#         return result.numpy()
#
#
# def normalize(I):
#     # 归一化梯度map，先归一化到 mean=0 std=1
#     norm = (I - I.mean()) / I.std()
#     # 把 std 重置为 0.1，让梯度map中的数值尽可能接近 0
#     norm = norm * 0.1
#     # 均值加 0.5，保证大部分的梯度值为正
#     norm = norm + 0.5
#     # 把 0，1 以外的梯度值分别设置为 0 和 1
#     norm = norm.clip(0, 1)
#     return norm











