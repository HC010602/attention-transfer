import torch


# print(torch.cuda.is_available())


a = torch.Tensor([[1, 2, 3, 0], [4, 5, 6, 0]])
print(a)
print(a.pow(2))
b = a.pow(2).mean(1)
print(b)