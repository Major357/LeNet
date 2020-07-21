# -*- coding: utf-8 -*-

# ============================ inference ============================
import os
from tools.my_dataset import RMBDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
# 导入模型
path_model="../saved_models/model.pkl"
net_loaded=torch.load(path_model)

# 均值和方差
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


# 数据预处理
test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std), # 标准化
])

# 测试数据文件夹
split_dir = os.path.join("..", "data", "rmb_split")
test_dir = os.path.join(split_dir, "test")


# 构建MyDataset实例
test_data = RMBDataset(data_dir=test_dir, transform=test_transform)
# 构建DataLoder
test_loader = DataLoader(dataset=test_data, batch_size=1)

for i, data in enumerate(test_loader):
    # forward
    inputs, labels = data
    outputs = net_loaded(inputs)
    _, predicted = torch.max(outputs.data, 1)

    rmb = 1 if predicted.numpy()[0] == 0 else 100
    print("模型获得{}元".format(rmb))