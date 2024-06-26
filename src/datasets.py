import torchvision as tv
from torch.utils.data import Subset

# 获取数据集
def get_dataset(dir, name):
    global train_dataset, eval_dataset
    if name == 'mnist':
        # root: 数据路径
        # train参数表示是否是训练集或者测试集
        # download=true表示从互联网上下载数据集并把数据集放在root路径中
        # transform：图像类型的转换
        train_dataset = tv.datasets.MNIST(dir, train=True, download=True, transform=tv.transforms.ToTensor())
        eval_dataset = tv.datasets.MNIST(dir, train=False, transform=tv.transforms.ToTensor())
    elif name == 'cifar':
        # 设置两个转换格式
        # transforms.Compose 是将多个transform组合起来使用（由transform构成的列表）
        transform_train = tv.transforms.Compose([
            # transforms.RandomCrop： 切割中心点的位置随机选取
            tv.transforms.RandomCrop(32, padding=4), tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            # transforms.Normalize： 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化
            tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = tv.datasets.CIFAR10(dir, train=True, download=True, transform=transform_train)
        eval_dataset = tv.datasets.CIFAR10(dir, train=False, transform=transform_test)
    return train_dataset, eval_dataset

def get_non_iid_subset(dataset, id):
    label_partition = [[0,1,2],[3,4],[5,6,7],[8,9]]
    indices = [i for i, (_, label) in enumerate(dataset) if label in label_partition[id%4]]
    subdataset = Subset(dataset, indices)
    return subdataset, len(subdataset)/len(dataset)