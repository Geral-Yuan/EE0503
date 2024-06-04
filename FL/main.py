import argparse
import json
import random

import datasets
from client import *
from server import *

if __name__ == '__main__':

    # 设置命令行程序
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    parser.add_argument('-i', '--id', dest='id', default=0)
    # 获取所有的参数
    args = parser.parse_args()

    # 读取配置文件
    with open(args.conf, 'r') as f:
        conf = json.load(f)
        
    id = args.id
    no_models = conf["no_models"]
    
    if no_models <= 1:
        print("Number of models should be greater than 1")
        exit(0)
    
    if id >= no_models:
        print("ID should be less than the number of models")
        exit(0)

    # 获取数据集, 加载描述信息
    full_train_datasets, fuLL_eval_datasets = datasets.get_dataset("./data/", conf["type"])
    train_datasets = datasets.get_non_iid_subset(full_train_datasets, id)
    eval_datasets = datasets.get_non_iid_subset(fuLL_eval_datasets, id)

    # 开启服务器
    server = Server(conf, eval_datasets)
    # 客户端列表
    clients = []

    # 添加10个客户端到列表
    for c in range(no_models):
        clients.append(Client(conf, server.global_model, train_datasets, c))

    print("\n\n")

    # 全局模型训练
    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)
        # 每次训练都是从clients列表中随机采样k个进行本轮训练
        candidates = random.sample(clients, conf["k"])
        print("select clients is: ")
        for c in candidates:
            print(c.client_id)

        # 权重累计
        weight_accumulator = {}

        # 初始化空模型参数weight_accumulator
        for name, params in server.global_model.state_dict().items():
            # 生成一个和参数矩阵大小相同的0矩阵
            weight_accumulator[name] = torch.zeros_like(params)

        # 遍历客户端，每个客户端本地训练模型
        for c in candidates:
            diff = c.local_train(server.global_model)

            # 根据客户端的参数差值字典更新总体权重
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])

        # 模型参数聚合
        server.model_aggregate(weight_accumulator)

        # 模型评估
        acc, loss = server.model_eval()

        print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
