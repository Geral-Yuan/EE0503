import argparse
import json
import socket
import numpy as np
import pickle
import struct
import threading

import datasets
from local_trainer import *
from model import *
from partition_diff import *

def send_data(sock, data):
    try:
        data = pickle.dumps(data)
        data_size = struct.pack('>I', len(data))  # 4 字节表示数据大小
        sock.sendall(data_size)
        # 分块发送数据
        for i in range(0, len(data), 4096):
            sock.sendall(data[i:i + 4096])
    except (socket.error, pickle.PicklingError) as e:
        print(f"Send error: {e}")
        # 根据需要添加重试或记录日志的逻辑

def receive_data(sock):
    try:
        data_size = struct.unpack('>I', sock.recv(4))[0]  # 读取 4 字节表示的数据大小
        recv_data = b''
        while len(recv_data) < data_size:
            packet = sock.recv(min(4096, data_size - len(recv_data)))
            if not packet:
                break
            recv_data += packet
        if len(recv_data) != data_size:
            raise ValueError(f"Expected {data_size} bytes, but received {len(recv_data)} bytes")
        return pickle.loads(recv_data)
    except (socket.error, pickle.UnpicklingError, ValueError) as e:
        print(f"Receive error: {e}")
        # 根据需要添加重试或记录日志的逻辑

# 用来发送数据的线程
def send_data_thread(sock, data):
    send_data(sock, data)
    print(f"Sent data size: {len(pickle.dumps(data))}")

if __name__ == '__main__':

    # 设置命令行程序
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    parser.add_argument('-i', '--id', dest='id', default=0, type=int)
    parser.add_argument("-p", "--port", dest="port", default=10000, type=int)
    # 获取所有的参数
    args = parser.parse_args()

    # 读取配置文件
    with open(args.conf, 'r') as f:
        conf = json.load(f)
    
    # ip_addr_list = ["192.168.137.14", "192.168.137.148", "192.168.137.210", "192.168.137.202"]
    
    id = args.id
    local_port = args.port
    no_models = conf["no_models"]
    
    if no_models <= 1:
        print("Number of models should be greater than 1")
        exit(0)
    
    if id >= no_models:
        print("ID should be less than the number of models")
        exit(0)
        
    # 创建本地服务器
    local_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ip_addr = "localhost"
    # ip_addr = ip_addr_list[id]
    local_socket.bind((ip_addr, local_port))
    local_socket.listen(5)
    print("Listening on port", local_port)
    
    local_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket = None
    
    server_addr = "localhost"
    # server_addr = ip_addr_list[(id+1)%no_models]
    server_port = int(input("Please input the server port: "))
    # server_port = local_port
    # 连接server结点同时接受client结点的连接请求
    if id != 0:
        # 连接server结点
        local_client_socket.connect((server_addr, server_port))
        
        # 接受client结点的连接请求
        client_socket, addr = local_socket.accept()
        print("Got a connection from %s" % str(addr))
    else:
        # 接受client结点的连接请求
        client_socket, addr = local_socket.accept()
        print("Got a connection from %s" % str(addr))
        
        input("Press Enter to continue...")
        
        # 连接server结点
        local_client_socket.connect((server_addr, server_port))
    

    # 获取数据集, 加载描述信息
    full_train_datasets, fuLL_eval_datasets = datasets.get_dataset("./data/", conf["type"])
    train_datasets, train_ratio = datasets.get_non_iid_subset(full_train_datasets, id)
    eval_datasets, eval_ratio = datasets.get_non_iid_subset(fuLL_eval_datasets, id)
    
    eval_loader = torch.utils.data.DataLoader(
        eval_datasets,
        # 设置单个批次大小
        batch_size=conf["batch_size"],
        # 打乱数据集
        shuffle=True
    )
    
    # 创建本地模型
    local_model = MNISTNet()
    
    for e in range(conf["global_epochs"]):
        # 拷贝本地模型进行本地训练得到参数差值
        diff = Local_Trainer(conf, local_model, train_datasets, id).local_train(local_model)
        flat_diff, shape_info = flatten_diff(diff)
        flat_diff *= train_ratio
        param_diff_blocks = split_flat_diff(flat_diff, no_models)
        
        for j in range(no_models-1):
            print(f"Round {j} of sending and receiving data ...")
            
            # 子线程发送参数差值给服务器节点
            data_to_send = param_diff_blocks[(id+no_models-j)%no_models]
            send_thread = threading.Thread(target=send_data_thread, args=(local_client_socket, data_to_send))
            send_thread.start()
            
            # 接收客户端节点传来的参数差值
            recv_np_array = receive_data(client_socket)
            print(f"Received data size: {len(pickle.dumps(recv_np_array))}")
            
            # 等待子线程结束
            send_thread.join()
            
            # 聚合参数差值
            param_diff_blocks[(id+no_models-j-1)%no_models] += recv_np_array
        
        for j in range(no_models-1):
            print(f"Round {j} of broadcasting data ...")
            
            # 子线程发送聚合后的参数差值给server结点
            data_to_send = param_diff_blocks[(id+no_models-j+1)%no_models]
            send_thread = threading.Thread(target=send_data_thread, args=(local_client_socket, data_to_send))
            send_thread.start()
            
            # 接收client结点传来的聚合后的参数差值
            recv_np_array = receive_data(client_socket)
            print(f"Received data size: {len(pickle.dumps(recv_np_array))}")
            
            # 等待子线程结束
            send_thread.join()
            
            # 更新为聚合后的参数差值
            param_diff_blocks[(id+no_models-j)%no_models] = np.copy(recv_np_array)
            
        global_flat_diff = np.concatenate(param_diff_blocks)
        global_diff = unflatten_diff(global_flat_diff, shape_info)
        
        for name, param in global_diff.items():
            global_diff[name] = param
        
        # 更新模型参数
        for name, data in local_model.state_dict().items():
            local_model.state_dict()[name].add_(global_diff[name])
        
        # 模型评估
        local_model.eval()  # 开启模型评估模式（不修改参数）
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        # 遍历评估数据集合
        for batch_id, batch in enumerate(eval_loader):
            data, target = batch
            # 获取所有的样本总量大小
            dataset_size += data.size()[0]
            # 存储到gpu
            # if torch.cuda.is_available():
            #     data = data.cuda()
            #     target = target.cuda()
            # 加载到模型中训练
            output = local_model(data)
            # 聚合所有的损失 cross_entropy交叉熵函数计算损失
            total_loss += torch.nn.functional.cross_entropy(
                output,
                target,
                reduction='sum'
            ).item()
            # 获取最大的对数概率的索引值， 即在所有预测结果中选择可能性最大的作为最终的分类结果
            pred = output.data.max(1)[1]
            # 统计预测结果与真实标签target的匹配总个数
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        acc = 100.0 * (float(correct) / float(dataset_size))  # 计算准确率
        total_l = total_loss / dataset_size  # 计算损失值

        print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, total_loss))
        
    local_socket.close()
    local_client_socket.close()
