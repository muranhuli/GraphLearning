import yaml
import os
import torch
import torch.nn as nn
import numpy as np
from numpy import ndarray
import networkx as nx
import matplotlib.pyplot as plt
import gc


def load_config(file_path: str) -> dict:
    """
    load the settings and return a dictionary
    ------
    Parameters:
        file_path: path to the configuration file
    Returns:
        dict: the settings dictionary
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found!")
    with open(file_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, yaml.SafeLoader)
    return config


def recursive_print_cfg(config: dict, key: str, level=0):
    if isinstance(config, dict):
        print("    "*level+key)
        for key in config.keys():
            # print(key+":")
            recursive_print_cfg(config[key], key, level=level+1)
    else:
        print("    "*level+f"[{key}]".ljust(25), "->", config)


def create_folder(folder_path: str) -> None:
    """
    Create the folder if not exisits
    ------
    parameters:
        folder_path: path to the folder
    Returns:
        None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
def flatten_params(model):
    """将模型参数平坦化为一个向量"""
    return torch.cat([param.data.view(-1) for param in model.parameters()])

def unflatten_params(model, flat_params):
    """将平坦化的参数重新整形为模型的形状"""
    offset = 0
    for param in model.parameters():
        param_length = torch.prod(torch.tensor(param.size()))
        param.data = flat_params[offset:offset + param_length].view(param.size())
        offset += param_length
        
def get_hessian(loss,model):
    params = list(model)
    first_grad = torch.autograd.grad(loss, params, create_graph=True)
    second_grad=[]
    for i in first_grad:
        #首先求出每个tensor中所含参数的个数 
        x=i.view(-1)
        #逐个对model.parameters()进行求导
        for i in range(len(x)):
            grad_= torch.autograd.grad(x[i],params,retain_graph=True)
            second_grad.append(grad_)
    second_grad = [torch.cat([j.view(-1) for j in i]) for i in second_grad]
    second_grad = torch.stack(second_grad)
    # second_grad = torch.transpose(second_grad, 0, 1)
    first_grad = torch.cat([i.view(-1) for i in first_grad]).view(-1,1)
    
        # 释放CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # print(first_grad)
    # print(second_grad)
    # # 逆矩阵
    # identity_matrix = (torch.eye(second_grad.shape[0])*0.000001).cuda()
    
    
    # inverse_matrix = torch.inverse(second_grad+identity_matrix)
    # print(inverse_matrix)
    
    return first_grad, second_grad

def gen_topology_picture(weight_matrix,path):
    G = nx.Graph()

    # 添加节点
    num_nodes = len(weight_matrix)
    G.add_nodes_from(range(num_nodes))

    # 添加带权重的边
    for i in range(num_nodes):
        for j in range(i+1,num_nodes):
            if weight_matrix[i][j] != 0:
                G.add_edge(i, j, weight=weight_matrix[i][j])

    # 绘制图形
    pos = nx.circular_layout(G)  # 设置节点位置，这里使用了圆形布局
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]  # 将权重放大，以便更好地显示

    nx.draw(G, pos, with_labels=True, node_size=100, node_color='skyblue', font_size=7, font_weight='normal')
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, alpha=0.7, edge_color='gray')

    # 显示权重标签
    labels = nx.get_edge_attributes(G, 'weight')
    formatted_labels = {(u, v): f"{labels[(u, v)]:.3f}" for u, v in edges}

    nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_labels, font_size=8)

    # 显示图形
    plt.savefig(os.path.join(path, 'weight_topology.png'), dpi=300)