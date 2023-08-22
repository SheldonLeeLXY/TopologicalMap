# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from scipy.signal import fftconvolve
from math import pi
import networkx as nx
import matplotlib.pyplot as plt

sum_imageset = 6
channel = "GrayScale"
path = "/Users/lixiangyu/WorkSpace/imageset/camvideo_6/"
imagesets = [path + "junction_6", path + "junction_3", path + "junction_1", path + "junction_7", path + "junction_1", path + "junction_2", path + "junction_1", path + "junction_3", path + "junction_4", path + "junction_3", path + "junction_5", path + "junction_3", path + "junction_6"]
threshold = 0.7


def normxcorr2(imagePath1, imagePath2, channel, mode="same"):
    if channel == "GrayScale":
        image1 = cv2.imread(imagePath1, 0)
        image2 = cv2.imread(imagePath2, 0)
    elif channel == "B":
        image1 = cv2.imread(imagePath1)
        image1 = image1[:, :, 0]
        image2 = cv2.imread(imagePath2)
        image2 = image2[:, :, 0]
    elif channel == "G":
        image1 = cv2.imread(imagePath1)
        image1 = image1[:, :, 1]
        image2 = cv2.imread(imagePath2)
        image2 = image2[:, :, 1]
    elif channel == "R":
        image1 = cv2.imread(imagePath1)
        image1 = image1[:, :, 2]
        image2 = cv2.imread(imagePath2)
        image2 = image2[:, :, 2]

    image_shape = image1.shape
    image1 = image1[0:int(image_shape[0] / 2)]
    image2 = image2[0:int(image_shape[0] / 2)]

    image1 = image1 - np.mean(image1)
    image2 = image2 - np.mean(image2)

    a1 = np.ones(image1.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(image1))
    out = fftconvolve(image2, ar.conj(), mode=mode)

    image2 = fftconvolve(np.square(image2), a1, mode=mode) - \
             np.square(fftconvolve(image2, a1, mode=mode)) / (np.prod(image1.shape))

    # Remove small machine precision errors after subtraction
    image2[np.where(image2 < 0)] = 0

    image1 = np.sum(np.square(image1))
    out = out / np.sqrt(image2 * image1)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    index = np.where(out == np.max(out))
    max_item = [np.max(out), imagePath2, int(index[1] - image_shape[1] / 2)]
    # print(max_item)
    return max_item


def ImageSetConvolution(imgpath1, imgpath2, channel):
    files1 = os.listdir(imgpath1)
    files2 = os.listdir(imgpath2)

    num_imgset1 = len(files1) - 1
    num_imgset2 = len(files2) - 1

    similarity_list = []
    shift_list = []
    sum_shift_list = []

    for i in range(0, num_imgset2):
        print("-------------" + str(i) + "--------------")
        sim_sum = 0
        item_shift_list = []
        for j in range(0, num_imgset1):
            imgFile1 = imgpath1 + "/camera_image" + str(j + 1) + ".jpeg"
            imgFile2 = imgpath2 + "/camera_image" + str((j + i) % num_imgset2 + 1) + ".jpeg"

            max_item = normxcorr2(imgFile1, imgFile2, channel)

            sim = max_item[0]
            sim_sum += sim
            shift = max_item[2]
            item_shift_list.append(shift)

        norm_sim = sim_sum / (len(files2) - 1)
        similarity_list.append(norm_sim)
        shift_list.append(item_shift_list)

    max_sim = max(similarity_list)
    max_index = similarity_list.index(max_sim)

    for i in shift_list:
        sum_shift = 0
        for j in i:
            sum_shift += j
        average_shift = sum_shift / len(i)
        sum_shift_list.append(average_shift)

    rotate_radians = max_index * 2 * pi / sum_imageset + sum_shift_list[max_index] * 1.3962634 / 800

    return_item = [imgpath1, imgpath2, max_sim, max_index, sum_shift_list[max_index], rotate_radians]

    return return_item


def create_topological_map(imagesets, channel):
    nodes = ["S"]
    dataset = [("S", imagesets[0])]
    edges = []
    last_junction = "S"
    count = ord("A")
    weight = 1

    for i in range(1, len(imagesets)):
        print(i)
        max_sim, max_node = recognize_junction(imagesets[i], dataset)

        if max_sim == 0:
            junction = chr(count)
            nodes.append(junction)
            new_edge = (last_junction, junction, weight)
            exist = check_exist_edge(new_edge, edges)
            if not exist:
                edges.append(new_edge)
                weight += 1  # 边的权重递增
            last_junction = junction
            dataset.append((junction, imagesets[i]))
            count += 1
        else:
            node = max_node
            new_edge = (last_junction, node, weight)
            exist = check_exist_edge(new_edge, edges)
            if not exist:
                edges.append(new_edge)
                weight += 1  # 边的权重递增
            last_junction = node

    # 使用函数创建拓扑地图
    print("Nodes:", nodes)
    print("Edges:", edges)
    print("Dataset:", dataset)

    # 创建一个无向图
    G = nx.Graph()

    # 添加节点
    G.add_nodes_from(nodes)

    # 添加边，并指定边的权重
    G.add_weighted_edges_from(edges)

    # 绘制图
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, with_labels=True, node_size=1000, node_color="skyblue", font_size=10, font_color="black")

    # 获取边的权重字典
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Topology Map with Edge Weights")
    plt.show()


def recognize_junction(current_imageset, dataset):
    max_sim = 0
    max_node = None
    for node, imageset in dataset:
        similarity = ImageSetConvolution(current_imageset, imageset, channel)[2]
        if similarity > threshold and similarity > max_sim:
            max_sim = similarity
            max_node = node
    return max_sim, max_node


def check_exist_edge(new_edge, edges):
    # 检查是否已经存在相同的边
    for edge in edges:
        print(edge)
        A, B, _ = edge
        if (A == new_edge[0] and B == new_edge[1]) or (B == new_edge[0] and A == new_edge[1]):
            return True
    return False


if __name__ == '__main__':
    create_topological_map(imagesets, channel)
