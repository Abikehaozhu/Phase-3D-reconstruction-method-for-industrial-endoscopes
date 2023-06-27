import matplotlib.pyplot as plt
import numpy as np
import time


def create_T_table(fx, fy, dx, dy, n):
    """
    创造一个解包裹的查找表
    :param fx:x频率
    :param fy:y频率
    :param dx:x距离
    :param dy:y距离
    :param n:表的大小
    :return:返回一个查找表
    """
    size = 2 * n + 1
    table = np.array([[0] * size] * size, dtype=np.double)
    for i in range(size):
        for j in range(size):
            table[i][j] = fx * dx / fy / dy * (j - n) - (i - n)
    # print(table)
    return table


def find_wrap_n(img_x, img_y, T_table, fx, fy, dx, dy):
    """
    寻找展开级数
    :param img_x:输入的x方向解相位
    :param img_y:输入的y方向解相位
    :param T_table：输入的复制表
    :param fx:
    :param fy:
    :param dx:
    :param dy:
    :return:展开的矩阵
    """
    start_time = time.perf_counter()
    length = np.shape(img_x)[0]
    n_num = np.shape(T_table)[0]
    theta = (img_x - fx * dx / fy / dy * img_y) / 2 / np.pi
    # plt.figure()
    # plt.imshow(theta, cmap="rainbow")
    # plt.grid(True)
    nx = np.array([[0] * length] * length)
    ny = np.array([[0] * length] * length)
    size_per_run = int(length / 10)
    for i in range(length):  # 开始逐像素寻找
        for j in range(length):
            """对每一个像素寻找展开的k"""
            min_error = np.abs(T_table-theta[i][j])  # 最小值矩阵
            index = int(min_error.argmin())  # 找到最小值的坐标
            x = int(index / n_num)  # 最小值的x坐标
            y = index % n_num  # 最小值的y坐标
            nx[i][j] = x - n_num / 2
            ny[i][j] = y - n_num / 2
        if i % size_per_run == 0:
            print("*", end="")  # 显示解相位进度
    end_time = time.perf_counter()
    print(f"\n解相位完成,花费的时间为:{end_time - start_time}", end="\n")
    return nx, ny
