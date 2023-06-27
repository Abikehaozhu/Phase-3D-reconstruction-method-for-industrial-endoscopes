"""
证明了双频投影的可行性以及普适性
随机生成的三维函数均可进行投影重建
"""

import numpy as np
import matplotlib.pyplot as plt
import find_wrapped_phase_2d
import phase_unwrap_2sin
import random
import other_function

fx = 1 / 12  # 实际条纹图中大概也是20个像素一个周期
fy = 1 / 15
px = 1 / fx  # 条纹之间的间距
py = 1 / fy
wx = 2 * np.pi * fx  # 角频率
wy = 2 * np.pi * fy
d = 15  # 投影仪与相机的距离
L = 80  # 相机到参考平面的距离
dsize = 512  # 图片大小为512

x = np.array([i for i in range(dsize)])  # x坐标
y = np.array([i for i in range(dsize)])  # y坐标
X, Y = np.meshgrid(x, y)  # 生成坐标

origin = (127.5 / 2 * (2 + np.cos(wx * X) + np.cos(wy * Y)))
origin = other_function.gauss_noise(origin, 0, 1)  # 添加高斯噪声
T_table = phase_unwrap_2sin.create_T_table(1 / 12, 1 / 15, 15, 15, 4)
while True:  # 生成包裹相位图
    z = [[0 * dsize] * dsize]  # z的三维
    num_mountain = int(random.uniform(3, 5))  # 山峰的数目
    for i in range(num_mountain):
        sigma = random.uniform(50, 70)  # 高斯函数的方差
        x_delta = random.uniform(100, dsize - 100)  # 高斯函数的偏移量
        y_delta = random.uniform(100, dsize - 100)
        h = random.uniform(30, 40)  # 函数的高度
        z += h * np.exp(-((X - x_delta) ** 2 + (Y - y_delta) ** 2) / (2 * sigma ** 2)) * ((-1) ** i)  # 高斯函数
    img = (127.5 / 2 * (2 + np.cos(wx * (X + z * d / (z - L))) + np.cos(wy * (Y + z * d / (z - L)))))  # 生成的弯曲条纹图
    img = other_function.gauss_noise(img, 0, 1)  # 添加高斯噪声
    plt.subplot(1, 2, 1), plt.imshow(origin, 'gray')
    plt.subplot(1, 2, 2), plt.imshow(img, 'gray')
    plt.show()
    origin_shift, img_shift = find_wrapped_phase_2d.fft(origin, img)  # 傅里叶变换
    origin_shift_x, img_shift_x = find_wrapped_phase_2d.butterworth_filter(origin_shift,
                                                                           img_shift,
                                                                           dsize, 41, 0,
                                                                           11, 15, 5)  # 巴特沃斯滤波
    wrapped_x = find_wrapped_phase_2d.wrapped_phase(origin_shift_x, img_shift_x)  # 求解相位差

    origin_shift_y, img_shift_y = find_wrapped_phase_2d.butterworth_filter(origin_shift,
                                                                           img_shift,
                                                                           dsize, 0, 34,
                                                                           12, 13, 5)  # 巴特沃斯滤波
    wrapped_y = find_wrapped_phase_2d.wrapped_phase(origin_shift_y, img_shift_y)  # 求解相位差

    """展开相位"""
    nx, ny = phase_unwrap_2sin.find_wrap_n(wrapped_x, wrapped_y, T_table, 1 / 12, 1 / 15, 15, 15)
    img_phase_x = wrapped_x + 2 * np.pi * nx
    img_phase_y = wrapped_y + 2 * np.pi * ny
    fig = plt.figure(1)
    plt.subplot(3, 2, 1), plt.imshow(wrapped_x, "gray")
    plt.subplot(3, 2, 2), plt.imshow(wrapped_y, "gray")
    plt.subplot(3, 2, 3), plt.imshow(nx, "gray")
    plt.subplot(3, 2, 4), plt.imshow(ny, "gray")
    plt.subplot(3, 2, 5), plt.imshow(img_phase_x, "gray")
    plt.subplot(3, 2, 6), plt.imshow(img_phase_y, "gray")
    plt.draw()
    plt.show()
    # plt.pause(5)
    # plt.close(fig)

    """还原相位高度"""
    delta_x = img_phase_x / np.pi / 2
    delta_y = img_phase_y / np.pi / 2
    h_x = 80 * delta_x / (delta_x - 15 / 12)  # 不同的相位高度还原方程
    h_y = 80 * delta_y / (delta_y - 15 / 15)
    fig1 = plt.figure(2)  # 定义新的三维坐标轴
    # ax3 = plt.axes(projection='3d')
    # 定义三维数据
    ax = fig1.add_subplot(1, 3, 1, projection='3d')
    ax.plot_surface(X, Y, z, cmap='rainbow')
    ax = fig1.add_subplot(1, 3, 2, projection='3d')
    ax.plot_surface(X, Y, h_x, cmap='rainbow')
    ax = fig1.add_subplot(1, 3, 3, projection='3d')
    ax.plot_surface(X, Y, h_y, cmap='rainbow')
    plt.draw()
    plt.show()

    plt.figure()
    plt.imshow(z - h_x, cmap='rainbow')
    plt.colorbar()
    plt.show()
    # plt.pause(5)
    # plt.close(fig1)
