import numpy as np
import scipy.signal


def write_height(file_path, h):
    """
    将高度信息写入txt文件中
    :param file_path:保存的路径
    :param h:高度数组
    :return:
    """
    dsize = np.size(h[0])
    with open(file_path, 'w') as file_object:
        for i in range(dsize):
            # file_object.write("[")
            for j in range(dsize):
                file_object.write(str(h[i][j]))
                if j != dsize - 1:
                    file_object.write(" ")
            file_object.write("\n")


def gauss_noise(img, mean=0, sigma=25):
    image = np.array(img / 255, dtype=float)  # 将原始图像的像素值进行归一化
    # 创建一个均值为mean，方差为sigma，呈高斯分布的图像矩阵
    noise = np.random.normal(mean, sigma / 255.0, image.shape)
    out = image + noise  # 将噪声和原始图像进行相加得到加噪后的图像
    res_img = np.clip(out, 0.0, 1.0)
    res_img = np.uint8(res_img * 255.0)

    return res_img


def mesh_height(height_x, height_y):
    """
    将两个高度图融合在一起，以加权平均的方式融合，权重取决于高度的平滑程度，
    以一阶梯度作为衡量
    :param height_x: 输入的height_x
    :param height_y: 输入的height_y
    :return: 输出融合过后的图像
    """
    dsize = np.size(height_y[0])
    kernel = np.array([[1, 1, 1],
                       [1, -8, 1],
                       [1, 1, 1]])  # 拉普拉斯算子
    grad_x = scipy.signal.convolve2d(height_x, kernel)  # 求梯度
    grad_y = scipy.signal.convolve2d(height_y, kernel)
    va_x = (np.sqrt(grad_x * grad_x) / dsize / dsize).sum()  # 求均方差
    va_y = (np.sqrt(grad_y * grad_y) / dsize / dsize).sum()
    height = (va_y / (va_x + va_y)) * height_x + (va_x / (va_x + va_y)) * height_y
    return height
