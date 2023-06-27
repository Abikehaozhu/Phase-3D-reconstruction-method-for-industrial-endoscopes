import numpy as np
import matplotlib.pyplot as plt


def fft(origin, img, show=False):
    """
    对图像进行二维傅里叶变换并展示
    :param origin:原始条纹图
    :param img:弯曲条纹图
    :param show:是否展示
    :return:中心化后的傅里叶
    """
    """对图像进行二维傅里叶变换"""
    fft_img = np.fft.fft2(img)
    fft_origin = np.fft.fft2(origin)

    """展示二维傅里叶变换结果"""
    img_shift = np.fft.fftshift(fft_img)
    origin_shift = np.fft.fftshift(fft_origin)
    if show:
        img_shift_show = np.log(np.abs(img_shift) + 1)
        origin_shift_show = np.log(np.abs(origin_shift) + 1)
        plt.subplot(1, 2, 1), plt.imshow(img_shift_show, "gray")
        plt.subplot(1, 2, 2), plt.imshow(origin_shift_show, "gray")
        plt.show()
    return origin_shift, img_shift


def butterworth_filter(origin, img, dsize, fx, fy, rx, ry, n, show=False):
    """
    对频谱图进行巴特沃斯滤波
    :param origin:原始条纹频谱图
    :param img:弯曲条纹频谱图
    :param dsize:频谱图大小
    :param fx:滤波器x
    :param fy:滤波器y
    :param rx:滤波器x半径
    :param ry:滤波器y半径
    :param n:滤波器阶数
    :param show:是否展示
    :return:滤后的频谱图，去中心化
    """
    """设计滤波器"""
    fx = fx + dsize / 2
    fy = fy + dsize / 2
    xx = np.arange(0, dsize)
    yy = np.arange(0, dsize)
    X, Y = np.meshgrid(xx, yy)
    # hanning = 0.5 + 0.5 * np.cos(2 * np.pi * (np.sqrt(((X - fx) / rx) ** 2 + ((Y - fy) / ry) ** 2)))
    butterworth = 1 / (1 + ((X - fx) / rx) ** (2 * n)) / (1 + ((Y - fy) / ry) ** (n * 2))
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(X, Y, butterworth, cmap='rainbow')
    # plt.show()
    """进行滤波"""
    img_shift = img * butterworth
    origin_shift = origin * butterworth
    if show:
        img_shift_show = np.log(np.abs(img_shift) + 1)
        origin_shift_show = np.log(np.abs(origin_shift) + 1)
        plt.subplot(1, 2, 1), plt.imshow(img_shift_show, "gray")
        plt.subplot(1, 2, 2), plt.imshow(origin_shift_show, "gray")
        plt.show()
    return origin_shift, img_shift


def wrapped_phase(origin, img, show_recover=False):
    """
    寻找包裹相位
    :param origin:滤波后的图像
    :param img:
    :param show_recover:
    :return:包裹相位
    """
    """进行反变换"""
    img_filter = np.fft.ifftshift(img)  # 反中心化
    origin_filter = np.fft.ifftshift(origin)
    img_out = (np.fft.ifft2(img_filter))  # 进行傅里叶逆变换
    origin_out = (np.fft.ifft2(origin_filter))
    if show_recover:
        plt.subplot(1, 2, 1), plt.imshow(np.real(img_out), "gray")
        plt.subplot(1, 2, 2), plt.imshow(np.real(origin_out), "gray")
        plt.show()

    """进行相位求解"""
    origin_out = origin_out.conjugate()  # 求共轭
    final_outcome = origin_out * img_out
    final_outcome = np.log(final_outcome)  # 相乘取对数
    final_outcome = final_outcome.imag  # 相位差为虚部
    return final_outcome


if __name__=='__main__':
    butterworth_filter([], [], 1000, 0, 0, 250, 200, 2, show=False)