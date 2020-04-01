from scipy import misc, ndimage
import matplotlib.pyplot as plt
import numpy as np


def ndimage_demo():
    f = misc.face()
    plt.imsave('face.png', f)  # uses the matplotlib module (plt)

    plt.imshow(f)
    plt.show()

    face = misc.face(gray=False)
    print(face.mean(), face.max(), face.min())

    # 图像执行一些几何变换，裁剪
    print(face.shape)
    lx, ly = face.shape[0:2]
    crop_face = face[int(lx / 4): -int(lx / 4), int(ly / 4): -int(ly / 4)]
    plt.imshow(crop_face)
    plt.show()

    # 倒置图像
    flip_ud_face = np.flipud(face)
    plt.imshow(flip_ud_face)
    plt.show()

    # 以指定的角度旋转图像
    rotate_face = ndimage.rotate(face, 45)
    plt.imshow(rotate_face)
    plt.show()

    # 滤镜-模糊
    blurred_face = ndimage.gaussian_filter(face, sigma=10)
    plt.imshow(blurred_face)
    plt.show()

    # 边缘检测
    # 图像看起来像一个方块的颜色
    im = np.zeros((256, 256))
    im[64:-64, 64:-64] = 1
    im[90:-90, 90:-90] = 2
    im = ndimage.gaussian_filter(im, 8)

    plt.imshow(im)
    plt.show()

    # 现在，检测这些彩色块的边缘
    sx = ndimage.sobel(im, axis=0, mode='constant')
    sy = ndimage.sobel(im, axis=1, mode='constant')
    sob = np.hypot(sx, sy)

    plt.imshow(sob)
    plt.show()


if __name__ == '__main__':
    ndimage_demo()
