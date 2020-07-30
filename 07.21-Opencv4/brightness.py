import numpy as np
import cv2

# 전체적인 밝기 조절
def brightness1():
    src = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        return

    dst = cv2.add(src, -150)

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 각각 부분적으로 엑세스하여 밝기를 조절하는것
def brightness2():
    src = cv2.imread('Lenna.bmp', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        return

    dst = np.empty(src.shape, src.dtype)
    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            dst[y, x] = src[y, x] + 100

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

    print('dst.shape:', dst.shape)
    print(dst)

def brightness4():
    src = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        return
    def update(pos):
        dst = cv2.add(src, pos)
        cv2.imshow('dst', dst)
    
    cv2.namedWindow('dst')
    cv2.createTrackbar('Brightness', 'dst', 0, 100, update)
    update(0)

    cv2.waitKey()
    cv2.destroyAllWindows()

def saturated(value):
    if value > 255:
        value = 255
    elif value < 0:
        value = 0
    return value

def brightness3():
    src = cv2.imread('Lenna.bmp', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        return
    
    dst = np.empty(src.shape, dtype=src.dtype)
    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            dst[y, x] = saturated(src[y, x] + 100)

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

def contrast1():
    src = cv2.imread('Lenna.bmp', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        return

    s = 1.5
    dst = cv2.multiply(src, s)

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

def contrast2():
    src = cv2.imread('Lenna.bmp', cv2.IMREAD_GRAYSCALE)
    
    if src is None:
        print('Image load failed!')
        return

    # dst(x,y) = src(x,y) + (src(x,y) - 128)*alpha
    alpha = 5.0
    dst = np.clip(src + (src - 128.)*alpha, 0, 255).astype(np.uint8)

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__== '__main__':
    contrast2()
    