import numpy as np
import cv2
import sys

def func1():
    img1 = cv2.imread('cat.jpg', flags=cv2.IMREAD_GRAYSCALE)

    if img1 is None:
        print('Image load failed')

    print('type(img1):', type(img1))
    print('img1.shape:', img1.shape)

    if len(img1.shape) == 2:
        print('img1 is a grayscale image')
    elif len(img1.shape) == 3:
        print('img1 is a truecolor image')
    cv2.imshow('img1', img1)
    cv2.waitKey()
    cv2.destroyAllWindows()

def func2():
    img1 = np.empty((480, 640), np.uint8) # grayscale imag
    img2 = np.zeros((480, 640, 3), np.uint8)
    img3 = np.ones((480, 640), np.uint8)
    img4 = np.full((480, 640), 0, np.float32)

    mat1 = np.array([[11, 12, 13, 14],
                     [21, 22, 23, 24],
                     [31, 32, 33, 34]]).astype(np.uint8)

    mat1[0, 1] = 100  # element at x=1, y=8
    mat1[2, :] = 200

    print(mat1)
    cv2.imshow('img1', img1)
    cv2.waitKey()
    cv2.imshow('img2', img2)
    cv2.waitKey()
    cv2.imshow('img3', img3)
    cv2.waitKey()
    cv2.imshow('img4', img4)
    cv2.waitKey()
    cv2.destroyAllWindows()

def func3():
    img1 = cv2.imread('cat2.bmp')

    img2 = img1
    img3 = img1.copy()

    img1[:, :] = (0, 255, 155) # yellow

    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.imshow('img3', img3)
    cv2.waitKey()
    cv2.destroyAllWindows

    # print(img1)
    # print(img2)
    # print(img3)
    # print(img4)

def func4():
    img1 = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)

    img2 = img1[250:400, 200:400]
    img3 = img1[200:400, 200:400].copy()

    img2 += 20

    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.imshow('img3', img3)
    cv2.waitKey()
    cv2.destroyAllWindows()

def func5():
    mat1 = np.array(np.arange(12)).reshape(3, 4)

    print('mat1:')
    print(mat1)

    h, w = mat1.shape[:2]

    mat2 = np.zeros(mat1.shape, type(mat1))

    for j in range(h):
        for i in range(w):
            mat2[j, i] = mat1[j, i] + 10

    print('mat2:')
    print(mat2)

def func6():
    mat1 = np.ones((3, 4), np.int32) # 1's matrix
    mat2 = np.arange(12).reshape(3, 4)
    mat3 = mat1 + mat2
    mat4 = mat2 + 2

    print("mat1:", mat1, sep='\n')
    print("mat2:", mat2, sep='\n')
    print("mat3:", mat3, sep='\n')
    print("mat4:", mat4, sep='\n')

if __name__== '__main__':
    func6()


# img = cv2.imread('cat.jpg', flags=cv2.IMREAD_GRAYSCALE)
# cv2.imwrite('cat1.jpg', img, params=[cv2.IMWRITE_PAM_FORMAT_RGB, 10])

# if img is None:
#     print('Image load failed!')
#     sys.exit()

# cv2.namedWindow('image')
# cv2.imshow('image', img)
# cv2.waitKey()

# cv2.destroyAllWindows()