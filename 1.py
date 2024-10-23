import cv2
import numpy as np


def show_img(win, img):
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, img)
    cv2.waitKey(0)


img = cv2.imread("./1/origin.jpeg")

# 显示图片
show_img("origin", img)
# 保存图片
cv2.imwrite("./1/copy.jpeg", img)


# 创建图象
img2 = np.zeros(img.shape, np.uint8)
# 复制图象
img2 = img.copy()


# 通道分离
if len(img.shape) == 3:
    # 3 通道
    if img.shape[2] == 3:
        b, g, r = cv2.split(img)
        show_img("b", b)
        show_img("g", g)
        show_img("r", r)
    # 4 通道
    elif img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        show_img("b", b)
        show_img("g", g)
        show_img("r", r)
        show_img("a", a)

# 通道合并
b = img[:, :, 0]
g = img[:, :, 1]
r = img[:, :, 2]

img2 = cv2.merge((b, g, r))
show_img("merged-bgr", img2)

# 调换通道顺序
img2 = cv2.merge((r, g, b))
show_img("merged-rgb", img2)

cv2.destroyAllWindows()
