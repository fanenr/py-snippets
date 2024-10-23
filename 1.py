import cv2
import numpy as np


def show_img(win, img):
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, img)


img = cv2.imread("./1/origin.jpeg")

# 显示图片
show_img("origin", img)
# 保存图片
cv2.imwrite("./1/copy.jpeg", img)


# 创建图象
img2 = np.zeros(img.shape, np.uint8)
# 复制图象
img2 = img.copy()

# 分离通道
channels = cv2.split(img)
names = ["b", "g", "r", "a"]
for i, ch in enumerate(channels):
    show_img(names[i], ch)

# 合并通道
bgr_img = cv2.merge((channels[0], channels[1], channels[2]))
show_img("merged-bgr", bgr_img)

# 调整顺序
rgb_img = cv2.merge((channels[2], channels[1], channels[0]))
show_img("merged-rgb", rgb_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
