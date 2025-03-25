import cv2


def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)


def apply_histogram_equalization(image):
    if len(image.shape) == 3:
        img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    else:
        return cv2.equalizeHist(image)
