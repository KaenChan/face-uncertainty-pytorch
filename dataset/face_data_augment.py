import cv2
from time import time
import numpy as np
import os
from PIL import Image, ImageFilter
from PIL import Image, ImageOps, ImageEnhance
from scipy import misc

def get_new_shape(images, size=None, n=None):
    shape = list(images.shape)
    if size is not None:
        h, w = tuple(size)
        shape[1] = h
        shape[2] = w
    if n is not None:
        shape[0] = n
    shape = tuple(shape)
    return shape

def random_crop_fix(image, size):
    images = np.array([image])
    n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    shape_new = get_new_shape(images, size)
    assert (_h>=h and _w>=w)

    images_new = np.ndarray(shape_new, dtype=images.dtype)

    y = np.random.randint(low=0, high=_h-h+1, size=(n))
    x = np.random.randint(low=0, high=_w-w+1, size=(n))

    for i in range(n):
        images_new[i] = images[i, y[i]:y[i]+h, x[i]:x[i]+w]

    return images_new[0]

def center_crop_fix_array(images, size):
    n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    assert (_h>=h and _w>=w)

    y = int(round(0.5 * (_h - h)))
    x = int(round(0.5 * (_w - w)))

    images_new = images[:, y:y+h, x:x+w]

    return images_new

def center_crop_fix(image, size):
    images = np.array([image])
    return center_crop_fix_array(images, size)[0]

# 细节增强
def detail_enhance(current_img, value=None):
    if value is None:
        value = np.random.random()*5 + 1
    value = value * 0.05
    return cv2.detailEnhance(current_img, sigma_s=50, sigma_r=value)

# 边缘保持
def edge_preserve(current_img, value=None):
    if value is None:
        value = np.random.random()*4 + 1
    value = value * 0.05
    return cv2.edgePreservingFilter(current_img, flags=1, sigma_s=50, sigma_r=value)

# 饱和度
def change_saturation(current_img, value=None):
    if value is None:
        value = -30 + np.random.random()*60
    img_hsv = cv2.cvtColor(current_img, cv2.COLOR_BGR2HLS)
    if value > 2:
        img_hsv[:, :, 2] = np.log(img_hsv[:, :, 2] /255* (value - 1)+1) / np.log(value + 1) * 255
    if value < 0:
        img_hsv[:, :, 2] = np.uint8(img_hsv[:, :, 2] / np.log(- value + np.e))
    return cv2.cvtColor(img_hsv, cv2.COLOR_HLS2BGR)

# 明度调节
def change_darker(current_img, value=None):
    if value is None:
        value = -13 + np.random.random()*40
    img_hsv = cv2.cvtColor(current_img, cv2.COLOR_BGR2HLS)
    if value > 3:
        img_hsv[:, :, 1] = np.log(img_hsv[:, :, 1] /255* (value - 1)+1) / np.log(value + 1) * 255
    if value < 0:
        img_hsv[:, :, 1] = np.uint8(img_hsv[:, :, 1] / np.log(- value + np.e))
    return cv2.cvtColor(img_hsv, cv2.COLOR_HLS2BGR)


def crop_image(img, x0, y0, w, h):
    """
    定义裁剪函数
    :param img: 要处理的图片
    :param x0: 左上角横坐标
    :param y0: 左上角纵坐标
    :param w: 裁剪宽度
    :param h: 裁剪高度
    :return: 裁剪后的图片
    """
    return img[x0:x0+w, y0:y0+h]

def random_pad(img, area_ratio, hw_vari):
    """
    定义随机pad函数
    :param img: 要处理的图片
    :param area_ratio: pad画面占原图片的比例
    :param hw_vari: 扰动占原宽高的比例
    :return: 裁剪后的图片
    """
    w, h = img.shape[:2]
    hw_delta = np.random.uniform(-hw_vari, hw_vari)
    hw_mult = 1 + hw_delta
    color = int(np.random.random()*200)
    w_pad = int(np.round(w * np.sqrt(area_ratio * hw_mult)))
    h_pad = int(np.round(h * np.sqrt(area_ratio / hw_mult)))
    top, bottom, left, right = h_pad, h_pad, w_pad, w_pad
    pad_image = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return pad_image

def random_crop(img, area_ratio, hw_vari):
    """
    定义随机裁剪函数
    :param img: 要处理的图片
    :param area_ratio: 裁剪画面占原图片的比例
    :param hw_vari: 扰动占原宽高的比例
    :return: 裁剪后的图片
    """
    w, h = img.shape[:2]
    hw_delta = np.random.uniform(-hw_vari, hw_vari)
    hw_mult = 1 + hw_delta

    # 裁剪宽度
    w_crop = int(np.round(w * np.sqrt(area_ratio * hw_mult)))
    # 裁剪宽度不能大于原图宽度
    if w_crop > w:
        w_crop = w
    # 裁剪高度
    h_crop = int(np.round(h * np.sqrt(area_ratio / hw_mult)))
    if h_crop > h:
        h_crop = h

    # 随机生成左上角的位置
    x0 = np.random.randint(0, w - w_crop + 1)
    y0 = np.random.randint(0, h - h_crop + 1)

    return crop_image(img, x0, y0, w_crop, h_crop)


def rotate_image(img, angle, crop):
    """
    定义旋转函数
    :param img: 要处理的图片
    :param angle: 图片旋转的角度
    :param crop: 是否对旋转后出现的黑边进行裁剪
    :return: 旋转后的图片
    """
    w, h = img.shape[:2]

    # 旋转角度的周期是360°
    angle %= 360

    # 计算仿射变换矩阵
    M_rotate = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    # 得到旋转后的图像
    img_rotated = cv2.warpAffine(img, M_rotate, (w, h))

    # 如果需要裁剪黑边
    if crop:
        # 裁剪角度的等效周期是180°
        angle_crop = angle % 180
        if angle_crop > 90:
            angle_crop = 180 - angle_crop

        # 角度转化为弧度
        theta = angle_crop * np.pi / 180

        # 计算高宽比
        hw_ratio = float(h) / float(w)

        # 计算裁剪边长系数的分子项
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * tan_theta

        # 计算分母项中相关的项
        r = hw_ratio if h > w else 1 / hw_ratio
        # 分母项
        denominator = r * tan_theta + 1

        # 最终的边长系数
        crop_mult = numerator / denominator

        # 得到裁剪区域
        w_crop = int(round(crop_mult * w))
        h_crop = int(round(crop_mult * h))
        x0 = int((w - w_crop) / 2)
        y0 = int((h - h_crop) / 2)

        img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)

    return img_rotated


def random_rotate(img, angle_vari, p_crop):
    """
    定义随机旋转函数
    :param img: 要处理的图片
    :param angle_vari: 旋转角度的范围
    :param p_crop: 要进行裁剪黑边的图片所占的比例
    :return: 随机旋转的图片
    """
    angle = np.random.uniform(-angle_vari, angle_vari)
    crop = False if np.random.random() > p_crop else True

    return rotate_image(img, angle, crop)


def hsv_transform(img, hue_delta, sat_mult, val_mult):
    """
    定义hsv转换函数
    :param img: 要处理的图片
    :param hue_delta: 色调变化
    :param sat_mult: 饱和度变化
    :param val_mult: 明暗度变化
    :return: hsv变化后的图片
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
    img_hsv[:, :, 1] *= sat_mult
    img_hsv[:, :, 2] *= val_mult
    img_hsv[img_hsv > 255] = 255

    return cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)


def random_hsv_transform(img, hue_vari, sat_vari, val_vari):
    """
    随机hsv变换
    :param img: 要处理的图片
    :param hue_vari: 色调比例变化范围
    :param sat_vari: 饱和度比例变化范围
    :param val_vari: 明暗度比例变化范围
    :return: 随机hsv变换后的图片
    """
    hue_delta = np.random.uniform(-hue_vari, hue_vari)
    sat_mult = np.random.uniform(-sat_vari, sat_vari) + 1
    val_mult = np.random.uniform(-val_vari, val_vari) + 1

    return hsv_transform(img, hue_delta, sat_mult, val_mult)


def gamma_transform(img, gamma):
    """
    定义gamma变换函数
    :param img: 要处理的函数
    :param gamma: gamma系数
    :return: gamma变换后的图片
    """
    gamma_table = [np.power(x / 255, gamma) * 255 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)

    return gamma_transform(img, gamma)

def fill_rect(img, pt1, pt2, color, transparency=0.2):
    x1, y1 = pt1[0], pt1[1]
    x2, y2 = pt2[0], pt2[1]
    pts = np.array([[[x1,y1], [x1,y2], [x2,y2], [x2,y1]]])
    img_fill = img.copy()
    cv2.fillPoly(img_fill, pts, color)
    img = cv2.addWeighted(img, transparency, img_fill, 1-transparency, 1.0)
    return img

def draw_liuhai(img):
    # img = cv2.imread('./im_160x160.jpg')
    img = np.copy(img)
    # img = cv2.resize(img, (100, 100))
    coord_point = np.array([[ 34.19, 46.16], [ 65.65, 45.98], [ 50.12, 82.40]], dtype=np.float32)
    coord_point = np.array([[31., 44.], [69., 44.], [50., 76.]], dtype=np.float32)
    width = img.shape[0]
    coord_point = coord_point / 100 * width
    h_diff = int(np.random.random()*0.03*width)
    left_eye = (int(coord_point[0][0]), int(coord_point[0][1] + h_diff))
    right_eye = (int(coord_point[1][0]), int(coord_point[1][1] + h_diff))
    glass_w = (0.15 + np.random.random()*0.05) * width
    glass_w = int(glass_w)
    glass_h = (0.1 + np.random.random()*0.10) * width
    glass_h = int(glass_h)
    color = int(np.random.random()*100)
    hh = 0.1 * width
    pt1 = (left_eye[0]-glass_w, int(max(left_eye[1]-glass_h-hh,0)))
    pt2 = (right_eye[0]+glass_w, int(right_eye[1]-hh))
    img = fill_rect(img, pt1, pt2, color, transparency=0.1)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.imwrite('img_glass.jpg', img)
    return img

def draw_beard(img):
    # img = cv2.imread('./im_160x160.jpg')
    img = np.copy(img)
    # img = cv2.resize(img, (100, 100))
    coord_point = np.array([[ 34.19, 46.16], [ 65.65, 45.98], [ 50.12, 82.40]], dtype=np.float32)
    coord_point = np.array([[31., 44.], [69., 44.], [50., 76.]], dtype=np.float32)
    width = img.shape[0]
    coord_point = coord_point / 100 * width
    h_diff = int(np.random.random()*0.03*width)
    left_eye = (int(coord_point[0][0]), int(coord_point[0][1] + h_diff))
    right_eye = (int(coord_point[1][0]), int(coord_point[1][1] + h_diff))
    glass_w = (0.02 + np.random.random()*0.04) * width
    glass_w = int(glass_w)
    glass_h = (0.1 + np.random.random()*0.10) * width
    glass_h = int(glass_h)
    color = int(np.random.random()*100)
    hh = 0.43 * width
    pt1 = (left_eye[0]-glass_w, int(max(left_eye[1]-glass_h+hh,0)))
    pt2 = (right_eye[0]+glass_w, int(min(right_eye[1]+hh,width)))
    img = fill_rect(img, pt1, pt2, color, transparency=0.5)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.imwrite('img_glass.jpg', img)
    return img

def draw_glass(img):
    # img = cv2.imread('./im_160x160.jpg')
    img = np.copy(img)
    # img = cv2.resize(img, (100, 100))
    coord_point = np.array([[ 34.19, 46.16], [ 65.65, 45.98], [ 50.12, 82.40]], dtype=np.float32)
    coord_point = np.array([[31., 44.], [69., 44.], [50., 76.]], dtype=np.float32)

    width = img.shape[0]
    coord_point = coord_point / 100 * width
    # img = annotate_shapes(img, coord_point, verbose=1)
    h_diff = int(np.random.random()*0.03*width)
    left_eye = (int(coord_point[0][0]), int(coord_point[0][1] + h_diff))
    right_eye = (int(coord_point[1][0]), int(coord_point[1][1] + h_diff))
    # cv2.circle(img, left_eye, 10, color=(0, 0, 0), thickness=4)
    # cv2.circle(img, right_eye, 10, color=(0, 0, 0), thickness=4)
    glass_h = (0.05 + np.random.random()*0.10) * width
    glass_w = (0.15 + np.random.random()*0.05) * width
    glass_h = int(glass_h)
    glass_w = int(glass_w)
    thickness = 1 + np.random.random()*5
    thickness = int(thickness)
    # print(thickness)
    if np.random.random() < 0.2:
        cv2.ellipse(img, left_eye, (glass_w, glass_h),0,0,360,0, thickness=thickness)
        cv2.ellipse(img, right_eye, (glass_w, glass_h),0,0,360,0, thickness=thickness)
    else:
        color = int(np.random.random()*255)
        glass_w -= 3
        glass_h += 1
        pt1 = (left_eye[0]-glass_w, left_eye[1]-glass_h)
        pt2 = (left_eye[0]+glass_w, left_eye[1]+glass_h)
        cv2.rectangle(img, pt1, pt2, color=color, thickness=thickness)
        img = fill_rect(img, pt1, pt2, color, transparency=0.6)
        pt1 = (right_eye[0]-glass_w, right_eye[1]-glass_h)
        pt2 = (right_eye[0]+glass_w, right_eye[1]+glass_h)
        cv2.rectangle(img, pt1, pt2, color=color, thickness=thickness)
        img = fill_rect(img, pt1, pt2, color, transparency=0.6)

    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.imwrite('img_glass.jpg', img)
    return img

def random_occlusion(img, area_ratio=0.05, hw_vari=0.5):
    w, h = img.shape[:2]
    hw_delta = np.random.uniform(-hw_vari, hw_vari)
    hw_mult = 1 + hw_delta
    # 裁剪宽度
    w_crop = int(np.round(w * np.sqrt(area_ratio * hw_mult)))
    # 裁剪宽度不能大于原图宽度
    if w_crop > w:
        w_crop = w
    # 裁剪高度
    h_crop = int(np.round(h * np.sqrt(area_ratio / hw_mult)))
    if h_crop > h:
        h_crop = h
    # 随机生成左上角的位置
    x0 = np.random.randint(10, w - w_crop + 1 - 10)
    y0 = np.random.randint(15, h - h_crop + 1 - 10)
    pt1 = (x0, y0)
    pt2 = (x0+w_crop, y0+h_crop)
    color = int(np.random.random()*100)
    img = fill_rect(img, pt1, pt2, color, transparency=0.1)
    return img


def cutout(img, length_ratio=0.3, n_holes=1):
    """
    Args:
        img (Tensor): Tensor image of size (C, H, W).
    Returns:
        Tensor: Image with n_holes of dimension length x length cut out of it.
    """
    h = img.shape[0]
    w = img.shape[1]
    length = int(h*length_ratio)

    mask = np.ones((h, w), np.float32)

    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)
        # y = np.random.randint(15, h - length + 1 - 10)
        # x = np.random.randint(10, w - length + 1 - 10)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.

    img = img * np.expand_dims(mask, -1)

    return img


def random_resize(img):
    w = img.shape[0]
    s = int(np.random.random() * w * 3 / 4 + w/4)
    img = misc.imresize(img, (s, s))
    img = misc.imresize(img, (w, w))
    return img


def rotate(img, level=3):
    pil_img = Image.fromarray(img.astype(np.uint8))  # Convert to PIL.Image
    degrees = int(level * 30 / 10)
    # print('rotate', degrees)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    pil_img = pil_img.rotate(degrees, resample=Image.BILINEAR)
    img = np.asarray(pil_img)
    return img.astype(np.uint8)


def gaussian_blur(img, radius=5.0):
    pil_img = Image.fromarray(img.astype(np.uint8))  # Convert to PIL.Image
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    img = np.asarray(pil_img)
    return img.astype(np.uint8)

def img_rgb_2_gray3ch(img):
    img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    img = np.expand_dims(img, -1)
    img = np.concatenate([img, img, img], axis=2)
    return img

def face_image_augment_cv(img_i, aug_proba=1.0, isRgbImage=False, num_sample_opt=2, verbose=0):
    img = img_i.copy()
    img_size = img.shape[:2]
    grayscale = img.shape[2] == 1

    if grayscale:
        img = np.concatenate([img, img, img], axis=2)
    if isRgbImage:
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img[..., ::-1]
    img = img.astype(np.uint8)

    augment_func_once = [
        # {'name':'draw_glass', 'func':lambda img: draw_glass(img), 'weight':0.2},
        # {'name':'draw_liuhai', 'func':lambda img: draw_liuhai(img), 'weight':0.01},
        # {'name':'draw_beard', 'func':lambda img: draw_beard(img), 'weight':0.05},
        {'name':'random_occlusion', 'func':lambda img: cutout(img), 'weight':0.5},
    ]
    sample_once_weights = np.array([l['weight'] for l in augment_func_once])
    sample_once_weights /= np.sum(sample_once_weights)
    augment_func = [
        # {'name':'detail_enhance', 'func':lambda img: detail_enhance(img, np.random.random()*4), 'weight':0.01},
        # {'name':'edge_preserve', 'func':lambda img: edge_preserve(img), 'weight':0.03},
        {'name':'random_resize', 'func':lambda img: random_resize(img), 'weight':0.8},
        {'name':'guassian_blur', 'func':lambda img: gaussian_blur(img, 0.5), 'weight':0.8},
        # {'name':'change_saturation', 'func':lambda img: change_saturation(img, -20 + np.random.random()*40), 'weight':0.1},
        # {'name':'change_darker', 'func':lambda img: change_darker(img, -8 + np.random.random()*16), 'weight':1},
        # {'name':'random_rotate', 'func':lambda img: random_rotate(img, 9, True), 'weight':0.1},
        {'name':'random_rotate', 'func':lambda img: rotate(img, 3), 'weight':0.1},
        # {'name':'random_pad', 'func':lambda img: random_pad(img, 0.005, 0.15), 'weight':0.1},
        {'name':'random_crop', 'func':lambda img: random_crop(img, 0.95, 0.15), 'weight':0.1},
        {'name':'random_bgr', 'func':lambda img: img[..., ::-1], 'weight':0.1},
        {'name':'random_to_gray', 'func':lambda img: img_rgb_2_gray3ch(img), 'weight':0.1},
    ]
    sample_weights = np.array([l['weight'] for l in augment_func])
    sample_weights /= np.sum(sample_weights)
    info = ''

    # sample once
    if np.random.random() < aug_proba:
        bi = np.random.choice(len(sample_once_weights), 1, p=sample_once_weights)[0]
        img = augment_func_once[bi]['func'](img)
        info += augment_func_once[bi]['name'] + ', '

    # sample multiple
    for _ in range(num_sample_opt):
        if np.random.random() < aug_proba:
            bi = np.random.choice(len(sample_weights), 1, p=sample_weights)[0]
            img = augment_func[bi]['func'](img)
            info += augment_func[bi]['name'] + ', '
            aug_proba *= 0.5
    if isRgbImage:
        img = img[..., ::-1]
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, img_size)
    img = misc.imresize(img, img_size)
    if grayscale:
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        img = np.expand_dims(img, -1)
    if len(info) > 0:
        # print(info)
        pass

    if np.random.random() < 0.5:
        # img = cv2.flip(img, 1) # 水平翻转
        img = np.fliplr(img)

    img = img.astype(np.uint8)

    if verbose > 0:
        if len(info) > 0:
            print(info)
        # img_disp = img.astype(np.uint8)
        # cv2.imshow('img', img_disp[...,::-1])
        # cv2.waitKey(0)
    return img


def demo_aug():
    import cv2
    img_path = r'Aaron_Eckhart_0001.jpg'
    img_path = r'im_160.jpg'
    # img_path = r'G:\data\zk_handan\xiaogu-videos-live\xiaogu-videos-live\face_112x112\sn_dualIR\nir\cam0\cam0_2020.06.07_11.20.09.712962_1280x960\cam0_2020.06.07_11.20.09.712962_1280x960_63236.jpg'
    img_raw = cv2.imread(img_path)
    while True:
        t1 = time()
        img = face_image_augment_cv(img_raw, aug_proba=1.0, verbose=1)
        print(time()-t1, 's')
        img = cv2.resize(img, (300, 300))
        cv2.imshow('img', img)
        data = np.split(img, 4, axis=0)
        cv2.imshow('img0', data[0])
        cv2.imshow('img1', data[1])
        cv2.imshow('img2', data[2])
        cv2.imshow('img3', data[3])
        key = cv2.waitKey(0)
        if key & 0xff == 27 or key & 0xff == 13:  # Esc or Enter
            break


def augment_dataset():
    from data.data_utils import get_dataset
    import cv2

    input_dir = r'E:\data\face-1vs1\test-5w-3d\ISOK-160'
    input_dir = r'E:\data\face-recognition\real\nanningyouji-160'
    input_dir = r'E:\data\face-recognition\MS-Celeb-1M\MsCelebV1-Faces-Aligned-160-Clean-Relabel-128Vec'
    input_dir = r'C:\work\data\face-data\lfw-112-mxnet'
    output_dir = r'C:\work\data\face-data\lfw-112-mxnet-n10-aug'
    dataset = get_dataset(input_dir)
    print(input_dir)

    nrof_classes = len(dataset)

    t1 = time()
    for ci, cls in enumerate(dataset):
        for image_path in cls.image_paths[:5]:
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            try:
                img = cv2.imread(image_path)
            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(image_path, e)
                print(errorMessage)
            else:
                if not isinstance(img, np.ndarray):
                    continue
                if img.ndim<2:
                    print('Unable to align "%s"' % image_path)
                    continue
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = img[:,:,0:3]
                # vs = filename.split('_')
                # yaw = int(vs[3])
                data = np.split(img, 4, axis=0)
                cv2.imshow('img0', data[0])
                cv2.imshow('img1', data[1])
                cv2.imshow('img2', data[2])
                cv2.imshow('img3', data[3])
                yaw = 0
                if yaw < 40:
                    output_class_dir = os.path.split(image_path)[0]
                    output_filename = os.path.join(output_dir, filename+'.jpg')
                    # print(image_path)
                    if os.path.exists(output_filename):
                        continue
                    # img = draw_glass(img)
                    img = face_image_augment_cv(img, aug_proba=1.0)
                    # cv2.imwrite(output_filename, img)
                    img = cv2.resize(img, (300, 300))
                    cv2.imshow('img', img)
                    key = cv2.waitKey(0)
                    if key & 0xff == 27 or key & 0xff == 13:  # Esc or Enter
                        break

        print('%.0f s - %.0f s  %d / %d %s'% (
            time()-t1, (time()-t1)/(ci+1)*(nrof_classes-ci), ci, nrof_classes, cls.name))

if __name__ == '__main__':
    augment_dataset()
    # demo_aug()
