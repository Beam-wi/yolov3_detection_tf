import cv2
import copy
import numpy as np


def imRotate(img,
             angle,
             points=None,
             center=None,
             scale=1.0,
             border_value=0,
             interpolation='bilinear',
             auto_bound=False,
             adaption=False):
    """Rotate an image or with points if given.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        points (ndarray (n, 2)): 为None时只返回旋转后的图片，
            否则返回 旋转图 和旋转后的 多边形点
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used.
        scale (float): Isotropic scale factor.
        border_value (int): Border value.
        interpolation (str): Same as :func:`resize`.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image.
        adaption(bool): 是否启用自适应 获得旋转后的最小外接矩形图像

    Returns:
        ndarray: The rotated image.
    """
    def _pointsRotate(matrix, points):
        Points = np.concatenate(
            (copy.deepcopy(points), np.ones((points.shape[0], 1))), axis=-1).T
        return np.int0(np.dot(matrix, Points).T)
    cv2_interp_codes = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
        'lanczos': cv2.INTER_LANCZOS4
    }
    if center is not None and auto_bound:
        raise ValueError('`auto_bound` conflicts with `center`')
    h, w = img.shape[:2]
    if center is None or adaption:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, (tuple, list, np.ndarray))
    rboardh, rboardw = h, w
    if adaption:
        matrixOr = cv2.getRotationMatrix2D(center, -angle, scale)
        WH = _pointsRotate(matrixOr, np.array([[0, 0], [0, h], [w, h], [w, 0]]))
        W_, H_ = WH[:, 0].max()-WH[:, 0].min(),  WH[:, 1].max()-WH[:, 1].min()
        W, H = max(W_, w), max(H_, h)
        imadapt = np.zeros((H, W, 3), dtype=np.uint8)+border_value
        boardH, boardW = int((H-h)/2), int((W-w)/2)
        imadapt[boardH:boardH+h, boardW:boardW+w, :] = img

        img = imadapt
        w, h = W, H
        center = ((W - 1) * 0.5, (H - 1) * 0.5)
        rboardh, rboardw = H_, W_

    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))

    rotated = cv2.warpAffine(
        img,
        matrix, (w, h),
        flags=cv2_interp_codes[interpolation],
        borderValue=(border_value, border_value, border_value))
    boardh, boardw = int((h - rboardh) / 2), int((w - rboardw) / 2)
    rotated = rotated[boardh:boardh+rboardh, boardw:boardw+rboardw, :]
    if points is None:
        return rotated
    else:
        Points = _pointsRotate(matrix, points)
        xmin, ymin = Points[:, 0].min(), Points[:, 1].min()
        Points[:, 0] += xmin if adaption else 0
        Points[:, 1] += ymin if adaption else 0
        return rotated, Points