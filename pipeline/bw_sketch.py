import cv2
import numpy as np
from skimage.morphology import skeletonize


def generate_bw_sketch(data: dict, config: dict) -> np.ndarray:
    flattened = data["flattened"]
    silhouette = data["silhouette"]
    construction_lines = data["construction_lines"]
    h, w = silhouette.shape

    edges = np.zeros((h, w), dtype=np.uint8)

    silhouette_contours = _extract_all_contours(silhouette)
    edges = cv2.bitwise_or(edges, silhouette_contours)

    structural = _detect_strong_structural_edges(flattened, silhouette, config)
    edges = cv2.bitwise_or(edges, structural)

    if construction_lines is not None and np.sum(construction_lines) > 0:
        edges = cv2.bitwise_or(edges, construction_lines)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)

    min_component = config.get("min_region_area", 100)
    edges = _remove_small_components(edges, min_component)

    skeleton = _skeletonize_edges(edges)
    stroke_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    stroked = cv2.dilate(skeleton, stroke_kernel, iterations=1)

    bw = np.full((h, w), 255, dtype=np.uint8)
    bw[stroked > 0] = 0

    return bw


def _extract_all_contours(silhouette: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(silhouette, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_img = np.zeros_like(silhouette)
    cv2.drawContours(contour_img, contours, -1, 255, 1)
    return contour_img


def _detect_strong_structural_edges(flattened: np.ndarray, silhouette: np.ndarray,
                                     config: dict) -> np.ndarray:
    gray = cv2.cvtColor(flattened, cv2.COLOR_BGR2GRAY)

    d = config.get("bilateral_d", 15)
    sc = config.get("bilateral_sigma_color", 75)
    ss = config.get("bilateral_sigma_space", 75)
    smoothed = gray.copy()
    for _ in range(4):
        smoothed = cv2.bilateralFilter(smoothed, d, sc, ss)

    canny = cv2.Canny(smoothed, config.get("canny_low", 40), config.get("canny_high", 120))
    canny[silhouette == 0] = 0

    grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    fg_grads = grad_mag[silhouette > 0]
    if len(fg_grads) == 0:
        return np.zeros_like(silhouette)

    threshold = np.percentile(fg_grads, 90)
    canny[grad_mag < threshold] = 0

    return canny


def _skeletonize_edges(edges: np.ndarray) -> np.ndarray:
    binary = (edges > 0).astype(np.uint8)
    skeleton = skeletonize(binary).astype(np.uint8) * 255
    return skeleton


def _remove_small_components(edges: np.ndarray, min_size: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    cleaned = np.zeros_like(edges)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 255
    return cleaned
