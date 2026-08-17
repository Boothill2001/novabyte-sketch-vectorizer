import cv2
import numpy as np
from skimage.morphology import skeletonize


def generate_bw_sketch(data: dict, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns (bw_raster, ambiguous_edge_mask)."""
    flattened = data["flattened"]
    original = data["bgr"]
    silhouette = data["silhouette"]
    construction_lines = data["construction_lines"]
    h, w = silhouette.shape

    edges = np.zeros((h, w), dtype=np.uint8)

    silhouette_contours = _extract_all_contours(silhouette)
    edges = cv2.bitwise_or(edges, silhouette_contours)

    structural = _detect_structural_edges(flattened, original, silhouette, config)
    edges = cv2.bitwise_or(edges, structural)

    ambiguous = _detect_ambiguous_edges(flattened, original, silhouette, edges, config)
    edges = cv2.bitwise_or(edges, ambiguous)

    if construction_lines is not None and np.sum(construction_lines) > 0:
        edges = cv2.bitwise_or(edges, construction_lines)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)

    min_component = config.get("min_region_area", 100) // 2
    edges = _remove_small_components(edges, min_component)

    skeleton = _skeletonize_edges(edges)
    stroke_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    stroked = cv2.dilate(skeleton, stroke_kernel, iterations=1)

    bw = np.full((h, w), 255, dtype=np.uint8)
    bw[stroked > 0] = 0

    return bw, ambiguous


def _extract_all_contours(silhouette: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(silhouette, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_img = np.zeros_like(silhouette)
    cv2.drawContours(contour_img, contours, -1, 255, 1)
    return contour_img


def _detect_structural_edges(flattened: np.ndarray, original: np.ndarray,
                              silhouette: np.ndarray, config: dict) -> np.ndarray:
    gray_flat = cv2.cvtColor(flattened, cv2.COLOR_BGR2GRAY)
    gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    d = config.get("bilateral_d", 15)
    sc = config.get("bilateral_sigma_color", 75)
    ss = config.get("bilateral_sigma_space", 75)

    smoothed_flat = gray_flat.copy()
    for _ in range(3):
        smoothed_flat = cv2.bilateralFilter(smoothed_flat, d, sc, ss)

    smoothed_orig = gray_orig.copy()
    for _ in range(2):
        smoothed_orig = cv2.bilateralFilter(smoothed_orig, d, sc, ss)

    canny_low = config.get("canny_low", 40)
    canny_high = config.get("canny_high", 120)

    edges_flat = cv2.Canny(smoothed_flat, canny_low, canny_high)
    edges_flat[silhouette == 0] = 0

    edges_orig = cv2.Canny(smoothed_orig, canny_low, canny_high)
    edges_orig[silhouette == 0] = 0

    grad_x = cv2.Sobel(smoothed_flat, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smoothed_flat, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    fg_grads = grad_mag[silhouette > 0]
    if len(fg_grads) == 0:
        return np.zeros_like(silhouette)

    strong_threshold = np.percentile(fg_grads, 70)
    strong_edges = edges_flat.copy()
    strong_edges[grad_mag < strong_threshold] = 0

    dilated_flat = cv2.dilate(edges_flat, np.ones((5, 5), np.uint8), iterations=1)
    confirmed = edges_orig & dilated_flat
    confirmed[silhouette == 0] = 0

    result = cv2.bitwise_or(strong_edges, confirmed)
    return result


def _detect_ambiguous_edges(flattened: np.ndarray, original: np.ndarray,
                             silhouette: np.ndarray, kept_edges: np.ndarray,
                             config: dict) -> np.ndarray:
    gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.bilateralFilter(gray_orig,
                                    config.get("bilateral_d", 15),
                                    config.get("bilateral_sigma_color", 75),
                                    config.get("bilateral_sigma_space", 75))

    edges_orig = cv2.Canny(smoothed, config.get("canny_low", 40), config.get("canny_high", 120))
    edges_orig[silhouette == 0] = 0

    dilated_kept = cv2.dilate(kept_edges, np.ones((5, 5), np.uint8), iterations=1)
    ambiguous = edges_orig & ~dilated_kept
    ambiguous[silhouette == 0] = 0

    min_size = config.get("min_region_area", 100)
    ambiguous = _remove_small_components(ambiguous, min_size // 3)

    return ambiguous


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
