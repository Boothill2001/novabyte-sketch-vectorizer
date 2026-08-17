import cv2
import numpy as np
from skimage.segmentation import slic


def load_image(path: str) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)
    elif img.shape[2] == 4:
        alpha = img[:, :, 3]
        bgr = img[:, :, :3]
    else:
        bgr = img
        alpha = _create_alpha_from_white_bg(bgr)

    return bgr, alpha


def _create_alpha_from_white_bg(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    bg_mask = (gray > 240) & (saturation < 15) & (value > 240)

    fg_mask = (~bg_mask).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    if num_labels > 1:
        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        min_component = total_pixels * 0.001
        cleaned = np.zeros_like(fg_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_component:
                cleaned[labels == i] = 255
        fg_mask = cleaned

    return fg_mask


def extract_silhouette(alpha: np.ndarray, threshold: int = 128) -> np.ndarray:
    mask = np.zeros_like(alpha, dtype=np.uint8)
    mask[alpha >= threshold] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def segment_materials(bgr: np.ndarray, mask: np.ndarray, sat_threshold: int = 80) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    material_map = np.zeros(mask.shape, dtype=np.uint8)
    material_map[(mask > 0) & (saturation >= sat_threshold)] = 2
    material_map[(mask > 0) & (saturation < sat_threshold)] = 1
    return material_map


def flatten_shading(bgr: np.ndarray, mask: np.ndarray, material_map: np.ndarray,
                    config: dict) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)

    rgb_for_slic = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fg_mask = mask > 0

    if np.sum(fg_mask) < 100:
        return bgr.copy()

    segments = slic(
        rgb_for_slic,
        n_segments=config.get("slic_n_segments", 200),
        compactness=config.get("slic_compactness", 25),
        mask=fg_mask,
        start_label=0,
    )

    flattened_lab = lab.copy()
    for seg_id in np.unique(segments):
        if seg_id < 0:
            continue
        seg_mask = segments == seg_id
        fg_pixels = seg_mask & fg_mask
        if np.sum(fg_pixels) == 0:
            continue

        material_vals = material_map[fg_pixels]
        is_gem = np.mean(material_vals == 2) > 0.5

        if is_gem:
            continue

        median_l = np.median(lab[fg_pixels, 0])
        flattened_lab[fg_pixels, 0] = median_l

    flattened_bgr = cv2.cvtColor(flattened_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    flattened_bgr[~fg_mask] = bgr[~fg_mask]
    return flattened_bgr


def detect_construction_lines(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thin_lines = np.zeros_like(mask)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    low_sat = hsv[:, :, 1] < 20
    mid_val = (gray > 100) & (gray < 220)

    edges = cv2.Canny(gray, 15, 50)
    kernel_thin = np.ones((1, 1), dtype=np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_thin)

    candidate = edges & low_sat.astype(np.uint8) * 255 & mid_val.astype(np.uint8) * 255

    near_bg = ~(mask > 0)
    dilated_bg = cv2.dilate(near_bg.astype(np.uint8) * 255,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
    candidate[dilated_bg > 0] = 0
    candidate[mask == 0] = 0

    return candidate


def preprocess(image_path: str, config: dict) -> dict:
    bgr, alpha = load_image(image_path)
    sat_thresh = config.get("gem_saturation_threshold", 80)

    silhouette = extract_silhouette(alpha, threshold=128)
    material_map = segment_materials(bgr, silhouette, sat_thresh)
    flattened = flatten_shading(bgr, silhouette, material_map, config)
    construction_lines = detect_construction_lines(bgr, silhouette)

    return {
        "bgr": bgr,
        "alpha": alpha,
        "silhouette": silhouette,
        "material_map": material_map,
        "flattened": flattened,
        "construction_lines": construction_lines,
        "height": bgr.shape[0],
        "width": bgr.shape[1],
    }
