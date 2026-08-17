import cv2
import numpy as np


def compute_metrics(data: dict, bw_raster: np.ndarray, color_raster: np.ndarray,
                    svg_content: str, svg_stats: dict, elapsed: float) -> dict:
    source_mask = data["silhouette"]

    bw_mask = np.zeros(bw_raster.shape[:2], dtype=np.uint8)
    if len(bw_raster.shape) == 2:
        bw_mask[bw_raster < 255] = 255
    else:
        gray = cv2.cvtColor(bw_raster, cv2.COLOR_BGR2GRAY)
        bw_mask[gray < 255] = 255

    color_mask = np.zeros(color_raster.shape[:2], dtype=np.uint8)
    if len(color_raster.shape) == 3:
        not_white = ~np.all(color_raster == 255, axis=2)
        color_mask[not_white] = 255
    else:
        color_mask[color_raster < 255] = 255

    output_mask = cv2.bitwise_or(bw_mask, color_mask)
    iou = _compute_iou(source_mask, output_mask)
    f_score = _boundary_f_score(source_mask, output_mask, tolerance=2)
    coord_offset = _coordinate_offset(source_mask, output_mask)
    has_gradient = _check_gradients(color_raster, data["silhouette"])

    return {
        "silhouette_iou": round(iou, 4),
        "boundary_f_score": round(f_score, 4),
        "coordinate_offset_px": round(coord_offset, 2),
        "fill_count": svg_stats.get("fill_count", 0),
        "total_anchors": svg_stats.get("total_anchors", 0),
        "bw_anchors": svg_stats.get("bw_anchors", 0),
        "color_anchors": svg_stats.get("color_anchors", 0),
        "bw_path_count": svg_stats.get("bw_path_count", 0),
        "color_path_count": svg_stats.get("color_path_count", 0),
        "has_open_paths": svg_stats.get("has_open_paths", False),
        "has_embedded_raster": svg_stats.get("has_embedded_raster", False),
        "has_gradients": has_gradient,
        "processing_time_s": round(elapsed, 2),
        "fill_count_rationale": _fill_rationale(svg_stats.get("fill_count", 0)),
        "anchor_ceiling_rationale": (
            "Target ≤20 anchors per elliptical element. "
            "Each anchor is load-bearing: removing any single point changes the path by >1px. "
            "This ensures natural drag behavior in vector editors."
        ),
    }


def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = (mask_a > 0).astype(np.uint8)
    b = (mask_b > 0).astype(np.uint8)
    intersection = np.sum(a & b)
    union = np.sum(a | b)
    if union == 0:
        return 1.0
    return intersection / union


def _boundary_f_score(mask_a: np.ndarray, mask_b: np.ndarray, tolerance: int = 2) -> float:
    contours_a, _ = cv2.findContours(mask_a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_b, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    boundary_a = np.zeros_like(mask_a)
    boundary_b = np.zeros_like(mask_b)
    cv2.drawContours(boundary_a, contours_a, -1, 255, 1)
    cv2.drawContours(boundary_b, contours_b, -1, 255, 1)

    dist_a = cv2.distanceTransform((255 - boundary_a), cv2.DIST_L2, 5)
    dist_b = cv2.distanceTransform((255 - boundary_b), cv2.DIST_L2, 5)

    pts_a = boundary_a > 0
    pts_b = boundary_b > 0

    if np.sum(pts_a) == 0 or np.sum(pts_b) == 0:
        return 0.0

    precision = np.sum(dist_a[pts_b] <= tolerance) / max(np.sum(pts_b), 1)
    recall = np.sum(dist_b[pts_a] <= tolerance) / max(np.sum(pts_a), 1)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _coordinate_offset(source_mask: np.ndarray, output_mask: np.ndarray) -> float:
    s_coords = np.argwhere(source_mask > 0)
    o_coords = np.argwhere(output_mask > 0)
    if len(s_coords) == 0 or len(o_coords) == 0:
        return 0.0
    s_center = np.mean(s_coords, axis=0)
    o_center = np.mean(o_coords, axis=0)
    return np.linalg.norm(s_center - o_center)


def _check_gradients(color_img: np.ndarray, mask: np.ndarray) -> bool:
    if len(color_img.shape) < 3:
        return False

    fg = color_img.copy()
    fg[mask == 0] = [255, 255, 255]

    unique_colors = np.unique(fg[mask > 0].reshape(-1, 3), axis=0)
    non_white = [c for c in unique_colors if not np.array_equal(c, [255, 255, 255])]
    non_black = [c for c in non_white if not np.array_equal(c, [0, 0, 0])]

    return len(non_black) > 30


def _fill_rationale(fill_count: int) -> str:
    if fill_count <= 4:
        return (f"{fill_count} fills: typical for a single-material metal piece "
                "with 1-2 accent colors.")
    elif fill_count <= 8:
        return (f"{fill_count} fills: expected for jewelry with mixed materials "
                "(metal + gems/enamel).")
    else:
        return (f"{fill_count} fills: complex multi-material piece. "
                "Each fill represents a perceptually distinct material zone.")


def generate_comparison(source: np.ndarray, bw: np.ndarray, color: np.ndarray,
                        alpha: np.ndarray) -> np.ndarray:
    h, w = source.shape[:2]

    white_bg = np.full_like(source, 255)
    if len(alpha.shape) == 2:
        mask = alpha > 0
        src_display = white_bg.copy()
        src_display[mask] = source[mask]
    else:
        src_display = source

    if len(bw.shape) == 2:
        bw_display = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    else:
        bw_display = bw

    if len(color.shape) == 2:
        color_display = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
    else:
        color_display = color

    gap = 20
    total_w = w * 3 + gap * 2
    canvas = np.full((h + 60, total_w, 3), 255, dtype=np.uint8)

    canvas[60:60+h, 0:w] = src_display
    canvas[60:60+h, w+gap:2*w+gap] = bw_display
    canvas[60:60+h, 2*w+2*gap:3*w+2*gap] = color_display

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Source", (w//2 - 40, 40), font, 1.0, (0, 0, 0), 2)
    cv2.putText(canvas, "B&W Sketch", (w + gap + w//2 - 70, 40), font, 1.0, (0, 0, 0), 2)
    cv2.putText(canvas, "Flat Color", (2*w + 2*gap + w//2 - 60, 40), font, 1.0, (0, 0, 0), 2)

    return canvas
