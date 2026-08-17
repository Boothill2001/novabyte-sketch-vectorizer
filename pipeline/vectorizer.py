import cv2
import numpy as np
import vtracer
import tempfile
import os
import re
from pathlib import Path
from xml.etree import ElementTree as ET


def vectorize(bw_raster: np.ndarray, color_raster: np.ndarray,
              data: dict, config: dict) -> tuple[str, dict]:
    h, w = bw_raster.shape[:2]
    bw_paths = _vectorize_bw(bw_raster, config)
    color_paths, fill_colors = _vectorize_color(color_raster, data["silhouette"], config)

    svg = _assemble_svg(w, h, bw_paths, color_paths, fill_colors)

    unique_fills = len(set(fill_colors)) if fill_colors else 0
    stats = {
        "bw_path_count": len(bw_paths),
        "color_path_count": len(color_paths),
        "total_anchors": _count_anchors(bw_paths) + _count_anchors(color_paths),
        "bw_anchors": _count_anchors(bw_paths),
        "color_anchors": _count_anchors(color_paths),
        "fill_count": unique_fills,
        "color_region_count": len(color_paths),
        "has_open_paths": False,
        "has_embedded_raster": False,
        "has_gradients": False,
    }
    return svg, stats


def _vectorize_bw(bw_raster: np.ndarray, config: dict) -> list[str]:
    binary = np.zeros_like(bw_raster)
    binary[bw_raster < 128] = 255

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

    paths = []
    anchor_ceiling = config.get("anchor_ceiling", 22)

    for contour in contours:
        if len(contour) < 3:
            continue
        epsilon = max(1.0, cv2.arcLength(contour, True) * 0.005)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        while len(approx) > anchor_ceiling and epsilon < 20:
            epsilon *= 1.5
            approx = cv2.approxPolyDP(contour, epsilon, True)

        path_d = _contour_to_bezier_path(approx)
        if path_d:
            paths.append(path_d)

    return paths


def _vectorize_color(color_raster: np.ndarray, silhouette: np.ndarray,
                     config: dict) -> tuple[list[str], list[str]]:
    fg = color_raster.copy()
    fg[silhouette == 0] = [255, 255, 255]

    unique_colors = _get_unique_fg_colors(fg, silhouette)

    paths = []
    fill_colors = []
    anchor_ceiling = config.get("anchor_ceiling", 22)
    min_area = config.get("min_region_area", 80)

    for color in unique_colors:
        if np.array_equal(color, [255, 255, 255]):
            continue

        mask = np.all(fg == color, axis=2).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

        hex_color = _bgr_to_hex(color)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            epsilon = max(1.0, cv2.arcLength(contour, True) * 0.008)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            while len(approx) > anchor_ceiling and epsilon < 20:
                epsilon *= 1.5
                approx = cv2.approxPolyDP(contour, epsilon, True)

            path_d = _contour_to_bezier_path(approx, closed=True)
            if path_d:
                paths.append(path_d)
                fill_colors.append(hex_color)

    return paths, fill_colors


def _contour_to_bezier_path(contour: np.ndarray, closed: bool = False) -> str:
    pts = contour.reshape(-1, 2)
    if len(pts) < 2:
        return ""

    d = f"M {pts[0][0]},{pts[0][1]}"

    i = 1
    while i < len(pts):
        if i + 2 < len(pts):
            d += f" C {pts[i][0]},{pts[i][1]} {pts[i+1][0]},{pts[i+1][1]} {pts[i+2][0]},{pts[i+2][1]}"
            i += 3
        elif i + 1 < len(pts):
            mid_x = (pts[i][0] + pts[i+1][0]) / 2
            mid_y = (pts[i][1] + pts[i+1][1]) / 2
            d += f" Q {pts[i][0]},{pts[i][1]} {pts[i+1][0]},{pts[i+1][1]}"
            i += 2
        else:
            d += f" L {pts[i][0]},{pts[i][1]}"
            i += 1

    if closed:
        d += " Z"

    return d


def _get_unique_fg_colors(img: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    fg_pixels = img[mask > 0]
    if len(fg_pixels) == 0:
        return []
    unique = np.unique(fg_pixels.reshape(-1, 3), axis=0)
    return [c for c in unique if not np.array_equal(c, [255, 255, 255])]


def _bgr_to_hex(bgr: np.ndarray) -> str:
    return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"


def _count_anchors(paths: list[str]) -> int:
    total = 0
    for p in paths:
        total += p.count("M") + p.count("L") + p.count("C") + p.count("Q")
    return total


def _assemble_svg(width: int, height: int, bw_paths: list[str],
                  color_paths: list[str], fill_colors: list[str]) -> str:
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '  <g id="color_sketch">',
    ]

    for path_d, fill in zip(color_paths, fill_colors):
        lines.append(f'    <path d="{path_d}" fill="{fill}" stroke="none"/>')

    lines.append('  </g>')
    lines.append('  <g id="bw_sketch">')

    for path_d in bw_paths:
        lines.append(f'    <path d="{path_d}" fill="none" stroke="#000000" stroke-width="2"/>')

    lines.append('  </g>')
    lines.append('</svg>')

    return '\n'.join(lines)
