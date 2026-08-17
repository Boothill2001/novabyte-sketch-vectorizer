import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from scipy.spatial.distance import cdist


def generate_flat_color(data: dict, config: dict) -> np.ndarray:
    flattened = data["flattened"]
    silhouette = data["silhouette"]
    h, w = silhouette.shape

    lab = cv2.cvtColor(flattened, cv2.COLOR_BGR2LAB).astype(np.float64)

    rgb_for_slic = cv2.cvtColor(flattened, cv2.COLOR_BGR2RGB)
    segments = slic(
        rgb_for_slic,
        n_segments=config.get("slic_n_segments", 350),
        compactness=config.get("slic_compactness", 20),
        mask=(silhouette > 0),
        start_label=0,
    )

    merged_labels, sp_means = _merge_superpixels(
        lab, segments, silhouette,
        delta_e_thresh=config.get("merge_delta_e", 7.0)
    )

    k = config.get("quantize_k", 5)
    valid_means = sp_means[~np.all(sp_means == 0, axis=1)]
    if len(valid_means) <= k:
        cluster_centers = valid_means
        k = len(valid_means)
    else:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(valid_means)
        cluster_centers = kmeans.cluster_centers_

    quantized_lab = np.full((h, w, 3), 255, dtype=np.uint8)

    for seg_id in np.unique(merged_labels):
        if seg_id < 0:
            continue
        fg = (merged_labels == seg_id) & (silhouette > 0)
        if np.sum(fg) == 0:
            continue

        mean_color = np.mean(lab[fg], axis=0)
        dists = np.linalg.norm(cluster_centers - mean_color, axis=1)
        nearest = np.argmin(dists)
        quantized_lab[fg] = cluster_centers[nearest].astype(np.uint8)

    quantized_bgr = cv2.cvtColor(quantized_lab, cv2.COLOR_LAB2BGR)

    unsegmented = (silhouette > 0) & np.all(quantized_bgr == 255, axis=2)
    if np.sum(unsegmented) > 0:
        unseg_pixels_lab = lab[unsegmented]
        dists = cdist(unseg_pixels_lab, cluster_centers)
        nearest = np.argmin(dists, axis=1)
        fill_lab = cluster_centers[nearest].astype(np.uint8)
        fill_lab_img = np.zeros((len(fill_lab), 1, 3), dtype=np.uint8)
        fill_lab_img[:, 0, :] = fill_lab
        fill_bgr_img = cv2.cvtColor(fill_lab_img, cv2.COLOR_LAB2BGR)
        quantized_bgr[unsegmented] = fill_bgr_img[:, 0, :]

    quantized_bgr[silhouette == 0] = 255

    min_area = config.get("min_region_area", 80)
    quantized_bgr = _fill_small_holes(quantized_bgr, silhouette, min_area)

    return quantized_bgr


def _merge_superpixels(lab: np.ndarray, segments: np.ndarray, mask: np.ndarray,
                       delta_e_thresh: float = 7.0) -> tuple[np.ndarray, np.ndarray]:
    unique_segs = sorted([s for s in np.unique(segments) if s >= 0])
    n = len(unique_segs)
    seg_to_idx = {s: i for i, s in enumerate(unique_segs)}

    means = np.zeros((n, 3), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    for seg_id in unique_segs:
        idx = seg_to_idx[seg_id]
        sp_mask = (segments == seg_id) & (mask > 0)
        count = np.sum(sp_mask)
        if count > 0:
            means[idx] = np.mean(lab[sp_mask], axis=0)
            counts[idx] = count

    adjacency = _compute_adjacency_fast(segments, unique_segs, seg_to_idx, mask)

    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1

    for (i, j) in adjacency:
        if np.linalg.norm(means[i] - means[j]) < delta_e_thresh:
            union(i, j)

    root_map = {}
    new_id = 0
    merged_labels = np.full_like(segments, -1)
    for seg_id in unique_segs:
        idx = seg_to_idx[seg_id]
        root = find(idx)
        if root not in root_map:
            root_map[root] = new_id
            new_id += 1
        merged_labels[segments == seg_id] = root_map[root]

    group_means = np.zeros((new_id, 3), dtype=np.float64)
    group_counts = np.zeros(new_id, dtype=np.float64)
    for idx in range(n):
        root = find(idx)
        gid = root_map[root]
        group_means[gid] += means[idx] * counts[idx]
        group_counts[gid] += counts[idx]

    nonzero = group_counts > 0
    group_means[nonzero] /= group_counts[nonzero, np.newaxis]

    return merged_labels, group_means


def _compute_adjacency_fast(segments: np.ndarray, unique_segs: list,
                            seg_to_idx: dict, mask: np.ndarray) -> set:
    adjacency = set()
    h, w = segments.shape

    for dy, dx in [(0, 1), (1, 0)]:
        y1, y2 = 0, h - dy
        x1, x2 = 0, w - dx
        s1 = segments[y1:y2, x1:x2]
        s2 = segments[dy:dy + y2, dx:dx + x2]
        m1 = mask[y1:y2, x1:x2]
        m2 = mask[dy:dy + y2, dx:dx + x2]

        diff_mask = (s1 != s2) & (m1 > 0) & (m2 > 0)
        pairs_s1 = s1[diff_mask]
        pairs_s2 = s2[diff_mask]

        if len(pairs_s1) > 0:
            stacked = np.column_stack([pairs_s1, pairs_s2])
            stacked.sort(axis=1)
            unique_pairs = np.unique(stacked, axis=0)
            for a, b in unique_pairs:
                if a in seg_to_idx and b in seg_to_idx:
                    adjacency.add((seg_to_idx[a], seg_to_idx[b]))

    return adjacency


def _fill_small_holes(img: np.ndarray, mask: np.ndarray, min_area: int) -> np.ndarray:
    result = img.copy()
    fg_mask = (mask > 0) & ~np.all(img == 255, axis=2)
    hole_mask = (mask > 0) & np.all(img == 255, axis=2)
    hole_mask_u8 = hole_mask.astype(np.uint8) * 255

    if np.sum(hole_mask) == 0:
        return result

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hole_mask_u8, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area * 5:
            region = labels == i
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated = cv2.dilate(region.astype(np.uint8), kernel, iterations=2)
            neighbor = dilated.astype(bool) & fg_mask & ~region
            if np.sum(neighbor) > 0:
                fill_color = np.median(img[neighbor], axis=0).astype(np.uint8)
                result[region] = fill_color

    return result
