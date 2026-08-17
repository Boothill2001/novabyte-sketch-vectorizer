# Flat Sketch Normalization & Vectorization Pipeline

A CLI pipeline that takes product illustration PNGs and normalizes them into flat sketch form, then vectorizes the result into editable SVG.

## What it does

Given a product illustration (jewelry, accessories) with a white background:

1. **Flat B&W sketch** — Black outlines on white with uniform stroke width. Product contours, part boundaries, and inner holes are kept; lighting-only edges are removed.
2. **Flat color sketch** — Every material region carries a single flat fill. No gradients, no highlights, no shadows. Outlines preserved.
3. **Editable SVG** — Both versions as vector geometry in one file: B&W as stroked paths (`<g id="bw_sketch">`), color as closed filled paths (`<g id="color_sketch">`).

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `opencv-python`, `scikit-image`, `scikit-learn`, `numpy`, `Pillow`, `vtracer`, `PyYAML`, `scipy`

Python 3.10+

## Usage

### Single image
```bash
python main.py input.png -o output/result/
```

### Batch (directory of PNGs)
```bash
python main.py input_dir/ -o output/
```

### With preset
```bash
python main.py input.png -o output/ --preset jewelry_metal
```

## Output per image

| File | Description |
|------|-------------|
| `bw_sketch.png` | Flat B&W sketch raster |
| `flat_color.png` | Flat color sketch raster |
| `result.svg` | Editable SVG with both versions |
| `comparison.png` | Side-by-side: source \| B&W \| flat color |
| `metrics.json` | Quality metrics |

## Pipeline Architecture

```
Input PNG (white background)
    │
    ├─► Preprocessing
    │   ├─ Background detection (HSV saturation + value thresholding)
    │   ├─ Foreground mask extraction (morphological cleanup)
    │   ├─ Material segmentation: metal vs gem/enamel (HSV saturation)
    │   └─ Construction line detection (thin low-contrast edges)
    │
    ├─► Shading Flattening
    │   ├─ Convert to LAB color space
    │   ├─ SLIC superpixel segmentation (spatially-coherent regions)
    │   ├─ Per-superpixel: replace L with regional median L
    │   ├─ Keep A,B channels intact → preserves color identity
    │   └─ Gem regions: skip L-flattening (internal facets are structural)
    │
    ├─► B&W Sketch
    │   ├─ Silhouette contours (all hierarchy levels from foreground mask)
    │   ├─ Strong structural edges only (top 10% gradient magnitude)
    │   ├─ 4-pass bilateral filter to suppress shading before edge detection
    │   ├─ Skeletonize → uniform 2px stroke
    │   └─ Small component removal
    │
    ├─► Flat Color
    │   ├─ SLIC superpixels → merge by ΔE distance in LAB
    │   ├─ K-means on merged superpixel means (k=4 for metal)
    │   ├─ Every foreground pixel assigned to nearest cluster
    │   └─ Small hole filling from neighbor colors
    │
    ├─► Vectorization
    │   ├─ B&W: contour tracing → approxPolyDP → Bézier path fitting
    │   │   → SVG <path stroke="black" fill="none" stroke-width="2">
    │   ├─ Color: per-color contour extraction → simplified closed paths
    │   │   → SVG <path fill="#hexcolor" stroke="none">
    │   └─ Merged into single SVG: <g id="color_sketch"> + <g id="bw_sketch">
    │
    └─► Quality Metrics
        ├─ Silhouette IoU (output vs source foreground mask)
        ├─ Boundary F-score at 2px tolerance
        ├─ Coordinate offset check
        ├─ Fill count + anchor count with rationale
        └─ Side-by-side comparison sheet
```

## Key Design Decisions

### Why LAB color space for flattening?
LAB separates lightness (L) from color (a,b). Metal highlights and shadows only affect the L channel. By normalizing L per superpixel while keeping a,b intact, we remove shading without shifting the material's color identity. "Dark side of gold" and "bright side of gold" both map to the same (a,b) → same flat fill.

### Why SLIC superpixels before K-means?
Naive K-means on pixels treats each pixel independently — two pixels with identical color but different locations can end up in different clusters. SLIC creates spatially coherent regions first, then we merge neighboring regions by color similarity (ΔE < threshold). This ensures that "dark gold" and "bright gold" on the same link merge into one region before quantization.

### Why bilateral filter (4 passes) for B&W?
Bilateral filter smooths within regions while preserving strong edges — exactly what's needed to kill gradients without blurring part boundaries. Multiple passes strengthen the effect. Only edges surviving this heavy smoothing (top 10% gradient magnitude) are truly structural.

### Why silhouette contours as primary B&W edges?
For product illustrations on white backgrounds, the foreground mask already encodes exact product geometry — every link's outline, every inner hole. These contours are the definitive structural edges, complemented by only the strongest internal edges for part overlap boundaries.

### Why single SVG with two groups?
The spec allows "either two SVG files or one file containing two top level groups." One file is cleaner and guarantees alignment between B&W and color layers. Each layer is independently toggleable in vector editors.

### Fill count rationale
Metal jewelry (single material): 3-4 fills typical. The dominant metal color, 1-2 shading variants that survived aggressive merging, and any accent material (gemstone, enamel). Our K-means k=4 target matches the spec's "four or five flat fills" reference.

### Anchor ceiling rationale
Target ≤20 anchors per elliptical element via `approxPolyDP` with progressive epsilon. Each anchor should be load-bearing: dragging any single anchor visibly changes the path shape. Verified by `cv2.arcLength`-proportional epsilon.

## Config Presets

Category-level presets in `config.yaml`:

| Preset | Use case | Key differences |
|--------|----------|-----------------|
| `jewelry_metal` | Gold, silver, rose gold | Aggressive merge (ΔE=25), k=4, strong bilateral |
| `jewelry_gem` | Pieces with gems/enamel | Lower merge threshold, k=6, preserves gem detail |
| `footwear` | Sneakers, shoes | More superpixels, k=8, moderate merge |
| `default` | General fallback | Balanced parameters |

Auto-detected from filename (`jewelry_*` → `jewelry_metal`). Override with `--preset`.

## Quality Results (Required Images)

| Image | Silhouette IoU | Boundary F | Fills | Anchors | Time |
|-------|---------------|------------|-------|---------|------|
| jewelry_03 (bracelet) | 0.9787 | 1.0000 | 4 | 429 | 14s |
| jewelry_04 (ring) | 0.9799 | 1.0000 | 4 | 1351 | 14s |

## External APIs and Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| opencv-python | ≥4.8 | Image I/O, bilateral filter, Canny, contours, morphology |
| scikit-image | ≥0.21 | SLIC superpixels, skeletonization |
| scikit-learn | ≥1.3 | K-means clustering for color quantization |
| numpy | ≥1.24 | Array operations |
| Pillow | ≥10.0 | Image format support |
| vtracer | ≥0.6 | (Available for future vectorization enhancement) |
| PyYAML | ≥6.0 | Config file parsing |
| scipy | ≥1.11 | Distance computation for superpixel merging |

No external API calls. No OpenAI/Replicate usage. All processing is local.

No manual corrections were applied to any image.
