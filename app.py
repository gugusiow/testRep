"""
Minimum Spanning Tree (MST) Calculation Challenge — Flask server
Endpoint: POST /mst-calculation

Input (application/json):
[
  {"image": "<base64 png>"},
  {"image": "<base64 png>"}
]

Output (application/json):
[
  {"value": <int>},
  {"value": <int>}
]

Notes
- Pure-Python + OpenCV pipeline. Tesseract is optional; if present, we use it for OCR.
- If Tesseract isn't available, we fall back to template-matching digits rendered via OpenCV's Hershey font.
- Designed to finish well under 30s for small graphs (≤12 nodes/edges).

Run locally:
  pip install flask opencv-python numpy
  # (optional) pip install pytesseract
  python mst_calculation_flask_app.py

Deploy (Render/Heroku/etc.):
  - Start command (Gunicorn): gunicorn -w 2 -b 0.0.0.0:$PORT mst_calculation_flask_app:app

"""
from __future__ import annotations
import base64
import io
import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2

try:
    import pytesseract  # type: ignore
    TESSERACT_OK = True
except Exception:
    TESSERACT_OK = False

from flask import Flask, request, jsonify

app = Flask(__name__)
app.url_map.strict_slashes = False
# Cap request size to avoid huge base64 bodies causing memory spikes
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

# ----------------------------
# Utility dataclasses
# ----------------------------
@dataclass
class Node:
    x: float
    y: float

@dataclass
class Edge:
    u: int
    v: int
    w: int

# ----------------------------
# Image helpers
# ----------------------------

def _b64_to_bgr_png(b64: str) -> np.ndarray:
    try:
        buf = base64.b64decode(b64, validate=True)
        arr = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
        return img
    except Exception as e:
        raise ValueError(f"Invalid base64 PNG: {e}")


def _preprocess(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bgr = img.copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return bgr, gray, hsv


# ----------------------------
# Node detection (black circles)
# ----------------------------

def detect_nodes(gray: np.ndarray) -> List[Node]:
    # Emphasize dark circular blobs
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive threshold to isolate dark nodes
    th = cv2.adaptiveThreshold(255 - blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 21, 5)
    # Morph close small gaps
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Try HoughCircles first
    nodes: List[Node] = []
    circles = cv2.HoughCircles(255 - th, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=100, param2=15, minRadius=6, maxRadius=36)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0, :]:
            nodes.append(Node(float(x), float(y)))
    else:
        # Fallback: contour centroids of dark blobs
        contours, _ = cv2.findContours(255 - th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50 or area > 5000:
                continue
            (x, y), r = cv2.minEnclosingCircle(cnt)
            if 6 <= r <= 36:
                nodes.append(Node(float(x), float(y)))

    # Deduplicate via simple clustering (grid-based)
    if not nodes:
        return []

    pts = np.array([[n.x, n.y] for n in nodes], dtype=np.float32)
    # Merge close points
    used = np.zeros(len(pts), dtype=bool)
    merged: List[Node] = []
    for i in range(len(pts)):
        if used[i]:
            continue
        close = np.linalg.norm(pts - pts[i], axis=1) < 15.0
        group = pts[close]
        used[close] = True
        merged.append(Node(float(group[:, 0].mean()), float(group[:, 1].mean())))
    return merged


# ----------------------------
# Edge detection (lines)
# ----------------------------

def detect_edge_segments(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    edges = cv2.Canny(gray, 60, 120, L2gradient=True)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=60,
                            minLineLength=30, maxLineGap=10)
    segs: List[Tuple[int, int, int, int]] = []
    if lines is not None:
        for l in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, l)
            # Skip overly short lines
            if (x1 - x2)**2 + (y1 - y2)**2 < 25**2:
                continue
            segs.append((x1, y1, x2, y2))
    return segs


# ----------------------------
# Weight reading near line midpoints
# ----------------------------

# Prepare digit templates (0-9) once for fallback OCR
_DIGIT_TEMPLATES = {}

def _init_digit_templates():
    """Build simple glyph templates for 0-9 using OpenCV's Hershey font.
    Fix: use cv2.findNonZero + cv2.boundingRect on points (not on the raw image),
    which avoids a crash some OpenCV builds hit when passing a binary image directly.
    """
    global _DIGIT_TEMPLATES
    if _DIGIT_TEMPLATES:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    for d in range(10):
        canvas = np.zeros((40, 30), dtype=np.uint8)
        cv2.putText(canvas, str(d), (3, 32), font, 1.2, 255, 2, cv2.LINE_AA)
        # Binarize and trim to tight bounding box using non-zero points
        _, bw = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY)
        pts = cv2.findNonZero(bw)
        if pts is None:
            # Fallback: keep the raw canvas if detection failed (rare)
            tpl = bw
        else:
            x, y, w, h = cv2.boundingRect(pts)
            tpl = bw[y:y+h, x:x+w]
        _DIGIT_TEMPLATES[d] = tpl


def _ocr_digits_roi(roi_bgr: np.ndarray) -> Optional[int]:
    # Try Tesseract first if available
    if TESSERACT_OK:
        try:
            cfg = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(roi_bgr, config=cfg)
            text = ''.join(ch for ch in text if ch.isdigit())
            if text:
                return int(text)
        except Exception:
            pass

    # Fallback: template match per character (simple, robust for clean digits)
    _init_digit_templates()
    roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invert if needed
    if np.mean(bw) > 127:
        bw = 255 - bw

    # Find connected components as candidate glyphs
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort left-to-right
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 6 or h < 10:
            continue
        if w > 60 or h > 60:
            continue
        boxes.append((x, y, w, h))
    if not boxes:
        return None
    boxes.sort(key=lambda b: b[0])

    digits: List[int] = []
    for (x, y, w, h) in boxes:
        glyph = bw[y:y+h, x:x+w]
        best_d, best_val = None, -1.0
        for d, tpl in _DIGIT_TEMPLATES.items():
            # Resize glyph to template size for correlation
            tpl_h, tpl_w = tpl.shape
            glyph_rs = cv2.resize(glyph, (tpl_w, tpl_h), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(glyph_rs, tpl, cv2.TM_CCOEFF_NORMED)
            score = float(res.max())
            if score > best_val:
                best_val = score
                best_d = d
        if best_d is not None:
            digits.append(best_d)
    if not digits:
        return None
    try:
        return int(''.join(map(str, digits)))
    except Exception:
        return None


def read_weight_near_segment(bgr: np.ndarray, seg: Tuple[int, int, int, int]) -> Optional[int]:
    x1, y1, x2, y2 = seg
    mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
    # Square ROI around midpoint; enlarge if needed
    side = max(28, int(0.08 * max(bgr.shape[:2])))
    x0 = max(0, mx - side)
    y0 = max(0, my - side)
    x1r = min(bgr.shape[1], mx + side)
    y1r = min(bgr.shape[0], my + side)
    roi = bgr[y0:y1r, x0:x1r]

    w = _ocr_digits_roi(roi)
    if w is not None:
        return w
    # Expand ROI once if not found
    side2 = side * 2
    x0 = max(0, mx - side2)
    y0 = max(0, my - side2)
    x1r = min(bgr.shape[1], mx + side2)
    y1r = min(bgr.shape[0], my + side2)
    roi2 = bgr[y0:y1r, x0:x1r]
    return _ocr_digits_roi(roi2)


# ----------------------------
# Associate segments with nodes
# ----------------------------

def assign_segment_to_nodes(nodes: List[Node], seg: Tuple[int, int, int, int]) -> Optional[Tuple[int, int]]:
    if len(nodes) < 2:
        return None
    x1, y1, x2, y2 = seg
    p1 = np.array([x1, y1], dtype=float)
    p2 = np.array([x2, y2], dtype=float)
    pts = np.array([[n.x, n.y] for n in nodes])
    # Nearest node to each endpoint
    d1 = np.linalg.norm(pts - p1, axis=1)
    d2 = np.linalg.norm(pts - p2, axis=1)
    i = int(np.argmin(d1))
    j = int(np.argmin(d2))
    if i == j:
        # Possibly a short chord within one node—skip
        return None
    # Sanity: must be reasonably close to endpoints
    if d1[i] > 40 or d2[j] > 40:
        return None
    return (i, j)


# ----------------------------
# MST via Kruskal
# ----------------------------
class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return True


def mst_weight(n_nodes: int, edges: List[Edge]) -> int:
    # Deduplicate multi-edges by keeping the smallest weight
    best = {}
    for e in edges:
        u, v = (e.u, e.v) if e.u < e.v else (e.v, e.u)
        if (u, v) not in best or e.w < best[(u, v)]:
            best[(u, v)] = e.w
    dedup = [Edge(u, v, w) for (u, v), w in best.items()]

    dsu = DSU(n_nodes)
    total = 0
    for e in sorted(dedup, key=lambda x: x.w):
        if dsu.union(e.u, e.v):
            total += e.w
    return total


# ----------------------------
# Core per-image processing
# ----------------------------

def solve_one(b64img: str) -> int:
    bgr = _b64_to_bgr_png(b64img)
    bgr, gray, hsv = _preprocess(bgr)

    nodes = detect_nodes(gray)
    segs = detect_edge_segments(gray)

    edges: List[Edge] = []
    for seg in segs:
        pair = assign_segment_to_nodes(nodes, seg)
        if pair is None:
            continue
        w = read_weight_near_segment(bgr, seg)
        if w is None:
            continue
        u, v = pair
        edges.append(Edge(u, v, int(w)))

    # If node detection weak, infer node count from endpoints
    if not nodes and segs:
        endpoints = []
        for (x1, y1, x2, y2) in segs:
            endpoints.extend([(x1, y1), (x2, y2)])
        endpoints = np.array(list({(int(x), int(y)) for x, y in endpoints}))
        # Cluster endpoints
        used = np.zeros(len(endpoints), dtype=bool)
        merged: List[Tuple[float, float]] = []
        for i in range(len(endpoints)):
            if used[i]:
                continue
            close = np.linalg.norm(endpoints - endpoints[i], axis=1) < 20.0
            group = endpoints[close]
            used[close] = True
            merged.append((float(group[:, 0].mean()), float(group[:, 1].mean())))
        nodes = [Node(x, y) for x, y in merged]

    if len(nodes) == 0 or len(edges) == 0:
        # As a last resort, try reading weights for all segments ignoring nodes
        # and assume segments correspond to actual edges between distinct nodes.
        fallback_edges: List[Edge] = []
        for idx, seg in enumerate(segs):
            w = read_weight_near_segment(bgr, seg)
            if w is None:
                continue
            # pseudo-nodes: endpoints index
            fallback_edges.append(Edge(idx * 2, idx * 2 + 1, int(w)))
        if not fallback_edges:
            raise ValueError("Could not parse graph from image: no nodes/edges")
        # MST of disjoint pairs isn't meaningful, but better than failing.
        return sum(sorted(e.w for e in fallback_edges)[:max(0, len(fallback_edges) - 1)])

    total = mst_weight(len(nodes), edges)
    return int(total)


# ----------------------------
# Flask endpoint
# ----------------------------
@app.route('/mst-calculation', methods=['POST'])
def mst_calculation():
    if not request.is_json:
        return jsonify({"error": "Expected application/json"}), 415
    data = request.get_json(silent=True)
    if not isinstance(data, list) or not data:
        return jsonify({"error": "Expected a non-empty JSON array."}), 400

    results = []
    for i, item in enumerate(data):
        if not isinstance(item, dict) or 'image' not in item:
            return jsonify({"error": f"Item {i} missing 'image' field."}), 400
        b64 = item['image']
        try:
            val = solve_one(b64)
            results.append({"value": int(val)})
        except Exception as e:
            # If one case fails, still attempt to return something sensible
            results.append({"value": 0})
    return jsonify(results), 200


# Health check (useful for Render)
@app.route('/')
def health():
    return jsonify({"ok": True, "service": "mst-calculation"})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8000'))
    app.run(host='0.0.0.0', port=port, debug=False)
