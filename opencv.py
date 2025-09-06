"""
Minimum Spanning Tree (MST) Calculation Challenge — Flask server
Endpoint: POST /mst-calculation

Input (application/json or text/plain):
[
  {"image": "<base64 png>"},
  {"image": "<base64 png>"}
]

Output:
[
  {"value": <int>},
  {"value": <int>}
]
"""
from __future__ import annotations
import base64
import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
from flask import Flask, request, jsonify

# Optional OCR libraries
try:
    import easyocr
    EASYOCR_READER = easyocr.Reader(['en'], gpu=False)
    EASYOCR_OK = True
except Exception:
    EASYOCR_READER = None
    EASYOCR_OK = False

app = Flask(__name__)
app.url_map.strict_slashes = False
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB cap

# ----------------------------
# Data classes
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
# Decoding & preprocessing
# ----------------------------
def _b64_to_bgr_png(b64: str) -> np.ndarray:
    buf = base64.b64decode(b64, validate=True)
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode returned None")
    return img

def _preprocess(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bgr = img
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return bgr, gray, hsv

# ----------------------------
# Node detection (black circles)
# ----------------------------
def detect_nodes(gray: np.ndarray) -> List[Node]:
    nodes: List[Node] = []

    # Hough circles (fast path)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.0, minDist=25,
                               param1=50, param2=30, minRadius=8, maxRadius=50)
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        for x, y, r in circles:
            nodes.append(Node(float(x), float(y)))

    # Dark blob fallback
    _, b1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    b2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 15, 8)
    combined = cv2.bitwise_or(b1, b2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)

    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 200 or area > 8000:
            continue
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circ = 4 * math.pi * area / (peri * peri)
        if circ < 0.3:
            continue
        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        nodes.append(Node(float(cx), float(cy)))

    if not nodes:
        return []

    # Dedup nearby detections
    pts = np.array([[n.x, n.y] for n in nodes], dtype=np.float32)
    used = np.zeros(len(pts), dtype=bool)
    merged: List[Node] = []
    for i in range(len(pts)):
        if used[i]:
            continue
        close = np.linalg.norm(pts - pts[i], axis=1) < 25.0
        group = pts[close]
        used[close] = True
        merged.append(Node(float(group[:,0].mean()), float(group[:,1].mean())))
    return merged

# ----------------------------
# Edge detection (prefer colored strokes)
# ----------------------------
def color_edge_mask(hsv: np.ndarray) -> np.ndarray:
    # Keep colored (non-dark) pixels: S high, V not too low
    mask = cv2.inRange(hsv, (0, 40, 60), (180, 255, 255))
    mask = cv2.medianBlur(mask, 3)
    return mask

def detect_edge_segments(bgr: np.ndarray, gray: np.ndarray, hsv: np.ndarray) -> List[Tuple[int,int,int,int]]:
    # Edges from colored mask (preferred) + grayscale (fallback), then union
    colmask = color_edge_mask(hsv)
    edges_col = cv2.Canny(colmask, 80, 160)
    edges_gray = cv2.Canny(cv2.GaussianBlur(gray,(3,3),0), 50, 150, apertureSize=3, L2gradient=True)
    edges = cv2.bitwise_or(edges_col, edges_gray)

    segs: List[Tuple[int,int,int,int]] = []
    def add_lines(lines, min_len=20):
        nonlocal segs
        if lines is None:
            return
        for (x1,y1,x2,y2) in lines[:,0]:
            length = math.hypot(x2-x1, y2-y1)
            if length >= min_len:
                seg = (int(x1), int(y1), int(x2), int(y2))
                # avoid near-duplicates
                dup = False
                for ex in segs:
                    if (abs(seg[0]-ex[0])<5 and abs(seg[1]-ex[1])<5 and
                        abs(seg[2]-ex[2])<5 and abs(seg[3]-ex[3])<5):
                        dup = True; break
                if not dup:
                    segs.append(seg)

    add_lines(cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=25, maxLineGap=15))
    if len(segs) < 3:
        add_lines(cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=15, maxLineGap=20), min_len=15)
    return segs

# ----------------------------
# OCR helpers (digits)
# ----------------------------
_DIGIT_TEMPLATES: Dict[int, np.ndarray] = {}

def _init_digit_templates():
    if _DIGIT_TEMPLATES:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    for d in range(10):
        canvas = np.zeros((40,30), dtype=np.uint8)
        cv2.putText(canvas, str(d), (3,32), font, 1.2, 255, 2, cv2.LINE_AA)
        _, bw = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY)
        pts = cv2.findNonZero(bw)
        if pts is None:
            tpl = bw
        else:
            x,y,w,h = cv2.boundingRect(pts)
            tpl = bw[y:y+h, x:x+w]
        _DIGIT_TEMPLATES[d] = tpl

def _ocr_digits_roi(roi_bgr: np.ndarray) -> Optional[int]:
    # Try EasyOCR if available
    if EASYOCR_OK and EASYOCR_READER is not None:
        try:
            # EasyOCR works better with grayscale for digits
            roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            
            # Try multiple preprocessing approaches
            variants = [roi_gray]
            
            # Enhance contrast
            enhanced = cv2.equalizeHist(roi_gray)
            variants.append(enhanced)
            
            # Threshold
            _, thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(thresh)
            
            for variant in variants:
                try:
                    results = EASYOCR_READER.readtext(variant, allowlist='0123456789', width_ths=0.7, height_ths=0.7)
                    for (bbox, text, confidence) in results:
                        if confidence > 0.5:  # EasyOCR confidence threshold
                            # Extract digits only
                            digits = "".join(ch for ch in text if ch.isdigit())
                            if digits:
                                val = int(digits)
                                if 1 <= val <= 999:
                                    return val
                except Exception:
                    continue
                    
        except Exception:
            pass

    # Template OCR fallback (existing code)
    _init_digit_templates()
    roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    variants = []
    # Otsu, adaptive, equalized
    _, bw1 = cv2.threshold(cv2.GaussianBlur(roi,(3,3),0), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    variants.append(bw1)
    variants.append(cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,8))
    _, bw3 = cv2.threshold(cv2.equalizeHist(roi), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    variants.append(bw3)

    for bw in variants:
        if np.mean(bw) > 127:  # invert to make digits white on black
            bw = 255 - bw
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if w < 4 or h < 8 or w > 80 or h > 80: 
                continue
            if h < 0.8*w: 
                continue
            boxes.append((x,y,w,h))
        if not boxes:
            continue
        boxes.sort(key=lambda b: b[0])
        digits = []
        for (x,y,w,h) in boxes:
            glyph = bw[y:y+h, x:x+w]
            best_d, best_val = None, -1.0
            for d, tpl in _DIGIT_TEMPLATES.items():
                th, tw = tpl.shape
                glyph_rs = cv2.resize(glyph, (tw, th), interpolation=cv2.INTER_AREA)
                score = float(cv2.matchTemplate(glyph_rs, tpl, cv2.TM_CCOEFF_NORMED).max())
                if score > best_val:
                    best_val, best_d = score, d
            if best_d is not None and best_val > 0.3:
                digits.append(best_d)
        if digits:
            try:
                val = int("".join(map(str, digits)))
                if 1 <= val <= 999:
                    return val
            except Exception:
                pass
    return None

def read_weight_near_segment(bgr: np.ndarray, hsv: np.ndarray, seg: Tuple[int,int,int,int]) -> Optional[int]:
    x1,y1,x2,y2 = seg
    mx, my = int((x1+x2)/2), int((y1+y2)/2)

    # Sample hue around midpoint to focus digits of same color as edge
    pad = 12
    H,W = hsv.shape[:2]
    x0, y0 = max(0, mx-pad), max(0, my-pad)
    x1r, y1r = min(W, mx+pad), min(H, my+pad)
    mid_patch = hsv[y0:y1r, x0:x1r]
    local_mask = None
    if mid_patch.size:
        S = mid_patch[:,:,1].reshape(-1)
        Hs = mid_patch[:,:,0].reshape(-1)
        Hs = Hs[S > 40]
        if Hs.size:
            hue = int(np.median(Hs))
            lower = (max(0, hue-10), 40, 60)
            upper = (min(180, hue+10), 255, 255)
            local_mask = cv2.inRange(hsv, lower, upper)

    # ROIs (center & perpendicular offsets), multi sizes
    base_side = max(35, int(0.12 * max(bgr.shape[:2])))
    dx, dy = x2-x1, y2-y1
    length = math.hypot(dx, dy)
    offs = []
    offs.append((mx, my))
    if length > 0:
        px, py = -dy/length, dx/length
        offs.append((int(mx+15*px), int(my+15*py)))
        offs.append((int(mx-15*px), int(my-15*py)))

    for mul in [1.0, 1.5, 2.0]:
        side = int(base_side * mul)
        for cx, cy in offs:
            x0, y0 = max(0, cx-side), max(0, cy-side)
            x1b, y1b = min(bgr.shape[1], cx+side), min(bgr.shape[0], cy+side)
            if x1b - x0 < 20 or y1b - y0 < 20:
                continue
            roi = bgr[y0:y1b, x0:x1b].copy()
            if local_mask is not None:
                mask_roi = local_mask[y0:y1b, x0:x1b]
                roi[mask_roi == 0] = (255,255,255)
            w = _ocr_digits_roi(roi)
            if w is not None and w > 0:
                return w
    return None

# ----------------------------
# Segment→nodes association
# ----------------------------
def assign_segment_to_nodes(nodes: List[Node], seg: Tuple[int,int,int,int]) -> Optional[Tuple[int,int]]:
    if len(nodes) < 2:
        return None
    x1,y1,x2,y2 = seg
    p1 = np.array([x1,y1], dtype=float)
    p2 = np.array([x2,y2], dtype=float)
    pts = np.array([[n.x, n.y] for n in nodes])

    d1 = np.linalg.norm(pts - p1, axis=1)
    d2 = np.linalg.norm(pts - p2, axis=1)
    i = int(np.argmin(d1))
    j = int(np.argmin(d2))
    if i == j:
        # try second nearest on one side
        di = np.argsort(d1)
        dj = np.argsort(d2)
        if len(di) > 1 and d1[di[1]] < 50: i = int(di[1])
        elif len(dj) > 1 and d2[dj[1]] < 50: j = int(dj[1])
        if i == j:
            return None
    # relaxed distance gate
    if d1[i] > 60 or d2[j] > 60:
        return None
    u, v = (i, j) if i < j else (j, i)
    return (u, v)

# ----------------------------
# MST (Kruskal)
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
    # dedupe multi-edges by min weight
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
    return int(total)

# ----------------------------
# Per-image solver
# ----------------------------
def solve_one(b64img: str) -> int:
    bgr = _b64_to_bgr_png(b64img)
    bgr, gray, hsv = _preprocess(bgr)

    nodes = detect_nodes(gray)
    segs = detect_edge_segments(bgr, gray, hsv)

    edges: List[Edge] = []
    for seg in segs:
        pair = assign_segment_to_nodes(nodes, seg)
        if pair is None:
            continue
        w = read_weight_near_segment(bgr, hsv, seg)
        if w is None:
            continue
        u, v = pair
        edges.append(Edge(u, v, int(w)))

    # Endpoint-cluster fallback if no nodes but lines exist
    if not nodes and segs:
        endpoints = []
        for (x1,y1,x2,y2) in segs:
            endpoints.extend([(x1,y1),(x2,y2)])
        pts = np.array(list({(int(x),int(y)) for x,y in endpoints}))
        used = np.zeros(len(pts), dtype=bool)
        merged = []
        for i in range(len(pts)):
            if used[i]: continue
            close = np.linalg.norm(pts - pts[i], axis=1) < 20.0
            group = pts[close]; used[close] = True
            merged.append((float(group[:,0].mean()), float(group[:,1].mean())))
        nodes = [Node(x,y) for x,y in merged]

    if len(nodes) == 0 or len(edges) == 0:
        # keep output aligned; return 0 for this case
        return 0

    return mst_weight(len(nodes), edges)

# ----------------------------
# Robust request parsing
# ----------------------------
def _coerce_to_array():
    data = request.get_json(silent=True)
    raw = (request.data or b"").decode("utf-8", "ignore").strip()
    if data is None and raw:
        try:
            data = json.loads(raw)
        except Exception:
            # NDJSON fallback (one JSON obj per line)
            objs = []
            for line in raw.splitlines():
                line = line.strip()
                if not line: continue
                try:
                    objs.append(json.loads(line))
                except Exception:
                    objs = []
                    break
            data = objs if objs else None

    # Accept array or object-wrapped arrays
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("tests","data","input","cases"):
            v = data.get(k)
            if isinstance(v, list):
                return v
    return []

# ----------------------------
# Flask endpoints
# ----------------------------
@app.route('/mst-calculation', methods=['POST'])
def mst_calculation():
    items = _coerce_to_array()
    results = []
    for i, item in enumerate(items):
        b64 = item.get("image") if isinstance(item, dict) else None
        try:
            val = solve_one(b64) if b64 else 0
        except Exception:
            val = 0
        results.append({"value": int(val)})
    return jsonify(results), 200

# Some judges POST to the base URL
@app.route('/', methods=['POST'])
def mst_calculation_alias():
    return mst_calculation()

# Health check
@app.route('/', methods=['GET'])
def health():
    return jsonify({"ok": True, "service": "mst-calculation"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8000'))
    app.run(host='0.0.0.0', port=port, debug=False)
