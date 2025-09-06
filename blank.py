# app.py
from flask import Flask, request, jsonify
import os
import math

app = Flask(__name__)
app.url_map.strict_slashes = False

# ---------------------------------------
# Savitzky–Golay (7-point, cubic) kernel
# y_smooth[i] = sum_k c[k] * y[i + k - 3], with reflection padding
# Coeffs sum to 1; classic (7,3) smoothing kernel:
SG_COEFFS = [-2/21, 3/21, 6/21, 7/21, 6/21, 3/21, -2/21]
HALF_WIN = 3  # (7-1)//2

def _reflect_index(j, n):
    # Mirror-reflection padding: ... 2 1 0 | 0 1 2 3 ... n-2 n-1 | n-1 n-2 ...
    while j < 0 or j >= n:
        if j < 0:
            j = -j - 1
        if j >= n:
            j = 2*n - j - 1
    return j

def sg_smooth(arr):
    n = len(arr)
    out = [0.0] * n
    for i in range(n):
        s = 0.0
        for k, c in enumerate(SG_COEFFS):
            j = i + (k - HALF_WIN)
            j = _reflect_index(j, n)
            s += c * arr[j]
        out[i] = s
    return out

# ---------------------------------------
# Linear interpolation with edge fill
def linear_fill(series):
    n = len(series)
    vals = series[:]  # copy (floats or None)
    # Find indices of known points
    known_idx = [i for i, v in enumerate(vals) if v is not None]
    if not known_idx:
        # No information—return zeros (safe fallback)
        return [0.0] * n
    # Forward fill from first known
    first = known_idx[0]
    for i in range(0, first):
        vals[i] = float(vals[first])
    # Backward fill from last known
    last = known_idx[-1]
    for i in range(last + 1, n):
        vals[i] = float(vals[last])
    # Linear interpolate between consecutive known points
    for a, b in zip(known_idx, known_idx[1:]):
        ya = float(vals[a])
        yb = float(vals[b])
        span = b - a
        if span > 1:
            step = (yb - ya) / span
            for t in range(1, span):
                vals[a + t] = ya + step * t
    # Ensure finite numbers
    for i, v in enumerate(vals):
        if v is None or not math.isfinite(v):
            vals[i] = 0.0
    return vals

# ---------------------------------------
# Clamp / sanitize numeric output
def sanitize_list(xs):
    out = []
    for v in xs:
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
            out.append(0.0)
        else:
            out.append(float(v))
    return out

# ---------------------------------------
@app.post("/blankety")
def blankety():
    payload = request.get_json(force=True, silent=True)
    if not isinstance(payload, dict) or "series" not in payload:
        return jsonify({"error": "Invalid body. Expected {'series': [[...], ...]}"}), 400

    series = payload["series"]
    if (not isinstance(series, list) or
        any(not isinstance(lst, list) for lst in series)):
        return jsonify({"error": "'series' must be a list of lists"}), 400

    answer = []
    for lst in series:
        # Snapshot which positions were missing
        missing_mask = [v is None for v in lst]

        # 1) Base fill via linear interpolation (+ edge fill)
        base = linear_fill(lst)

        # 2) Smooth baseline (SG 7,3) to capture local polynomial/periodic structure
        smooth = sg_smooth(base)

        # 3) Construct final: keep observed values; replace only original nulls
        final = []
        for was_missing, orig, sm in zip(missing_mask, lst, smooth):
            if was_missing:
                # Guard against tiny numerical noise & extremes
                y = float(sm)
                if not math.isfinite(y):
                    y = 0.0
                final.append(y)
            else:
                final.append(float(orig))

        # 4) Final sanitize
        final = sanitize_list(final)
        answer.append(final)

    return jsonify({"answer": answer}), 200

# ---------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Single-process is fine; no shared state.
    app.run(host="0.0.0.0", port=port)
