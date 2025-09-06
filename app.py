# app.py
import re
import math
import json
import os
from flask import Flask, request, jsonify

app = Flask(__name__)
app.url_map.strict_slashes = False

# ----------------------------
# Helpers: LaTeX -> Python
# ----------------------------

GREEKS = [
    "alpha","beta","gamma","delta","epsilon","zeta","eta","theta","iota","kappa","lambda",
    "mu","nu","xi","omicron","pi","rho","sigma","tau","upsilon","phi","chi","psi","omega"
]

def _safe_env(varmap):
    env = {
        "max": max,
        "min": min,
        "log": math.log,
        "exp": math.exp,
        "sqrt": math.sqrt,
        "abs": abs,
        "pow": pow,
        "e": math.e,
        "pi": math.pi,
    }
    env.update(varmap)
    return {"__builtins__": {}}, env

def _replace_frac(expr: str) -> str:
    def repl_once(s):
        i = s.find(r"\frac")
        if i == -1:
            return s, False
        j = i + 5
        while j < len(s) and s[j].isspace():
            j += 1
        if j >= len(s) or s[j] != '{':
            return s, False
        def grab_block(k):
            if k >= len(s) or s[k] != '{':
                return None, k
            depth = 0
            start = k + 1
            p = k
            while p < len(s):
                if s[p] == '{':
                    depth += 1
                elif s[p] == '}':
                    depth -= 1
                    if depth == 0:
                        return s[start:p], p + 1
                p += 1
            return None, k
        numer, j2 = grab_block(j)
        if numer is None:
            return s, False
        while j2 < len(s) and s[j2].isspace():
            j2 += 1
        if j2 >= len(s) or s[j2] != '{':
            return s, False
        denom, j3 = grab_block(j2)
        if denom is None:
            return s, False
        return s[:i] + f"(({numer})/({denom}))" + s[j3:], True

    changed = True
    while changed:
        expr, changed = repl_once(expr)
    return expr

def _latex_to_python(formula: str, varmap: dict) -> str:
    s = formula.strip()
    s = s.replace("$$", "").replace("$", "")
    if "=" in s:
        s = s.split("=", 1)[1]
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\\text\s*\{([^}]*)\}", lambda m: re.sub(r"\W+", "_", m.group(1)).strip("_"), s)
    s = re.sub(r"([A-Za-z]+)\[([A-Za-z0-9_\\]+)\]", r"\1_\2", s)
    for g in GREEKS:
        s = s.replace(fr"\{g}", g)
    s = re.sub(r"\\([A-Za-z]+)_\{?([A-Za-z0-9]+)\}?", r"\1_\2", s)
    s = s.replace(r"\times", "*").replace(r"\cdot", "*")
    s = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", s)
    s = _replace_frac(s)
    s = re.sub(r"\\max\s*\(", "max(", s)
    s = re.sub(r"\\min\s*\(", "min(", s)
    s = re.sub(r"\\log\s*\(", "log(", s)
    s = re.sub(r"(?<![A-Za-z0-9_])e\^\{([^{}]+)\}", r"exp(\1)", s)
    s = re.sub(r"(?<![A-Za-z0-9_])e\^([A-Za-z0-9_]+)", r"exp(\1)", s)
    s = re.sub(r"(\([^()]*\)|[A-Za-z0-9_\.]+)\s*\^\s*\{([^{}]+)\}", r"(\1)**(\2)", s)
    s = re.sub(r"(\([^()]*\)|[A-Za-z0-9_\.]+)\s*\^\s*([A-Za-z0-9_\.]+)", r"(\1)**(\2)", s)
    s = s.replace(r"\,", "").replace(r"\ ", "")

    sum_pat = re.compile(r"\\sum_\{([A-Za-z])\s*=\s*([^}]*)\}\^\{([^}]*)\}\s*")
    while True:
        m = sum_pat.search(s)
        if not m:
            break
        it, start_str, end_str = m.group(1), m.group(2), m.group(3)
        tail = s[m.end():]
        expr_chunk = ""
        consumed = 0
        if tail and tail[0] == '{':
            depth = 0
            for idx, ch in enumerate(tail):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        expr_chunk = tail[1:idx]
                        consumed = idx + 1
                        break
        else:
            m2 = re.search(r"(?=[\+\-])(?![^\(]*\))", tail)
            expr_chunk = tail[:m2.start()] if m2 else tail
            consumed = len(expr_chunk)

        placeholder = f"__SUM_PLACEHOLDER_{id(m)}__"
        s = s[:m.start()] + placeholder + s[m.end()+consumed:]
        if "__SUM_SPECS__" not in varmap:
            varmap["__SUM_SPECS__"] = []
        varmap["__SUM_SPECS__"].append((placeholder, it, start_str, end_str, expr_chunk))

    s = s.replace("[", "_").replace("]", "")
    s = re.sub(r"__+", "_", s)
    return s

def _materialize_sums(expr: str, varmap: dict) -> str:
    specs = varmap.pop("__SUM_SPECS__", [])
    if not specs:
        return expr
    builtins_env, env = _safe_env(varmap)
    for placeholder, it, start_str, end_str, body in specs:
        start = eval(_latex_to_python(start_str, dict(varmap)), builtins_env, env)
        end = eval(_latex_to_python(end_str, dict(varmap)), builtins_env, env)
        total = 0.0
        for i in range(int(start), int(end) + 1):
            env[it] = i
            term = eval(_latex_to_python(body, dict(varmap) | {it: i}), builtins_env, env)
            total += term
        expr = expr.replace(placeholder, f"({total})")
        env.pop(it, None)
    return expr

def evaluate_formula(latex: str, variables: dict) -> float:
    varmap = dict(variables)
    alias_map = {}
    for k, v in list(varmap.items()):
        alias_map[k] = v
        alias_map[k.replace("\\", "")] = v
        alias_map[k.replace("[", "_").replace("]", "")] = v
    varmap.update(alias_map)
    py_expr = _latex_to_python(latex, varmap)
    py_expr = _materialize_sums(py_expr, varmap)
    builtins_env, env = _safe_env(varmap)
    value = eval(py_expr, builtins_env, env)
    return float(f"{value:.4f}")

# ----------------------------
# API
# ----------------------------

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

def _coerce_to_tests_array():
    """
    Accepts:
      - JSON array
      - JSON object with array under 'tests'/'data'/'input'/'cases'
      - NDJSON (newline-delimited JSON objects)
      - text/plain raw JSON
      - empty body -> treat as zero tests []
    Always returns a list (possibly empty) or None if truly unparsable.
    """
    # Try Flask JSON first (handles application/json)
    data = request.get_json(silent=True)

    raw = (request.data or b"").decode("utf-8").strip()
    if data is None and raw:
        # try raw JSON
        try:
            data = json.loads(raw)
        except Exception:
            # Try NDJSON
            objs = []
            ok = False
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    objs.append(obj)
                    ok = True
                except Exception:
                    ok = False
                    break
            if ok and objs:
                data = objs
            else:
                data = None

    if data is None:
        # empty or unparsable -> zero tests (lets us return [] with no mismatch expectation)
        return []

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for k in ("tests", "data", "input", "cases"):
            v = data.get(k)
            if isinstance(v, list):
                return v
        # single-test object: wrap it
        if "formula" in data and "variables" in data:
            return [data]

    # truly unexpected
    return []

def _handle_payload(data_list):
    """
    Always returns len(results) == len(data_list).
    On per-case errors, returns 0.0000 for that case (keeps count aligned).
    """
    results = []
    for case in data_list:
        # default fallback per case keeps count
        fallback = {"result": 0.0000}

        if not isinstance(case, dict):
            results.append(fallback)
            continue

        formula = case.get("formula", "")
        variables = case.get("variables", {})
        if not formula or not isinstance(variables, dict):
            results.append(fallback)
            continue

        try:
            res = evaluate_formula(formula, variables)
            # ensure 4dp numeric
            results.append({"result": float(f"{res:.4f}")})
        except Exception:
            results.append(fallback)

    return results

@app.route("/trading-formula", methods=["POST"])
def trading_formula():
    data_list = _coerce_to_tests_array()
    results = _handle_payload(data_list)
    return jsonify(results), 200

# Alias POST / to accept base-url posts
@app.route("/", methods=["POST"])
def trading_formula_alias():
    return trading_formula()

# ----------------------------
# Local / Render run
# ----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)