# app.py
import re
import math
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
    """Restricted eval environment."""
    env = {
        "max": max,
        "min": min,
        "log": math.log,
        "exp": math.exp,
        "sqrt": math.sqrt,
        "abs": abs,
        "pow": pow,
        # expose constants if ever needed
        "e": math.e,
        "pi": math.pi,
    }
    env.update(varmap)
    return {"__builtins__": {}}, env

def _replace_frac(expr: str) -> str:
    # Replace \frac{a}{b} with (a)/(b), supporting nesting by looping
    pattern = re.compile(r"\\frac\s*\{([^{}]+|\{[^{}]*\})+?\}\s*\{([^{}]+|\{[^{}]*\})+?\}")
    # To robustly handle nested braces, do a manual parse
    def repl_once(s):
        i = s.find(r"\frac")
        if i == -1:
            return s, False
        # parse from i
        j = i + 5
        # skip spaces
        while j < len(s) and s[j].isspace():
            j += 1
        # expect {numer}
        if j >= len(s) or s[j] != '{':
            return s, False
        # grab first { ... }
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
        # skip spaces
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

    # strip $$, $, and any leading identifier like "Fee ="
    s = s.replace("$$", "").replace("$", "")
    if "=" in s:
        # keep right-hand side only
        s = s.split("=", 1)[1]

    # remove \left and \right
    s = s.replace(r"\left", "").replace(r"\right", "")

    # \text{Var Name} -> VarName
    s = re.sub(r"\\text\s*\{([^}]*)\}", lambda m: re.sub(r"\W+", "_", m.group(1)).strip("_"), s)

    # Expectation-style names: E[R_m] -> E_R_m ; also nested brackets
    s = re.sub(r"([A-Za-z]+)\[([A-Za-z0-9_\\]+)\]", r"\1_\2", s)

    # Greek names in variables: \alpha -> alpha, etc.
    for g in GREEKS:
        s = s.replace(fr"\{g}", g)

    # Subscripts: \foo_{bar} -> foo_bar ; also simple \foo_bar
    s = re.sub(r"\\([A-Za-z]+)_\{?([A-Za-z0-9]+)\}?", r"\1_\2", s)

    # Products: \times, \cdot -> *
    s = s.replace(r"\times", "*").replace(r"\cdot", "*")

    # sqrt: \sqrt{...} -> sqrt(...)
    s = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", s)

    # frac: \frac{a}{b}
    s = _replace_frac(s)

    # max/min: \max( ... ) -> max(...)
    s = re.sub(r"\\max\s*\(", "max(", s)
    s = re.sub(r"\\min\s*\(", "min(", s)

    # log: \log(...) -> log(...)
    s = re.sub(r"\\log\s*\(", "log(", s)

    # exponential: e^{x} or e^x -> exp(x)
    # handle e^{...}
    s = re.sub(r"(?<![A-Za-z0-9_])e\^\{([^{}]+)\}", r"exp(\1)", s)
    # handle e^x (single token x)
    s = re.sub(r"(?<![A-Za-z0-9_])e\^([A-Za-z0-9_]+)", r"exp(\1)", s)

    # general power: a^{b} -> (a)**(b)
    # 1) parenthesized/base tokens with ^{...}
    s = re.sub(r"(\([^()]*\)|[A-Za-z0-9_\.]+)\s*\^\s*\{([^{}]+)\}", r"(\1)**(\2)", s)
    # 2) and a^b (simple)
    s = re.sub(r"(\([^()]*\)|[A-Za-z0-9_\.]+)\s*\^\s*([A-Za-z0-9_\.]+)", r"(\1)**(\2)", s)

    # Remove LaTeX thin spaces etc.
    s = s.replace(r"\,", "").replace(r"\ ", "")

    # Basic \sum_{i=1}^{n} expr  (inclusive bounds). Support simple "i" iterator.
    # We'll repeatedly find and evaluate innermost sums.
    sum_pat = re.compile(r"\\sum_\{([A-Za-z])\s*=\s*([^}]*)\}\^\{([^}]*)\}\s*")
    while True:
        m = sum_pat.search(s)
        if not m:
            break
        it, start_str, end_str = m.group(1), m.group(2), m.group(3)
        # Find the following expression chunk to sum: assume it's a balanced parenthesis/group or a token sequence.
        # Strategy: if next char is { ... }, take that block; else take next "token" until a top-level '+' or '-' etc.
        tail = s[m.end():]
        expr_chunk = ""
        consumed = 0
        if tail and tail[0] == '{':
            # grab balanced braces
            depth = 0
            for idx, ch in enumerate(tail):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        expr_chunk = tail[1:idx]  # inside braces
                        consumed = idx + 1
                        break
        else:
            # take until we hit a top-level + or - (very rough, but ok for simple terms)
            m2 = re.search(r"(?=[\+\-])(?![^\(]*\))", tail)
            expr_chunk = tail[:m2.start()] if m2 else tail
            consumed = len(expr_chunk)

        # Evaluate bounds using current varmap-less safe env (we'll evaluate later with variables), so keep them symbolic
        # We'll substitute a placeholder and evaluate later:
        placeholder = f"__SUM_PLACEHOLDER_{id(m)}__"
        s = s[:m.start()] + placeholder + s[m.end()+consumed:]

        # Store a lambda-like spec for later materialization:
        if "__SUM_SPECS__" not in varmap:
            varmap["__SUM_SPECS__"] = []
        varmap["__SUM_SPECS__"].append((placeholder, it, start_str, end_str, expr_chunk))

    # After all syntactic transforms, map any [ ] to underscores again (safety), and kill stray braces
    s = s.replace("[", "_").replace("]", "")
    # clean double underscores
    s = re.sub(r"__+", "_", s)

    return s

def _materialize_sums(expr: str, varmap: dict) -> str:
    """Replace sum placeholders with computed numeric values."""
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
        # cleanup loop var
        env.pop(it, None)
    return expr

def evaluate_formula(latex: str, variables: dict) -> float:
    # Prepare a variables dict that contains both given keys and some tolerant aliases
    varmap = dict(variables)

    # Also create aliases removing backslashes if keys have them (unlikely) and ensure names like E[R_m] supported
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
    # round to 4 decimals
    return float(f"{value:.4f}")

# ----------------------------
# API
# ----------------------------

@app.route("/trading-formula", methods=["POST"])
def trading_formula():
    try:
        data = request.get_json(force=True, silent=False)
        if not isinstance(data, list):
            return jsonify({"error": "Input must be a JSON array of test cases."}), 400

        results = []
        for case in data:
            if not isinstance(case, dict):
                return jsonify({"error": "Each test case must be an object."}), 400
            formula = case.get("formula", "")
            variables = case.get("variables", {})
            if not formula or not isinstance(variables, dict):
                return jsonify({"error": f"Bad test case: {case.get('name','(unnamed)')}"}), 400
            try:
                res = evaluate_formula(formula, variables)
            except Exception as e:
                return jsonify({"error": f"Evaluation failed for {case.get('name','(unnamed)')}: {e}"}), 400
            results.append({"result": res})

        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": f"Malformed JSON or server error: {e}"}), 400

# ----------------------------
# Local run
# ----------------------------
if __name__ == "__main__":
    # For local testing
    app.run(host="0.0.0.0", port=8000, debug=True)
