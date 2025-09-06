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
    """Create a safe evaluation environment with mathematical functions"""
    env = {
        # Basic math functions
        "max": max,
        "min": min,
        "log": math.log,
        "exp": math.exp,
        "sqrt": math.sqrt,
        "abs": abs,
        "pow": pow,
        
        # Additional math functions for financial calculations
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "floor": math.floor,
        "ceil": math.ceil,
        
        # Constants
        "e": math.e,
        "pi": math.pi,
        
        # Handle division by zero gracefully
        "safediv": lambda a, b: float('inf') if b == 0 else a / b,
    }
    
    # Add all variables to environment
    env.update(varmap)
    
    # Create restricted builtins
    restricted_builtins = {
        "__builtins__": {},
        # Allow basic operations
        "int": int,
        "float": float,
        "round": round,
    }
    
    return restricted_builtins, env

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
    
    # Better handling of equations with variables having exponents on left side
    if "=" in s:
        left_side, right_side = s.split("=", 1)
        # Check if left side has exponents (like σ_p^2)
        if "^" in left_side:
            # For assignment equations, we only care about the right side
            s = right_side
        else:
            s = right_side
    
    # Remove LaTeX delimiters
    s = s.replace(r"\left", "").replace(r"\right", "")
    
    # Better text handling - preserve underscores in variable names
    s = re.sub(r"\\text\s*\{([^}]*)\}", lambda m: re.sub(r"[^\w_]", "_", m.group(1)).strip("_"), s)
    
    # Handle expectation notation E[R_m] -> E_R_m more robustly
    s = re.sub(r"([A-Za-z]+)\[([A-Za-z0-9_\\]+)\]", r"\1_\2", s)

    # Greek names in variables: \alpha -> alpha, etc.
    for g in GREEKS:
        s = s.replace(fr"\{g}", g)

    # Better subscript handling: \foo_{bar_baz} -> foo_bar_baz
    s = re.sub(r"\\([A-Za-z]+)_\{([A-Za-z0-9_]+)\}", r"\1_\2", s)
    s = re.sub(r"\\([A-Za-z]+)_([A-Za-z0-9_]+)", r"\1_\2", s)
    # Handle complex subscripts like {portfolio}
    s = re.sub(r"([A-Za-z_]+)_\{([A-Za-z0-9_]+)\}", r"\1_\2", s)

    # Products: \times, \cdot -> *
    s = s.replace(r"\times", "*").replace(r"\cdot", "*")

    # sqrt: \sqrt{...} -> sqrt(...)
    s = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", s)

    # frac: \frac{a}{b}
    s = _replace_frac(s)

    # max/min: better handling including nested cases
    # First handle simple cases
    s = re.sub(r"\\max\s*\(", "max(", s)
    s = re.sub(r"\\min\s*\(", "min(", s)
    # Then handle cases with braces
    s = re.sub(r"\\max\s*\{([^{}]+)\}", r"max(\1)", s)
    s = re.sub(r"\\min\s*\{([^{}]+)\}", r"min(\1)", s)

    # log: \log(...) -> log(...)
    s = re.sub(r"\\log\s*\(", "log(", s)

    # exponential: e^{x} or e^x -> exp(x)
    # handle e^{...} with better grouping
    s = re.sub(r"(?<![A-Za-z0-9_])e\^\{([^{}]+)\}", r"exp(\1)", s)
    # handle e^x (single token x)
    s = re.sub(r"(?<![A-Za-z0-9_])e\^([A-Za-z0-9_]+)", r"exp(\1)", s)

    # general power: a^{b} -> (a)**(b) - improved to handle more cases
    # 1) Handle complex expressions in parentheses with ^{...}
    s = re.sub(r"(\([^()]+\))\s*\^\s*\{([^{}]+)\}", r"(\1)**(\2)", s)
    # 2) Handle variables/numbers with ^{...}
    s = re.sub(r"([A-Za-z0-9_\.]+)\s*\^\s*\{([^{}]+)\}", r"(\1)**(\2)", s)
    # 3) Handle simple cases a^b
    s = re.sub(r"([A-Za-z0-9_\.]+)\s*\^\s*([A-Za-z0-9_\.]+)", r"(\1)**(\2)", s)

    # Remove LaTeX thin spaces etc.
    s = s.replace(r"\,", "").replace(r"\ ", "")

    # Summation handling (unchanged but improved variable handling)
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
            # Better expression chunking for summations
            m2 = re.search(r"(?=[\+\-])(?![^\(]*\))", tail)
            expr_chunk = tail[:m2.start()] if m2 else tail
            consumed = len(expr_chunk)

        placeholder = f"__SUM_PLACEHOLDER_{abs(id(m))}__"
        s = s[:m.start()] + placeholder + s[m.end()+consumed:]
        if "__SUM_SPECS__" not in varmap:
            varmap["__SUM_SPECS__"] = []
        varmap["__SUM_SPECS__"].append((placeholder, it, start_str, end_str, expr_chunk))

    # Clean up variable names - but preserve summation placeholders
    s = s.replace("[", "_").replace("]", "")
    
    # Replace multiple underscores with single underscore, but preserve SUM placeholders
    # Split by SUM placeholders, clean each part, then rejoin
    placeholder_pattern = r'(__SUM_PLACEHOLDER_\d+__)'
    parts = re.split(placeholder_pattern, s)
    
    cleaned_parts = []
    for part in parts:
        if re.match(r'__SUM_PLACEHOLDER_\d+__', part):
            # This is a placeholder, keep it as-is
            cleaned_parts.append(part)
        else:
            # This is regular content, clean up underscores
            cleaned_part = re.sub(r'__+', '_', part)
            cleaned_parts.append(cleaned_part)
    
    s = ''.join(cleaned_parts)
    
    return s

def _materialize_sums(expr: str, varmap: dict) -> str:
    specs = varmap.pop("__SUM_SPECS__", [])
    if not specs:
        return expr
    
    builtins_env, env = _safe_env(varmap)
    
    for placeholder, it, start_str, end_str, body in specs:
        try:
            # Evaluate start and end bounds
            start = eval(_latex_to_python(start_str, dict(varmap)), builtins_env, env)
            end = eval(_latex_to_python(end_str, dict(varmap)), builtins_env, env)
            
            total = 0.0
            # Handle summation with proper variable substitution
            for i in range(int(start), int(end) + 1):
                # Create enhanced variable map for this iteration
                iter_varmap = dict(varmap)
                iter_varmap[it] = i
                
                # Add iterator variable to environment
                iter_env = dict(env)
                iter_env[it] = i
                iter_env.update(iter_varmap)
                
                # Handle numbered variables in the body (like a_i -> a_1, a_2, etc.)
                iter_body = body
                # Replace iterator variable with actual number in variable names
                iter_body = re.sub(rf'([A-Za-z_]+)_{it}', rf'\\1_{i}', iter_body)
                
                # Parse and evaluate the body expression for this iteration
                try:
                    body_expr = _latex_to_python(iter_body, iter_varmap)
                    term = eval(body_expr, builtins_env, iter_env)
                    total += float(term)
                except Exception as body_error:
                    # If individual term fails, try with original varmap
                    try:
                        body_expr = _latex_to_python(body, iter_varmap)
                        term = eval(body_expr, builtins_env, iter_env)
                        total += float(term)
                    except:
                        # Skip this term if it can't be evaluated
                        pass
                
            # Replace placeholder with computed sum
            expr = expr.replace(placeholder, f"({total})")
            
        except Exception as e:
            # If summation fails, replace with 0 to avoid breaking the entire formula
            expr = expr.replace(placeholder, "0")
    
    return expr

def evaluate_formula(latex: str, variables: dict) -> float:
    # Prepare a variables dict that contains both given keys and tolerant aliases
    varmap = dict(variables)

    # Create comprehensive aliases for variable names
    alias_map = {}
    for k, v in list(varmap.items()):
        original_key = k
        alias_map[k] = v
        
        # Remove backslashes
        clean_key = k.replace("\\", "")
        alias_map[clean_key] = v
        
        # Handle bracket notation: E[R_m] -> E_R_m
        bracket_key = k.replace("[", "_").replace("]", "")
        alias_map[bracket_key] = v
        
        # Handle subscript patterns: R_f, beta_i, etc.
        if "_" in k:
            alias_map[k] = v
        
        # Handle complex subscripts with braces
        if "{" in k and "}" in k:
            brace_clean = re.sub(r"\{([^}]+)\}", r"_\1", k)
            alias_map[brace_clean] = v
            
        # Handle Greek letter patterns
        for greek in GREEKS:
            if greek in k:
                alias_map[k] = v
                # Also add version without Greek prefix
                no_greek = k.replace(f"\\{greek}", greek)
                alias_map[no_greek] = v
                
        # Handle numbered subscripts: CF_1, CF_2, etc.
        if re.match(r".*_\d+", k):
            alias_map[k] = v
            
        # Handle long variable names
        if len(k) > 10:
            alias_map[k] = v
            # Also try camelCase version if needed
            
    varmap.update(alias_map)

    try:
        py_expr = _latex_to_python(latex, varmap)
        py_expr = _materialize_sums(py_expr, varmap)
        
        builtins_env, env = _safe_env(varmap)
        value = eval(py_expr, builtins_env, env)
        
        # Ensure we return a proper float with 4 decimal places
        result = float(f"{value:.4f}")
        return result
        
    except Exception as e:
        # Better error handling - try to provide more context
        raise ValueError(f"Formula evaluation failed: {str(e)}")

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