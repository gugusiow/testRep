# app.py
from flask import Flask, request, jsonify
import re
import os

app = Flask(__name__)
app.url_map.strict_slashes = False

# ------------- Roman numeral parsing -------------

ROMAN_MAP = [
    ("M", 1000),
    ("CM", 900),
    ("D", 500),
    ("CD", 400),
    ("C", 100),
    ("XC", 90),
    ("L", 50),
    ("XL", 40),
    ("X", 10),
    ("IX", 9),
    ("V", 5),
    ("IV", 4),
    ("I", 1),
]
ROMAN_VALUE = dict(ROMAN_MAP)
ROMAN_RE = re.compile(r"^(M{0,3})(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")

def is_roman(s: str) -> bool:
    if not s:
        return False
    return bool(ROMAN_RE.match(s.upper()))

def roman_to_int(s: str) -> int:
    s = s.upper()
    i = 0
    n = 0
    while i < len(s):
        if i + 1 < len(s) and s[i:i+2] in ROMAN_VALUE:
            n += ROMAN_VALUE[s[i:i+2]]
            i += 2
        else:
            n += ROMAN_VALUE.get(s[i], 0)
            i += 1
    return n

# ------------- Arabic (decimal) -------------

ARABIC_RE = re.compile(r"^\d+$")

def is_arabic(s: str) -> bool:
    return bool(ARABIC_RE.match(s))

def arabic_to_int(s: str) -> int:
    return int(s)

# ------------- English number parsing -------------

EN_SMALL = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
    "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,
    "seventeen":17,"eighteen":18,"nineteen":19
}
EN_TENS = {
    "twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90
}
EN_SCALE = {
    "hundred":100, "thousand":1000, "million":1_000_000, "billion":1_000_000_000
}

def looks_english(s: str) -> bool:
    t = s.lower().replace("-", " ")
    ws = [w for w in t.split() if w]
    if not ws:
        return False
    for w in ws:
        if w not in EN_SMALL and w not in EN_TENS and w not in EN_SCALE and w != "and":
            return False
    return True

def english_to_int(s: str) -> int:
    t = s.lower().replace("-", " ")
    ws = [w for w in t.split() if w and w != "and"]
    total = 0
    current = 0
    for w in ws:
        if w in EN_SMALL:
            current += EN_SMALL[w]
        elif w in EN_TENS:
            current += EN_TENS[w]
        elif w == "hundred":
            current *= 100
        elif w in ("thousand","million","billion"):
            scale = EN_SCALE[w]
            total += current * scale
            current = 0
        else:
            raise ValueError(f"Unknown English token: {w}")
    return total + current

# ------------- German number parsing -------------

def de_norm(s: str) -> str:
    s = s.lower()
    s = s.replace("ß", "ss").replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    s = s.replace("-", " ")
    s = s.replace(" und ", "und")
    s = s.replace(" ", "")
    return s

DE_UNITS = {
    "null":0,"eins":1,"ein":1,"eine":1,"einen":1,"zwei":2,"drei":3,"vier":4,"fuenf":5,"fünf":5,
    "sechs":6,"sieben":7,"acht":8,"neun":9
}
DE_TEENS = {
    "zehn":10,"elf":11,"zwoelf":12,"zwölf":12,"dreizehn":13,"vierzehn":14,"fuenfzehn":15,
    "fünfzehn":15,"sechzehn":16,"siebzehn":17,"achtzehn":18,"neunzehn":19
}
DE_TENS = {
    "zwanzig":20,"dreissig":30,"dreißig":30,"vierzig":40,"fuenfzig":50,"fünfzig":50,
    "sechzig":60,"siebzig":70,"achtzig":80,"neunzig":90
}

def looks_german(s: str) -> bool:
    t = de_norm(s)
    return any(w in t for w in [
        "und","zig","ssig","hundert","tausend","million","milliarde",
        "ein","zwei","drei","vier","fuenf","sechs","sieben","acht","neun",
        "zehn","elf","zwoelf","dreizehn","vierzehn","fuenfzehn","sechzehn",
        "siebzehn","achtzehn","neunzehn"
    ])

def parse_de_below_100(t: str) -> int:
    if t in DE_TEENS: return DE_TEENS[t]
    if t in DE_TENS: return DE_TENS[t]
    if t in DE_UNITS: return DE_UNITS[t]
    m = re.match(r"^(ein|eins|zwei|drei|vier|fuenf|sechs|sieben|acht|neun)und(.+)$", t)
    if m:
        u = m.group(1)
        tens = m.group(2)
        if tens in DE_TENS and (u in DE_UNITS):
            return DE_UNITS[u] + DE_TENS[tens]
    raise ValueError(f"Unrecognized German <100: {t}")

def german_to_int(s: str) -> int:
    t = de_norm(s)

    def parse_chunk_upto_999(x: str) -> int:
        if "hundert" in x:
            pre, post = x.split("hundert", 1)
            if pre == "": pre = "ein"
            val = (DE_UNITS.get(pre, parse_de_below_100(pre)) if pre else 1) * 100
            tail = post
            if tail:
                val += (DE_UNITS.get(tail) or DE_TEENS.get(tail) or DE_TENS.get(tail)
                        or parse_de_below_100(tail))
            return val
        if x == "": return 0
        return DE_UNITS.get(x) or DE_TEENS.get(x) or DE_TENS.get(x) or parse_de_below_100(x)

    def split_scale(word: str, plural_suffix: str, txt: str):
        if word not in txt:
            return None
        left, rem = txt.split(word, 1)
        # Strip optional plural suffix immediately following the scale
        if plural_suffix and rem.startswith(plural_suffix):
            rem = rem[len(plural_suffix):]
        return left, rem

    total = 0
    # Descend big -> small scales; handle plural 'en'/'n'
    for word, val, suffix in [("milliarde", 1_000_000_000, "n"),
                              ("million", 1_000_000, "en"),
                              ("tausend", 1000, "")]:
        split = split_scale(word, suffix, t)
        if split:
            left, t = split
            left_val = parse_chunk_upto_999(left if left else "ein")
            total += left_val * val
    total += parse_chunk_upto_999(t)
    return total

# ------------- Chinese number parsing -------------

CN_DIGITS = {
    "零":0,"〇":0,"○":0,"O":0,"０":0,
    "一":1,"二":2,"兩":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,
}
CN_UNITS = {"十":10,"百":100,"千":1000}
CN_LARGE = [("兆", 10**12), ("億", 10**8), ("亿", 10**8), ("萬", 10**4), ("万", 10**4)]

def looks_chinese(s: str) -> bool:
    return any(ch in s for ch in list(CN_DIGITS.keys()) + list(CN_UNITS.keys()) + [lu for lu,_ in CN_LARGE])

def chinese_section_to_int(sec: str) -> int:
    if sec == "": return 0
    total = 0
    num = 0
    i = 0
    if len(sec) >= 1 and sec[0] == "十":
        total += 10
        i += 1
    while i < len(sec):
        ch = sec[i]
        if ch in CN_DIGITS:
            num = CN_DIGITS[ch]
            i += 1
            if i < len(sec) and sec[i] in CN_UNITS:
                total += num * CN_UNITS[sec[i]]
                i += 1
                num = 0
        elif ch in CN_UNITS:
            total += max(1, num) * CN_UNITS[ch]
            num = 0
            i += 1
        else:
            i += 1
    total += num
    return total

def chinese_to_int(s: str) -> int:
    rest = s
    total = 0
    for token, val in CN_LARGE:
        if token in rest:
            parts = rest.split(token)
            left = parts[0]
            rest = token.join(parts[1:])
            total += chinese_section_to_int(left) * val
    total += chinese_section_to_int(rest)
    return total

def classify_chinese_trad_or_simp(s: str) -> str:
    if any(ch in s for ch,_ in [("萬",10_000),("億",100_000_000),("兩",2)]):
        return "zh_trad"
    if any(ch in s for ch,_ in [("万",10_000),("亿",100_000_000),("两",2)]):
        return "zh_simp"
    return "zh_trad"

# ------------- Language detection & parsing -------------

LANG_ROMAN = "roman"
LANG_EN = "english"
LANG_ZH_TRAD = "zh_trad"
LANG_ZH_SIMP = "zh_simp"
LANG_DE = "german"
LANG_ARABIC = "arabic"

def detect_language(s: str) -> str:
    if is_roman(s):
        return LANG_ROMAN
    if is_arabic(s):
        return LANG_ARABIC
    if looks_chinese(s):
        return classify_chinese_trad_or_simp(s)
    if looks_english(s):
        return LANG_EN
    if looks_german(s):
        return LANG_DE
    if s.isdigit():
        return LANG_ARABIC
    return LANG_EN

def to_int_by_lang(s: str, lang: str) -> int:
    if lang == LANG_ROMAN:
        return roman_to_int(s)
    if lang == LANG_ARABIC:
        return arabic_to_int(s)
    if lang == LANG_EN:
        return english_to_int(s)
    if lang == LANG_DE:
        return german_to_int(s)
    if lang in (LANG_ZH_TRAD, LANG_ZH_SIMP):
        return chinese_to_int(s)
    raise ValueError(f"Unsupported language for value: {s}")

# ------------- Tie-break priority for Part TWO -------------

# Roman < English < Traditional Chinese < Simplified Chinese < German < Arabic
LANG_PRIORITY = {
    LANG_ROMAN: 0,
    LANG_EN: 1,
    LANG_ZH_TRAD: 2,
    LANG_ZH_SIMP: 3,
    LANG_DE: 4,
    LANG_ARABIC: 5,
}

# ------------- Endpoint -------------

@app.post("/duolingo-sort")
def duolingo_sort():
    body = request.get_json(force=True, silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "invalid body"}), 400

    part = str(body.get("part", "")).upper()
    ch_input = body.get("challengeInput", {}) or {}
    arr = ch_input.get("unsortedList", [])
    if not isinstance(arr, list):
        return jsonify({"error":"unsortedList must be a list"}), 400

    try:
        if part == "ONE":
            # Part 1: Roman + Arabic -> return Arabic numerals (strings), sorted asc
            items = []
            for s in arr:
                s2 = s.strip()
                if is_roman(s2):
                    val = roman_to_int(s2)
                elif is_arabic(s2):
                    val = int(s2)
                else:
                    # be permissive but still return numeric strings
                    val = to_int_by_lang(s2, detect_language(s2))
                items.append(val)
            items.sort()
            return jsonify({"sortedList": [str(v) for v in items]})

        elif part == "TWO":
            # Part 2: Mixed languages -> return original reps sorted by value,
            # with tie-break by language priority
            annotated = []
            for s in arr:
                s2 = s.strip()
                lang = detect_language(s2)
                if lang in (LANG_ZH_TRAD, LANG_ZH_SIMP):
                    lang = classify_chinese_trad_or_simp(s2)
                val = to_int_by_lang(s2, lang)
                annotated.append((val, LANG_PRIORITY.get(lang, 99), s2))
            annotated.sort(key=lambda t: (t[0], t[1], t[2]))
            return jsonify({"sortedList": [t[2] for t in annotated]})

        else:
            return jsonify({"error": "part must be 'ONE' or 'TWO'"}), 400

    except Exception as e:
        return jsonify({"error": f"parse error: {e}"}), 400

# ------------- Run -------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
