"""Country-agnostic text normalisation for names and addresses (rule based, no external lookups)."""
import re
import pandas as pd
from unidecode import unidecode

LEGAL_MAP = {"incorporated": "inc", "corporation": "corp", "limited": "ltd", "private": "pvt", "company": "co",
             "pty": "pvt", "prive": "pvt", "societe": "sa", "llc": "llc", "lp": "llp"}
LEGAL = {"inc", "llc", "ltd", "pvt", "corp", "co", "llp", "lp", "sarl", "sas", "sa", "sasu", "eurl", "sci", "gmbh",
         "plc", "pllc", "pc", "ag", "and", "the", "of", "de", "du", "la", "le", "les", "des", "et", "opc", "ltda"}
STREET_MAP = {"rd": "road", "st": "street", "str": "street", "dr": "drive", "drve": "drive", "ave": "avenue", "av": "avenue",
              "blvd": "boulevard", "bd": "boulevard", "bvd": "boulevard", "ct": "court", "ln": "lane", "pl": "place",
              "hwy": "highway", "pkwy": "parkway", "cir": "circle", "trl": "trail", "ter": "terrace", "sq": "square",
              "expy": "expressway", "cres": "crescent", "r": "rue", "ave.": "avenue", "n": "north", "s": "south",
              "e": "east", "w": "west", "mt": "mount", "ft": "fort", "nagar": "nagar", "bldg": "building", "flr": "floor"}
ADDR_STOP = {"unit", "apt", "suite", "ste", "pmb", "po", "box", "floor", "plot", "near", "opp", "behind", "no", "city",
             "c", "o", "at", "the", "of", "and", "next", "to", "beside", "opposite", "flat", "room", "bldg", "building",
             "sector", "regional", "office", "dist", "district", "tq", "taluk", "suburban", "urban", "ward", "block"}
STATES = {"alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
          "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
          "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
          "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
          "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
          "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
          "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
          "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
          "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
          "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc"}
INDIA_ST = {"mh": "maharashtra", "tn": "tamil nadu", "dl": "delhi", "ka": "karnataka", "up": "uttar pradesh",
            "gj": "gujarat", "wb": "west bengal", "rj": "rajasthan", "mp": "madhya pradesh", "ap": "andhra pradesh",
            "ts": "telangana", "kl": "kerala", "hr": "haryana", "pb": "punjab", "br": "bihar", "od": "odisha",
            "or": "odisha", "jh": "jharkhand", "cg": "chhattisgarh", "uk": "uttarakhand", "hp": "himachal pradesh",
            "as": "assam", "jk": "jammu and kashmir", "ga": "goa"}
_state_re = re.compile(r"\b(" + "|".join(sorted(STATES, key=len, reverse=True)) + r")\b")
_dom = re.compile(r"\b(?:www\.)?([a-z0-9\-]+)\.(?:com|in|net|org|fr|co\.in|co|io|biz|info)\b")
_nonal = re.compile(r"[^a-z0-9 ]+")
_ws = re.compile(r"\s+")
_hasnum = re.compile(r"\d")


def is_nonlatin(s):
    return any(ord(c) > 0x24F and c.isalpha() for c in s)


def norm_name(raw):
    """-> (core_tokens_str, nospace_str, has_domain, leading_junk_digits)"""
    s = unidecode(raw).lower().replace("&", " and ")
    dom = _dom.search(s)
    has_dom = 0
    if dom:
        has_dom = 1
        s = s.replace(dom.group(0), " " + dom.group(1) + " ")
    s = _ws.sub(" ", _nonal.sub(" ", s)).strip()
    toks = [LEGAL_MAP.get(t, t) for t in s.split()]
    toks = [t for t in toks if not (t.isdigit() and len(t) >= 6)]        # phone-number junk
    core = [t for t in toks if t not in LEGAL]
    if not core:
        core = toks
    return " ".join(core), "".join(core), has_dom, " ".join(t for t in toks if t in LEGAL and t not in ("and", "the", "of", "de", "du", "la", "le", "les", "des", "et"))


def _num(t):
    m = re.match(r"^0*(\d+)[a-z]?$", t)
    return m.group(1) if m else None


def norm_addr(raw):
    """-> (tokens_str, nums_str, zip_str, state_str, has_addr)"""
    if not raw.strip():
        return "", "", "", "", 0
    s = unidecode(raw).lower()
    s = _state_re.sub(lambda m: STATES[m.group(1)], s)
    s = re.sub(r"(\d)\s*(st|nd|rd|th)\b", r"\1", s)                    # 112th / 112ST -> 112
    s = _ws.sub(" ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()
    toks, nums, zips, state = [], [], "", ""
    for t in s.split():
        if t in ADDR_STOP:
            continue
        n = _num(t)
        if n is not None:
            if len(n) in (5, 6) and not zips and len(t) == len(n):
                zips = n
            nums.append(n); toks.append(n); continue
        t = STREET_MAP.get(t, t)
        if t in INDIA_ST and len(t) == 2:
            t = INDIA_ST[t]
        toks.append(t)
    st = [t for t in toks if len(t) == 2 and t in STATES.values()]
    state = st[-1] if st else ""
    return " ".join(toks), " ".join(nums), zips, state, 1


def normalize_df(df):
    n = [norm_name(x) for x in df.business_name.values]
    a = [norm_addr(x) for x in df.business_address.values]
    out = pd.DataFrame({
        "entity_id": df.entity_id.values, "country": df.country.values,
        "nm_core": [x[0] for x in n], "nm_nospace": [x[1] for x in n], "nm_dom": [x[2] for x in n], "nm_legal": [x[3] for x in n],
        "nm_nonlatin": [int(is_nonlatin(x)) for x in df.business_name.values],
        "ad_toks": [x[0] for x in a], "ad_nums": [x[1] for x in a], "ad_zip": [x[2] for x in a],
        "ad_state": [x[3] for x in a], "ad_has": [x[4] for x in a],
        "nm_raw": df.business_name.values, "ad_raw": df.business_address.values})
    return out


if __name__ == "__main__":
    for r in ["Frontier-Pacific Acadia Inc.", "kéystonebiomedical.com", "3037043925 Bollinger & Torres -", "Foods Jai Financial Private Ltd", "माय एक्सपोर्ट्स प्रा. लि."]:
        print(norm_name(r))
    for r in ["1502- 112ST STREET, TACOMA, WA", "Tacoam, Washington, 112th St", "19404 1/2 Fox Chase Drive, Indiana, Noblesville CITY", "Plot ##624 At.kandari Partur, Jalna, Ghansawangi, MH"]:
        print(norm_addr(r))
