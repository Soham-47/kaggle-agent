"""Multilingual report → 12 binary labels (public-recipe extractor)."""

import re
import unicodedata

try:
    from .schema import LABELS
except ImportError:  # standalone on a Kaggle kernel folder
    from schema import LABELS  # type: ignore

# --------------------------------------------------------------------------- #
# Multilingual rule-based label extractor.
# Weak on purpose. This is the artifact an LLM extraction has to beat.
# --------------------------------------------------------------------------- #

_CHAR_MAP = str.maketrans({"ı": "i", "İ": "i", "ß": "ss",
                           "ø": "o", "Ø": "o",
                           "đ": "d", "Đ": "d"})

TEAR = [
    "tear", "torn", "tearing", "rupture", "ruptured", "disruption", "discontinuity",
    "rotura", "ruptura", "desgarro", "roto", "rota",
    "dechirure", "dechire", "lesion meniscale",
    "scheur", "ruptuur", "gescheurd",
    "riss", "rissbildung", "ruptur", "zerreissung", "einriss",
    "yirtik", "yirtig", "kopma", "butunluk kaybi",
    "ρηξη", "ρηξις", "ρηγμα",
    "руптура", "разкъсв",
    "разрив", "puknuce", "prekid",
]
SPRAIN = [
    "sprain", "esguince", "entorse", "verstauchung", "verstuiking",
    "distorsiyon", "burkulma", "διαστρεμμα",
    "навяхван",
    "injury", "lesion", "letsel", "verletzung", "zedelenme",
]
OA_FIND = [
    "osteoarthritis", "osteoarthrosis", "arthrosis", "arthritic",
    "artrosis", "artrose", "arthrose", "gonartrose", "gonartroz", "gonarthrose",
    "gonartro", "artroz",
    "chondrosis", "chondral", "chondropathy", "chondropathie", "chondropatie",
    "chondromalacia", "kondromalazi", "condropatia", "condral", "chondropatia",
    "kraakbeenlijden", "kraakbeenverlies", "kraakbeenschade",
    "knorpelschaden", "knorpeldefekt", "knorpelverlust", "knorpelbelag",
    "cartilage loss", "cartilage thinning", "cartilage fissuring",
    "cartilage defect", "cartilage fissure", "chondral defect", "chondral loss",
    "osteophyt", "osteofit", "osteofyt", "osteophyte", "spurring", "osteofyte",
    "joint space narrowing", "gelenkspaltverschmalerung",
    "kikirdak kaybi", "kikirdak incelme", "kondral",
    "ulcera condral", "hrskavice", "denudacija",
    "χονδροπαθ", "οστεοαρθρ",
    "οστεοφυτ", "αρθριτ",
    "артроз", "хондропат",
    "остеофит", "хрущялн",
]
MEDIAL_Q = ["medial", "interno", "interna", "interne", "mediaal", "mediale",
            "innen", "medyal", "medijaln", "εσω",
            "медиал", "вътреш"]
LATERAL_Q = ["lateral", "externo", "externa", "externe", "buiten", "aussen",
             "lateraln", "εξω", "латерал",
             "външ"]
PF_Q = ["patellofemoral", "patelofemoral", "femoropatellar", "femoropatelar",
        "femoropatellair", "retropatellar", "retrorotulian", "patellar facet",
        "trochlea", "troclea", "trochlee", "rotula", "rotulian", "patella",
        "patellaire", "patellar", "diz kapagi", "patelofemoraln",
        "επιγονατιδ", "τροχιλ",
        "пател", "ретропател"]
TRICOMP = ["tricompartmental", "tricompartimental", "three compartments",
           "three compartmens", "all compartments", "gonartrose", "gonartroz",
           "gonarthrose", "gonartrosis", "gonartro", "pangonartro"]

SPECIFIC = {
    "ACL": [
        "acl", "anterior cruciate", "lca", "ligamento cruzado anterior",
        "ligament croise anterieur", "croise anterieur",
        "voorste kruisband", "vkb", "vorderes kreuzband", "vorderen kreuzband",
        "vordere kreuzband", "on capraz bag", "anterior capraz bag",
        "προσθιο χιαστ",
        "προσθιου χιαστ",
        "προσθιος χιαστ",
        "предна кръстна",
        "предната кръстна",
        "prednji ukrizeni",
    ],
    "MCL": [
        "mcl", "medial collateral", "lcm", "ligamento colateral medial",
        "ligamento colateral interno", "ligament collateral medial",
        "collateral medial", "mediale collaterale", "mediaal collateraal",
        "innenband", "mediales kollateralband", "medialen kollateralband",
        "mediale kollateralband", "medyal kollateral", "ic yan bag",
        "εσω πλαγιο",
        "медиален колатерален",
        "вътрешна колатерална",
    ],
    "Medial Meniscus": [
        "medial meniscus", "meniscus medialis", "menisco medial", "menisco interno",
        "mediale meniscus", "binnenmeniscus", "innenmeniskus", "meniskus medialis",
        "menisque interne", "menisque medial", "medial menisk", "medyal menisk",
        "εσω μηνισκ",
        "медиалния менискус",
        "медиален мениск",
        "вътрешния мениск",
        "medial and lateral menisc", "medial ve lateral menisk",
        "menisco medial y lateral", "menisco interno y externo",
        "menisco interno y lateral", "mediale en laterale meniscus",
        "innen und aussenmeniskus", "medijalnog meniskusa",
    ],
    "Lateral Meniscus": [
        "lateral meniscus", "meniscus lateralis", "menisco lateral", "menisco externo",
        "laterale meniscus", "buitenmeniscus", "aussenmeniskus", "meniskus lateralis",
        "menisque externe", "menisque lateral", "lateral menisk",
        "εξω μηνισκ",
        "латералния менискус",
        "латерален мениск",
        "външния мениск",
        "medial and lateral menisc", "medial ve lateral menisk",
        "menisco medial y lateral", "menisco interno y externo",
        "menisco interno y lateral", "mediale en laterale meniscus",
        "innen und aussenmeniskus", "lateralnog meniskusa",
    ],
}

GENERIC = {
    "ACL": ["cruciate", "cruzado", "croise", "kruisband", "kreuzband",
            "χιαστ", "кръстн",
            "capraz bag", "ukrizen"],
    "MCL": ["collateral", "colateral", "kollateral", "collaterale",
            "πλαγι", "колатерал",
            "yan bag"],
    "Medial Meniscus": ["menisc", "menisk", "μηνισκ",
                        "мениск"],
    "Lateral Meniscus": ["menisc", "menisk", "μηνισκ",
                         "мениск"],
}
SIDE_Q = {
    "ACL": ["anterior", "anterieur", "voorste", "vorder",
            "προσθι", "предн", "prednj"],
    "MCL": MEDIAL_Q,
    "Medial Meniscus": MEDIAL_Q,
    "Lateral Meniscus": LATERAL_Q,
}

DIRECT = {
    "Effusion": [
        "effusion", "joint fluid", "hemarthrosis", "haemarthrosis", "hydrops",
        "derrame", "epanchement", "gewrichtsvocht", "vocht in het gewricht",
        "erguss", "gelenkerguss", "gelenkserguss",
        "eklem ici sivi", "sivi artisi", "efuzyon", "eklem mesafesinde sivi",
        "eklem ici serbest sivi", "eklem sivisi",
        "ενδαρθρικ",
        "αρθρικο υγρο",
        "συλλογη υγρου",
        "ставен излив",
        "излив", "хидропс",
        "zglobni izljev", "izljev",
    ],
    "Synovitis": [
        "synovitis", "synovial thickening", "synovial hypertrophy",
        "thickened synovial", "hypertrophy of the synovium",
        "synovial proliferation", "proliferation of the synovium",
        "sinovitis", "synovite", "synovitiden", "synovialitis",
        "verdikking van het synovium", "verdikkingen van het synovium",
        "synoviale verdikking", "synovialisverdickung", "synovialverdickung",
        "sinovit", "hoffitis", "sinovyal kalinlasma", "sinovyal proliferasyon",
        "συνοβιτ", "υμενιτ",
        "синовит", "sinovij", "sinovije",
    ],
    "Baker's": [
        "baker", "popliteal cyst", "poplitealcyst", "popliteal cysts",
        "quiste popliteo", "quistes popliteos", "quiste de baker",
        "kyste de baker", "kyste poplite", "popliteale cyste", "popliteale cyst",
        "bakercyste", "bakerzyste", "poplitealzyste", "popliteazyste",
        "baker kisti", "popliteal kist",
        "κυστη baker", "κυστη του baker",
        "киста на бейкър",
        "бейкърова киста",
        "poplitealna cista",
    ],
    "Contusion": [
        "contusion", "contusiones", "contusie", "kontusyon", "kontuzyon",
        "bone bruise", "bone bruising", "knochenprellung", "prellung",
        "botcontusie", "μωλωπ", "θλαση",
        "контузи", "kontuzij", "nagnjecen",
    ],
    "Fracture": [
        "fracture", "fractur", "fractura", "fraktur", "fractuur", "breuk",
        "kirik", "kirig", "καταγμα",
        "καταγματ",
        "фрактура", "счупван",
        "avulsion", "avulsie", "avulsiyon", "prijelom",
    ],
}

# Negation cues, word-boundary anchored. A bare "no " substring matches inside the
# Spanish word "cuerno", which silently killed most Spanish positives before this
# was anchored.
NEG_SRC = [
    r"\bno\b", r"\bnot\b", r"\bnon\b", r"\bwithout\b", r"\babsen\w*",
    r"\bnegative for\b", r"\bunremarkable\b", r"\bintact\w*", r"\bnormal\w*",
    r"\bpreserved\b", r"\bfree of\b", r"\bexcluded\b",
    r"\bno hay\b", r"\bsin\b", r"\bausen\w*", r"\bno se\b", r"\bconservad\w*",
    r"\bintegr\w*", r"\bindemne\b",
    r"\bpas de\b", r"\baucun\w*", r"\bsans\b",
    r"\bgeen\b", r"\bzonder\b", r"\bonopvallend\w*", r"\bnormaal\b",
    r"\bvrij van\b", r"\bintacte\b",
    r"\bkein\w*", r"\bohne\b", r"\bunauffallig\w*", r"\bregelrecht\w*",
    r"\bnicht\b", r"\bintakt\w*",
    r"\byok\w*", r"\bizlenmemis\w*", r"\bizlenmedi\w*", r"\bsaptanmamis\w*",
    r"\bgozlenmemis\w*", r"\bgorulmemis\w*", r"\bkorunmus\w*",
    r"\bmevcut degil\b", r"\bnormaldir\b", r"\bdogaldir\b",
    r"\bδεν\b", r"\bχωρις\b",
    r"\bφυσιολογικ\w*",
    r"\bακεραι\w*",
    r"\bбез\b", r"\bняма\b",
    r"\bне се\b", r"\bнормал\w*",
    r"\bзапазен\w*",
    r"\bсъхранен\w*",
    r"\bsenza\b", r"\bsem\b", r"\bnao\b", r"\bbez\b", r"\buredn\w*",
]
NEG_RE = re.compile("|".join(NEG_SRC))

NEG_BACK = 70
NEG_FWD = 45
QUAL_WINDOW = 90
PAIR_WINDOW = 160
CLAUSE_SPLIT = re.compile(r"[.;:\n\r•·]+|\s-\s|\s>\s|\s\*\s")
PF_MASK = re.compile(r"(medial|lateral)\s+(patellar|patella|facet|trochlea|trochlear|retinac)\w*")


def fold_text(text):
    """Lowercase, normalise script-specific letters, and strip diacritics.

    Turkish dotless i, the German sharp s and the Croatian barred d have no
    combining-mark decomposition, so they are mapped explicitly before NFKD.

    Args:
        text: Any report text.

    Returns:
        A lowercase, accent-free string safe for substring matching.
    """
    t = str(text).lower().translate(_CHAR_MAP)
    t = unicodedata.normalize("NFKD", t)
    return "".join(c for c in t if not unicodedata.combining(c))


def _find_any(clause, terms):
    """Return the earliest index at which any term occurs, or -1.

    Args:
        clause: Folded clause text.
        terms: Iterable of folded surface forms.

    Returns:
        Character index of the earliest hit, or -1 when none matches.
    """
    best = -1
    for t in terms:
        i = clause.find(t)
        if i >= 0 and (best < 0 or i < best):
            best = i
    return best


def _negated(clause, pos, span):
    """Test whether a negation cue sits near a matched finding term.

    Args:
        clause: Folded clause text.
        pos: Start index of the finding term.
        span: Length of the finding term.

    Returns:
        True when a cue appears in the preceding or following window.
    """
    back = clause[max(0, pos - NEG_BACK):pos]
    fwd = clause[pos + span:pos + span + NEG_FWD]
    return bool(NEG_RE.search(back) or NEG_RE.search(fwd))


def _fire_direct(clause, terms):
    """Test a direct finding term, retrying later occurrences past a negation.

    Args:
        clause: Folded clause text.
        terms: Surface forms that are themselves the finding.

    Returns:
        True when at least one occurrence is unnegated.
    """
    for t in terms:
        start = 0
        while True:
            i = clause.find(t, start)
            if i < 0:
                break
            if not _negated(clause, i, len(t)):
                return True
            start = i + 1
    return False


def _anatomy_index(clause, label):
    """Locate the anatomy mention for a paired label.

    Falls back to a generic organ term paired with a side qualifier, which is what
    catches Greek and Bulgarian phrasing that names the compartment rather than the
    structure.

    Args:
        clause: Folded clause text.
        label: One of the four paired label names.

    Returns:
        Character index of the anatomy mention, or -1.
    """
    i = _find_any(clause, SPECIFIC[label])
    if i >= 0:
        return i
    g = _find_any(clause, GENERIC[label])
    if g >= 0 and _find_any(clause, SIDE_Q[label]) >= 0:
        return g
    return -1


def extract_labels(report):
    """Extract twelve binary findings from one free-text radiology report.

    Args:
        report: Report text in any of the languages present in the corpus.

    Returns:
        Dict mapping each of the twelve label names to 0 or 1.
    """
    out = {lab: 0 for lab in LABELS}
    if not isinstance(report, str) or not report.strip():
        return out
    text = fold_text(report)
    for raw in CLAUSE_SPLIT.split(text):
        clause = raw.strip()
        if len(clause) < 3:
            continue
        for lab in ("Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"):
            if not out[lab] and _fire_direct(clause, DIRECT[lab]):
                out[lab] = 1
        for lab in SPECIFIC:
            if out[lab]:
                continue
            ai = _anatomy_index(clause, lab)
            if ai < 0:
                continue
            finds = TEAR + SPRAIN if lab in ("ACL", "MCL") else TEAR
            for t in finds:
                fi = clause.find(t)
                if fi < 0 or abs(fi - ai) > PAIR_WINDOW:
                    continue
                if not _negated(clause, fi, len(t)):
                    out[lab] = 1
                    break
        oi = _find_any(clause, OA_FIND)
        if oi >= 0 and not _negated(clause, oi, 8):
            if _find_any(clause, TRICOMP) >= 0:
                out["Medial OA"] = 1
                out["Lateral OA"] = 1
                out["PF OA"] = 1
            win = clause[max(0, oi - QUAL_WINDOW):oi + QUAL_WINDOW]
            if _find_any(win, PF_Q) >= 0:
                out["PF OA"] = 1
            masked = PF_MASK.sub(" ", win)
            if _find_any(masked, MEDIAL_Q) >= 0:
                out["Medial OA"] = 1
            if _find_any(masked, LATERAL_Q) >= 0:
                out["Lateral OA"] = 1
    return out


print("extractor ready:", len(SPECIFIC), "paired labels,",
      len(DIRECT), "direct labels,", len(NEG_SRC), "negation cues")