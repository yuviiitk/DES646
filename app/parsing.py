import re
from typing import List
from models import Storyboard, Panel

# --- Helpers for segmentation ---

# Split sentences on punctuation like . ! ? ;
_SENT_SPLIT = re.compile(r"[.!?;]+\s+")

# Split clauses on connectors and commas/semicolons
# (we'll later merge lead-in phrases like "After the workout" with the next clause)
_CLAUSE_CONNECTOR = re.compile(
    r"(?:,\s*|\s*;\s*|\s*(?:then|and then|after that|afterwards|followed by|finally)\s+)",
    flags=re.IGNORECASE,
)

# Lead-in phrases that should attach to the following clause
_PREP_LEADIN = re.compile(r"^(after|before|when|while|as soon as)\b", re.IGNORECASE)

# If a clause starts directly with a verb, prepend "I " (reads better as a panel caption)
_VERB_START = re.compile(
    r"^(came|had|did|went|ran|walked|returned|cooked|ate|slept|started|finished|"
    r"studied|worked|trained|saw|noticed|grabbed|made|prepared|took|drove|wrote|called)\b",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> List[str]:
    s = (text or "").strip().replace("\n", " ")
    if not s:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(s) if p.strip()]
    return parts


def _normalize_clause(cl: str) -> str:
    c = cl.strip().lstrip(",; ").strip()
    # remove leading discourse markers
    c = re.sub(r"^(then|and then|and|after that|afterwards|finally|so|also)\s+", "", c, flags=re.IGNORECASE).strip()
    # add subject if starting with a verb (makes nicer captions)
    if _VERB_START.match(c):
        c = "I " + c
    # capitalize first letter
    if c and not c[0].isupper():
        c = c[0].upper() + c[1:]
    return c


def _split_clauses(sent: str) -> List[str]:
    # First, split by connectors and punctuation
    raw = [chunk.strip() for chunk in _CLAUSE_CONNECTOR.split(sent) if chunk and chunk.strip()]
    if not raw:
        return []

    # Merge lead-in phrases (e.g., "After the workout") with the next action clause
    merged: List[str] = []
    i = 0
    while i < len(raw):
        cur = raw[i].strip()
        if _PREP_LEADIN.match(cur) and i + 1 < len(raw):
            nxt = raw[i + 1].strip()
            merged.append(f"{cur}, {nxt}")
            i += 2
        else:
            merged.append(cur)
            i += 1

    # Further split overly long pieces on " and " as a last resort
    refined: List[str] = []
    for piece in merged:
        # Heuristic: only split long multi-action chunks
        if len(piece.split()) > 10 and re.search(r"\band\b", piece, flags=re.IGNORECASE):
            parts = [p.strip() for p in re.split(r"\band\b", piece, flags=re.IGNORECASE) if p.strip()]
            refined.extend(parts)
        else:
            refined.append(piece)

    # Normalize each clause
    out = [_normalize_clause(x) for x in refined if x.strip()]
    return out


def _expand_to_n(units: List[str], n: int) -> List[str]:
    """
    Ensure we have exactly n unique-ish units:
    - Deduplicate while preserving order.
    - If still short, try splitting the longest ones by commas/and.
    - Finally, pad with the last unique if absolutely necessary.
    """
    # Dedup (case-insensitive)
    seen = set()
    uniq: List[str] = []
    for u in units:
        key = u.lower()
        if key not in seen:
            uniq.append(u)
            seen.add(key)
    units = uniq

    if len(units) >= n:
        return units[:n]

    # Try to expand by splitting existing items
    i = 0
    while len(units) < n and i < len(units):
        u = units[i]
        # split by comma/semicolon/" and " as a fallback
        extras = [p.strip() for p in re.split(r"[,;]\s+|\band\b\s+", u, flags=re.IGNORECASE) if p.strip()]
        # normalize and keep only genuinely new pieces
        new_bits = []
        for e in extras:
            e_norm = _normalize_clause(e)
            if e_norm and e_norm.lower() not in (x.lower() for x in units):
                new_bits.append(e_norm)
        # insert right after current position
        for e in new_bits:
            if len(units) >= n:
                break
            units.insert(i + 1, e)
        i += 1

    # Final pad only if we truly ran out
    while len(units) < n and units:
        units.append(units[-1])

    return units[:n]


def parse_story(text: str, n_panels: int) -> Storyboard:
    """
    Parse free-form text into n_panels succinct panel descriptions.
    Strategy:
      1) sentence split (. ! ? ;)
      2) clause split (commas, 'then/and then/after that/...' etc.)
      3) merge lead-ins like 'After the workout' into the next clause
      4) normalize ('I came back', capitalize, trim fillers)
      5) expand uniquely to n_panels
    """
    sentences = _split_sentences(text)
    units: List[str] = []
    for s in sentences:
        units.extend(_split_clauses(s))

    units = _expand_to_n(units, n_panels)
    panels = [Panel(id=i + 1, description=units[i]) for i in range(n_panels)]
    return Storyboard(title="Storyboard", style="realistic", panels=panels)
