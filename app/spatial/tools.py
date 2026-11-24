"""Deterministic spatial reasoning over the building table.

The LLM only *chooses* one of these tools and *phrases* the result; it never counts.
Every tool returns {..., "ids": [...highlight ids], "text": "<plain-English fact>"}.
"""
from collections import Counter

QUADRANTS = ["NW", "NE", "SW", "SE"]
REGIONS = QUADRANTS + ["left", "right", "top", "bottom", "all"]
DAMAGE_CLASSES = ["No Damage", "Minor", "Major", "Destroyed", "Unclassified"]
DAMAGED = ["Minor", "Major", "Destroyed"]
SEVERE = ["Major", "Destroyed"]

_DAMAGE_ALIASES = {
    "no damage": ["No Damage"], "none": ["No Damage"], "undamaged": ["No Damage"], "intact": ["No Damage"],
    "minor": ["Minor"], "major": ["Major"], "destroyed": ["Destroyed"], "collapsed": ["Destroyed"],
    "unclassified": ["Unclassified"], "unknown": ["Unclassified"],
    "damaged": DAMAGED, "any damage": DAMAGED, "severe": SEVERE, "severely damaged": SEVERE,
    "all": DAMAGE_CLASSES,
}
_REGION_ALIASES = {
    "northwest": "NW", "north-west": "NW", "north west": "NW", "nw": "NW", "top-left": "NW", "top left": "NW",
    "northeast": "NE", "north-east": "NE", "north east": "NE", "ne": "NE", "top-right": "NE", "top right": "NE",
    "southwest": "SW", "south-west": "SW", "south west": "SW", "sw": "SW", "bottom-left": "SW", "bottom left": "SW",
    "southeast": "SE", "south-east": "SE", "south east": "SE", "se": "SE", "bottom-right": "SE", "bottom right": "SE",
    "left": "left", "west": "left", "right": "right", "east": "right",
    "top": "top", "north": "top", "upper": "top", "bottom": "bottom", "south": "bottom", "lower": "bottom",
    "all": "all", "whole": "all", "entire": "all", "image": "all",
}


def norm_damage(v, default=None):
    if v is None or v == "" or v == []:
        return list(default) if default else None
    items = v if isinstance(v, (list, tuple)) else [v]
    out: list[str] = []
    for it in items:
        key = str(it).strip().lower()
        hit = _DAMAGE_ALIASES.get(key) or ([str(it)] if str(it) in DAMAGE_CLASSES else None)
        if hit is None:
            raise ValueError(f"unknown damage class: {it}")
        out += [h for h in hit if h not in out]
    return out


def norm_region(v):
    if v is None or v == "":
        return "all"
    key = str(v).strip().lower()
    r = _REGION_ALIASES.get(key) or (str(v) if str(v) in REGIONS else None)
    if r is None:
        raise ValueError(f"unknown region: {v}")
    return r


def in_region(b: dict, region: str) -> bool:
    q = b["quadrant"]
    return {
        "all": True,
        "left": q.endswith("W"), "right": q.endswith("E"),
        "top": q.startswith("N"), "bottom": q.startswith("S"),
    }.get(region, q == region)


def _label(damage: list[str]) -> str:
    if damage == DAMAGED:
        return "damaged"
    if damage == SEVERE:
        return "severely damaged (Major or Destroyed)"
    if damage == DAMAGE_CLASSES:
        return ""
    return "/".join(damage)


def _where(region: str) -> str:
    return {"all": "in the whole image", "left": "in the left half", "right": "in the right half",
            "top": "in the top half", "bottom": "in the bottom half"}.get(region, f"in the {region} quadrant")


def _name(region: str) -> str:
    return region + " half" if region in ("left", "right", "top", "bottom") else region + (" quadrant" if region in QUADRANTS else "")


def _sel(buildings, damage, region):
    return [b for b in buildings if b["damage"] in damage and in_region(b, region)]


# ----------------------------------------------------------------------------- tools
def count_buildings(buildings, damage=None, region="all"):
    damage, region = norm_damage(damage, DAMAGED), norm_region(region)
    sel = _sel(buildings, damage, region)
    lab = _label(damage)
    return {"count": len(sel), "damage": damage, "region": region, "ids": [b["id"] for b in sel],
            "text": f"There are {len(sel)} {lab + ' ' if lab else ''}buildings {_where(region)}."}


def damage_distribution(buildings, region="all"):
    region = norm_region(region)
    sel = [b for b in buildings if in_region(b, region)]
    c = Counter(b["damage"] for b in sel)
    dist = {k: c.get(k, 0) for k in DAMAGE_CLASSES}
    parts = ", ".join(f"{k}: {v}" for k, v in dist.items() if v)
    return {"total": len(sel), "distribution": dist, "region": region, "ids": [b["id"] for b in sel],
            "text": f"{len(sel)} buildings {_where(region)} ({parts or 'none'})."}


def quadrant_ranking(buildings, damage=None):
    damage = norm_damage(damage, ["Destroyed"])
    counts = {q: sum(1 for b in buildings if b["quadrant"] == q and b["damage"] in damage) for q in QUADRANTS}
    top = max(counts.values())
    lab = _label(damage) or "total"
    if top == 0:
        return {"counts": counts, "top": [], "damage": damage, "ids": [],
                "text": f"No {lab} buildings were found in any quadrant."}
    winners = [q for q, v in counts.items() if v == top]
    ids = [b["id"] for b in buildings if b["quadrant"] in winners and b["damage"] in damage]
    who = " and ".join(winners) + (" quadrants are tied for the most" if len(winners) > 1 else " quadrant has the most")
    return {"counts": counts, "top": winners, "top_count": top, "damage": damage, "ids": ids,
            "text": f"The {who} {lab} buildings, with {top} each." if len(winners) > 1
            else f"The {who} {lab} buildings, with {top}."}


def compare_regions(buildings, region_a="left", region_b="right", damage=None):
    damage, a, b_ = norm_damage(damage, DAMAGED), norm_region(region_a), norm_region(region_b)
    sa, sb = _sel(buildings, damage, a), _sel(buildings, damage, b_)
    lab = _label(damage) or "total"
    if len(sa) == len(sb):
        verdict = f"Both sides are equal with {len(sa)} {lab} buildings."
    else:
        w, wn, l, ln = (a, len(sa), b_, len(sb)) if len(sa) > len(sb) else (b_, len(sb), a, len(sa))
        verdict = f"The {_name(w)} has more {lab} buildings ({wn}) than the {_name(l)} ({ln})."
    return {"region_a": a, "count_a": len(sa), "region_b": b_, "count_b": len(sb), "damage": damage,
            "ids": [x["id"] for x in sa + sb], "text": verdict}


def list_buildings(buildings, damage=None, region="all", limit=20):
    damage, region = norm_damage(damage, DAMAGED), norm_region(region)
    try:
        limit = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("limit must be an integer") from exc
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    sel = sorted(_sel(buildings, damage, region), key=lambda b: -float(b.get("conf", 0)))
    shown = sel[:limit]
    return {"count": len(sel), "ids": [b["id"] for b in shown],
            "items": [{"id": b["id"], "damage": b["damage"], "conf": b["conf"], "quadrant": b["quadrant"]} for b in shown],
            "text": f"Found {len(sel)} matching buildings {_where(region)}; showing the top {len(shown)} by confidence."}


def building_details(buildings, building_id):
    bid = str(building_id).upper()
    b = next((x for x in buildings if x["id"].upper() == bid), None)
    if not b:
        return {"found": False, "ids": [], "text": f"No building with ID {building_id} exists."}
    return {"found": True, "building": {k: b[k] for k in ("id", "damage", "conf", "quadrant", "bbox", "area")}, "ids": [b["id"]],
            "text": f"{b['id']} is classified as {b['damage']} (confidence {b['conf']:.2f}) in the {b['quadrant']} quadrant."}


def unanswerable(buildings, reason="This question cannot be answered from the building-level data."):
    return {"ids": [], "reason": reason, "text": f"I can't answer that from the analysis data. {reason}"}


TOOLS = {
    "count_buildings": (count_buildings,
        'Count buildings. args: damage (class name or "damaged"/"severe"/"all"; default "damaged"), region (NW|NE|SW|SE|left|right|top|bottom|all)'),
    "damage_distribution": (damage_distribution,
        "Breakdown of buildings by damage class. args: region"),
    "quadrant_ranking": (quadrant_ranking,
        'Which quadrant has the most buildings of a damage level. args: damage (default "Destroyed")'),
    "compare_regions": (compare_regions,
        "Compare two regions (e.g. left vs right half). args: region_a, region_b, damage"),
    "list_buildings": (list_buildings,
        "List/highlight specific buildings. args: damage, region, limit"),
    "building_details": (building_details,
        "Details for one building. args: building_id (e.g. B012)"),
    "unanswerable": (unanswerable,
        "Use when the data cannot answer (cause, casualties, roads, flooding, future, anything not about buildings). args: reason"),
}
