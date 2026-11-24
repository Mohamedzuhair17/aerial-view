"""Question -> tool choice -> deterministic result -> natural-language answer.

Works with or without an LLM: if none is configured (or it errors / returns junk),
a keyword router and the tools' built-in sentences are used instead.
"""
import inspect
import json
import logging
import re

from app.spatial import tools as T

from .client import LLM

log = logging.getLogger("agent")

PLANNER_SYSTEM = """You route questions about a post-disaster satellite damage analysis to exactly ONE tool.
Reply with JSON only, no prose, no markdown: {{"tool": "<name>", "args": {{...}}}}

Tools:
{tools}

Damage classes: No Damage, Minor, Major, Destroyed, Unclassified.
Quadrants: NW, NE, SW, SE (north is the top of the image). Halves: left, right, top, bottom.
Rules: only use args listed for the tool. If the question cannot be answered from building counts,
damage classes and positions, use "unanswerable"."""

def _tool_list() -> str:
    return "\n".join(f"- {n}: {d}" for n, (_, d) in T.TOOLS.items())


# ------------------------------------------------------------------ rule-based router
_DMG_PATTERNS = [
    (r"no damage|undamaged|intact|unharmed", "No Damage"), (r"destroy|collaps|flatten", "Destroyed"),
    (r"\bmajor", "Major"), (r"\bminor", "Minor"), (r"severe", "severe"),
    (r"unclassified|unknown", "Unclassified"), (r"damage|affected|hit", "damaged"),
]
_REGION_KEYS = sorted((k for k in T._REGION_ALIASES if k not in ("all", "whole", "entire", "image")), key=len, reverse=True)


def _find_damage(t: str):
    for pat, val in _DMG_PATTERNS:
        if re.search(pat, t):
            return val
    return None


def _find_regions(t: str) -> list[str]:
    found: list[tuple[int, str]] = []
    for k in _REGION_KEYS:
        for m in re.finditer(rf"(?<![\w-]){re.escape(k)}(?![\w-])", t):
            found.append((m.start(), T._REGION_ALIASES[k]))
    out: list[str] = []
    for _, r in sorted(found):
        if r not in out:
            out.append(r)
    return out


def rule_plan(q: str) -> tuple[str, dict]:
    t = q.lower()
    if m := re.search(r"\bb[\s-]?0*(\d{1,4})\b", t):
        return "building_details", {"building_id": f"B{int(m.group(1)):03d}"}
    if re.search(r"\b(why|cause|casualt|dead|death|injur|people|population|road|bridge|flood|cost|rebuild|predict|forecast|when|how long)\b", t) \
            and "building" not in t:
        return "unanswerable", {"reason": "That is outside what the building-level analysis measures."}
    damage, regions = _find_damage(t), _find_regions(t)
    if len(regions) >= 2 or re.search(r"\b(compare|versus|vs\.?|which half|more .* than)\b", t):
        a, b = (regions + ["left", "right"])[:2] if len(regions) >= 2 else (("top", "bottom") if "top" in t or "bottom" in t else ("left", "right"))
        return "compare_regions", {"region_a": a, "region_b": b, "damage": damage}
    if "quadrant" in t or re.search(r"\b(most|worst|highest|heaviest)\b.*\b(region|area|part|zone|side)\b", t):
        return "quadrant_ranking", {"damage": damage}
    if re.search(r"distribution|breakdown|summary|overview|how bad|each (class|category)|categories", t):
        return "damage_distribution", {"region": regions[0] if regions else "all"}
    if re.search(r"\b(list|show|highlight|which buildings|ids)\b", t):
        return "list_buildings", {"damage": damage, "region": regions[0] if regions else "all"}
    if re.search(r"how many|count|number of|total", t) or damage:
        return "count_buildings", {"damage": damage, "region": regions[0] if regions else "all"}
    return "unanswerable", {"reason": "Try asking about building counts, damage levels or quadrants."}


# ------------------------------------------------------------------ LLM planning
def _parse_json(s: str):
    s = re.sub(r"```(?:json)?", "", s)
    a, b = s.find("{"), s.rfind("}")
    return json.loads(s[a:b + 1]) if a != -1 and b > a else None


def _llm_plan(llm: LLM, q: str, total: int):
    raw = llm.complete(PLANNER_SYSTEM.format(tools=_tool_list()), f"Total buildings in image: {total}\nQuestion: {q}")
    plan = _parse_json(raw)
    if not isinstance(plan, dict) or plan.get("tool") not in T.TOOLS:
        return None
    args = plan.get("args") if isinstance(plan.get("args"), dict) else {}
    # Keep the LLM strictly within a small, typed argument surface.
    allowed = set(inspect.signature(T.TOOLS[plan["tool"]][0]).parameters) - {"buildings"}
    args = {key: value for key, value in args.items() if key in allowed}
    if plan["tool"] == "unanswerable":
        args = {"reason": "The available building-level analysis cannot answer that question."}
    if plan["tool"] == "list_buildings" and "limit" in args:
        try:
            args["limit"] = max(1, min(100, int(args["limit"])))
        except (TypeError, ValueError):
            args["limit"] = 20
    return plan["tool"], args


def _run_tool(name: str, args: dict, buildings: list[dict]) -> dict:
    fn = T.TOOLS[name][0]
    allowed = set(inspect.signature(fn).parameters) - {"buildings"}
    clean = {k: v for k, v in args.items() if k in allowed and v is not None}
    return fn(buildings, **clean)


def _explicitly_out_of_scope(question: str) -> bool:
    return bool(re.search(
        r"\b(casualt(?:y|ies)|fatalit(?:y|ies)|deaths?|injur(?:y|ies)|population|people|"
        r"roads?|bridges?|flood(?:ing| depth)?|cause of (?:the )?disaster|reconstruction cost|"
        r"repair cost|future damage|predict(?:ion)?|forecast)\b",
        question.lower(),
    ))


def answer_question(question: str, buildings: list[dict], llm: LLM) -> dict:
    # Do not let an LLM route explicitly unsupported questions into a plausible-looking tool.
    if _explicitly_out_of_scope(question):
        result = T.unanswerable(buildings, reason="The available analysis contains building-level damage classes and image positions, not this information.")
        return {"answer": result["text"], "tool": "unanswerable", "args": {},
                "highlight_ids": [], "evidence": result, "reasoning_mode": "rules"}

    mode, plan = "rules", None
    if llm.available:
        try:
            plan = _llm_plan(llm, question, len(buildings))
            mode = "llm" if plan else "rules"
        except Exception as e:  # network, quota, bad JSON...
            log.warning("LLM planning failed: %s", e)
    name, args = plan or rule_plan(question)

    try:
        result = _run_tool(name, args, buildings)
    except (ValueError, TypeError) as e:
        log.info("tool args rejected (%s); falling back to rules", e)
        name, args = rule_plan(question)
        mode = "rules"
        try:
            result = _run_tool(name, args, buildings)
        except (ValueError, TypeError):
            name, args = "unanswerable", {"reason": "I couldn't interpret the damage level or region in that question."}
            result = T.unanswerable(buildings, **args)

    text = result["text"]
    # Final text is generated by deterministic spatial tools; LLM only plans tool selection.

    return {"answer": text, "tool": name, "args": args, "highlight_ids": result.get("ids", []),
            "evidence": result, "reasoning_mode": mode}
