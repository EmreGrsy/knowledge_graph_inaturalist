"""GraphRAG pipeline: natural language -> SPARQL -> grounded answer.

The Streamlit app calls these in order per question:
``extract_place_name`` -> ``geocode_in_hamburg`` -> ``nl_to_sparql``
-> (graph executes the SPARQL) -> ``summarize_rows`` -> ``extract_citations``.
"""

from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache

import requests
from openai import OpenAI

from kg.logging_config import get_logger
from rag.prompts import (
    NL_TO_SPARQL_SYSTEM,
    ROWS_TO_ANSWER_SYSTEM,
    SCHEMA_DESCRIPTION,
)

log = get_logger("rag")

MODEL = "gpt-4o-mini"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_UA  = "ask-hamburg-biodiversity (github.com/EmreGrsy/knowledge_graph_inaturalist)"


def extract_place_name(client: OpenAI, question: str) -> str | None:
    """Ask the LLM to extract a Hamburg place/park/neighbourhood name, if any."""
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system",
                 "content": (
                     "Extract a single Hamburg place name (park, neighbourhood, "
                     "district, street, landmark) from the user's question. "
                     "Return a JSON object: {\"place\": \"<name>\"} if there is "
                     "one, or {\"place\": null} if no specific place is mentioned. "
                     "Return JSON only."
                 )},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        place = data.get("place")
        return place if place else None
    except Exception as exc:
        log.warning("place extraction failed: %s", exc)
        return None


@lru_cache(maxsize=512)
def geocode_in_hamburg(place: str) -> tuple[float, float, float, float] | None:
    """Geocode a place via Nominatim, scoped to Hamburg. Returns (lat_min, lat_max, lon_min, lon_max) or None."""
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={
                "q": f"{place}, Hamburg, Germany",
                "format": "json",
                "limit": 1,
            },
            headers={"User-Agent": NOMINATIM_UA},
            timeout=10,
        )
        results = resp.json() or []
        if not results:
            return None
        bbox_strs = results[0].get("boundingbox") or []
        if len(bbox_strs) != 4:
            return None
        # Nominatim order: [south, north, west, east]
        south, north, west, east = (float(x) for x in bbox_strs)
        return (south, north, west, east)
    except Exception as exc:
        log.warning("geocoding %r failed: %s", place, exc)
        return None

# Prefixes the model is allowed to use. If a generated query uses one of
# these without declaring it (a common LLM slip), `ensure_prefixes` patches
# the query rather than letting Oxigraph reject it.
STANDARD_PREFIXES: dict[str, str] = {
    "bio":  "https://example.org/bio-kg/",
    "dwc":  "http://rs.tdwg.org/dwc/terms/",
    "bfo":  "http://purl.obolibrary.org/obo/BFO_",
    "prov": "http://www.w3.org/ns/prov#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "rdf":  "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xsd":  "http://www.w3.org/2001/XMLSchema#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
}


def ensure_prefixes(sparql: str) -> str:
    """Prepend any standard PREFIX declarations the query uses but doesn't declare."""
    declared = {
        m.group(1)
        for m in re.finditer(r"PREFIX\s+(\w+)\s*:", sparql, re.IGNORECASE)
    }
    used = {p for p in STANDARD_PREFIXES if re.search(rf"\b{p}:\w", sparql)}
    missing = used - declared
    if not missing:
        return sparql
    log.info("auto-declaring missing prefixes: %s", ", ".join(sorted(missing)))
    prelude = "\n".join(
        f"PREFIX {p}: <{STANDARD_PREFIXES[p]}>" for p in sorted(missing)
    )
    return prelude + "\n" + sparql


def _strip_markdown_fence(text: str) -> str:
    """If the LLM wrapped its SPARQL in ``` fences despite instructions, strip them."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        # drop the opening fence
        lines = lines[1:]
        # drop the closing fence if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


def nl_to_sparql(client: OpenAI, question: str) -> str:
    """Translate an NL question to SPARQL via the LLM. Returns SPARQL text.

    If the question mentions a Hamburg place, look it up via Nominatim and
    inject the real bounding box into the prompt so the LLM doesn't have to
    guess coordinates (it tends to invent them for less-famous places).
    """
    log.info("nl -> sparql: %r", question)

    place = extract_place_name(client, question)
    bbox = geocode_in_hamburg(place) if place else None
    if place:
        log.info("detected place: %r -> bbox=%s", place, bbox)

    user_content = question
    if bbox:
        south, north, west, east = bbox
        user_content = (
            f"{question}\n\n"
            f"(Geocoded location hint for '{place}': bounding box is latitude "
            f"[{south:.4f}, {north:.4f}], longitude [{west:.4f}, {east:.4f}]. "
            f"Use these exact bounds in a FILTER on ?lat and ?lon.)"
        )

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": NL_TO_SPARQL_SYSTEM.format(
                 schema=SCHEMA_DESCRIPTION,
                 today=date.today().isoformat(),
             )},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
    )
    sparql = _strip_markdown_fence(completion.choices[0].message.content or "")
    return ensure_prefixes(sparql)


def summarize_rows(
    client: OpenAI,
    question: str,
    sparql: str,
    rows: list[dict],
) -> str:
    """Ask the LLM to compose a grounded answer from the SPARQL result rows."""
    log.info("summarizing %d rows", len(rows))
    payload = {
        "question": question,
        "sparql": sparql,
        "rows": rows[:50],  # cap so the prompt doesn't blow up
    }
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": ROWS_TO_ANSWER_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.3,
    )
    return (completion.choices[0].message.content or "").strip()


def extract_citations(rows: list[dict]) -> list[str]:
    """Pull iNaturalist observation URIs from result rows for use as citations."""
    cites: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for v in row.values():
            if (
                isinstance(v, str)
                and v.startswith("https://www.inaturalist.org/observations/")
                and v not in seen
            ):
                cites.append(v)
                seen.add(v)
    return cites


