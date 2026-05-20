"""GraphRAG pipeline: natural language -> SPARQL -> grounded answer.

The pipeline runs two LLM calls with a deterministic SPARQL query in
between. Call ``ask`` for the full end-to-end pipeline; or use
``nl_to_sparql``, ``rows_from_query``, ``summarize_rows`` separately if
you want to display each stage's progress live (as the Streamlit app does).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date

import pyoxigraph as ox
from openai import OpenAI

from kg.logging_config import get_logger
from rag.prompts import (
    NL_TO_SPARQL_SYSTEM,
    ROWS_TO_ANSWER_SYSTEM,
    SCHEMA_DESCRIPTION,
)

log = get_logger("rag")

MODEL = "gpt-4o-mini"

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


@dataclass
class AskResult:
    question: str
    sparql: str
    rows: list[dict]
    answer: str
    citations: list[str] = field(default_factory=list)
    error: str | None = None


def _term_to_py(term) -> object:
    if term is None:
        return None
    return term.value if hasattr(term, "value") else str(term)


def rows_from_query(qr) -> list[dict]:
    """Materialize a pyoxigraph QuerySolutions iterator into a list of dicts."""
    var_names = [v.value for v in qr.variables]
    return [{v: _term_to_py(sol[v]) for v in var_names} for sol in qr]


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
    """Translate an NL question to SPARQL via the LLM. Returns SPARQL text."""
    log.info("nl -> sparql: %r", question)
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": NL_TO_SPARQL_SYSTEM.format(
                 schema=SCHEMA_DESCRIPTION,
                 today=date.today().isoformat(),
             )},
            {"role": "user", "content": question},
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


def ask(question: str, store: ox.Store, client: OpenAI) -> AskResult:
    """Convenience wrapper: run the full pipeline and return an AskResult."""
    sparql = ""
    rows: list[dict] = []
    try:
        sparql = nl_to_sparql(client, question)
        rows = rows_from_query(store.query(sparql))
        answer = summarize_rows(client, question, sparql, rows)
        return AskResult(
            question=question,
            sparql=sparql,
            rows=rows,
            answer=answer,
            citations=extract_citations(rows),
        )
    except Exception as exc:
        log.exception("graphRAG pipeline failed")
        return AskResult(
            question=question,
            sparql=sparql,
            rows=rows,
            answer="",
            error=str(exc),
        )
