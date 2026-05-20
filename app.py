"""Streamlit app: natural-language and SPARQL interface over the biodiversity KG."""

from pathlib import Path

import pandas as pd
import pyoxigraph as ox
import streamlit as st
from openai import OpenAI

from kg.logging_config import get_logger
from rag.pipeline import extract_citations, nl_to_sparql, summarize_rows

log = get_logger("app")

st.set_page_config(
    page_title="Biodiversity Knowledge Graph",
    layout="wide",
)

DATA_FILE   = Path("data/observations.ttl")
SCHEMA_FILE = Path("kg/schema.owl")

DEFAULT_SPARQL = """\
PREFIX bio: <https://example.org/bio-kg/>
PREFIX dwc: <http://rs.tdwg.org/dwc/terms/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?species ?wiki (COUNT(?obs) AS ?n) WHERE {
  ?obs   a dwc:Occurrence ;
         bio:observedTaxon  ?taxon .
  ?taxon dwc:scientificName ?species .
  OPTIONAL { ?taxon rdfs:seeAlso ?wiki }
}
GROUP BY ?species ?wiki
ORDER BY DESC(?n)
LIMIT 10"""


# --- helpers ---------------------------------------------------------------


@st.cache_resource(show_spinner="Loading knowledge graph...")
def load_store(data_mtime: float, schema_mtime: float) -> ox.Store:
    # mtimes go into the cache key so the store auto-reloads when files change.
    del data_mtime, schema_mtime
    store = ox.Store()
    store.bulk_load(path=str(DATA_FILE),   format=ox.RdfFormat.TURTLE)
    store.bulk_load(path=str(SCHEMA_FILE), format=ox.RdfFormat.TURTLE)
    log.info("loaded oxigraph store: %d triples", len(store))
    return store


def _cell(term) -> object:
    if term is None:
        return None
    return term.value if hasattr(term, "value") else str(term)


def run_sparql(store: ox.Store, query: str) -> pd.DataFrame:
    qr = store.query(query)
    var_names = [v.value for v in qr.variables]
    rows = [{v: _cell(sol[v]) for v in var_names} for sol in qr]
    return pd.DataFrame(rows, columns=var_names)


def link_column_config(df: pd.DataFrame) -> dict:
    config = {}
    for col in df.columns:
        values = df[col].dropna().astype(str)
        if not values.empty and all(v.startswith(("http://", "https://")) for v in values):
            config[col] = st.column_config.LinkColumn(col, display_text=r"^.*/([^/]+)/?$")
    return config


@st.cache_data(show_spinner="Loading map...")
def get_map_data(_store: ox.Store, data_mtime: float) -> pd.DataFrame:
    del data_mtime  # part of the cache key only
    qr = _store.query(
        "PREFIX dwc: <http://rs.tdwg.org/dwc/terms/> "
        "SELECT ?lat ?lon WHERE { "
        "  ?obs a dwc:Occurrence ; "
        "       dwc:decimalLatitude  ?lat ; "
        "       dwc:decimalLongitude ?lon . "
        "}"
    )
    rows = []
    for sol in qr:
        try:
            rows.append({
                "lat": float(sol["lat"].value),
                "lon": float(sol["lon"].value),
            })
        except (ValueError, KeyError):
            pass
    return pd.DataFrame(rows)


# --- page body -------------------------------------------------------------


store = load_store(
    DATA_FILE.stat().st_mtime,
    SCHEMA_FILE.stat().st_mtime,
)

with st.sidebar:
    st.markdown("### Graph stats")
    st.metric("Total triples", f"{len(store):,}")
    for r in store.query(
        "PREFIX dwc: <http://rs.tdwg.org/dwc/terms/> "
        "SELECT (COUNT(?o) AS ?n) WHERE { ?o a dwc:Occurrence }"
    ):
        st.metric("Observations", f"{int(r['n'].value):,}")
        break

    st.markdown("### About")
    st.markdown(
        "Research-grade iNaturalist observations from the Hamburg "
        "bounding box. Stack: LinkML, OWL, SHACL, Oxigraph, Streamlit, "
        "OpenAI `gpt-4o-mini`."
    )

st.title("Biodiversity Knowledge Graph")
st.caption(
    "Natural-language and SPARQL access over research-grade iNaturalist "
    "observations around Hamburg."
)

# --- Map -------------------------------------------------------------------

st.markdown("#### Observation map")
st.map(get_map_data(store, DATA_FILE.stat().st_mtime), size=5)

# --- GraphRAG --------------------------------------------------------------

_api_key = ""
_secrets_error: str | None = None
try:
    _api_key = (st.secrets.get("OPENAI_API_KEY", "") or "").strip()
except Exception as _exc:
    _secrets_error = str(_exc)
_has_key = _api_key.startswith("sk-")

st.markdown("#### Ask in English (GraphRAG)")
if _secrets_error:
    st.error(
        f"Could not read `.streamlit/secrets.toml`: {_secrets_error}. "
        'TOML string values must be in double quotes, e.g. `OPENAI_API_KEY = "sk-..."`.'
    )
elif not _has_key:
    st.info(
        "Add `OPENAI_API_KEY` to `.streamlit/secrets.toml` to enable the "
        "natural-language interface. You can still write SPARQL directly below."
    )

with st.form("nl_ask_form", clear_on_submit=False, border=False):
    nl_question = st.text_input(
        "Ask in English",
        placeholder="e.g., What's the most observed bird in Hamburg?",
        key="nl_question",
        label_visibility="collapsed",
        disabled=not _has_key,
    )
    nl_submitted = st.form_submit_button(
        "Ask",
        type="primary",
        disabled=not _has_key,
    )

if nl_submitted and nl_question and nl_question.strip():
    # 60s per OpenAI call so a hung request fails visibly rather than spinning forever.
    client = OpenAI(api_key=_api_key, timeout=60.0)
    nl_sparql, nl_df, nl_answer, nl_citations = "", None, "", []
    with st.status("Running GraphRAG pipeline...", expanded=True) as status:
        try:
            st.write("**Step 1** — Translating to SPARQL...")
            nl_sparql = nl_to_sparql(client, nl_question.strip())
            st.code(nl_sparql, language="sparql")

            st.write("**Step 2** — Querying the graph...")
            nl_df = run_sparql(store, nl_sparql)
            st.write(f"Got **{len(nl_df)}** row(s).")
            if not nl_df.empty:
                st.dataframe(
                    nl_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config=link_column_config(nl_df),
                )

            st.write("**Step 3** — Composing grounded answer...")
            nl_rows = nl_df.to_dict("records")
            nl_answer = summarize_rows(
                client, nl_question.strip(), nl_sparql, nl_rows,
            )
            nl_citations = extract_citations(nl_rows)
            status.update(label="GraphRAG pipeline complete", state="complete")
        except Exception as exc:
            status.update(label=f"Pipeline failed: {exc}", state="error")
            log.warning("graphRAG failed: %s", exc)

    if nl_sparql:
        st.session_state.sparql_text = nl_sparql

    if nl_answer:
        st.markdown("#### Answer")
        st.markdown(nl_answer)
        if nl_citations:
            st.markdown(
                "**Sources:** "
                + " · ".join(
                    f"[obs/{c.rsplit('/', 1)[-1]}]({c})"
                    for c in nl_citations[:10]
                )
            )

st.markdown("---")

# --- SPARQL editor ---------------------------------------------------------

st.markdown("#### SPARQL")
sparql = st.text_area(
    "SPARQL query",
    value=DEFAULT_SPARQL,
    height=220,
    key="sparql_text",
    label_visibility="collapsed",
)

if st.button("Run query", type="primary"):
    with st.spinner("Running..."):
        try:
            df = run_sparql(store, sparql)
        except Exception as exc:
            st.error(f"Query failed: {exc}")
            log.warning("query failed: %s", exc)
        else:
            st.markdown(f"#### Results — {len(df)} row(s)")
            if df.empty:
                st.info("No matches.")
            else:
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config=link_column_config(df),
                )
