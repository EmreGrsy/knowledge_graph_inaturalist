"""Streamlit app: natural-language interface over the biodiversity knowledge graph."""

from pathlib import Path

import pandas as pd
import pydeck as pdk
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

PALETTE = [
    [255,  99, 132], [ 54, 162, 235], [255, 206,  86], [ 75, 192, 192],
    [153, 102, 255], [255, 159,  64], [ 99, 200, 255], [255,  99, 255],
    [ 99, 255, 132], [120, 120, 120],
]

_OBS_URI_PREFIX = "https://www.inaturalist.org/observations/"


# --- helpers ---------------------------------------------------------------


@st.cache_resource(show_spinner="Loading knowledge graph...")
def load_store(data_mtime: float, schema_mtime: float) -> ox.Store:
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
def get_all_observation_points(_store: ox.Store, data_mtime: float) -> pd.DataFrame:
    del data_mtime
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
            rows.append({"lat": float(sol["lat"].value), "lon": float(sol["lon"].value)})
        except (ValueError, KeyError):
            pass
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_species_points(_store: ox.Store, species_key: tuple, data_mtime: float) -> pd.DataFrame:
    del data_mtime
    species_inline = " ".join(f'"{s}"' for s in species_key)
    qr = _store.query(
        "PREFIX bio: <https://example.org/bio-kg/> "
        "PREFIX dwc: <http://rs.tdwg.org/dwc/terms/> "
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
        f"SELECT ?species ?wiki ?lat ?lon WHERE {{ "
        f"  VALUES ?species {{ {species_inline} }} "
        f"  ?obs   a dwc:Occurrence ; "
        f"         bio:observedTaxon  ?taxon ; "
        f"         dwc:decimalLatitude  ?lat ; "
        f"         dwc:decimalLongitude ?lon . "
        f"  ?taxon dwc:scientificName ?species . "
        f"  OPTIONAL {{ ?taxon rdfs:seeAlso ?wiki }} "
        f"}}"
    )
    rows = []
    for sol in qr:
        try:
            rows.append({
                "species": sol["species"].value,
                "wiki":    sol["wiki"].value if sol["wiki"] is not None else "",
                "lat":     float(sol["lat"].value),
                "lon":     float(sol["lon"].value),
            })
        except (ValueError, KeyError):
            pass
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_obs_points(_store: ox.Store, obs_key: tuple, data_mtime: float) -> pd.DataFrame:
    del data_mtime
    obs_inline = " ".join(f"<{u}>" for u in obs_key)
    qr = _store.query(
        "PREFIX bio: <https://example.org/bio-kg/> "
        "PREFIX dwc: <http://rs.tdwg.org/dwc/terms/> "
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
        f"SELECT ?obs ?species ?wiki ?lat ?lon WHERE {{ "
        f"  VALUES ?obs {{ {obs_inline} }} "
        f"  ?obs   a dwc:Occurrence ; "
        f"         dwc:decimalLatitude  ?lat ; "
        f"         dwc:decimalLongitude ?lon ; "
        f"         bio:observedTaxon    ?taxon . "
        f"  ?taxon dwc:scientificName ?species . "
        f"  OPTIONAL {{ ?taxon rdfs:seeAlso ?wiki }} "
        f"}}"
    )
    rows = []
    for sol in qr:
        try:
            rows.append({
                "obs":     sol["obs"].value,
                "species": sol["species"].value,
                "wiki":    sol["wiki"].value if sol["wiki"] is not None else "",
                "lat":     float(sol["lat"].value),
                "lon":     float(sol["lon"].value),
            })
        except (ValueError, KeyError):
            pass
    return pd.DataFrame(rows)


def render_species_map(df: pd.DataFrame, species_list: list[str], key: str) -> None:
    """Pydeck scatter map, one colour per species. Hover shows the scientific
    name; click a species in the legend below to open its Wikipedia article."""
    colors = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(species_list)}
    df = df.copy()
    df["color"] = df["species"].map(lambda s: colors.get(s, [128, 128, 128]) + [230])

    # Species -> first wiki, for the legend below the map.
    species_to_wiki: dict[str, str] = {}
    for s in species_list:
        wikis = [w for w in df.loc[df["species"] == s, "wiki"].dropna().astype(str) if w]
        if wikis:
            species_to_wiki[s] = wikis[0]

    deck = pdk.Deck(
        map_style="light",
        initial_view_state=pdk.ViewState(
            latitude=float(df["lat"].mean()),
            longitude=float(df["lon"].mean()),
            zoom=10,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_line_color=[40, 40, 40, 200],
                line_width_min_pixels=1,
                get_radius=80,
                radius_min_pixels=6,
                radius_max_pixels=14,
                stroked=True,
                pickable=True,
            ),
        ],
        tooltip={
            "html": "<b><i>{species}</i></b>",
            "style": {"backgroundColor": "white", "color": "black",
                      "padding": "6px 10px", "border": "1px solid #ccc",
                      "borderRadius": "4px", "fontFamily": "sans-serif"},
        },
    )
    st.pydeck_chart(deck, use_container_width=True, key=key)

    # Legend, clickable (no arrows).
    cols = st.columns(min(len(species_list), 5))
    for i, s in enumerate(species_list):
        r, g, b = colors[s]
        col = cols[i % len(cols)]
        wiki = species_to_wiki.get(s)
        name_html = (
            f'<a href="{wiki}" target="_blank" style="text-decoration:none;color:inherit">'
            f'<em>{s}</em></a>'
            if wiki else f'<em>{s}</em>'
        )
        col.markdown(
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">'
            f'<div style="width:14px;height:14px;background:rgb({r},{g},{b});'
            f'border:1px solid #333;border-radius:50%;flex-shrink:0"></div>'
            f'<small>{name_html}</small></div>',
            unsafe_allow_html=True,
        )


def update_map_from_result(df: pd.DataFrame | None) -> None:
    """Prefer observation URIs (most specific); fall back to species."""
    if df is None or df.empty:
        return
    for col in df.columns:
        vals = df[col].dropna().astype(str)
        if vals.empty:
            continue
        if vals.str.startswith(_OBS_URI_PREFIX).all():
            st.session_state["map_obs_uris"] = vals.unique().tolist()[:200]
            st.session_state.pop("map_species", None)
            return
    if "species" in df.columns:
        species = df["species"].dropna().astype(str).unique().tolist()
        if species:
            st.session_state["map_species"] = species[:10]
            st.session_state.pop("map_obs_uris", None)


# --- page body -------------------------------------------------------------


store = load_store(
    DATA_FILE.stat().st_mtime,
    SCHEMA_FILE.stat().st_mtime,
)

with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "Data from [iNaturalist](https://www.inaturalist.org/). Observations "
        "are modelled as a knowledge graph that you can query in plain English."
    )

st.title("Biodiversity Knowledge Graph")
st.caption(
    "Ask in plain English. The map below switches from all observations "
    "to the species in your answer; click a name in the legend to open its Wikipedia article."
)

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
        "natural-language interface."
    )

with st.form("nl_ask_form", clear_on_submit=False, border=False):
    nl_question = st.text_input(
        "Ask in English",
        placeholder="e.g., What are the top 5 birds in Hamburg?",
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
    client = OpenAI(api_key=_api_key, timeout=60.0)
    nl_sparql, nl_df, nl_answer, nl_citations = "", None, "", []
    with st.status("Running pipeline...", expanded=True) as status:
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
            status.update(label="Done", state="complete")
        except Exception as exc:
            status.update(label=f"Pipeline failed: {exc}", state="error")
            log.warning("graphRAG failed: %s", exc)

    update_map_from_result(nl_df)

    if nl_answer:
        # Persist the answer so it survives reruns triggered by map / button clicks.
        st.session_state["last_answer"] = {
            "question":  nl_question.strip(),
            "answer":    nl_answer,
            "citations": nl_citations,
        }

# --- Render the most recent answer (from session state) --------------------

_last = st.session_state.get("last_answer")
if _last:
    st.markdown("#### Answer")
    st.markdown(_last["answer"])
    if _last.get("citations"):
        st.markdown(
            "**Sources:** "
            + " · ".join(
                f"[obs/{c.rsplit('/', 1)[-1]}]({c})"
                for c in _last["citations"][:10]
            )
        )

# --- Map (under the answer) -------------------------------------------------

map_obs_uris = st.session_state.get("map_obs_uris")
map_species  = st.session_state.get("map_species")

if map_obs_uris:
    df_pts = get_obs_points(store, tuple(map_obs_uris), DATA_FILE.stat().st_mtime)
    species_in_view = df_pts["species"].dropna().unique().tolist() if not df_pts.empty else []
    st.markdown(
        f"#### Observation map — **{len(df_pts)}** observation(s) from your last query"
    )
    if df_pts.empty:
        st.info("Your query returned observation URIs but their coordinates aren't in the snapshot.")
    else:
        render_species_map(df_pts, species_in_view, key="obs_map")
elif map_species:
    df_pts = get_species_points(store, tuple(map_species), DATA_FILE.stat().st_mtime)
    st.markdown(
        f"#### Observation map — **{len(map_species)}** species from your last query"
    )
    if df_pts.empty:
        st.info("Your query returned species, but the snapshot has no observations for them.")
    else:
        render_species_map(df_pts, map_species, key="species_map")
else:
    st.markdown("#### Observation map")
    st.map(get_all_observation_points(store, DATA_FILE.stat().st_mtime), size=8)
