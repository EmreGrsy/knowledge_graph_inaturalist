"""Streamlit chat app: natural-language interface over the biodiversity knowledge graph."""

import re
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
    page_title="Ask Hamburg Biodiversity",
    layout="wide",
)

# Headings in forest green to fit the nature aspect; make past user questions
# share the same cream background as the chat_input box for a uniform look.
st.markdown(
    """
    <style>
        h1, h2, h3, h4, h5, h6 {
            color: #2D5016 !important;
        }
        /* Past user question bubbles share the cream tone with the input. */
        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
            background-color: #E8E2D5 !important;
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
        }
        /* Drop the white avatar circle so the leaf blends into whatever
           background it's sitting on (cream bubble or page). */
        [data-testid="stChatMessageAvatar"],
        [data-testid="chatAvatarIcon-user"],
        [data-testid="chatAvatarIcon-assistant"] {
            background-color: transparent !important;
            box-shadow: none !important;
        }
        /* Phone-friendly map height (default is ~600px, way too tall on a phone). */
        @media (max-width: 768px) {
            div[data-testid="stDeckGlJsonChart"],
            .stDeckGlJsonChart,
            iframe[title*="pydeck"] {
                height: 320px !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
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
def get_species_points(
    _store: ox.Store,
    species_key: tuple,
    data_mtime: float,
    bbox: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    del data_mtime
    species_inline = " ".join(f'"{s}"' for s in species_key)
    bbox_filter = ""
    if bbox:
        lat_min, lat_max, lon_min, lon_max = bbox
        bbox_filter = (
            f"  FILTER (?lat >= {lat_min} && ?lat <= {lat_max} && "
            f"?lon >= {lon_min} && ?lon <= {lon_max}) "
        )
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
        f"{bbox_filter}"
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
    colors = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(species_list)}
    df = df.copy()
    df["color"] = df["species"].map(lambda s: colors.get(s, [128, 128, 128]) + [230])

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


_CMP_RE = re.compile(r"\?(\w+)\s*(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)")


def extract_bbox(sparql: str) -> tuple[float, float, float, float] | None:
    """If the SPARQL contains lat/lon range FILTERs, return (lat_min, lat_max,
    lon_min, lon_max). The map's follow-up query reuses the same bbox so the
    plotted observations stay inside the user's geographic constraint."""
    lat_lo = lat_hi = lon_lo = lon_hi = None
    for var, op, num in _CMP_RE.findall(sparql):
        try:
            n = float(num)
        except ValueError:
            continue
        v = var.lower()
        if "lat" in v:
            if op in (">=", ">"):
                lat_lo = n if lat_lo is None else min(lat_lo, n)
            else:
                lat_hi = n if lat_hi is None else max(lat_hi, n)
        elif "lon" in v or "long" in v:
            if op in (">=", ">"):
                lon_lo = n if lon_lo is None else min(lon_lo, n)
            else:
                lon_hi = n if lon_hi is None else max(lon_hi, n)
    if None not in (lat_lo, lat_hi, lon_lo, lon_hi):
        return (lat_lo, lat_hi, lon_lo, lon_hi)
    return None


def update_map_from_result(df: pd.DataFrame | None, sparql: str = "") -> None:
    if df is None or df.empty:
        return

    bbox = extract_bbox(sparql) if sparql else None
    if bbox:
        st.session_state["map_bbox"] = bbox
    else:
        st.session_state.pop("map_bbox", None)

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


def render_map_for_state(store: ox.Store) -> None:
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
        map_bbox = st.session_state.get("map_bbox")
        df_pts = get_species_points(
            store, tuple(map_species), DATA_FILE.stat().st_mtime, bbox=map_bbox,
        )
        st.markdown(
            f"#### Observation map — **{len(map_species)}** species from your last query"
        )
        if df_pts.empty:
            st.info("Your query returned species, but the snapshot has no observations for them.")
        else:
            render_species_map(df_pts, map_species, key="species_map")
    else:
        st.markdown("#### Observation map")
        df_all = get_all_observation_points(store, DATA_FILE.stat().st_mtime)
        df_all = df_all.copy()
        # Forest green dots matching the theme; semi-transparent so the ~10K
        # points form a density gradient instead of one solid blob.
        df_all["color"] = [[45, 80, 22, 100]] * len(df_all)
        all_deck = pdk.Deck(
            map_style="light",
            initial_view_state=pdk.ViewState(
                latitude=float(df_all["lat"].mean()),
                longitude=float(df_all["lon"].mean()),
                zoom=10,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df_all,
                    get_position="[lon, lat]",
                    get_fill_color="color",
                    get_line_color=[40, 40, 40, 200],
                    line_width_min_pixels=1,
                    get_radius=80,
                    radius_min_pixels=6,
                    radius_max_pixels=14,
                    stroked=True,
                    pickable=False,
                ),
            ],
        )
        st.pydeck_chart(all_deck, use_container_width=True, key="all_obs_map")


def render_assistant_message(msg: dict) -> None:
    """Render a saved assistant message, including its details expander."""
    sparql = msg.get("sparql")
    df_records = msg.get("df_records")
    if sparql or df_records:
        with st.expander("Details — generated SPARQL and result rows", expanded=False):
            if sparql:
                st.code(sparql, language="sparql")
            if df_records:
                df_disp = pd.DataFrame(df_records)
                st.dataframe(
                    df_disp,
                    use_container_width=True,
                    hide_index=True,
                    column_config=link_column_config(df_disp),
                )
    if msg.get("content"):
        st.markdown(msg["content"])
    if msg.get("citations"):
        st.markdown(
            "**Sources:** "
            + " · ".join(
                f"[obs/{c.rsplit('/', 1)[-1]}]({c})"
                for c in msg["citations"][:10]
            )
        )


# --- page body -------------------------------------------------------------


store = load_store(
    DATA_FILE.stat().st_mtime,
    SCHEMA_FILE.stat().st_mtime,
)

with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "**Ask Hamburg Biodiversity** is a semantic search app for "
        "wildlife observations in Hamburg, created by Emre Gürsoy.\n\n"
        "Species observations are sourced from [iNaturalist](https://www.inaturalist.org/), "
        "validated, and represented as an RDF knowledge graph using "
        "Darwin Core, BFO, and PROV-O ontologies.\n\n"
        "Users can ask questions in natural language. An LLM translates "
        "the question into SPARQL, queries the knowledge graph, and "
        "generates an answer from the retrieved results."
    )

st.title("Ask Hamburg Biodiversity")
st.caption(
    "Each dot is an observation of a species. "
    "The map updates based on user questions."
)

# --- Map (at the top, reflects the latest state) ---------------------------

render_map_for_state(store)

# --- Chat thread -----------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# OpenAI key check
_api_key = ""
_secrets_error: str | None = None
try:
    _api_key = (st.secrets.get("OPENAI_API_KEY", "") or "").strip()
except Exception as _exc:
    _secrets_error = str(_exc)
_has_key = _api_key.startswith("sk-")

if _secrets_error:
    st.error(
        f"Could not read `.streamlit/secrets.toml`: {_secrets_error}. "
        'TOML string values must be in double quotes, e.g. `OPENAI_API_KEY = "sk-..."`.'
    )
elif not _has_key:
    st.info(
        "Add `OPENAI_API_KEY` to `.streamlit/secrets.toml` to enable the chat."
    )

# Render the saved conversation. Both roles get the same leaf avatar so the
# thread reads in one consistent colour rather than red-face vs yellow-robot.
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="🌿"):
        if msg["role"] == "assistant":
            render_assistant_message(msg)
        else:
            st.markdown(msg["content"])

# New input (pinned at bottom of the page by Streamlit)
prompt = st.chat_input(
    "Ask about Hamburg biodiversity...",
    disabled=not _has_key,
)

if prompt and prompt.strip():
    prompt = prompt.strip()

    # 1) Echo the user's question immediately in the thread.
    with st.chat_message("user", avatar="🌿"):
        st.markdown(prompt)

    # 2) Run the pipeline inside an assistant bubble.
    client = OpenAI(api_key=_api_key, timeout=60.0)
    sparql, df, answer, citations = "", None, "", []
    with st.chat_message("assistant", avatar="🌿"):
        with st.status("Thinking...", expanded=True) as status:
            try:
                st.write("**Step 1** — Translating to SPARQL...")
                sparql = nl_to_sparql(client, prompt)
                st.code(sparql, language="sparql")

                st.write("**Step 2** — Querying the graph...")
                df = run_sparql(store, sparql)
                st.write(f"Got **{len(df)}** row(s).")
                if not df.empty:
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        column_config=link_column_config(df),
                    )

                st.write("**Step 3** — Writing the answer...")
                rows = df.to_dict("records")
                answer = summarize_rows(client, prompt, sparql, rows)
                citations = extract_citations(rows)
                status.update(label="Done", state="complete")
            except Exception as exc:
                status.update(label=f"Pipeline failed: {exc}", state="error")
                log.warning("graphRAG failed: %s", exc)
                answer = "Sorry, I couldn't answer that. See the status panel above."
        if answer:
            st.markdown(answer)
            if citations:
                st.markdown(
                    "**Sources:** "
                    + " · ".join(
                        f"[obs/{c.rsplit('/', 1)[-1]}]({c})"
                        for c in citations[:10]
                    )
                )

    # 3) Persist messages and refresh the map state.
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({
        "role":      "assistant",
        "content":   answer,
        "sparql":    sparql,
        "df_records": df.to_dict("records") if df is not None else None,
        "citations": citations,
    })
    update_map_from_result(df, sparql=sparql)
    st.rerun()
