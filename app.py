"""Single-page Streamlit app: SPARQL query interface over the biodiversity KG."""

from pathlib import Path

import pandas as pd
import pyoxigraph as ox
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

from kg.logging_config import get_logger

log = get_logger("app")

st.set_page_config(
    page_title="Biodiversity Knowledge Graph",
    layout="wide",
)

DATA_FILE   = Path("data/observations.ttl")
SCHEMA_FILE = Path("kg/schema.owl")


@st.cache_resource(show_spinner="Loading knowledge graph...")
def load_store() -> ox.Store:
    store = ox.Store()
    store.bulk_load(path=str(DATA_FILE),   format=ox.RdfFormat.TURTLE)
    store.bulk_load(path=str(SCHEMA_FILE), format=ox.RdfFormat.TURTLE)
    log.info("loaded oxigraph store: %d triples", len(store))
    return store


# --- helpers -----------------------------------------------------------------


def _cell(term) -> object:
    if term is None:
        return None
    return term.value if hasattr(term, "value") else str(term)


def _short(uri: str) -> str:
    """Last path segment or fragment of a URI, for compact labels."""
    return uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1] or uri


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


def render_subgraph(store: ox.Store, root_iri: str) -> None:
    """Draw the 1-hop outgoing neighborhood of `root_iri` as a node-edge graph."""
    nodes: dict[str, Node] = {}
    edges: list[Edge] = []

    nodes[root_iri] = Node(id=root_iri, label=_short(root_iri), color="#1E5631", size=28)

    try:
        results = list(store.query(f"SELECT ?p ?o WHERE {{ <{root_iri}> ?p ?o }}"))
    except Exception as exc:
        st.error(f"Could not query that URI: {exc}")
        return

    if not results:
        st.warning("That URI has no outgoing edges in the graph.")
        return

    for r in results:
        pred = _short(r["p"].value)
        obj = r["o"]
        if isinstance(obj, ox.NamedNode):
            obj_iri = obj.value
            if obj_iri not in nodes:
                nodes[obj_iri] = Node(id=obj_iri, label=_short(obj_iri), color="#7AA17A", size=20)
            edges.append(Edge(source=root_iri, target=obj_iri, label=pred))
        else:
            # Literal
            text = str(obj.value)
            shown = text if len(text) <= 30 else text[:27] + "..."
            lit_id = f"lit::{r['p'].value}::{text}"
            if lit_id not in nodes:
                nodes[lit_id] = Node(id=lit_id, label=f'"{shown}"', color="#CCCCCC", size=14)
            edges.append(Edge(source=root_iri, target=lit_id, label=pred))

    config = Config(height=500, width=900, directed=True, physics=True)
    agraph(nodes=list(nodes.values()), edges=edges, config=config)


# --- canned queries ----------------------------------------------------------


CANNED_QUERIES = {
    "Count observations": """\
PREFIX dwc: <http://rs.tdwg.org/dwc/terms/>

SELECT (COUNT(?o) AS ?n) WHERE {
  ?o a dwc:Occurrence .
}""",
    "Top 10 species": """\
PREFIX dwc: <http://rs.tdwg.org/dwc/terms/>

SELECT ?species (COUNT(?obs) AS ?n) WHERE {
  ?obs a dwc:Occurrence ;
       dwc:scientificName ?species .
}
GROUP BY ?species
ORDER BY DESC(?n)
LIMIT 10""",
    "Count by iconic group": """\
PREFIX bio: <https://example.org/bio-kg/>
PREFIX dwc: <http://rs.tdwg.org/dwc/terms/>

SELECT ?group (COUNT(?obs) AS ?n) WHERE {
  ?obs a dwc:Occurrence ;
       bio:iconicGroup ?group .
}
GROUP BY ?group
ORDER BY DESC(?n)""",
    "Recent bird observations": """\
PREFIX bio: <https://example.org/bio-kg/>
PREFIX dwc: <http://rs.tdwg.org/dwc/terms/>

SELECT ?obs ?species ?date WHERE {
  ?obs a dwc:Occurrence ;
       bio:iconicGroup     "Aves" ;
       dwc:scientificName  ?species ;
       dwc:eventDate       ?date .
}
ORDER BY DESC(?date)
LIMIT 20""",
}


def use_canned(query: str) -> None:
    st.session_state.sparql_text = query


# --- page body ---------------------------------------------------------------


store = load_store()

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
        "Data: ~2000 research-grade iNaturalist observations from the "
        "Hamburg bounding box. Stack: LinkML, OWL, SHACL, Oxigraph, Streamlit."
    )

st.title("Biodiversity Knowledge Graph")
st.caption(
    "SPARQL runs against an Oxigraph store containing the iNaturalist "
    "snapshot plus the LinkML-generated OWL ontology. URI columns in "
    "results are clickable. Visualize any URI as a graph in the section "
    "at the bottom."
)

st.markdown("#### Example queries")
cols = st.columns(len(CANNED_QUERIES))
for col, (name, query) in zip(cols, CANNED_QUERIES.items()):
    col.button(name, on_click=use_canned, args=(query,), use_container_width=True)

st.markdown("#### SPARQL")
sparql = st.text_area(
    "SPARQL query",
    value=CANNED_QUERIES["Top 10 species"],
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
            df = None
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
                # Stash first URI in any result column for the graph viewer below
                for col in df.columns:
                    vals = df[col].dropna().astype(str)
                    if not vals.empty and vals.iloc[0].startswith(("http://", "https://")):
                        st.session_state.inspect_uri_default = vals.iloc[0]
                        break

st.markdown("---")
st.markdown("### Inspect a URI as a graph")
st.caption(
    "Paste any URI from a result above (or one you know). The viewer "
    "draws its 1-hop outgoing neighborhood: the URI as a central green "
    "node, each property as an edge labelled with the predicate, and "
    "each target as either another URI node (light green) or a literal "
    "box (grey)."
)

default_uri = st.session_state.get(
    "inspect_uri_default",
    "https://www.inaturalist.org/observations/363131092",
)
inspect_uri = st.text_input(
    "URI to inspect",
    value=default_uri,
    key="inspect_uri_input",
)

if inspect_uri:
    render_subgraph(store, inspect_uri.strip())
