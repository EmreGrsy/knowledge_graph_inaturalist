# Ask Hamburg Biodiversity

A semantic search app for wildlife observations in the Hamburg region. Ask
in plain language; the app translates the question into SPARQL, queries a
knowledge graph built from iNaturalist data, and returns a grounded answer
with citations and a map.

## How it works

**Offline ingest** (`make schema && make load`):

1. `kg/schema.yaml` defines the data model in **LinkML** with three classes
   (`Observation`, `Taxon`, `Agent`) using terms from **Darwin Core**, **BFO**
   and **PROV-O**.
2. `gen-owl` and `gen-shacl` derive `kg/schema.owl` (loaded into the SPARQL
   store for class-hierarchy reasoning) and `kg/shapes.ttl` (used to validate
   every record at ingest).
3. `kg/load.py` fetches research-grade observations from iNaturalist inside a
   Hamburg bounding box, maps each one to RDF using the vocabularies above
   plus `rdfs:seeAlso` for Wikipedia links, validates with **pyshacl**, and
   writes the survivors to `data/observations.ttl` (~10K records).

**Online, per question:**

1. The Streamlit app loads `data/observations.ttl` and `kg/schema.owl` into an
   in-memory **Oxigraph** store.
2. A small `gpt-4o-mini` call extracts any place name from the question. If
   found, **OpenStreetMap Nominatim** returns its bounding box.
3. A second `gpt-4o-mini` call translates the question into SPARQL, using the
   schema description and the geocoded bbox as context.
4. Oxigraph runs the SPARQL on the local graph.
5. A third `gpt-4o-mini` call writes a short answer from the result rows,
   citing iNaturalist URIs.
6. The map redraws with the species from the answer, inside the same bbox.

Cost per question is about $0.001 (three `gpt-4o-mini` calls plus a free,
cached Nominatim lookup).

## Repository layout

```
app.py                  Streamlit chat UI and map
kg/
  schema.yaml           LinkML schema (single source of truth)
  schema.owl            generated OWL ontology
  shapes.ttl            generated SHACL shapes
  load.py               iNat fetch, SHACL validation, Turtle output
rag/
  pipeline.py           place extraction, geocoding, NL to SPARQL, summary
  prompts.py            LLM prompt templates
data/observations.ttl   committed RDF snapshot
```
