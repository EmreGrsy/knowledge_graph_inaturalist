# Biodiversity Knowledge Graph App

A Streamlit app that turns iNaturalist observations into a knowledge graph and exposes it through SPARQL and a natural language interface.

## Pipeline

1. **LinkML schema** (`kg/schema.yaml`) defines the data model as the single source of truth.
2. **OWL** (`kg/schema.owl`) and **SHACL** (`kg/shapes.ttl`) are generated from the LinkML schema. OWL carries the class hierarchy for SPARQL reasoning. SHACL carries the constraints for record validation.
3. **Loader** (`kg/load.py`) fetches iNaturalist observations, maps fields onto standard RDF terms (DarwinCore, NCBITaxon, PROV-O), validates each record with pyshacl, and writes passing records to `data/observations.ttl`.
4. **Oxigraph** loads the Turtle snapshot together with the OWL ontology into an in memory SPARQL store.
5. **GraphRAG** (`rag/pipeline.py`) translates natural language questions to SPARQL using OpenAI `gpt-4o-mini`, executes the SPARQL on Oxigraph, then summarizes the result rows back as text with iNaturalist URIs as citations.
6. **Streamlit** serves the UI. The Query page accepts SPARQL or natural language. The Dashboard page renders Altair charts driven by canned SPARQL queries against the same graph.

## Running locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # set OPENAI_API_KEY
make schema                         # generate OWL + SHACL from LinkML
make load                           # fetch + validate iNaturalist data
make run                            # start the Streamlit app
```
