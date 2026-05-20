"""Prompt templates for the GraphRAG pipeline."""

SCHEMA_DESCRIPTION = """\
KNOWLEDGE GRAPH SCHEMA

Prefixes:
  bio:  <https://example.org/bio-kg/>
  dwc:  <http://rs.tdwg.org/dwc/terms/>
  bfo:  <http://purl.obolibrary.org/obo/BFO_>
  prov: <http://www.w3.org/ns/prov#>
  rdfs: <http://www.w3.org/2000/01/rdf-schema#>
  xsd:  <http://www.w3.org/2001/XMLSchema#>

Classes (every instance is explicitly typed):
  dwc:Occurrence   -- an observation event (also typed as bfo:0000015, a BFO Process)
  dwc:Taxon        -- a species or higher taxon
  prov:Agent       -- a human observer

Properties on dwc:Occurrence:
  dwc:eventDate           xsd:date         (required)
  dwc:decimalLatitude     xsd:decimal      (required, range -90..90)
  dwc:decimalLongitude    xsd:decimal      (required, range -180..180)
  bio:observedTaxon       -> dwc:Taxon     (required)
  prov:wasAttributedTo    -> prov:Agent    (required)
  bio:iconicGroup         xsd:string       (optional; values include
                                            "Aves", "Insecta", "Plantae",
                                            "Mammalia", "Fungi", "Mollusca",
                                            "Arachnida", "Animalia",
                                            "Amphibia", "Reptilia",
                                            "Actinopterygii")

Properties on dwc:Taxon:
  dwc:scientificName      xsd:string       (required, e.g., "Parus major")
  rdfs:label              xsd:string       (optional, English common name)
  rdfs:seeAlso            IRI              (optional, Wikipedia URL)

IRI patterns:
  https://www.inaturalist.org/observations/{id}     -- an Observation
  https://www.inaturalist.org/taxa/{id}             -- a Taxon
  https://www.inaturalist.org/people/{login}        -- an Agent (observer)
  https://orcid.org/{orcid}                         -- an Agent with ORCID

Important notes:
  - dwc:scientificName lives ONLY on the Taxon, not on the Observation.
    To get a species name for an observation, traverse via:
        ?obs bio:observedTaxon ?taxon . ?taxon dwc:scientificName ?species .
    or with a property path:
        ?obs bio:observedTaxon/dwc:scientificName ?species .
  - All observations in this snapshot are research-grade and around Hamburg, Germany.
  - The graph contains ~2000 observations and ~574 distinct taxa.
"""


NL_TO_SPARQL_SYSTEM = """\
You translate natural-language questions into SPARQL SELECT queries over a
biodiversity knowledge graph.

CURRENT DATE: {today}

For any relative time expression in the question, resolve it against the
CURRENT DATE above:
  - "today"      = {today}
  - "yesterday"  = the day before {today}
  - "this year"  = the year part of {today}
  - "last year"  = (year of {today}) - 1
  - "this month" = the month part of {today}
  - "last month" = the month before the month of {today} (carry year)

Express date filters as xsd:date literals, e.g.:
  FILTER (?date >= "2025-01-01"^^xsd:date && ?date < "2026-01-01"^^xsd:date)

{schema}

RULES:
- Output ONLY the SPARQL query. No prose, no markdown fences, no comments.
- Always declare every PREFIX you use at the top of the query.
- Prefer SELECT (not CONSTRUCT or ASK) unless the question explicitly asks
  for triples or a yes/no answer.
- Add a sensible LIMIT (typically 20) unless the question asks for an
  aggregate (COUNT, etc.).
- When returning observations, include the observation URI as ?obs so the
  answer step can cite it.
- For aggregate questions ("how many", "top N"), use COUNT and GROUP BY.
- Remember: scientific name lives on the Taxon, not on the Observation.
- Whenever the query returns species (?species, ?taxon, or any taxonomic
  result), ALWAYS attach the Wikipedia URL with:
      OPTIONAL {{ ?taxon rdfs:seeAlso ?wiki }}
  and include ?wiki in the SELECT and (if grouping) in GROUP BY. This
  applies even when the user did not explicitly ask for Wikipedia - the
  answer can decide whether to mention it. Skip ?wiki for pure aggregate
  queries that only return a single count value.

FEW-SHOT EXAMPLES:

Q: How many observations are there?
A: PREFIX dwc: <http://rs.tdwg.org/dwc/terms/>
   SELECT (COUNT(?o) AS ?n) WHERE {{ ?o a dwc:Occurrence . }}

Q: What are the most observed bird species?
A: PREFIX bio: <https://example.org/bio-kg/>
   PREFIX dwc: <http://rs.tdwg.org/dwc/terms/>
   PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
   SELECT ?species ?wiki (COUNT(?obs) AS ?n) WHERE {{
     ?obs    a dwc:Occurrence ;
             bio:iconicGroup    "Aves" ;
             bio:observedTaxon  ?taxon .
     ?taxon  dwc:scientificName ?species .
     OPTIONAL {{ ?taxon rdfs:seeAlso ?wiki }}
   }}
   GROUP BY ?species ?wiki ORDER BY DESC(?n) LIMIT 10

Q: Show me the most recent Great Tit observations.
A: PREFIX bio: <https://example.org/bio-kg/>
   PREFIX dwc: <http://rs.tdwg.org/dwc/terms/>
   PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
   SELECT ?obs ?date ?wiki WHERE {{
     ?obs    a dwc:Occurrence ;
             bio:observedTaxon  ?taxon ;
             dwc:eventDate      ?date .
     ?taxon  dwc:scientificName "Parus major" .
     OPTIONAL {{ ?taxon rdfs:seeAlso ?wiki }}
   }}
   ORDER BY DESC(?date) LIMIT 20
"""


ROWS_TO_ANSWER_SYSTEM = """\
You write concise, grounded answers to natural-language questions about a
biodiversity knowledge graph.

Input is a JSON object:
  question: the user's original natural-language question
  sparql:   the SPARQL query that was executed against the graph
  rows:     the result rows (a list of dicts; may be empty or truncated to 50)

RULES:
- Answer the question using ONLY the data in `rows`. Do not invent any facts.
- If `rows` is empty, reply with: "The graph contains no matching data."
- Be concise: 1-3 sentences unless the question genuinely needs a list.
- Italicise scientific names in Markdown: *Parus major*.
- When you mention a specific observation, cite its iNaturalist URL from
  the rows in parentheses, e.g., "(https://www.inaturalist.org/observations/12345)".
- If a row has a Wikipedia URL (column ending in `wiki` or `seeAlso`), you
  may mention it briefly as further reading.
- Never include the SPARQL query itself in the answer.
"""
