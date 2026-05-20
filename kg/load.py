"""Fetch iNaturalist observations, validate against SHACL, write RDF Turtle.

Run as ``python -m kg.load --limit 2000``. Defaults target a bounding box
around Hamburg. Each record is mapped onto the LinkML schema (DwC + BFO +
PROV-O + NCBITaxon prefixes), validated with pyshacl, and merged into the
output graph only if it conforms.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator

import requests
from pyshacl import validate as shacl_validate
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from kg.logging_config import get_logger

log = get_logger("loader")

BIO  = Namespace("https://example.org/bio-kg/")
DWC  = Namespace("http://rs.tdwg.org/dwc/terms/")
BFO  = Namespace("http://purl.obolibrary.org/obo/BFO_")
PROV = Namespace("http://www.w3.org/ns/prov#")
DCT  = Namespace("http://purl.org/dc/terms/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

INAT_API   = "https://api.inaturalist.org/v2/observations"
USER_AGENT = "bio-kg-loader (github.com/EmreGrsy/knowledge_graph_inaturalist)"
PER_PAGE   = 30
RATE_LIMIT = 1.0  # seconds between page requests

FIELDS = ",".join([
    "id", "uuid", "observed_on", "location", "quality_grade",
    "license_code", "created_at", "uri",
    "taxon.id", "taxon.name", "taxon.preferred_common_name", "taxon.iconic_taxon_name",
    "user.id", "user.login", "user.orcid",
])


def fetch_page(page: int, bbox: tuple[float, float, float, float]) -> list[dict]:
    swlat, swlng, nelat, nelng = bbox
    params = {
        "swlat": swlat, "swlng": swlng,
        "nelat": nelat, "nelng": nelng,
        "quality_grade": "research",
        "per_page": PER_PAGE,
        "page": page,
        "fields": FIELDS,
    }
    resp = requests.get(
        INAT_API,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def fetch_observations(limit: int, bbox: tuple[float, float, float, float]) -> Iterator[dict]:
    yielded = 0
    page = 1
    while yielded < limit:
        log.info("fetching page %d", page)
        try:
            results = fetch_page(page, bbox)
        except requests.HTTPError as e:
            log.error("API error on page %d: %s", page, e)
            return
        if not results:
            log.info("no more results")
            return
        for rec in results:
            if yielded >= limit:
                return
            yield rec
            yielded += 1
        page += 1
        time.sleep(RATE_LIMIT)


def build_record(rec: dict, graph: Graph) -> URIRef | None:
    """Translate one iNat record into triples in `graph`. Returns the Observation IRI."""
    uri = rec.get("uri")
    if not uri:
        return None
    obs = URIRef(uri)

    graph.add((obs, RDF.type, DWC.Occurrence))
    graph.add((obs, RDF.type, BFO["0000015"]))

    if d := rec.get("observed_on"):
        graph.add((obs, DWC.eventDate, Literal(d, datatype=XSD.date)))

    loc = rec.get("location")
    if loc and "," in loc:
        lat_s, lng_s = (s.strip() for s in loc.split(","))
        try:
            float(lat_s); float(lng_s)
        except ValueError:
            log.warning("unparseable location on %s: %r", uri, loc)
        else:
            # Pass strings so rdflib constructs Decimal values; passing
            # a Python float keeps the value-type as float and SHACL rejects.
            graph.add((obs, DWC.decimalLatitude,  Literal(lat_s, datatype=XSD.decimal)))
            graph.add((obs, DWC.decimalLongitude, Literal(lng_s, datatype=XSD.decimal)))

    taxon = rec.get("taxon") or {}
    if name := taxon.get("name"):
        graph.add((obs, DWC.scientificName, Literal(name)))

    if tid := taxon.get("id"):
        taxon_iri = URIRef(f"https://www.inaturalist.org/taxa/{tid}")
        graph.add((obs, BIO.observedTaxon, taxon_iri))
        graph.add((taxon_iri, RDF.type, DWC.Taxon))
        if name:
            graph.add((taxon_iri, DWC.scientificName, Literal(name)))
        if common := taxon.get("preferred_common_name"):
            graph.add((taxon_iri, RDFS.label, Literal(common)))

    user = rec.get("user") or {}
    orcid = user.get("orcid")
    login = user.get("login")
    if orcid:
        agent: URIRef | None = URIRef(f"https://orcid.org/{orcid}")
    elif login:
        agent = URIRef(f"https://www.inaturalist.org/people/{login}")
    else:
        agent = None
    if agent is not None:
        graph.add((obs, PROV.wasAttributedTo, agent))
        graph.add((agent, RDF.type, PROV.Agent))
        if login:
            graph.add((agent, BIO.inatLogin, URIRef(f"https://www.inaturalist.org/people/{login}")))
        if orcid:
            graph.add((agent, BIO.orcid, URIRef(f"https://orcid.org/{orcid}")))

    if inat_id := rec.get("id"):
        graph.add((obs, PROV.wasDerivedFrom,
                   URIRef(f"https://api.inaturalist.org/v2/observations/{inat_id}")))

    if q := rec.get("quality_grade"):
        graph.add((obs, BIO.qualityGrade, Literal(q)))
    if icon := taxon.get("iconic_taxon_name"):
        graph.add((obs, BIO.iconicGroup, Literal(icon)))
    if lic := rec.get("license_code"):
        graph.add((obs, DCT.license, Literal(lic)))
    if created := rec.get("created_at"):
        graph.add((obs, DCT.created, Literal(created, datatype=XSD.dateTime)))

    return obs


def _bind_prefixes(g: Graph) -> None:
    g.bind("bio", BIO)
    g.bind("dwc", DWC)
    g.bind("bfo", BFO)
    g.bind("prov", PROV)
    g.bind("dcterms", DCT)
    g.bind("skos", SKOS)


def _summarize_report(report_text: str) -> str:
    msgs = [
        ln.strip().removeprefix("Message:").strip()
        for ln in report_text.splitlines()
        if ln.strip().startswith("Message:")
    ]
    return "; ".join(msgs[:3]) if msgs else "(no message)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=2000,
                        help="Max observations to fetch (default 2000).")
    parser.add_argument("--out", type=Path, default=Path("data/observations.ttl"),
                        help="Output Turtle file.")
    parser.add_argument("--bbox", type=str, default="53.4,9.7,53.7,10.3",
                        help="'swlat,swlng,nelat,nelng' (default Hamburg).")
    parser.add_argument("--shapes", type=Path, default=Path("kg/shapes.ttl"),
                        help="SHACL shapes file.")
    args = parser.parse_args()

    parts = args.bbox.split(",")
    if len(parts) != 4:
        log.error("invalid --bbox; expected 4 floats")
        return 2
    bbox = tuple(float(p) for p in parts)

    log.info("loading shapes from %s", args.shapes)
    shapes_graph = Graph()
    shapes_graph.parse(args.shapes, format="turtle")

    main_graph = Graph()
    _bind_prefixes(main_graph)

    n_fetched = n_kept = n_dropped = 0
    for rec in fetch_observations(args.limit, bbox):
        n_fetched += 1
        candidate = Graph()
        _bind_prefixes(candidate)
        obs_iri = build_record(rec, candidate)
        if obs_iri is None:
            n_dropped += 1
            log.warning("record %s had no URI; dropped", rec.get("id"))
            continue

        conforms, _, report = shacl_validate(
            data_graph=candidate,
            shacl_graph=shapes_graph,
            inference="none",
            meta_shacl=False,
            debug=False,
        )
        if not conforms:
            n_dropped += 1
            log.warning("validation failed for %s: %s", obs_iri, _summarize_report(report))
            continue

        for triple in candidate:
            main_graph.add(triple)
        n_kept += 1
        if n_kept % 50 == 0:
            log.info("kept %d / fetched %d", n_kept, n_fetched)

    log.info("done: fetched=%d kept=%d dropped=%d", n_fetched, n_kept, n_dropped)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    main_graph.serialize(destination=args.out, format="turtle")
    log.info("wrote %d triples to %s", len(main_graph), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
