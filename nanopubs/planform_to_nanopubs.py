#!/usr/bin/env python3
"""
planform_to_nanopubs.py

Create nanopublications from a Planform SQLite DB and (optionally) publish them.

One nanopublication per observed outcome (ResultantMorphology row with Frequency > 0):
- Assertion: a minimal claim that a given result set observed a morphology with a given frequency.
- Provenance: links the assertion to the source publication; attributes the assertion to your ORCID.
- PubInfo: timestamp, author (your ORCID), license.

USAGE
-----
# Dry-run: write .trig files only (no publishing)
python planform_to_nanopubs.py --db planformDB_2.5.0.edb --out ./nanopubs

# Publish to the nanopub **test** server first (recommended)
python planform_to_nanopubs.py --db planformDB_2.5.0.edb --publish test --limit 10

# Publish to **production** (be sure!)
python planform_to_nanopubs.py --db planformDB_2.5.0.edb --publish prod --min-year 1995

Before publishing, run once:
    pip install nanopub rdflib
    python3 -m nanopub setup   # interactive: sets RSA keys + ORCID
"""

import argparse
import os
import re
import sqlite3
from datetime import datetime, timezone

import rdflib
from rdflib import Graph, ConjunctiveGraph, URIRef, Literal, BNode, Namespace
from rdflib.namespace import RDF, RDFS, XSD, DCTERMS
from nanopub import Publication, NanopubClient, namespaces

# ---- Namespaces ----
NP   = namespaces.NP           # http://www.nanopub.org/nschema#
PROV = namespaces.PROV         # http://www.w3.org/ns/prov#
ORCID = namespaces.ORCID       # https://orcid.org/

def slug(s: str, max_len: int = 80) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:max_len] or "na"

def mint(base: str, *parts: str) -> URIRef:
    path = "/".join(slug(p) for p in parts if p is not None)
    return URIRef(f"{base.rstrip('/')}/{path}")

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to the Planform SQLite database (e.g., planformDB_2.5.0.edb)")
    ap.add_argument("--out", default="./nanopubs", help="Output directory for .trig files")
    ap.add_argument("--base", default="https://example.org/planform",
                    help="Base URI for minted Planform entities (change to your w3id/namespace)")
    ap.add_argument("--license", default="https://creativecommons.org/publicdomain/zero/1.0/",
                    help="License URI to set in pubinfo (default CC0)")
    ap.add_argument("--min-year", type=int, default=None, help="Only include rows with publication year >= min-year")
    ap.add_argument("--max-year", type=int, default=None, help="Only include rows with publication year <= max-year")
    ap.add_argument("--limit", type=int, default=None, help="Max number of nanopubs to generate/publish")
    ap.add_argument("--publish", choices=["none", "test", "prod"], default="none",
                    help="Publish destination: none (default), test, or prod")
    ap.add_argument("--dry", action="store_true", help="Do not publish even if --publish is set (write files only)")
    return ap

def fetch_rows(conn, min_year=None, max_year=None, limit=None):
    q = """
    SELECT
        rm.Id            AS rm_id,
        rm.Frequency     AS freq,
        rs.Id            AS rs_id,
        e.Id             AS e_id,
        e.Manipulation   AS manipulation,
        s.Id             AS species_id,
        s.Name           AS species_name,
        p.Id             AS pub_id,
        p.Year           AS year,
        p.Title          AS pub_title,
        m.Id             AS morph_id,
        m.Name           AS morph_name
    FROM ResultantMorphology rm
    JOIN ResultSet   rs ON rs.Id = rm.ResultSet
    JOIN Experiment  e  ON e.Id  = rs.Experiment
    JOIN Publication p  ON p.Id  = e.Publication
    JOIN Morphology  m  ON m.Id  = rm.Morphology
    LEFT JOIN Species s ON s.Id  = e.Species
    WHERE rm.Frequency > 0
      AND p.Year IS NOT NULL
      {year_clause}
    ORDER BY p.Year, e.Id, rs.Id, rm.Id
    {limit_clause}
    """
    year_clause = []
    params = []
    if min_year is not None:
        year_clause.append("p.Year >= ?")
        params.append(min_year)
    if max_year is not None:
        year_clause.append("p.Year <= ?")
        params.append(max_year)
    year_sql = " AND " + " AND ".join(year_clause) if year_clause else ""
    limit_sql = f" LIMIT {int(limit)}" if limit else ""
    q = q.format(year_clause=year_sql, limit_clause=limit_sql)
    cur = conn.execute(q, params)
    cols = [c[0] for c in cur.description]
    for row in cur.fetchall():
        yield dict(zip(cols, row))

def build_publication(row, base_uri: str, license_uri: str) -> Publication:
    """Make a Nanopub Publication from one ResultantMorphology row."""
    # Mint URIs for domain entities
    experiment = mint(base_uri, "experiment", str(row["e_id"]))
    resultset  = mint(base_uri, "resultset",  str(row["rs_id"]))
    morphology = mint(base_uri, "morphology", f"{row['morph_id']}-{row['morph_name'] or ''}")
    observation = mint(base_uri, "observation", str(row["rm_id"]))
    publication = mint(base_uri, "publication", str(row["pub_id"]))
    species = mint(base_uri, "species", f"{row['species_id']}-{row['species_name'] or ''}") if row["species_id"] else None

    # Assertion graph (the core claim)
    A = Graph()
    PF = Namespace(f"{base_uri.rstrip('/')}/vocab/")
    A.bind("pf", PF)
    A.bind("rdfs", RDFS)
    A.bind("dct", DCTERMS)

    # Types & labels for readability (optional but helpful)
    A.add((observation, RDF.type, PF.Observation))
    A.add((resultset,  RDF.type, PF.ResultSet))
    A.add((morphology, RDF.type, PF.Morphology))
    A.add((experiment, RDF.type, PF.Experiment))
    if species:
        A.add((species, RDF.type, PF.Species))

    if row["morph_name"]:
        A.add((morphology, RDFS.label, Literal(row["morph_name"])))
    if row["species_name"]:
        A.add((species, RDFS.label, Literal(row["species_name"])))
    if row["pub_title"]:
        A.add((publication, DCTERMS.title, Literal(row["pub_title"])))

    # Minimal observation claim
    A.add((observation, PF.inResultSet, resultset))
    A.add((observation, PF.hasOutcome,  morphology))
    A.add((observation, PF.frequency,   Literal(row["freq"], datatype=XSD.decimal)))
    A.add((observation, PF.year,        Literal(int(row["year"]), datatype=XSD.gYear)))
    A.add((resultset,   PROV.wasGeneratedBy, experiment))
    if species:
        A.add((experiment, PF.hasSpecies, species))
    if row["manipulation"]:
        A.add((experiment, PF.hasManipulation, Literal(row["manipulation"])))

    # Extra provenance triples (beyond 'derived_from' and attribution)
    P = Graph()
    P.bind("prov", PROV)
    P.bind("dct", DCTERMS)
    # We’ll let nanopub add the assertion link; here we capture some contextual provenance:
    P.add((experiment, PROV.generated, resultset))
    P.add((resultset, PROV.wasDerivedFrom, publication))
    if species:
        P.add((experiment, PROV.used, species))

    # PubInfo (timestamp, license; authorship handled by profile)
    I = Graph()
    I.bind("dct", DCTERMS)
    I.add((URIRef("this:placeholder"), DCTERMS.license, URIRef(license_uri)))
    # Timestamp is auto-added by the library; leaving as-is.

    # Build nanopublication
    pub = Publication.from_assertion(
        assertion_rdf=A,
        derived_from=publication,                    # provenance: prov:wasDerivedFrom
        attribute_assertion_to_profile=True,         # provenance: prov:wasAttributedTo <your ORCID>
        attribute_publication_to_profile=True,       # pubinfo: prov:wasAttributedTo <your ORCID>
        provenance_rdf=P,
        pubinfo_rdf=I,
        add_generated_at_time=True,
    )
    return pub

def main():
    args = build_argparser().parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Optional client for publishing
    if args.publish == "test":
        client = NanopubClient(use_test_server=True)
    elif args.publish == "prod":
        client = NanopubClient(use_test_server=False)
    else:
        client = None

    conn = sqlite3.connect(args.db)
    try:
        rows = list(fetch_rows(conn, min_year=args.min_year, max_year=args.max_year, limit=args.limit))
        if not rows:
            print("No eligible rows found (check year filters and Frequency>0).")
            return

        manifest_lines = []
        for i, row in enumerate(rows, start=1):
            pub = build_publication(row, base_uri=args.base, license_uri=args.license)

            # Save TriG locally
            trig_path = os.path.join(args.out, f"np_rs{row['rs_id']}_m{row['morph_id']}_rm{row['rm_id']}.trig")
            pub.rdf.serialize(destination=trig_path, format="trig")
            nanopub_uri = None

            # Optional publish
            if client and not args.dry:
                info = client.publish(pub)  # returns dict with 'nanopub_uri', etc.
                nanopub_uri = info.get("nanopub_uri")
                print(f"[{i}/{len(rows)}] Published: {nanopub_uri}")
            else:
                print(f"[{i}/{len(rows)}] Wrote: {trig_path}")

            manifest_lines.append("|".join([
                str(row["year"]),
                str(row["e_id"]), str(row["rs_id"]), str(row["morph_id"]), str(row["rm_id"]),
                row["morph_name"] or "",
                nanopub_uri or trig_path
            ]))

        # Write a simple manifest (year|experiment|resultset|morph|rm|morph_name|uri)
        with open(os.path.join(args.out, "manifest.txt"), "w", encoding="utf-8") as f:
            f.write("# year|experiment|resultset|morph|rm|morph_name|nanopub_or_file\n")
            f.write("\n".join(manifest_lines))
        print(f"Manifest written to {os.path.join(args.out, 'manifest.txt')}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
