#!/usr/bin/env python3
"""
np_patch_restructure.py
Transform our earlier wave TriG files to canonical nanopubs:
 - Add a named Head graph per nanopub (4-graph layout).
 - Move ex:EvidenceItem descriptions from Provenance into the Assertion graph.
 - Replace prov:wasSupportedBy edges with ex:hasEvidence inside the Assertion graph.
 - In Provenance, keep only links from the ASSERTION GRAPH to sources (prov:wasDerivedFrom).

USAGE:
  # Single-file mode (unchanged):
  python np_patch_restructure.py in.trig out.trig

  # Directory mode (new):
  python np_patch_restructure.py /path/to/dir
    -> writes np_*.trig alongside *.trig in that dir

  python np_patch_restructure.py /path/in_dir /path/out_dir
    -> writes np_*.trig into /path/out_dir (created if needed)

Requires: rdflib >= 6.0
"""
import sys
import os
from pathlib import Path
from typing import Dict
from rdflib import ConjunctiveGraph, Namespace, URIRef
from rdflib.namespace import RDF, DCTERMS

NP = Namespace("http://www.nanopub.org/nschema#")
PROV = Namespace("http://www.w3.org/ns/prov#")
CITO = Namespace("http://purl.org/spar/cito/")
EX = Namespace("https://w3id.org/levin-kg/")

def base_from_graph_name(gname: URIRef):
    s = str(gname)
    for suf in ["-assertion", "-provenance", "-pubinfo"]:
        if s.endswith(suf):
            return URIRef(s[: -len(suf)])
    # fallback: treat as base already
    return URIRef(s)

def transform_file(inp: Path, outp: Path):
    ds = ConjunctiveGraph()
    ds.parse(str(inp), format="trig")

    out = ConjunctiveGraph()

    # group graphs by nanopub base
    groups: Dict[URIRef, Dict[URIRef, ConjunctiveGraph]] = {}
    for g in ds.contexts():
        base = base_from_graph_name(g.identifier)
        groups.setdefault(base, {})[g.identifier] = g

    for base, gmap in groups.items():
        # locate specific graphs if they follow -assertion/-provenance/-pubinfo naming
        assertion_iri = URIRef(str(base) + "-assertion")
        provenance_iri = URIRef(str(base) + "-provenance")
        pubinfo_iri = URIRef(str(base) + "-pubinfo")
        head_iri = URIRef(str(base) + "#Head")

        assertion_named = out.get_context(assertion_iri)
        provenance_named = out.get_context(provenance_iri)
        pubinfo_named = out.get_context(pubinfo_iri)
        head_named = out.get_context(head_iri)

        # 1) Head graph
        head_named.add((base, RDF.type, NP.Nanopublication))
        head_named.add((base, NP.hasAssertion, assertion_iri))
        head_named.add((base, NP.hasProvenance, provenance_iri))
        head_named.add((base, NP.hasPublicationInfo, pubinfo_iri))

        # 2) Copy original assertion triples (if any)
        if assertion_iri in gmap:
            for t in gmap[assertion_iri].triples((None, None, None)):
                assertion_named.add(t)

        # 3) Move EvidenceItems from provenance -> assertion graph; change linkage
        if provenance_iri in gmap:
            prov_graph = gmap[provenance_iri]
            for s, p, o in prov_graph.triples((None, None, None)):
                # collect EvidenceItem descriptions
                if (s, RDF.type, EX.EvidenceItem) in prov_graph or (p == RDF.type and o == EX.EvidenceItem):
                    for t2 in prov_graph.triples((s, None, None)):
                        assertion_named.add(t2)
                    continue
                # rewrite wasSupportedBy
                if p == PROV.wasSupportedBy:
                    assertion_named.add((s, EX.hasEvidence, o))
                    continue

            # 4) Build minimal provenance that points from ASSERTION GRAPH to source DOI(s)
            sources = set()
            # sources from pubinfo
            if pubinfo_iri in gmap:
                for _, _, src in gmap[pubinfo_iri].triples((None, DCTERMS.source, None)):
                    sources.add(src)
            # sources from assertion via cito:citesAsEvidence
            for _, _, src in assertion_named.triples((None, CITO.citesAsEvidence, None)):
                sources.add(src)
            # provenance should refer to the assertion GRAPH node
            for src in sources:
                provenance_named.add((assertion_iri, PROV.wasDerivedFrom, src))

        # 5) Copy original pubinfo triples
        if pubinfo_iri in gmap:
            for t in gmap[pubinfo_iri].triples((None, None, None)):
                pubinfo_named.add(t)

    out.serialize(str(outp), format="trig")

def process_directory(in_dir: Path, out_dir: Path | None = None):
    in_dir = in_dir.resolve()
    if out_dir is None:
        out_dir = in_dir
    else:
        out_dir = out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    trig_files = sorted(p for p in in_dir.iterdir()
                        if p.is_file() and p.suffix.lower() == ".trig" and not p.name.startswith("np_"))

    if not trig_files:
        print(f"[info] No .trig files found to process in: {in_dir}")
        return

    print(f"[info] Processing {len(trig_files)} file(s) from {in_dir} -> {out_dir}")
    successes = 0
    for src in trig_files:
        dst = out_dir / f"np_{src.name}"
        try:
            transform_file(src, dst)
            print(f"[ok]   {src.name} -> {dst.name}")
            successes += 1
        except Exception as e:
            print(f"[err]  {src.name}: {e.__class__.__name__}: {e}")

    print(f"[done] {successes}/{len(trig_files)} file(s) written to {out_dir}")

def main(argv):
    if len(argv) < 2 or len(argv) > 3:
        print(__doc__)
        return 0

    p1 = Path(argv[1])

    # Directory mode
    if p1.is_dir():
        if len(argv) == 3:
            p2 = Path(argv[2])
            process_directory(p1, p2)
        else:
            process_directory(p1, None)
        return 0

    # Single-file mode (backward compatible)
    if len(argv) != 3:
        print(__doc__)
        return 0

    inp = p1
    outp = Path(argv[2])
    outp.parent.mkdir(parents=True, exist_ok=True)

    try:
        transform_file(inp, outp)
        print("Wrote", outp)
    except Exception as e:
        print(f"[err] {inp.name}: {e.__class__.__name__}: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
