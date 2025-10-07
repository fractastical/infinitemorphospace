#!/usr/bin/env python3
"""
np_patch_restructure.py
Transform our earlier wave TriG files to canonical nanopubs:
 - Add a named Head graph per nanopub (4-graph layout).
 - Move ex:EvidenceItem descriptions from Provenance into the Assertion graph.
 - Replace prov:wasSupportedBy edges with ex:hasEvidence inside the Assertion graph.
 - In Provenance, keep only links from the ASSERTION GRAPH to sources (prov:wasDerivedFrom).
Requires: rdflib >= 6.0
Usage:
  python np_patch_restructure.py in.trig out.trig
"""
import sys
from rdflib import ConjunctiveGraph, Namespace, URIRef, BNode, Literal
from rdflib.namespace import RDF, XSD, DCTERMS

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

def main(inp, outp):
    ds = ConjunctiveGraph()
    ds.parse(inp, format="trig")

    out = ConjunctiveGraph()

    # group graphs by nanopub base
    groups = {}
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
        #    ex:claim-XYZ prov:wasSupportedBy ex:evidence-... -> ex:claim-XYZ ex:hasEvidence ex:evidence-...
        if provenance_iri in gmap:
            prov_graph = gmap[provenance_iri]
            for s,p,o in prov_graph.triples((None, None, None)):
                # collect EvidenceItem descriptions
                if (s, RDF.type, EX.EvidenceItem) in prov_graph or p == RDF.type and o == EX.EvidenceItem:
                    # copy entire description of this EvidenceItem into assertion graph
                    # gather all triples about s from provenance
                    for t2 in prov_graph.triples((s, None, None)):
                        assertion_named.add(t2)
                    continue
                # rewrite wasSupportedBy
                if p == PROV.wasSupportedBy:
                    assertion_named.add((s, EX.hasEvidence, o))
                    continue

            # 4) Build minimal provenance that points from ASSERTION GRAPH to source DOI(s)
            # Find any DOIs in pubinfo or assertion graph
            sources = set()
            if pubinfo_iri in gmap:
                for _,_,src in gmap[pubinfo_iri].triples((None, DCTERMS.source, None)):
                    sources.add(src)
            for _,_,src in assertion_named.triples((None, CITO.citesAsEvidence, None)):
                sources.add(src)
            # provenance should refer to the assertion GRAPH node
            for src in sources:
                provenance_named.add((assertion_iri, PROV.wasDerivedFrom, src))

        # 5) Copy original pubinfo triples
        if pubinfo_iri in gmap:
            for t in gmap[pubinfo_iri].triples((None, None, None)):
                pubinfo_named.add(t)

    out.serialize(outp, format="trig")
    print("Wrote", outp)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(0)
    main(sys.argv[1], sys.argv[2])
