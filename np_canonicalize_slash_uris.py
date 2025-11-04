#!/usr/bin/env python3
"""
np_canonicalize_slash_uris.py
---------------------------------
Fix common issues that cause nanopub parsing errors:

- Replace relative graph IRIs (e.g., <Nabcd#Head>) with absolute w3id bases.
- Convert mixed '#Head' and '-assertion'/-provenance/-pubinfo forms to slash-style:
    BASE/Head, BASE/assertion, BASE/provenance, BASE/pubinfo
- Unify example.org namespace to w3id.org for the "ex" vocabulary and instances.
- Ensure Head graph objects exactly match the named graph IRIs.
- Insert dcterms:conformsTo <https://w3id.org/morphopkg/spc/1.0> into each Head.

Usage:
  python np_canonicalize_slash_uris.py input.trig output.trig

Requires: rdflib >= 6.0
"""
import re, sys, uuid
from urllib.parse import quote
from rdflib import ConjunctiveGraph, URIRef, Namespace, RDF, Literal
from rdflib.namespace import DCTERMS

NP   = Namespace("http://www.nanopub.org/nschema#")
PROV = Namespace("http://www.w3.org/ns/prov#")
EX_W3ID = Namespace("https://w3id.org/levin-kg/")
EX_EXAMPLE = Namespace("https://example.org/levin-kg/")

def local_id_from_old(assertion_graph_iri: str) -> str:
    s = assertion_graph_iri
    # cases: ...-assertion, ...#assertion, .../assertion
    for sep in ["-assertion", "#assertion", "/assertion"]:
        if s.endswith(sep):
            s = s[: -len(sep)]
            break
    # take last path segment
    return s.rstrip("/").split("/")[-1]

def canonical_base(old_assertion_iri: str) -> str:
    local = local_id_from_old(old_assertion_iri)
    # sanitize local id for URL path
    safe_local = quote(local, safe="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
    return f"https://w3id.org/levin-kg/np/{safe_local}/"

def rewrite_ns_example_to_w3id(node):
    if isinstance(node, URIRef) and str(node).startswith(str(EX_EXAMPLE)):
        return URIRef(str(node).replace(str(EX_EXAMPLE), str(EX_W3ID), 1))
    return node

def main(inp, outp):
    ds = ConjunctiveGraph()
    ds.parse(inp, format="trig")

    out = ConjunctiveGraph()
    out.bind("np", NP)
    out.bind("prov", PROV)
    out.bind("dcterms", DCTERMS)
    out.bind("ex", EX_W3ID)

    # Find nanopubs via any graph that declares np:Nanopublication
    heads = []
    for g in ds.contexts():
        for (npnode, _, _) in g.triples((None, RDF.type, NP.Nanopublication)):
            heads.append((g.identifier, npnode))
    # If none found, try any graph whose name ends with Head/hash Head
    if not heads:
        for g in ds.contexts():
            sid = str(g.identifier)
            if sid.endswith("/Head") or sid.endswith("#Head") or sid.endswith("-Head"):
                # synthesize a nanopub node = base (without suffix)
                base = sid.rsplit("/",1)[0] if sid.endswith("/Head") else sid.rsplit("#",1)[0]
                heads.append((g.identifier, URIRef(base)))

    for head_graph_iri, npnode in heads:
        # discover assertion/provenance/pubinfo targets
        g_head_in = ds.get_context(head_graph_iri)
        assertion_target = None
        prov_target = None
        pubinfo_target = None
        for _, p, o in g_head_in.triples((npnode, None, None)):
            if p == NP.hasAssertion: assertion_target = o
            if p == NP.hasProvenance: prov_target = o
            if p == NP.hasPublicationInfo: pubinfo_target = o
        # if not found, try to guess from existing graph names
        if assertion_target is None:
            # pick a graph that ends with -assertion/#assertion or /assertion and shares the stem
            for g in ds.contexts():
                si = str(g.identifier)
                if si.endswith(("-assertion","#assertion","/assertion")):
                    stem = si.replace("-assertion","").replace("#assertion","").replace("/assertion","")
                    if stem in str(npnode):
                        assertion_target = g.identifier
                        break
        if assertion_target is None:
            # last resort: skip this head
            continue

        # Compute canonical base & graph IRIs
        base = canonical_base(str(assertion_target))
        head_canon = URIRef(base + "Head")
        asrt_canon = URIRef(base + "assertion")
        prov_canon = URIRef(base + "provenance")
        pubi_canon = URIRef(base + "pubinfo")

        # Write Head
        g_head_out = out.get_context(head_canon)
        g_head_out.add((URIRef(base), RDF.type, NP.Nanopublication))
        g_head_out.add((URIRef(base), NP.hasAssertion, asrt_canon))
        g_head_out.add((URIRef(base), NP.hasProvenance, prov_canon))
        g_head_out.add((URIRef(base), NP.hasPublicationInfo, pubi_canon))
        g_head_out.add((URIRef(base), DCTERMS.conformsTo, URIRef("https://w3id.org/morphopkg/spc/1.0")))

        # Copy assertion graph content (rewrite example.org to w3id)
        g_assert_in = ds.get_context(assertion_target)
        g_assert_out = out.get_context(asrt_canon)
        for s,p,o in g_assert_in.triples((None,None,None)):
            g_assert_out.add((rewrite_ns_example_to_w3id(s), rewrite_ns_example_to_w3id(p), rewrite_ns_example_to_w3id(o)))

        # Copy provenance and pubinfo if present; otherwise synthesize minimal provenance pointing to any DOI in pubinfo
        if prov_target is not None:
            g_prov_in = ds.get_context(prov_target)
        else:
            g_prov_in = None
        g_prov_out = out.get_context(prov_canon)
        if g_prov_in:
            for s,p,o in g_prov_in.triples((None,None,None)):
                # Ensure subject is the assertion GRAPH IRI
                if p == PROV.wasDerivedFrom:
                    g_prov_out.add((asrt_canon, p, rewrite_ns_example_to_w3id(o)))
                else:
                    g_prov_out.add((rewrite_ns_example_to_w3id(s), rewrite_ns_example_to_w3id(p), rewrite_ns_example_to_w3id(o)))

        if pubinfo_target is not None:
            g_pub_in = ds.get_context(pubinfo_target)
        else:
            # try guess -pubinfo/#pubinfo
            guess = str(assertion_target).replace("-assertion","-pubinfo").replace("#assertion","#pubinfo").replace("/assertion","/pubinfo")
            g_pub_in = ds.get_context(URIRef(guess))
        g_pub_out = out.get_context(pubi_canon)
        if g_pub_in:
            for s,p,o in g_pub_in.triples((None,None,None)):
                g_pub_out.add((rewrite_ns_example_to_w3id(s), rewrite_ns_example_to_w3id(p), rewrite_ns_example_to_w3id(o)))

        # If provenance missing derivedFrom, try to add from pubinfo source
        has_derived = any(True for _ in g_prov_out.triples((asrt_canon, PROV.wasDerivedFrom, None)))
        if not has_derived:
            for _,_,src in g_pub_out.triples((None, DCTERMS.source, None)):
                g_prov_out.add((asrt_canon, PROV.wasDerivedFrom, src))

    out.serialize(outp, format="trig")
    print("Wrote", outp)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
