#!/usr/bin/env python3
"""
np_canonicalize_single_np.py
----------------------------
Canonicalize nanopubs and emit EXACTLY ONE nanopublication per output file
(by default) *or* split each input into one output per nanopub (with --explode).
Also unifies namespaces and normalizes graph IRIs.

Fixes:
- Replace relative graph IRIs (e.g., <Nabcd#Head>) with absolute w3id bases.
- Convert mixed '#Head' / '-assertion' / '-provenance' / '-pubinfo' forms to slash-style:
    BASE/Head, BASE/assertion, BASE/provenance, BASE/pubinfo
- Unify https://example.org/levin-kg/ -> https://w3id.org/levin-kg/
- (Extra) If any model classes/properties were mistakenly minted under https://w3id.org/levin-kg/np/,
  rewrite those *specific* terms back to https://w3id.org/levin-kg/ (clean vocabulary boundary).
- Ensure Head graph IRIs and links match exactly.
- Insert dcterms:conformsTo <https://w3id.org/morphopkg/spc/1.0> into /Head.

Selection (when an input contains multiple nanopubs):
- File mode: choose one with --select-*, else the first.
- Directory mode:
    * Default: one output per input file (choose first NP unless --select-* matches).
    * With --explode: write one output file per nanopub.

Usage:
  # File -> File (choose first NP)
  python np_canonicalize_single_np.py input.trig output.trig

  # File -> File (choose by DOI or base substring)
  python np_canonicalize_single_np.py input.trig output.trig --select-doi 10.1242/dev.086900
  python np_canonicalize_single_np.py input.trig output.trig --select-base-substr B2013-C1

  # Directory -> Directory (one output per input, choose first NP)
  python np_canonicalize_single_np.py input_dir output_dir

  # Directory -> Directory (explode: one output per NP)
  python np_canonicalize_single_np.py input_dir output_dir --explode

Requires: rdflib >= 6.0
"""
import sys, re
from pathlib import Path
from urllib.parse import quote
from rdflib import ConjunctiveGraph, URIRef, Namespace, RDF
from rdflib.namespace import DCTERMS

NP   = Namespace("http://www.nanopub.org/nschema#")
PROV = Namespace("http://www.w3.org/ns/prov#")
EX_W3ID = Namespace("https://w3id.org/levin-kg/")
EX_W3ID_NP = "https://w3id.org/levin-kg/np/"
EX_EXAMPLE = Namespace("https://example.org/levin-kg/")

# Recognized model terms (classes & properties) that belong under https://w3id.org/levin-kg/
MODEL_TERMS = {
    # classes
    "Assertion","EvidenceItem","BayesianAssessment","Hypothesis","EconomicHypothesis","EconomicAssessment",
    # properties (core)
    "hasEvidence","hasAssessment","supportsHypothesis","hypothesisContribution","mappingConfidence","mappingMethod",
    "assessmentAgent","context",
    # numeric properties used in assessments/evidence
    "bayesFactorCombined","weightOfEvidence_deciban","calibrationMethod","publicationYear",
    "pValue","bayesFactorVS_MPR","countN","totalN","frequency","timeDays"
}

def local_id_from_old(assertion_graph_iri: str) -> str:
    s = assertion_graph_iri
    for sep in ("-assertion","#assertion","/assertion"):
        if s.endswith(sep):
            s = s[: -len(sep)]
            break
    return s.rstrip("/").split("/")[-1] or "np"

def canonical_base(old_assertion_iri: str) -> str:
    local = local_id_from_old(old_assertion_iri)
    safe_local = quote(local, safe="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
    return f"https://w3id.org/levin-kg/np/{safe_local}/"

def rewrite_example_to_w3id(node):
    if isinstance(node, URIRef) and str(node).startswith(str(EX_EXAMPLE)):
        return URIRef(str(node).replace(str(EX_EXAMPLE), str(EX_W3ID), 1))
    return node

def rewrite_model_terms(node):
    """If a URI is under .../np/ and its local name is a known model term, rewrite to .../"""
    if isinstance(node, URIRef):
        s = str(node)
        if s.startswith(EX_W3ID_NP):
            local = s[len(EX_W3ID_NP):]
            # keep only simple local names (no further '/')
            if "/" not in local and local in MODEL_TERMS:
                return URIRef(str(EX_W3ID) + local)
    return node

def unify_node(node):
    node = rewrite_example_to_w3id(node)
    node = rewrite_model_terms(node)
    return node

def find_nanopubs(ds: ConjunctiveGraph):
    """Return list of dicts with keys: head, npnode, base, asg, prg, pig, dois"""
    out = []
    # Prefer declared heads by type
    for g in ds.contexts():
        for (npnode, _, _) in g.triples((None, RDF.type, NP.Nanopublication)):
            head_iri = g.identifier
            s = str(head_iri)
            if s.endswith(("/Head","#Head","-Head")):
                base = s[: -5]
            else:
                base = str(npnode)
            asg = prg = pig = None
            for _, p, o in g.triples((npnode, None, None)):
                if p == NP.hasAssertion: asg = o
                if p == NP.hasProvenance: prg = o
                if p == NP.hasPublicationInfo: pig = o
            dois = set()
            if pig:
                gp = ds.get_context(pig)
                for _, p, o in gp.triples((None, DCTERMS.source, None)):
                    dois.add(str(o))
            out.append(dict(head=head_iri, npnode=npnode, base=base, asg=asg, prg=prg, pig=pig, dois=sorted(dois)))
    # If none found, infer heads by graph-name suffix
    if not out:
        for g in ds.contexts():
            sid = str(g.identifier)
            if sid.endswith(("/Head","#Head","-Head")):
                out.append(dict(head=g.identifier, npnode=URIRef(sid[: -5]), base=sid[: -5], asg=None, prg=None, pig=None, dois=[]))
    return out

def select_one(candidates, select_base_substr=None, select_doi=None, select_index=None):
    if not candidates:
        return None
    if select_doi:
        for c in candidates:
            if any(select_doi in d for d in c["dois"]):
                return c
    if select_base_substr:
        for c in candidates:
            if select_base_substr in c["base"]:
                return c
    if isinstance(select_index, int) and 0 <= select_index < len(candidates):
        return candidates[select_index]
    return candidates[0]

def emit_single(ds: ConjunctiveGraph, cand, out_file: Path):
    head_graph_iri = cand["head"]
    base_guess = cand["base"]
    asg_in = cand["asg"]
    prg_in = cand["prg"]
    pig_in = cand["pig"]

    # If assertion graph wasn't linked, try to guess
    if asg_in is None:
        for g in ds.contexts():
            s = str(g.identifier)
            if s.endswith(("-assertion","#assertion","/assertion")) and base_guess in s:
                asg_in = g.identifier
                break
    if asg_in is None:
        raise RuntimeError(f"Could not locate assertion graph for nanopub with head {head_graph_iri}")

    # Compute canonical slash IRIs
    base = canonical_base(str(asg_in))
    head_canon = URIRef(base + "Head")
    asrt_canon = URIRef(base + "assertion")
    prov_canon = URIRef(base + "provenance")
    pubi_canon = URIRef(base + "pubinfo")

    out = ConjunctiveGraph()
    out.bind("np", NP)
    out.bind("prov", PROV)
    out.bind("dcterms", DCTERMS)
    out.bind("ex", EX_W3ID)

    # HEAD
    gH = out.get_context(head_canon)
    gH.add((URIRef(base), RDF.type, NP.Nanopublication))
    gH.add((URIRef(base), NP.hasAssertion, asrt_canon))
    gH.add((URIRef(base), NP.hasProvenance, prov_canon))
    gH.add((URIRef(base), NP.hasPublicationInfo, pubi_canon))
    gH.add((URIRef(base), DCTERMS.conformsTo, URIRef("https://w3id.org/morphopkg/spc/1.0")))

    # ASSERTION
    gA_in = ds.get_context(asg_in)
    gA = out.get_context(asrt_canon)
    for s,p,o in gA_in.triples((None,None,None)):
        gA.add((unify_node(s), unify_node(p), unify_node(o)))

    # PROVENANCE
    if prg_in is None:
        # guess
        guess = str(asg_in).replace("-assertion","-provenance").replace("#assertion","#provenance").replace("/assertion","/provenance")
        prg_in = URIRef(guess)
    gP_in = ds.get_context(prg_in)
    gP = out.get_context(prov_canon)
    if gP_in:
        for s,p,o in gP_in.triples((None,None,None)):
            if p == PROV.wasDerivedFrom:
                gP.add((asrt_canon, p, unify_node(o)))
            else:
                gP.add((unify_node(s), unify_node(p), unify_node(o)))

    # PUBINFO
    if pig_in is None:
        guess = str(asg_in).replace("-assertion","-pubinfo").replace("#assertion","#pubinfo").replace("/assertion","/pubinfo")
        pig_in = URIRef(guess)
    gI_in = ds.get_context(pig_in)
    gI = out.get_context(pubi_canon)
    if gI_in:
        for s,p,o in gI_in.triples((None,None,None)):
            gI.add((unify_node(s), unify_node(p), unify_node(o)))

    # Ensure at least one derivedFrom (if a DOI exists in pubinfo)
    has_derived = any(True for _ in gP.triples((asrt_canon, PROV.wasDerivedFrom, None)))
    if not has_derived:
        for _,_,src in gI.triples((None, DCTERMS.source, None)):
            gP.add((asrt_canon, PROV.wasDerivedFrom, src))

    out.serialize(str(out_file), format="trig")
    return out_file

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Input TriG file OR directory")
    ap.add_argument("output", help="Output TriG file OR directory")
    ap.add_argument("--select-base-substr", help="Select nanopub whose base IRI contains this substring")
    ap.add_argument("--select-doi", help="Select nanopub that cites this DOI (in dcterms:source)", default=None)
    ap.add_argument("--select-index", type=int, help="Select nanopub by index (0-based)", default=None)
    ap.add_argument("--explode", action="store_true", help="If input is a directory, write one output per nanopub")
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)

    if inp.is_file():
        # File -> File
        ds = ConjunctiveGraph()
        ds.parse(str(inp), format="trig")
        cands = find_nanopubs(ds)
        if not cands:
            print(f"[WARN] No nanopublications found in {inp}")
            sys.exit(2)
        chosen = select_one(cands, args.select_base_substr, args.select_doi, args.select_index)
        out_file = outp
        out_file.parent.mkdir(parents=True, exist_ok=True)
        emit_single(ds, chosen, out_file)
        print(f"[OK] Wrote single-NP file: {out_file}")
        sys.exit(0)

    if inp.is_dir():
        outp.mkdir(parents=True, exist_ok=True)
        trig_files = sorted([p for p in inp.iterdir() if p.is_file() and p.suffix.lower() == ".trig"])
        if not trig_files:
            print(f"[WARN] No *.trig files found in directory: {inp}")
            sys.exit(1)

        for f in trig_files:
            try:
                ds = ConjunctiveGraph()
                ds.parse(str(f), format="trig")
                cands = find_nanopubs(ds)
                if not cands:
                    print(f"[SKIP] {f.name}: no nanopubs found")
                    continue

                if args.explode:
                    # one output per nanopub
                    used_names = set()
                    for idx, cand in enumerate(cands):
                        # get a reasonable name from assertion graph
                        asg = cand["asg"] or ""
                        loc = local_id_from_old(str(asg)) if asg else f"np{idx}"
                        base_name = loc if loc else f"np{idx}"
                        # avoid collisions
                        name = base_name
                        k = 1
                        while name in used_names:
                            name = f"{base_name}-{k}"
                            k += 1
                        used_names.add(name)
                        out_file = outp / f"{name}.trig"
                        emit_single(ds, cand, out_file)
                        print(f"[OK] {f.name} -> {out_file.name}")
                else:
                    # one output per input file (choose one NP)
                    chosen = select_one(cands, args.select_base_substr, args.select_doi, args.select_index)
                    out_file = outp / f.name
                    emit_single(ds, chosen, out_file)
                    print(f"[OK] {f.name} -> {out_file.name} (one NP)")
            except Exception as e:
                print(f"[ERROR] {f.name}: {e}")
        sys.exit(0)

    print(f"[ERR] Input path not found: {inp}")
    sys.exit(2)

if __name__ == "__main__":
    main()
