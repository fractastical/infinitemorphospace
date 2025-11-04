#!/usr/bin/env python3
"""
np_canonicalize_single_np.py  (LLM-provenance enhanced)
-------------------------------------------------------
- Normalizes graph IRIs (slash style), unifies namespaces, fixes model term namespace.
- Skips empty assertions; guarantees required /provenance and /pubinfo structures.
- Optional normalization of hypothesis mappings (--normalize-mappings).
- Optional backfill of LLM provenance on assessments (--ensure-llm-provenance).

LLM provenance policy (when --ensure-llm-provenance is used):
For each ex:BayesianAssessment node in /pubinfo, ensure:
  dcterms:creator "GPT-5 Pro" ;
  ex:assessmentAgent <https://w3id.org/levin-kg/agent/gpt-5-pro> ;
  prov:wasAttributedTo <https://w3id.org/levin-kg/agent/gpt-5-pro> ;
  dcterms:created "YYYY-MM-DD"^^xsd:date   (if missing).
Also ensure the agent IRI is typed prov:SoftwareAgent (in /pubinfo).

Usage examples:
  python np_canonicalize_single_np.py waves/ np-waves_fixed/ --explode --normalize-mappings --ensure-llm-provenance
  python np_canonicalize_single_np.py input.trig output.trig --ensure-llm-provenance
"""
import sys, re, datetime
from pathlib import Path
from urllib.parse import quote
from rdflib import ConjunctiveGraph, URIRef, Namespace, RDF, Literal
from rdflib.namespace import DCTERMS, XSD

NP   = Namespace("http://www.nanopub.org/nschema#")
PROV = Namespace("http://www.w3.org/ns/prov#")
CITO = Namespace("http://purl.org/spar/cito/")
EX_W3ID = Namespace("https://w3id.org/levin-kg/")
EX_W3ID_NP = "https://w3id.org/levin-kg/np/"
EX_EXAMPLE = Namespace("https://example.org/levin-kg/")
PROFILE = URIRef("https://w3id.org/morphopkg/spc/1.0")
ACT_CANON = URIRef("https://w3id.org/levin-kg/activity/canonicalize-v1")

MODEL_TERMS = {
    "Assertion","EvidenceItem","BayesianAssessment","Hypothesis","EconomicHypothesis","EconomicAssessment",
    "hasEvidence","hasAssessment","supportsHypothesis","hypothesisContribution","mappingConfidence","mappingMethod",
    "assessmentAgent","context","bayesFactorCombined","weightOfEvidence_deciban","calibrationMethod","publicationYear",
    "pValue","bayesFactorVS_MPR","countN","totalN","frequency","timeDays"
}
MAPPING_ALIASES = {
    URIRef(str(EX_W3ID) + "supportsHypothesis"),
    URIRef(str(EX_W3ID) + "hasHypothesis"),
    URIRef(str(EX_W3ID) + "mapsToHypothesis"),
    URIRef("https://example.org/levin-kg/hasHypothesis"),
    URIRef("https://example.org/levin-kg/mapsToHypothesis"),
    URIRef("https://example.org/levin-kg/supportsHypotheses"),
}

AGENT_IRI = URIRef("https://w3id.org/levin-kg/agent/gpt-5-pro")
AGENT_LABEL = "GPT-5 Pro"

def local_id_from_old(assertion_graph_iri: str) -> str:
    s = assertion_graph_iri
    for sep in ("-assertion","#assertion","/assertion"):
        if s.endswith(sep):
            s = s[: -len(sep)]
            break
    return s.rstrip("/").split("/")[-1] or "np"

def canonical_base(old_assertion_iri: str) -> str:
    local = local_id_from_old(old_assertion_iri)
    from urllib.parse import quote
    safe_local = quote(local, safe="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
    return f"https://w3id.org/levin-kg/np/{safe_local}/"

def rewrite_example_to_w3id(node):
    if isinstance(node, URIRef) and str(node).startswith(str(EX_EXAMPLE)):
        return URIRef(str(node).replace(str(EX_EXAMPLE), str(EX_W3ID), 1))
    return node

def rewrite_model_terms(node):
    if isinstance(node, URIRef):
        s = str(node)
        if s.startswith(EX_W3ID_NP):
            local = s[len(EX_W3ID_NP):]
            if "/" not in local and local in MODEL_TERMS:
                return URIRef(str(EX_W3ID) + local)
    return node

def unify_hypothesis_iri(node):
    if not isinstance(node, URIRef):
        return node
    s = str(node)
    m = re.search(r"L([1-9])$", s)
    if m:
        return URIRef(str(EX_W3ID) + "L" + m.group(1))
    if s.startswith(str(EX_EXAMPLE)):
        return URIRef(s.replace(str(EX_EXAMPLE), str(EX_W3ID), 1))
    return node

def unify_node(node):
    return rewrite_model_terms(rewrite_example_to_w3id(node))

def find_nanopubs(ds: ConjunctiveGraph):
    out = []
    for g in ds.contexts():
        for (npnode, _, _) in g.triples((None, RDF.type, NP.Nanopublication)):
            head_iri = g.identifier
            s = str(head_iri)
            base = s[:-5] if s.endswith(("/Head","#Head","-Head")) else str(npnode)
            asg = prg = pig = None
            for _, p, o in g.triples((npnode, None, None)):
                if p == NP.hasAssertion:       asg = o
                if p == NP.hasProvenance:      prg = o
                if p == NP.hasPublicationInfo: pig = o
            dois = set()
            if pig:
                gp = ds.get_context(pig)
                for _, p, o in gp.triples((None, DCTERMS.source, None)):
                    dois.add(str(o))
            asrt_triples = 0
            if asg is not None:
                asrt_triples = sum(1 for _ in ds.get_context(asg).triples((None,None,None)))
            out.append(dict(head=head_iri, base=base, asg=asg, prg=prg, pig=pig, dois=sorted(dois), asrt_triples=asrt_triples))
    if not out:
        for g in ds.contexts():
            sid = str(g.identifier)
            if sid.endswith(("/Head","#Head","-Head")):
                out.append(dict(head=g.identifier, base=sid[:-5], asg=None, prg=None, pig=None, dois=[], asrt_triples=0))
    return out

def select_candidates(cands):
    return [c for c in cands if c.get("asg") is not None and c.get("asrt_triples",0) > 0]

def ensure_llm_provenance(gI, assessment):
    """Ensure LLM provenance on the given assessment node in /pubinfo."""
    today = datetime.date.today().isoformat()
    # creator literal
    if not any(True for _ in gI.triples((assessment, DCTERMS.creator, None))):
        gI.add((assessment, DCTERMS.creator, Literal(AGENT_LABEL)))
    # created date
    if not any(True for _ in gI.triples((assessment, DCTERMS.created, None))):
        gI.add((assessment, DCTERMS.created, Literal(today, datatype=XSD.date)))
    # assessmentAgent custom property
    gI.add((assessment, URIRef(str(EX_W3ID) + "assessmentAgent"), AGENT_IRI))
    # prov:wasAttributedTo
    gI.add((assessment, PROV.wasAttributedTo, AGENT_IRI))
    # Type the agent (once is fine)
    gI.add((AGENT_IRI, RDF.type, PROV.SoftwareAgent))

def emit_single(ds: ConjunctiveGraph, cand, out_file: Path, normalize_mappings=False, ensure_llm=False):
    base_guess = cand["base"]
    asg_in = cand["asg"]; prg_in = cand["prg"]; pig_in = cand["pig"]

    if asg_in is None:
        for g in ds.contexts():
            s = str(g.identifier)
            if s.endswith(("-assertion","#assertion","/assertion")) and base_guess in s:
                asg_in = g.identifier; break
    if asg_in is None:
        raise RuntimeError(f"No assertion graph found for base {base_guess}")

    base = canonical_base(str(asg_in))
    head_canon = URIRef(base + "Head")
    asrt_canon = URIRef(base + "assertion")
    prov_canon = URIRef(base + "provenance")
    pubi_canon = URIRef(base + "pubinfo")

    out = ConjunctiveGraph()
    out.bind("np", NP); out.bind("prov", PROV); out.bind("dcterms", DCTERMS)
    out.bind("ex", EX_W3ID); out.bind("cito", CITO)

    # HEAD
    gH = out.get_context(head_canon)
    gH.add((URIRef(base), RDF.type, NP.Nanopublication))
    gH.add((URIRef(base), NP.hasAssertion, asrt_canon))
    gH.add((URIRef(base), NP.hasProvenance, prov_canon))
    gH.add((URIRef(base), NP.hasPublicationInfo, pubi_canon))
    gH.add((URIRef(base), DCTERMS.conformsTo, PROFILE))

    # ASSERTION
    gA_in = ds.get_context(asg_in); gA = out.get_context(asrt_canon)
    for s,p,o in gA_in.triples((None,None,None)):
        s2,p2,o2 = unify_node(s), unify_node(p), unify_node(o)
        if normalize_mappings and p2 in MAPPING_ALIASES:
            p2 = URIRef(str(EX_W3ID) + "supportsHypothesis")
            o2 = unify_hypothesis_iri(o2)
        gA.add((s2,p2,o2))

    # PROVENANCE
    if prg_in is None:
        prg_in = URIRef(str(asg_in).replace("-assertion","-provenance").replace("#assertion","#provenance").replace("/assertion","/provenance"))
    gP_in = ds.get_context(prg_in); gP = out.get_context(prov_canon)
    if gP_in:
        for s,p,o in gP_in.triples((None,None,None)):
            s2,p2,o2 = unify_node(s), unify_node(p), unify_node(o)
            if p2 == PROV.wasDerivedFrom:
                gP.add((asrt_canon, p2, o2))
            else:
                gP.add((s2,p2,o2))

    # PUBINFO
    if pig_in is None:
        pig_in = URIRef(str(asg_in).replace("-assertion","-pubinfo").replace("#assertion","#pubinfo").replace("/assertion","/pubinfo"))
    gI_in = ds.get_context(pig_in); gI = out.get_context(pubi_canon)
    if gI_in:
        for s,p,o in gI_in.triples((None,None,None)):
            gI.add((unify_node(s), unify_node(p), unify_node(o)))

    # Ensure at least one BASE-subject triple in pubinfo
    if not any(True for _ in gI.triples((URIRef(base), None, None))):
        today = datetime.date.today().isoformat()
        gI.add((URIRef(base), DCTERMS.created, Literal(today, datatype=XSD.date)))

    # Try to ensure a DOI in pubinfo if available
    if not any(True for _ in gI.triples((URIRef(base), DCTERMS.source, None))):
        doi = None
        for _,_,o in gA.triples((None, CITO.citesAsEvidence, None)):
            doi = o; break
        if doi is None and gP_in:
            for _,p,o in gP_in.triples((None, PROV.wasDerivedFrom, None)):
                doi = o; break
        if doi is not None:
            gI.add((URIRef(base), DCTERMS.source, doi))

    # Ensure at least one triple in provenance with ASSERTION as subject
    if not any(True for _ in gP.triples((asrt_canon, None, None))):
        src = None
        for _,_,o in gI.triples((URIRef(base), DCTERMS.source, None)):
            src = o; break
        if src is not None:
            gP.add((asrt_canon, PROV.wasDerivedFrom, src))
        else:
            gP.add((asrt_canon, PROV.wasGeneratedBy, ACT_CANON))

    # Ensure LLM provenance on assessments
    if ensure_llm:
        # Find assessment nodes in pubinfo
        ASSESS_CLS = URIRef(str(EX_W3ID) + "BayesianAssessment")
        for a,_,_ in gI.triples((None, RDF.type, ASSESS_CLS)):
            ensure_llm_provenance(gI, a)

    out.serialize(str(out_file), format="trig")
    return out_file

def process_file(input_path: Path, output_path: Path, explode=False, **kwargs):
    ds = ConjunctiveGraph()
    ds.parse(str(input_path), format="trig")
    cands = find_nanopubs(ds)
    cands = [c for c in cands if c.get("asg") is not None and c.get("asrt_triples",0) > 0]
    if not cands:
        print(f"[WARN] {input_path.name}: no usable nanopubs (empty assertions)")
        return
    if not explode:
        out_file = output_path if output_path.suffix else (output_path / input_path.name)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        emit_single(ds, cands[0], out_file, **kwargs)
        print(f"[OK] {input_path.name} -> {out_file.name} (one NP)")
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        used = set()
        for idx, cand in enumerate(cands):
            asg = cand["asg"]
            loc = local_id_from_old(str(asg)) if asg else f"np{idx}"
            name = loc; k=1
            while name in used:
                name = f"{loc}-{k}"; k+=1
            used.add(name)
            out_file = output_path / f"{name}.trig"
            emit_single(ds, cand, out_file, **kwargs)
            print(f"[OK] {input_path.name} -> {out_file.name}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Input TriG file OR directory")
    ap.add_argument("output", help="Output TriG file OR directory")
    ap.add_argument("--explode", action="store_true", help="Directory mode: write one output per nanopub")
    ap.add_argument("--normalize-mappings", action="store_true", help="Rewrite mapping aliases and canonicalize hypothesis IRIs")
    ap.add_argument("--ensure-llm-provenance", action="store_true", help="Backfill LLM provenance on assessments")
    args = ap.parse_args()

    inp = Path(args.input); outp = Path(args.output)
    kwargs = dict(normalize_mappings=args.normalize_mappings, ensure_llm=args.ensure_llm_provenance)

    if inp.is_file():
        process_file(inp, outp, explode=False, **kwargs)
    elif inp.is_dir():
        outp.mkdir(parents=True, exist_ok=True)
        for f in sorted(p for p in inp.iterdir() if p.suffix.lower()==".trig"):
            try:
                process_file(f, outp, explode=args.explode, **kwargs)
            except Exception as e:
                print(f"[ERROR] {f.name}: {e}")
    else:
        print(f"[ERR] Input not found: {inp}")
        sys.exit(2)

if __name__ == "__main__":
    main()
