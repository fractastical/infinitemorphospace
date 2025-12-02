#!/usr/bin/env python3
"""
np_canonicalize_single_np.py  (LLM-provenance, repair, wave-friendly names)
----------------------------------------------------------------------------
Canonicalize nanopubs and emit EXACTLY ONE nanopublication per output file
(by default) *or* split each input into one output per nanopub (with --explode).
Also unifies namespaces and normalizes graph IRIs.

Fixes & Guarantees:
- Replace relative graph IRIs (e.g., <Nabcd#Head>) with absolute w3id bases.
- Convert mixed '#assertion'/'-provenance' forms to slash layout:
    BASE/Head, BASE/assertion, BASE/provenance, BASE/pubinfo
- Unify https://example.org/levin-kg/ -> https://w3id.org/levin-kg/
- If model terms (Assertion, hasEvidence, ...) were minted under /np/, move to https://w3id.org/levin-kg/
- Skip empty assertions by default (won't emit broken nanopubs).
- Ensure /Head includes:  dcterms:conformsTo <https://w3id.org/morphopkg/spc/1.0>
- Ensure /pubinfo has a BASE-subject triple (adds dcterms:created if missing)
  and tries to set dcterms:source (DOI) from assertion/provenance/pubinfo.
- Ensure /provenance contains at least one triple with the ASSERTION GRAPH as subject:
    <BASE/assertion> prov:wasDerivedFrom <DOI>           (preferred)
    else <BASE/assertion> prov:wasGeneratedBy <https://w3id.org/levin-kg/activity/canonicalize-v1>

Options:
  --normalize-mappings
      Rewrite alias properties to ex:supportsHypothesis and canonicalize hypothesis IRIs to ex:L1..ex:L9.
  --ensure-llm-provenance
      Backfill LLM provenance on ex:BayesianAssessment in /pubinfo:
        dcterms:creator "GPT-5 Pro" ;
        dcterms:created "YYYY-MM-DD"^^xsd:date ;
        ex:assessmentAgent <https://w3id.org/levin-kg/agent/gpt-5-pro> ;
        prov:wasAttributedTo <https://w3id.org/levin-kg/agent/gpt-5-pro> ;
        <https://w3id.org/levin-kg/agent/gpt-5-pro> a prov:SoftwareAgent .
  --repair-empty-assertions
      If an assertion graph is empty, synthesize a minimal assertion by:
        * locating a claim node (typed ex:Assertion or a claim-* IRI) from other graphs
        * copying all triples about that claim into /assertion
        * ensuring rdf:type ex:Assertion
        * adding cito:citesAsEvidence <DOI> if a DOI can be inferred
        * linking any ex:BayesianAssessment found in /pubinfo via ex:hasAssessment

Naming (new defaults):
  - If an input file has ONE nanopub: write output named like the input (keeps your wave filename).
  - If an input file has MULTIPLE nanopubs: name as <input_stem>__<slug>.trig, where slug prefers DOI.

Flags to control naming:
  --prefer-input-name / --no-prefer-input-name   (default: prefer when exactly one NP)
  --suffix {doi,assertion}                       (default: doi)

Usage:
  # File -> File (one NP)
  python np_canonicalize_single_np.py input.trig output.trig \
    --normalize-mappings --ensure-llm-provenance

  # Directory -> Directory (explode: one output per NP, keep wave names)
  python np_canonicalize_single_np.py waves/ waves_fixed/ \
    --explode --normalize-mappings --ensure-llm-provenance --repair-empty-assertions

Requires: rdflib >= 6.0
"""
import sys, re, datetime
from pathlib import Path
from urllib.parse import quote

try:
    from rdflib import Dataset, Graph, URIRef, Namespace, RDF, Literal
except ImportError:
    # Fallback for older rdflib: Dataset may not exist; use ConjunctiveGraph as a drop-in
    from rdflib import ConjunctiveGraph as Dataset, Graph, URIRef, Namespace, RDF, Literal

from rdflib.namespace import DCTERMS, XSD

# Namespaces
NP    = Namespace("http://www.nanopub.org/nschema#")
PROV  = Namespace("http://www.w3.org/ns/prov#")
CITO  = Namespace("http://purl.org/spar/cito/")
EX    = Namespace("https://w3id.org/levin-kg/")
EX_NP = "https://w3id.org/levin-kg/np/"
EX_EXAMPLE = Namespace("https://example.org/levin-kg/")

PROFILE   = URIRef("https://w3id.org/morphopkg/spc/1.0")
ACT_CANON = URIRef("https://w3id.org/levin-kg/activity/canonicalize-v1")

# Model terms that sometimes were minted under /np/ by mistake
MODEL_TERMS = {
    # classes
    "Assertion","EvidenceItem","BayesianAssessment","Hypothesis","EconomicHypothesis","EconomicAssessment",
    # properties
    "hasEvidence","hasAssessment","supportsHypothesis","hypothesisContribution","mappingConfidence","mappingMethod",
    "assessmentAgent","context",
    # numeric properties
    "bayesFactorCombined","weightOfEvidence_deciban","calibrationMethod","publicationYear",
    "pValue","bayesFactorVS_MPR","countN","totalN","frequency","timeDays"
}

# Mapping alias properties to normalize
MAPPING_ALIASES = {
    URIRef(str(EX) + "supportsHypothesis"),
    URIRef(str(EX) + "hasHypothesis"),
    URIRef(str(EX) + "mapsToHypothesis"),
    URIRef("https://example.org/levin-kg/hasHypothesis"),
    URIRef("https://example.org/levin-kg/mapsToHypothesis"),
    URIRef("https://example.org/levin-kg/supportsHypotheses"),
}

# LLM agent constants
AGENT_IRI   = URIRef("https://w3id.org/levin-kg/agent/gpt-5-pro")
AGENT_LABEL = "GPT-5 Pro"

# --------------------------
# Utility / normalization
# --------------------------

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
        return URIRef(str(node).replace(str(EX_EXAMPLE), str(EX), 1))
    return node

def rewrite_model_terms(node):
    if isinstance(node, URIRef):
        s = str(node)
        if s.startswith(EX_NP):
            local = s[len(EX_NP):]
            if "/" not in local and local in MODEL_TERMS:
                return URIRef(str(EX) + local)
    return node

def unify_hypothesis_iri(node):
    if not isinstance(node, URIRef):
        return node
    s = str(node)
    m = re.search(r"L([1-9])$", s)  # L1..L9
    if m:
        return URIRef(str(EX) + "L" + m.group(1))
    if s.startswith(str(EX_EXAMPLE)):
        return URIRef(s.replace(str(EX_EXAMPLE), str(EX), 1))
    return node

def unify_node(node):
    # example.org -> w3id, and /np/ModelTerm -> EX namespace
    return rewrite_model_terms(rewrite_example_to_w3id(node))

def _slug_from_doi(iri_or_literal):
    s = str(iri_or_literal)
    s = re.sub(r'^https?://(dx\.)?doi\.org/', '', s, flags=re.I)
    s = re.sub(r'^doi:', '', s, flags=re.I)
    return s.replace('/', '_').replace(':', '_')

def _extract_wave_prefix(stem: str) -> str:
    """
    Try to preserve your wave name. Examples:
      'np_levin_nanopubs_waveB' -> 'np_levin_nanopubs_waveB'
      'waveG' -> 'waveG'
    Otherwise return the full stem.
    """
    m = re.search(r'(.*wave[A-Za-z0-9]+)$', stem)
    return m.group(1) if m else stem

# --------------------------
# Discovery
# --------------------------

def find_nanopubs(ds: Dataset):
    """Return list of dicts: head, base, asg, prg, pig, dois, asrt_triples"""
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
            if pig is not None:
                gp = ds.get_context(pig)
                for _, p, o in gp.triples((None, DCTERMS.source, None)):
                    dois.add(str(o))
            asrt_triples = 0
            if asg is not None:
                asrt_triples = sum(1 for _ in ds.get_context(asg).triples((None,None,None)))
            out.append(dict(head=head_iri, base=base, asg=asg, prg=prg, pig=pig, dois=sorted(dois), asrt_triples=asrt_triples))

    # fallback: graphs named like .../Head, even if they don’t declare the type
    if not out:
        for g in ds.contexts():
            sid = str(g.identifier)
            if sid.endswith(("/Head","#Head","-Head")):
                out.append(dict(head=g.identifier, base=sid[:-5], asg=None, prg=None, pig=None, dois=[], asrt_triples=0))
    return out

def select_candidates(cands, accept_empty: bool):
    if accept_empty:
        return [c for c in cands if c.get("asg") is not None]
    return [c for c in cands if c.get("asg") is not None and c.get("asrt_triples",0) > 0]

# --------------------------
# Repair helpers
# --------------------------

def _collect_doi_from(ds: Dataset, gI_in: Graph, gP_in: Graph, gA_out: Graph):
    """Try to infer a DOI (as any IRI) from pubinfo source, assertion citesAsEvidence, or provenance wasDerivedFrom."""
    # 1) pubinfo dcterms:source (any subject)
    if gI_in is not None:
        for _, _, o in gI_in.triples((None, DCTERMS.source, None)):
            return o
    # 2) assertion citesAsEvidence (from already-copied assertion)
    for _, _, o in gA_out.triples((None, CITO.citesAsEvidence, None)):
        return o
    # 3) provenance wasDerivedFrom (input)
    if gP_in is not None:
        for _, _, o in gP_in.triples((None, PROV.wasDerivedFrom, None)):
            return o
    return None

def _find_candidate_claims(ds: Dataset):
    """Search across all graphs for plausible claim subjects."""
    claims = set()
    # typed ex:Assertion anywhere
    for g in ds.contexts():
        for s, _, _ in g.triples((None, RDF.type, URIRef(str(EX) + "Assertion"))):
            claims.add(s)
    # subjects that link to an assessment (likely the claim)
    for g in ds.contexts():
        for s, _, _ in g.triples((None, URIRef(str(EX) + "hasAssessment"), None)):
            if isinstance(s, URIRef):
                claims.add(s)
    return list(claims)

def _rehydrate_assertion(ds: Dataset, asg_in_iri: URIRef, prg_in_iri: URIRef, pig_in_iri: URIRef, base_uri: str):
    """
    Reconstruct a minimal assertion graph as a list of triples.
    Strategy:
      - choose a claim node from elsewhere (typed ex:Assertion or subject having ex:hasAssessment)
      - copy all triples about that claim from any graph
      - ensure rdf:type ex:Assertion
      - add cito:citesAsEvidence <DOI> when we can infer one
      - if any ex:BayesianAssessment found in pubinfo, link via ex:hasAssessment
    """
    triples = []

    # choose claim
    candidates = _find_candidate_claims(ds)
    if candidates:
        claim = candidates[0]
    else:
        claim = URIRef(str(EX) + "claim-" + local_id_from_old(str(asg_in_iri)))

    # copy triples about claim from any graph
    for g in ds.contexts():
        for s, p, o in g.triples((claim, None, None)):
            triples.append((s, p, o))

    # ensure type ex:Assertion
    if not any(p == RDF.type and str(o) == str(EX) + "Assertion" for _, p, o in triples):
        triples.append((claim, RDF.type, URIRef(str(EX) + "Assertion")))

    # attempt to add DOI if missing
    gI_in = ds.get_context(pig_in_iri) if pig_in_iri is not None else None
    gP_in = ds.get_context(prg_in_iri) if prg_in_iri is not None else None

    # temp graph to reuse _collect_doi_from logic with current triples
    gA_temp = Graph()
    for t in triples:
        gA_temp.add(t)

    doi = _collect_doi_from(ds, gI_in, gP_in, gA_temp)
    if doi is not None and not any(p == CITO.citesAsEvidence for _, p, _ in triples):
        triples.append((claim, CITO.citesAsEvidence, doi))

    # link assessment from pubinfo if present
    if gI_in is not None:
        for a, _, _ in gI_in.triples((None, RDF.type, URIRef(str(EX) + "BayesianAssessment"))):
            if not any(p == URIRef(str(EX) + "hasAssessment") for _, p, _ in triples):
                triples.append((claim, URIRef(str(EX) + "hasAssessment"), a))
                break

    return triples

# --------------------------
# Emission
# --------------------------

def ensure_llm_provenance(gI_out: Graph, assessment: URIRef):
    """Ensure LLM provenance on the given assessment node in /pubinfo."""
    today = datetime.date.today().isoformat()
    # creator literal
    if not any(True for _ in gI_out.triples((assessment, DCTERMS.creator, None))):
        gI_out.add((assessment, DCTERMS.creator, Literal(AGENT_LABEL)))
    # created date
    if not any(True for _ in gI_out.triples((assessment, DCTERMS.created, None))):
        gI_out.add((assessment, DCTERMS.created, Literal(today, datatype=XSD.date)))
    # assessmentAgent custom property
    gI_out.add((assessment, URIRef(str(EX) + "assessmentAgent"), AGENT_IRI))
    # prov:wasAttributedTo
    gI_out.add((assessment, PROV.wasAttributedTo, AGENT_IRI))
    # Type the agent (once is fine)
    gI_out.add((AGENT_IRI, RDF.type, PROV.SoftwareAgent))

def _choose_slug(ds: Dataset, cand: dict, gA_out: Graph, mode: str, default_local: str):
    """Return a string slug for filename based on mode: 'doi' or 'assertion'."""
    if mode == "doi":
        # Prefer DOI from pubinfo discovered earlier
        dois = cand.get("dois") or []
        if dois:
            return _slug_from_doi(dois[0])
        # Else try DOI from assertion via citesAsEvidence
        doi2 = next((o for _, _, o in gA_out.triples((None, CITO.citesAsEvidence, None))), None)
        if doi2 is not None:
            return _slug_from_doi(doi2)
        # Fall back
        return default_local
    # assertion mode
    return default_local

def emit_single(ds: Dataset, cand, out_file: Path,
                normalize_mappings=False, ensure_llm=False, repair_empty=False):
    base_guess = cand["base"]
    asg_in = cand["asg"]
    prg_in = cand["prg"]
    pig_in = cand["pig"]

    # assert graph fallback discovery
    if asg_in is None:
        for g in ds.contexts():
            s = str(g.identifier)
            if s.endswith(("-assertion","#assertion","/assertion")) and base_guess in s:
                asg_in = g.identifier
                break
    if asg_in is None:
        raise RuntimeError(f"No assertion graph found for base {base_guess}")

    # canonical IRIs
    base = canonical_base(str(asg_in))
    head_canon = URIRef(base + "Head")
    asrt_canon = URIRef(base + "assertion")
    prov_canon = URIRef(base + "provenance")
    pubi_canon = URIRef(base + "pubinfo")

    # output dataset
    out = Dataset()
    out.bind("np", NP); out.bind("prov", PROV); out.bind("dcterms", DCTERMS)
    out.bind("ex", EX); out.bind("cito", CITO)

    # HEAD
    gH = out.get_context(head_canon)
    gH.add((URIRef(base), RDF.type, NP.Nanopublication))
    gH.add((URIRef(base), NP.hasAssertion, asrt_canon))
    gH.add((URIRef(base), NP.hasProvenance, prov_canon))
    gH.add((URIRef(base), NP.hasPublicationInfo, pubi_canon))
    gH.add((URIRef(base), DCTERMS.conformsTo, PROFILE))

    # ASSERTION
    gA_in = ds.get_context(asg_in)
    gA = out.get_context(asrt_canon)
    src_triples = list(gA_in.triples((None, None, None)))

    if not src_triples and repair_empty:
        # synthesize a minimal assertion
        repaired = _rehydrate_assertion(ds, asg_in, prg_in, pig_in, base)
        for s, p, o in repaired:
            s2, p2, o2 = unify_node(s), unify_node(p), unify_node(o)
            if normalize_mappings and p2 in MAPPING_ALIASES:
                p2 = URIRef(str(EX) + "supportsHypothesis")
                o2 = unify_hypothesis_iri(o2)
            gA.add((s2, p2, o2))
    else:
        # copy as-is (with normalization)
        for s, p, o in src_triples:
            s2, p2, o2 = unify_node(s), unify_node(p), unify_node(o)
            if normalize_mappings and p2 in MAPPING_ALIASES:
                p2 = URIRef(str(EX) + "supportsHypothesis")
                o2 = unify_hypothesis_iri(o2)
            gA.add((s2, p2, o2))

    # PROVENANCE (input graph)
    if prg_in is None:
        prg_in = URIRef(str(asg_in).replace("-assertion","-provenance").replace("#assertion","#provenance").replace("/assertion","/provenance"))
    gP_in = ds.get_context(prg_in)
    gP = out.get_context(prov_canon)
    for s, p, o in gP_in.triples((None, None, None)):
        s2, p2, o2 = unify_node(s), unify_node(p), unify_node(o)
        if p2 == PROV.wasDerivedFrom:
            gP.add((asrt_canon, p2, o2))
        else:
            gP.add((s2, p2, o2))

    # PUBINFO (input graph)
    if pig_in is None:
        pig_in = URIRef(str(asg_in).replace("-assertion","-pubinfo").replace("#assertion","#pubinfo").replace("/assertion","/pubinfo"))
    gI_in = ds.get_context(pig_in)
    gI = out.get_context(pubi_canon)
    for s, p, o in gI_in.triples((None, None, None)):
        gI.add((unify_node(s), unify_node(p), unify_node(o)))

    # Ensure at least one BASE-subject triple in pubinfo
    if not any(True for _ in gI.triples((URIRef(base), None, None))):
        today = datetime.date.today().isoformat()
        gI.add((URIRef(base), DCTERMS.created, Literal(today, datatype=XSD.date)))

    # Ensure a DOI in pubinfo if available
    if not any(True for _ in gI.triples((URIRef(base), DCTERMS.source, None))):
        doi = _collect_doi_from(ds, gI_in, gP_in, gA)
        if doi is not None:
            gI.add((URIRef(base), DCTERMS.source, doi))

    # Ensure at least one triple in provenance with ASSERTION as subject
    if not any(True for _ in gP.triples((asrt_canon, None, None))):
        src = None
        for _, _, o in gI.triples((URIRef(base), DCTERMS.source, None)):
            src = o
            break
        if src is not None:
            gP.add((asrt_canon, PROV.wasDerivedFrom, src))
        else:
            gP.add((asrt_canon, PROV.wasGeneratedBy, ACT_CANON))

    # Ensure LLM provenance on assessments (in /pubinfo)
    if ensure_llm:
        ASSESS_CLS = URIRef(str(EX) + "BayesianAssessment")
        for a, _, _ in gI.triples((None, RDF.type, ASSESS_CLS)):
            ensure_llm_provenance(gI, a)

    # Write file
    out.serialize(str(out_file), format="trig")
    return out_file

# --------------------------
# Driver
# --------------------------

def process_file(input_path: Path, output_path: Path,
                 explode=False, normalize_mappings=False, ensure_llm=False, repair_empty=False,
                 prefer_input_name=True, suffix_mode="doi"):
    ds = Dataset()
    ds.parse(str(input_path), format="trig")
    cands = find_nanopubs(ds)
    cands = select_candidates(cands, accept_empty=repair_empty)

    if not cands:
        print(f"[WARN] {input_path.name}: no usable nanopubs (empty assertions)")
        return

    # Single-file output (not explode): keep 'output_path' as given
    if not explode:
        out_file = output_path if output_path.suffix else (output_path / input_path.name)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        emit_single(ds, cands[0], out_file,
                    normalize_mappings=normalize_mappings,
                    ensure_llm=ensure_llm,
                    repair_empty=repair_empty)
        print(f"[OK] {input_path.name} -> {out_file.name} (one NP)")
        return

    # Explode mode (one output per NP)
    output_path.mkdir(parents=True, exist_ok=True)
    used_names = set()

    # If EXACTLY ONE NP and prefer_input_name=True, keep the wave filename
    if len(cands) == 1 and prefer_input_name:
        out_file = output_path / f"{input_path.stem}.trig"
        emit_single(ds, cands[0], out_file,
                    normalize_mappings=normalize_mappings,
                    ensure_llm=ensure_llm,
                    repair_empty=repair_empty)
        print(f"[OK] {input_path.name} -> {out_file.name}")
        return

    # Otherwise, derive a friendly name: <wave_prefix>__<slug>.trig
    wave_prefix = _extract_wave_prefix(input_path.stem)

    for idx, cand in enumerate(cands):
        asg = cand["asg"]
        default_local = local_id_from_old(str(asg)) if asg else f"np{idx}"

        # First, build a transient output to compute a DOI-based slug if needed
        # We emit to a temp file in memory? Simpler: pick slug without emitting.
        # For DOI slug, we might need assertion content; but we can read it here:
        # Build a tiny Graph same way 'emit_single' will
        base = canonical_base(str(asg)) if asg else "https://w3id.org/levin-kg/np/np/"
        asrt_canon = URIRef(base + "assertion")
        gA_out = Graph()
        gA_in = ds.get_context(asg) if asg else Graph()
        src_triples = list(gA_in.triples((None, None, None)))
        if not src_triples and repair_empty:
            for t in _rehydrate_assertion(ds, asg, cand.get("prg"), cand.get("pig"), base):
                gA_out.add(t)
        else:
            for t in src_triples:
                gA_out.add(t)

        slug = _choose_slug(ds, cand, gA_out, suffix_mode, default_local)
        base_name = f"{wave_prefix}__{slug}"
        name = base_name
        k = 1
        while name in used_names:
            name = f"{base_name}-{k}"; k += 1
        used_names.add(name)

        out_file = output_path / f"{name}.trig"
        emit_single(ds, cand, out_file,
                    normalize_mappings=normalize_mappings,
                    ensure_llm=ensure_llm,
                    repair_empty=repair_empty)
        print(f"[OK] {input_path.name} -> {out_file.name}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Input TriG file OR directory")
    ap.add_argument("output", help="Output TriG file OR directory")
    ap.add_argument("--explode", action="store_true", help="Directory mode: write one output per nanopub")
    ap.add_argument("--normalize-mappings", action="store_true", help="Rewrite mapping aliases and canonicalize hypothesis IRIs")
    ap.add_argument("--ensure-llm-provenance", action="store_true", help="Backfill LLM provenance on assessments")
    ap.add_argument("--repair-empty-assertions", action="store_true", help="Synthesize assertion graph if empty")
    ap.add_argument("--no-prefer-input-name", dest="prefer_input_name", action="store_false",
                    help="Do not keep the input filename when there is exactly one NP in a file")
    ap.add_argument("--suffix", choices=["doi","assertion"], default="doi",
                    help="When a file has multiple NPs, choose filename suffix by 'doi' (default) or 'assertion' id")
    args = ap.parse_args()

    inp = Path(args.input); outp = Path(args.output)
    kwargs = dict(
        explode=args.explode,
        normalize_mappings=args.normalize_mappings,
        ensure_llm=args.ensure_llm_provenance,
        repair_empty=args.repair_empty_assertions,
        prefer_input_name=args.prefer_input_name,
        suffix_mode=args.suffix
    )

    if inp.is_file():
        process_file(inp, outp, **kwargs)
    elif inp.is_dir():
        outp.mkdir(parents=True, exist_ok=True)
        for f in sorted(p for p in inp.iterdir() if p.is_file() and p.suffix.lower() == ".trig"):
            try:
                process_file(f, outp, **kwargs)
            except Exception as e:
                print(f"[ERROR] {f.name}: {e}")
    else:
        print(f"[ERR] Input not found: {inp}")
        sys.exit(2)

if __name__ == "__main__":
    main()
