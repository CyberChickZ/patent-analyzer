"""Indefiniteness categories and likelihood expressions from PEDANTIC
(Knappich, Friedrich, Hätty, Razniewski, arXiv 2505.21342), copied from
github.com/boschresearch/pedantic-patentsemtech
src/pedantic/dataset_creation/rejection_schema.py and src/pedantic/predict_llm.py
(MIT licence, Bosch). Descriptions verbatim; the examples are theirs too.
Used by app.llm.definiteness_advisory (the draft's LLM advisory) and by
evals/pedantic_definiteness_eval.py (gate 2 detector)."""

CATEGORIES = [
    ("antecedent_basis",
     "Claim contains a term referencing an element lacking a clear prior introduction, creating ambiguity what it references.",
     "Claim 11 recites the limitations 'the text input...' in line 7, and 'the one or more annotated tokens...' in line 13. "
     "There is insufficient antecedent basis for these limitations in the claim."),
    ("undefined_term",
     "A term lacks a clear, accepted and/or unambiguous meaning to a person of ordinary skill in the art (POSITA), making the claim's scope uncertain.",
     "Claims 7 and 18 recite the limitations 'applying a first weighting for the inside domain data size and applying a second weighting "
     "for outside domain data size for each group.' The claims neither the specification does not disclose what an inside domain or an "
     "outside domain are. As such the claims are indefinite."),
    ("relative_term",
     "A relative or subjective term (e.g., 'thin,' 'substantial') is used without providing a clear point of comparison, rendering the claim's scope indefinite.",
     "The term 'faster' in claims 1, 7 and 10 is a relative term which renders the claim indefinite. The term is not defined by the claim, "
     "the specification does not provide a standard for ascertaining the requisite degree, and one of ordinary skill in the art would not be "
     "reasonably apprised of the scope of the invention."),
    ("exemplary_phrasing",
     "Claim uses 'such as,' or similar phrasing, making it unclear whether the listed items are exhaustive or merely examples, leading to indefiniteness.",
     "Regarding claims 1-9, the phrase 'such as' renders the claim indefinite because it is unclear whether the limitations following "
     "the phrase are part of the claimed invention. See MPEP § 2173.05(d)"),
    ("functional_claiming",
     "Claim recites 'means for' or 'step for' without disclosing adequate corresponding structure, material, or acts in the specification, as required under 35 U.S.C. 112(f).",
     "Claim 1 limitations 'user intent detection control logic, identifying user intents ...' have been interpreted under 35 U.S.C. 112(f) "
     "because they use the generic placeholder 'user intent detection control logic' coupled with functional language without reciting "
     "sufficient structure, material or acts to achieve the function."),
    ("contradicting_limitations",
     "Claim includes an element that contradicts or is inconsistent with other claim limitations, making the claim's scope unclear.",
     "Regarding claim 20, this claim states that 'the instructions further cause the computer system to perform the repeating a predetermined "
     "number of times, regardless of the confidence score.' However, parent claim 19 explicitly claims that the steps are repeated 'until no "
     "documents have a confidence score below a threshold.' Thus, dependent claim 20 directly contradicts parent claim 19."),
    ("omission_of_essential_element_or_step",
     "Claim fails to recite an element, step, or cooperative relationship between elements/steps that is essential to the invention as disclosed.",
     "Claims 1-19 are rejected under 35 U.S.C. 112(b) as being incomplete for omitting essential structural cooperative relationships of "
     "elements, such omission amounting to a gap between the necessary structural connections. See MPEP § 2172.01."),
]
CATEGORY_KEYS = [k for k, _, _ in CATEGORIES]

# predict_llm.likelihood_expressions (verbatim strings and their probabilities)
LIKELIHOOD = {
    "almost certain": 0.96, "highly likely": 0.9, "very good chance": 0.8, "likely": 0.7, "more likely than even": 0.6,
    "about even": 0.5, "less likely than even": 0.4, "probably not": 0.25, "unlikely": 0.2, "little chance": 0.1,
    "highly unlikely": 0.05, "almost no chance": 0.02,
}


def format_categories(show_example: bool = False) -> str:
    """rejection_schema.format_category_descriptions(indent='', ignore={dependence, other})."""
    out = []
    for key, desc, ex in CATEGORIES:
        block = f' * "{key.replace("_", " ")}"\n   Description: {desc}'
        if show_example:
            block += f"\n   Example: {ex}"
        out.append(block)
    return "\n\n".join(out)


def likelihood_p(expr: str) -> float:
    e = " ".join(str(expr or "").lower().split())
    if e in LIKELIHOOD:
        return LIKELIHOOD[e]
    try:
        v = float(e.rstrip("%"))
        return v / 100.0 if v > 1 else v
    except ValueError:
        return 0.5


def normalize_category(c: str) -> str:
    c = " ".join(str(c or "").lower().replace("_", " ").split())
    for key in CATEGORY_KEYS:
        if c == key.replace("_", " ") or c == key:
            return key
    for key in CATEGORY_KEYS:
        if key.split("_")[0] in c:
            return key
    return "other"
