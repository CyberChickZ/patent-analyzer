"""Disclosure adapter (Dis2Pat form): title / problem / core_idea / how_it_works /
novelty / benefits / optional_variants -> Doc (the seven HF `lj408/Dis2Pat`
disclosure fields). core_idea seeds the A1 concept, how_it_works is the
element source, optional_variants become dependent_hints."""

FIELDS = (
    ("problem", "Problem"),
    ("core_idea", "Core Idea"),
    ("how_it_works", "How It Works"),
    ("novelty", "Novelty"),
    ("benefits", "Benefits"),
    ("optional_variants", "Optional Variants"),
)


def _paragraphs(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v or "").strip()]
    text = str(value).replace("\r\n", "\n")
    paras = [" ".join(p.split()) for p in text.split("\n\n")]
    return [p for p in paras if p]


def doc_from_fields(problem: str = "", core_idea: str = "", how_it_works: str = "",
                    novelty: str = "", optional_variants=None, title: str = "", benefits: str = "") -> dict:
    values = {"problem": problem, "core_idea": core_idea, "how_it_works": how_it_works,
              "novelty": novelty, "benefits": benefits, "optional_variants": optional_variants}
    sections = [{"title": label, "paragraphs": _paragraphs(values[key]), "subsections": []}
                for key, label in FIELDS if _paragraphs(values[key])]
    return {"title": title or "", "abstract": " ".join(_paragraphs(core_idea))[:1000],
            "sections": sections, "kind": "disclosure",
            "concept_seed": " ".join(_paragraphs(core_idea)),
            "dependent_hints": _paragraphs(optional_variants)}
