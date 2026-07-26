"""Query validator: is the hit count usable? Practitioner rule of thumb —
a boolean query with zero hits is malformed/too narrow; one with tens of
thousands is a keyword dump that ranking will not save."""

CAP_BROAD = 20_000
FLOOR = 1


def validate(total: int | None, n_hits: int) -> str:
    if n_hits <= 0 and (total is None or total <= 0):
        return "zero"
    if total is not None and total > CAP_BROAD:
        return "too_broad"
    if n_hits < FLOOR:
        return "too_narrow"
    return "ok"
