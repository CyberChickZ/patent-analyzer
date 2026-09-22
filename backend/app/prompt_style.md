# Prompt style

2026-09-20. Harry on the pipeline prompts: "之前的太难看了".

This is the house style for every template in the prompt registry
(`app/prompts.py`, `register_default(...)`). It is about **shape**, not about
what a prompt asks for — a rewrite that changes what the model is told to do is
not a rewrite, it is a change, and it needs its own evidence.

---

## 0. The one thing to read if you read nothing else

**A rule is a behaviour and a reason, in one sentence.** Everything below is
downstream of that.

```
Bad   - Be specific.
Bad   - IMPORTANT: Always use **precise** terminology!!!
Good  - Use the words patents actually use, because a term nobody drafting a
        patent wrote matches nothing.
```

---

## 1. Shape

- One level of headings, `# Like This`. No `##`, no numbered outlines.
- List items start with ` - ` — a space, a hyphen, a space. Never `*`, `1.`, or `•`.
- Paragraphs are separated by a blank line. No line is longer than 100 characters.
- No emoji. No `**bold**` for emphasis; bold is for a term being defined, and
  most prompts need none at all.
- No ALL-CAPS shouting (`IMPORTANT`, `YOU MUST`). If a rule needs shouting, it
  is not written clearly enough.

## 2. Wording

- Negative constraints: **never**, **do not**, **don't**. Positive defaults:
  **prefer**, **default to**.
- Say the reason. A rule with no reason gets followed literally in the case it
  was not written for, which is the case that matters.
- One rule per line. Two rules in one sentence means the model can satisfy the
  first and drop the second.
- Name things verbatim: field names, JSON keys, file names, thresholds,
  CPC groups, MPEP sections. Never "the relevant field" or "a reasonable
  number" — write `learned_terms`, write `total < 200`.
- Examples are real. A made-up example teaches a made-up vocabulary; the ReAct
  prompt's counterexamples (`kinetic conferencing proxy`) are real failures
  that really happened.

## 3. Static before dynamic

Every prompt splits in two, and the static half comes first:

```
# Role            \
# Task             |  static: identical for every job, byte for byte
# Rules            |
# Output           /
                      <- blank line
# Document         \
# Elements          |  dynamic: this job's text
# Results so far   /
```

This is not only tidiness. Vertex and Anthropic both cache on a **prefix**
match, so a template whose first N thousand tokens never change is a template
whose first N thousand tokens can be billed at the cached rate. Interleaving
one `{field}` into the rules section forfeits everything after it.

Rules for the split:

- The static half contains no `{placeholder}` at all.
- Each dynamic section gets its own `# Heading`, so the model can tell the
  job's data from the instructions. Never paste a document straight after a
  rule with nothing between them.
- The output schema belongs in the static half, at its end: it is the last
  thing the model should have read before it starts reading the job.

## 4. Output schema

- Give the schema as literal JSON with the braces doubled (`{{`), because these
  are `str.format_map` templates.
- Every key gets a length or type bound in the same line: `"observation":
  "<=40 words on what the last results showed"`.
- Never ask for prose and JSON in the same response.

## 5. The contract

Each registered prompt carries a `contract` next to it — plain English, four
things, for the person editing it and for the revise assistant:

- **Does** — what this prompt is for, in one sentence.
- **Must output** — the exact fields or format the caller parses.
- **Consumed by** — the function or node that reads the output, by name.
- **Never** — the constraints that cannot be relaxed, with the reason.

The contract is the thing a rewrite must not break. It is also what the
`POST /prompts/{name}/revise` endpoint sends to the model along with this file.

---

## 6. Worked example

Excerpt from `search.react_step`, before:

```
Decide the next query. Rules of thumb:
- `total` is a diagnostic, NOT a target. Do not reshape a query to move the number — a phrase
  invented to widen or narrow the count is a phrase that matches the wrong documents.
- total < 200 → far too narrow: drop an item, or use a synonym / stemmed form; never repeat a query.
- Titles all off-topic at a large total → the neighbourhood group is wrong, not too wide.
- READ the returned titles: when they use patent vocabulary for what the paper calls something
  else (e.g. the paper says "kinetic proxy", patents say "teleconferencing robot",
  "swiveling monitor"), put those words in `learned_terms` and use them next.
```

After:

```
# Choosing the next query

 - Read `total` as a diagnostic, never as a target: a phrase invented to move
   the count is a phrase that matches the wrong documents.
 - When `total` is under 200 the query is too narrow, so drop one item from
   `specific` or use a stemmed form. Never repeat a query already in HISTORY.
 - When `total` is large and every returned title is off-topic, the `broad`
   group names the wrong field. Replace it; do not widen it.
 - Read the returned titles for the words patents use where the paper uses its
   own, and put those words in `learned_terms` so the next query can use them.
   The paper's "kinetic proxy" is "teleconferencing robot" in a patent title.
```

What changed: `- ` became ` - `; the arrow shorthand became a sentence; "NOT a
target" lost its capitals and kept its reason; each line names a field
(`total`, `specific`, `broad`, `learned_terms`, `HISTORY`) instead of
describing it. **Nothing was added to or removed from what the model is asked
to do.**

---

## 7. What a rewrite does not prove

The output schema surviving a rewrite proves the shape is unchanged. It does
not prove the behaviour is unchanged.

This is not hypothetical. Commit `4bb191a` fixed a ReAct prompt edit whose
schema was untouched and whose tests were all green: the model started
inventing compound phrases and first-reach went to zero (H.md 勘误五). The
prompt's function is what the model does, and a unit test cannot see it.

So every batch of rewrites is accepted the same way: **one local end-to-end
job**, comparing element count, checklist size, the vocabulary of the emitted
queries, and the determination label against the run before it. Report it as
"schema unchanged; behaviour accepted on one e2e", never as "unchanged".
