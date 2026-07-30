# Finding-level rubric annotation

Rate every item independently. Do not infer the generating method; method labels and
automated ratings are intentionally withheld.

Use `items_blinded.jsonl` for the research question, finding, earlier findings, and
the evidence available when the system produced the finding (cited tables and
passages, retrieval text, and/or SQL results). Record ratings in a separate copy of
`ratings_template.csv`.

## Scales

### Groundedness

- `yes`: The answer provides the factual information requested by the sub-question,
  and that information is supported by the supplied table, SQL-result, or passage
  evidence.
- `no`: The answer is unsupported, contradicted by the evidence, not tied to the
  available evidence, or only reports that information was not found.

### Relevance to the research question

- `none`: Unrelated or does not help answer the research question.
- `minimal`: Barely related; mostly off-topic for the research goals.
- `partial`: Tangentially related but misses main analytical goals.
- `substantial`: Mostly relevant with minor gaps.
- `full`: Directly addresses an important part of the research question.

### Distinctness from earlier findings

- `none`: Duplicate or near-verbatim rephrase of an earlier finding.
- `minimal`: Mostly repeats earlier findings with trivial wording changes.
- `partial`: Overlaps heavily with earlier findings but adds a small new detail.
- `substantial`: Mostly new angle or evidence with some overlap.
- `full`: Clearly new insight not covered by earlier findings.

### Report usefulness

- `none`: Not worth including; noise, redundant in a report, or only states that
  information was not found.
- `minimal`: Marginally informative; unlikely to help the reader.
- `partial`: Somewhat helpful but low priority for the report.
- `substantial`: Useful; addresses requested aspects or adds complementary insight.
- `full`: Highly useful; directly answers part of the query or adds valuable
  complementary insight.

Use the notes columns to briefly explain unclear, unsupported, or borderline cases.
Do not open `answer_key.json` until both annotators have finalized their ratings.
