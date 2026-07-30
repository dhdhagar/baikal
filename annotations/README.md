# Finding annotation interface

Open `annotations/index.html` directly in a browser. The generated
`finding_rubric/items_blinded.js` bundle allows the interface to work from a
`file://` URL without a local server.

Alternatively, from the repository root, start a static server:

```bash
python -m http.server 8000
```

Then open <http://localhost:8000/annotations/>.

The interface reads the generated browser bundle in
`annotations/finding_rubric/items_blinded.js` (with the JSONL file as a server-mode
fallback). Progress is saved automatically in the browser under the annotator ID,
so reopening the page and entering the same ID resumes the session. Because browser
storage is scoped to the page URL, resume using the same access method (`file://`
or localhost) throughout an annotation session.

Use **Export CSV** when annotation is complete. The exported columns exactly match
`annotations/finding_rubric/ratings_template.csv`:

```text
sample_id,annotator_id,grounded,relevance,distinctness,report_usefulness,grounded_notes,relevance_notes,distinctness_notes,report_usefulness_notes
```

The JSON export is a richer backup containing the same ratings plus export metadata
and completion counts. Do not expose `annotations/finding_rubric/answer_key.json`
to annotators until both annotation sessions are finalized.
