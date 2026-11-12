# Phase 10.5 — Evidence Resolver V2

## Confirmed Phase 10.4 failure

The sanitized Phase 10.4 diagnostic recorded `OverviewEvidenceInvalid` for
`segment_id=1`, a 19-character quote, and `quote_not_verbatim`. The validator's
legacy path required `_evidence_quote_matches()` to find the GPT-provided quote
in the cleaned segment after NFC and whitespace normalization. That match
failed. The saved Phase 10.4 JSON contains only the review status and reason;
it has no analysis items or raw response. The exact rejected wording and the
specific Thai spacing or punctuation difference therefore cannot be confirmed.

## Response and persisted evidence formats

New model responses use this item structure:

```json
{
  "text": "Summary claim",
  "owner": null,
  "deadline": null,
  "evidence": [
    {"segment_id": 1, "start_char": 12, "end_char": 27}
  ]
}
```

`segment_id` is the one-based position in the exact ordered segments supplied
to the model. `start_char` is inclusive and `end_char` exclusive; both are
Python string character indexes into that segment's cleaned `text`. They are
not UTF-8 byte offsets and are not calculated against NFC-normalized text.
Multiple spans, including spans in different segments, are supported.

The resolver reads only the referenced source slice. It rejects missing or
boolean IDs, non-integer/boolean offsets, negative or out-of-range values,
empty or whitespace-only spans, and references that contain a GPT-supplied
`quote`. It never searches for a replacement quote. Persisted V2 evidence
retains the segment and offsets, not a copied quote. The Phase 10 result also
records the source Phase 9 JSON path and SHA-256 so the offsets can be checked
against the unchanged cleaned transcript.

Historical quote-based records remain readable only through the legacy path:
the quote must be present in its cited cleaned segment after the existing
conservative NFC/whitespace normalization. Invalid historical evidence is
still rejected; the parser does not infer that a stored `validated` status
makes a citation valid.

## Deterministic checks and review behavior

- Claim numbers, recognized units, and alphanumeric equipment/code identifiers
  must be supported by the referenced slices. Units attached to a mentioned
  numeric value must match; dropping a source unit is rejected.
- Recognized date/time numerals and temporal terms (including Thai relative
  dates and weekdays) must also occur in the cited spans.
- Explicit `owner` and `deadline` fields must occur in the referenced source
  text. Owner/deadline wording in a claim without an explicit supported field
  is rejected when recognized by the validator.
- A conservative shared-word/Thai-character-gram check rejects clearly
  unrelated evidence. It is a heuristic, not a semantic entailment proof.

Valid V2 offsets establish where evidence came from; they cannot prove that a
paraphrased claim is entailed by that evidence. Accordingly, a structurally
and numerically valid V2 analysis is retained with `status=review_required`
and `review_reason=SemanticReviewRequired`. Phase 10 will not generate a
summary PDF from it. A human must compare each claim with its referenced
source spans before any separately implemented approval/publishing path can
treat it as a summary. Full semantic entailment, unmarked owner mentions, and
all possible date/deadline paraphrases cannot be determined reliably by these
offline checks.

## Offline verification

The resolver is covered with synthetic tests for Unicode and combining marks,
bad IDs/offsets, multiple spans, numeric/code/unit/date mismatches, unsupported
owners/deadlines, and legacy compatibility. The Phase 10 resume regression
uses a mocked HTTP session and asserts no transcription, cleanup, or PDF
writer call for a V2 response requiring semantic review. No Azure service is
contacted by these tests.
