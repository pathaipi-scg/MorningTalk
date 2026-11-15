# Progress: Sentence-Level Audio Review

## Phase 1 prototype — measured run 2026-11-15

- Read-only inspection completed for the Azure OpenAI transcription adapter, transcript dataclasses, cleanup flow, optional Azure Speech diarization, automatic scan integration, SQL persistence/schema, production batch workflow, configuration, output path mapping and focused tests.
- Confirmed fixed-duration chunking defaults to 180 seconds with 10-second overlap; both are configurable. FFmpeg cuts/converts audio and does not detect speech.
- Confirmed the OpenAI transcription adapter currently consumes transcript text only. It does not retain timestamps from that response. Existing chunk intervals are persisted, but do not indicate where individual words/sentences occur.
- Confirmed Azure Speech diarization can produce phrase-level timestamps and speaker labels; it is a separate Azure-dependent path, not VAD. The fallback OpenAI path in automatic scan wraps its transcript in one untimed segment.
- Confirmed overlap merge attempts suffix/prefix deduplication; if no sufficiently strong match is found, both texts remain.
- Confirmed the repository currently has no declared FastAPI/Flask/web UI dependency or route.
- Added isolated `app/vad_prototype.py` with offline WebRTC VAD framing, silence-region merging, pause-aware/maximum-duration segmentation, configurable playback padding, JSON metadata, and FFplay segment preview. The module is not imported by the production transcription pipeline.
- Added `tests/test_vad_prototype.py` for silence, short speech, long continuous speech, boundary regions, pause merging, contiguous core ranges, manifest shape, deterministic output and no-overwrite behavior.
- `webrtcvad-wheels` 2.0.14.post1 imports successfully using `D:\AI\MorningTalk\.venv\Scripts\python.exe`. The sibling path `D:\AI\MorningTalk.venv` is not present; all project Python commands used the repository's `.venv`.
- Confirmed the source recording exists at `D:\KM\Vault\MorningTalk\Record File\CB\Nov26\meeting-20261107-0817.mp3`. FFmpeg, ffprobe and ffplay are present. No source audio was changed.
- Focused offline tests pass using the project interpreter: `python -m pytest tests\test_vad_prototype.py -q` — **6 passed**.
- Generated manifest: `D:\AI\MorningTalk\workspace\vad_prototype\meeting-20261107-0817.vad.json`.
- **Measured results:** audio duration 1,131,540 ms (18:51.54); 235 detected speech regions; 244 review segments; average segment duration 4,265 ms; minimum 590 ms; maximum 14,570 ms; 66.82% of audio frames classified as speech; 0 segments longer than 15 seconds; 0 duration-only hard splits. Nine segments beyond one-per-region were split at pauses.
- **Timestamp validation:** all 244 segment ranges satisfy `0 <= start_ms < end_ms <= audio_duration_ms`; stored duration equals end minus start; maximum segment end equals the source duration. Core segment ranges remain contiguous within each VAD region. Transcript alignment remains `not_aligned` and all `transcript_text` values are null.
- **Gap and boundary review:** 234 inter-region gaps were recorded; the largest are 3,720, 3,450, 3,330, 3,000 and 2,460 ms. VAD activity touches both recording boundaries, so the first and last regions may contain clipped words and should be manually checked. The 590 ms minimum and 4,265 ms average show that many naturally short regions are below the 5-second target; the prototype intentionally retains those short regions rather than joining across longer silences.
- **Playback smoke test:** FFplay completed with exit code 0 for `seg-0001` (beginning), `seg-0123` (middle, with 1-second context and 0.75x rate), and `seg-0244` (end, with 1-second preceding context), all using the original MP3 and manifest timestamps.
- The previous PyPI timeout was resolved manually; no dependency was installed automatically during this run. No Azure, SQL Server, production pipeline, Vault output, or feature flag was used or changed.
- Updated `README.md`, `IMPLEMENTATION_PLAN.md`, and this status file. No database, Azure, feature flag, transcript, PDF, or Vault output was changed.

## Implemented command

```text
python -m app.vad_prototype analyze
python -m app.vad_prototype analyze <recording.mp3>
python -m app.vad_prototype preview <manifest.json> <segment_id> [--context-before SEC] [--context-after SEC] [--rate 0.75]
```

The analysis command writes a new manifest under `D:\AI\MorningTalk\workspace\vad_prototype`; it refuses to overwrite an existing manifest. FFplay is needed for audio preview.

## Next action

Phase 1 measurement is complete for the requested recording. Further recordings or boundary-quality evaluation require a separate approved review. No cloud transcription, SQL, pipeline integration, production feature flag, or UI is part of this phase.

## Not yet done

- VAD segments are acoustic activity regions, not sentences, speaker turns, or text/audio alignment. WebRTC VAD reports binary frame decisions and no confidence probabilities.
- Background machinery, short utterances, overlapping speakers and codec seek precision may cause false activations or missed speech. Boundary-touching regions and all gaps/segments should receive human listening review before employee use.
- Preview was verified as a successful FFplay process for beginning/middle/end; this does not establish transcription or boundary accuracy.
- No transcript alignment, employee identity, approval workflow, production integration, or reviewed exports are implemented.
- No SQL Server or Azure calls were made.

## Phase 2 — Local Audio Review Web UI — 2026-11-15

- Added `app/vad_review.py` and `app/vad_review.html`. The prototype reads manifests from `D:\AI\MorningTalk\workspace\vad_prototype` and displays Plant → Month → Recording. The existing CB/Nov26 recording loads all 244 VAD segments.
- Implemented segment ID/time/duration display, Play/Pause, Replay, 0.75x/1x/1.25x, up to 60 seconds of adjacent context, Previous/Next, and editable start/end millisecond boundaries. Short segments remain selectable and playable.
- Boundary edits are stored separately as `workspace\vad_prototype\<recording_id>.adjustments.json`; the VAD manifest and original timestamps remain unchanged. Reloading the catalog reapplies validated adjustment values.
- MP3 playback uses the original source through an HTTP endpoint supporting single byte ranges, with bounded streaming reads. No preview audio copies are created. Manifest audio paths are restricted to the configured Vault recording root; route identifiers are allow-listed and traversal attempts fail. No general file/static-root route exists.
- The server binds to `127.0.0.1` by default, rejects non-loopback bind addresses, and only accepts loopback Host headers. This is a local prototype, not an authenticated or production-hardened service.
- FastAPI and Uvicorn are absent from the project environment and dependency manifest. To avoid a new framework/dependency solely for this isolated prototype, the backend uses Python's standard-library `ThreadingHTTPServer`; the client is plain HTML/CSS/JavaScript.
- Validation/tests using `D:\AI\MorningTalk\.venv\Scripts\python.exe`: `pytest tests\test_vad_review.py -q` — **14 passed** (including audio range streaming, path traversal, timestamp validation, short-segment playback bounds, adjustment sidecar, invalid IDs, and recording navigation).
- Live loopback smoke test: page and recording APIs returned HTTP 200; browser catalog exposed CB / Nov26 / `meeting-20261107-0817.mp3` with 244 segments; a request for bytes 0–31 returned HTTP 206 and exactly 32 MP3 bytes. The server was stopped after verification.
- No Azure or SQL calls; no production pipeline, feature flag, Vault file, or source MP3 was changed. No Git commit or push was made.

### Phase 2 limitations and next approval gate

- The UI displays “Not aligned” and no transcript text. It has no correction, verification, submission, approval, authentication, user attribution, or multi-user conflict workflow; those belong to Phase 3 and require approval.
- Range streaming provides efficient browser seeks, but MP3 codec/frame seek precision and perceived segment boundaries should be reviewed by employees on representative machines.
- The local server intentionally supports only the configured manifest directory and recording hierarchy. It is not suitable for LAN or internet exposure.
- Phase 2 is complete. Stop here and await approval before Phase 3.

## Phase 2 segmentation evaluation — short-segment optimization — 2026-11-15

- Added the separate deterministic post-processor `app/vad_segment_optimizer.py`. It reads the Phase 1 manifest and writes `D:\AI\MorningTalk\workspace\vad_prototype\meeting-20261107-0817.vad.optimized-review.json` using exclusive creation; it does not rewrite the VAD manifest.
- The merge rule applies to current review intervals shorter than 1,000 ms. It considers the immediately previous and next groups, measures gaps using VAD core intervals when present (otherwise input intervals), accepts gaps up to 1,000 ms, chooses the smaller eligible gap (previous wins ties), and rejects any combined playback interval over 15,000 ms.
- A merged group spans the full union of its source review intervals, preserving intervening audio. It carries all source segment IDs and each source segment's original start/end/duration. Any unmerged sub-1-second segment is retained and marked `needs_review`; no short speech is discarded.
- **Before:** 244 segments; 26 shorter than 1 second; 81 shorter than 2 seconds; duration min/average/max 590 / 4,265 / 14,570 ms.
- **After:** 219 segments; 0 shorter than 1 second; 51 shorter than 2 seconds; duration min/average/max 1,010 / 4,768 / 14,570 ms; 24 merged groups; no unmerged sub-1-second segments. The 24 groups comprise 23 pairs and one group of three, hence the total segment reduction of 25.
- Every one of the 244 original segment IDs appears exactly once in the derived manifest, with original boundaries verified. The maximum recorded joined core gap is 960 ms. All output ranges are valid and within the original 1,131,540 ms audio duration.
- No `*.adjustments.json` sidecar existed for this recording at run time. The optimizer did not apply or remap adjustments and records `manual_adjustments.status: "not_remapped"`; its compatibility test verifies an existing sidecar remains byte-for-byte unchanged.
- The optimized file name does not match the review catalog's `*.vad.json` pattern. The running/default Phase 2 review catalog still lists the original 244-segment manifest; switching the UI to optimized segments remains gated on inspection and approval.
- Focused regression tests using the project `.venv`: `pytest tests\test_vad_prototype.py tests\test_vad_review.py tests\test_vad_segment_optimizer.py -q` — **29 passed**.
- No Azure or SQL calls; no production pipeline, Vault files, source MP3, or original manifest were modified. No commit or push was made.

### Remaining review

- The 1,000 ms core-gap threshold is a prototype heuristic, not a validated speech/linguistic boundary. The merged output should be listened to and approved before changing the UI's default manifest.
- Merging can produce a longer playback range that contains a silence gap; this is intentional to preserve the entire source interval. Transcript alignment remains absent, and no UI-default change or automatic adjustment remapping has been made.

## Original / Optimized comparison mode — 2026-11-15

- Added the compact segmentation selector to the review UI. Original VAD remains selected by default; the actual catalog counts are 244 original segments and 219 optimized segments.
- The catalog independently loads the original `.vad.json` and derived `.optimized-review.json` manifests under separate `(mode, recording_id)` keys. Optimized entries are validated for source lineage, merge status, and source ID consistency.
- Playback, context bounds, recording details, and audio requests carry the selected mode. Both modes stream the same original MP3; the optimized timestamps are used for optimized review playback. The UI identifies merged and unchanged optimized groups and displays source IDs for merged groups.
- Timing edits are isolated in `<recording_id>.adjustments.json` for Original and `<recording_id>.optimized.adjustments.json` for Optimized. Existing legacy Original sidecars remain readable; optimized adjustments are not copied or remapped from Original.
- Automated regression suite: `D:\AI\MorningTalk\.venv\Scripts\python.exe -m pytest tests\test_vad_review.py tests\test_vad_prototype.py tests\test_vad_segment_optimizer.py -q` — **33 passed** (including mode-specific playback, lineage validation, and sidecar isolation).
- Embedded UI JavaScript syntax validation with `node --check` passed.
- Loopback browser verification: Original default displayed 244 segments; switching to Optimized displayed 219. The first optimized row showed `review-0001`, timestamps 0:00.00–0:04.96, the merged badge, and source IDs `seg-0001, seg-0002`. Browser playback through the original MP3 endpoint was active in Optimized mode and advanced from 0 seconds. Switching back restored the original list and its `seg-0001` timestamps.
- The existing manifests and source MP3 were not modified; no timing changes were saved during browser verification. No Azure, SQL Server, production pipeline, Vault writes, feature flags, commit, or push.

### Comparison-mode limitations

- There is no transcript alignment or text correction. Optimized boundaries remain VAD-derived; the selector is for side-by-side listening and review, not a claim that merged groups are sentences.
- Playback was smoke-tested in the local browser for the optimized sample segment. Mode-specific playback bounds are also covered by automated tests; broader employee/device compatibility remains unverified.
- Mode-specific adjustment sidecars are local prototype data with no approval, identity, synchronization, or multi-user conflict workflow.

## Phase 3 — transcript alignment feasibility assessment — 2026-11-15

- Read the Phase 3 baseline documentation and inspected the Azure OpenAI adapter, optional Azure Speech diarizer, transcript data model, scan/production serialization paths, current local phase9 artifacts, and the target recording's transcript JSON/TXT/PDF. This was read-only analysis; no Azure API, SQL Server, or file-generation operation was performed.
- `AzureOpenAITranscriber` sends audio chunks and reads only the response `text`; it does not request or retain word/phrase timestamps. `ChunkProgress` and database chunk rows refer to request audio ranges and whole chunk text, not the location of speech words. Overlap merging outputs a text string.
- `AzureSpeechDiarizer` can parse Azure Speech phrase entries with `offsetMilliseconds`, `durationMilliseconds`, text, and optional speaker. That is phrase-level timing available only when this separate Azure path succeeds; it is not word-level alignment and was not present in the target recording's persisted transcript.
- Target artifact: the Phase 10 curated record links to `workspace\previews\live-test-20261107-resume-v3\meeting-20261107-0817.phase9.json`. The referenced JSON's SHA-256 and source fingerprint match the curated record. It contains one transcript segment, with `start_seconds` and `end_seconds` both null; `diarization_status` is `unavailable`. Raw text length is 8,267 characters and cleaned text length is 8,358 characters. That one untimed segment is not the 219 optimized VAD segments.
- The target Vault TXT is 8,358 characters across nine lines and has no timestamp tokens. The full-transcript PDF has three pages. Extracted PDF text contains one time-like token, `08:17` (matching the recording identifier), and no timing labels; it is not segment timing.
- A structural scan of 25 existing `*.phase9.json` workspace artifacts found 25 segment objects, all without non-null start/end times. Some payloads say `diarization_status: "available"`; that label alone does not establish that timings were persisted. An existing production-batch raw artifact stores a plain raw transcript string and source/config metadata, not word/phrase timings.
- **Conclusion:** there is no verified timestamp source to map this target transcript onto Original or Optimized VAD segments. Do not spread text proportionally or label any such mapping accurate. Keep transcript status unaligned unless a timing-bearing source or validated alignment is provided.
- **Proposed next phase:** after assessment approval, run a separate offline forced-alignment feasibility experiment using the exact transcript and local audio; first verify Thai model coverage, licensing, runtime, and quality against manually listened references. Dependency installation or model-weight acquisition requires separate approval. Keep uncertain results unaligned and provenance-bearing. Correction UI and isolated correction sidecars are a subsequent separately approved step.
- Updated `README.md` and `IMPLEMENTATION_PLAN.md` with evidence, limitations, proposed data-safety rules, implementation stages, and acceptance checks. No application/UI logic or correction storage was implemented. No source MP3, TXT, PDF, VAD manifest, production pipeline, or Vault file was modified; no Git commit or push was made.

### Phase 3 approval gate

This deliverable is an alignment feasibility assessment and implementation proposal only. Wait for user review and approval before implementing an alignment prototype, acquiring dependencies/model weights, adding correction persistence, or changing the review UI.

## Phase 3B — adaptive VAD chunking dry run — 2026-11-15

- Added isolated `app/vad_transcription_chunks.py` and focused tests in `tests/test_vad_transcription_chunks.py`. The planner reads the existing 219-segment optimized VAD manifest, groups in chronological order, targets 30–45 seconds, prefers wider natural pauses, includes intervening audio, and never splits an optimized segment.
- To prevent duplicated speech/audio at chunk boundaries, a boundary is accepted only when adjacent complete playback windows do not overlap. If that constraint prevents a 30-second group, the planner emits a shorter chunk rather than introducing overlap. Individually oversized source segments would remain unsplit and be flagged.
- Generated `workspace/vad_prototype/meeting-20261107-0817.azure-chunk-plan.json` with exclusive creation. It records original recording-relative chunk intervals, all optimized segment IDs, original VAD source IDs and timestamps, a separate null transcript field for each chunk, and configured deployment `gpt-4o-transcribe` / API version `2025-03-01-preview`. No keys or endpoints are written.
- **Measured chunk count:** 31. Duration minimum / average / maximum: **10.660 / 35.603 / 43.610 seconds**. The three below-target chunks are `chunk-0013` (27.320 s), `chunk-0024` (13.550 s), and the recording-end `chunk-0031` (10.660 s). No chunk exceeds 45 seconds.
- **Coverage:** 219 optimized segments carry 244 original VAD source IDs; all 244 source IDs appear exactly once. Detected speech-core coverage is **935,160 / 935,160 ms (100%)**. The union of chunk audio ranges covers **1,103,700 ms (97.5396%)** of the 1,131,540 ms recording.
- **Gaps/boundaries:** 30 inter-chunk gaps total **27,840 ms**, minimum **10 ms**, average **928 ms**, maximum **2,950 ms**. There are **zero overlapping chunk windows** and zero duplicated or unassigned source IDs. All chunk timestamps are valid and within the recording duration.
- Azure response status is `not_requested`; transcript fields are null and metadata explicitly says `timestamp_granularity: "chunk_only"`. No Azure request was made. Existing transcripts, source MP3, optimized/original VAD manifests, SQL Server and production chunking were not changed.
- Added an explicit `export-audio` command that uses existing FFmpeg to create MP3 clips only beneath `workspace/vad_prototype`, with path validation and no-overwrite behavior. Audio clips were **not** generated during this dry-run. No dependency was installed.
- Focused regression tests using `D:\AI\MorningTalk\.venv\Scripts\python.exe`: `pytest tests\test_vad_prototype.py tests\test_vad_review.py tests\test_vad_segment_optimizer.py tests\test_vad_transcription_chunks.py -q` — **41 passed**.
- The chunk manifest does not align its chunk-level transcript strings to the previously saved Azure transcript, nor does it provide sentence/word timestamps. Any later Azure request requires separate approval; do not alter the production 180-second/10-second-overlap chunking behavior.

## Phase 3C — one Azure transcription test — 2026-11-15

- With explicit approval, selected `chunk-0026` from the existing plan: **929,930–967,210 ms** (37.280 s), with 36,090 ms VAD speech core (**96.81%** of chunk duration).
- Exported only this chunk from the original MP3 using FFmpeg to `workspace/vad_prototype/meeting-20261107-0817.chunk-0026.mp3`. ffprobe reported 37.280 s. The source MP3 and VAD manifests were read-only.
- Reused `AzureOpenAITranscriber._transcribe_chunk` with the existing configured `gpt-4o-transcribe` deployment and `2025-03-01-preview` API version. A one-POST guard disabled redirects and HTTP retries.
- Exactly **one** Azure request completed: HTTP 200, request timestamp `2026-11-15T08:35:40.083621+00:00`, elapsed 5,392 ms. Response request IDs and content type are preserved in the per-chunk result JSON; no key, endpoint, or secret is stored.
- Saved the UTF-8 result separately at `workspace/vad_prototype/meeting-20261107-0817.chunk-0026.transcription.json`. It records recording/chunk IDs, exact chunk boundaries, clip reference, Azure metadata, request status/timestamps, response metadata, and the 66-character Thai transcript.
- The result is a **chunk-level transcription only**. It provides no word-level or sentence-level timestamps and is not an alignment of the pre-existing transcript.
- The initial console print encountered a Windows cp1252 encoding error after the successful response had been saved; the saved JSON was then read as UTF-8 and validated. No retry or second Azure request was made.
- The focused local suite remains **41 passed**; the saved result was independently checked for valid JSON, one-request count, HTTP 200, chunk identity/timestamps and non-empty Thai text.
- No other chunks were processed. No SQL, production chunking, original transcript, Vault file, source audio, VAD manifest, commit, or push was changed.
