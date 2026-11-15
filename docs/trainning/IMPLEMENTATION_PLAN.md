# Implementation Plan: Sentence-Level Audio Review

## Status and verified baseline

Phase 1's isolated VAD analyzer and Phase 2's local-only review browser are implemented without changing the production pipeline. Database integration, employee authentication/approval, transcript alignment, Azure behavior, and reviewed exports remain future work.

Verified details:

1. Chunking defaults to 180 seconds with 10 seconds overlap (`app/config.py`); configurable values drive FFmpeg seeks/cuts in `app/transcriber.py`.
2. The Azure OpenAI transcription endpoint returns a parsed `text` field only in current code. Chunk metadata has chunk-level start/end values, not the actual speech time of words or phrases.
3. Text overlap deduplication uses a normalized significant-character suffix/prefix match. If it cannot confidently remove matching text, it retains both chunk texts.
4. No VAD is present in the inspected implementation. `ffmpeg`/`ffprobe` are used for audio manipulation and duration; neither supplies speech segmentation.
5. The optional Azure Speech diarization adapter does return phrase-level start/end times and labels. It still requires Azure and does not provide the requested offline-first VAD option.
6. Existing `TranscriptSegment` already represents optional timestamps and raw/cleaned content. SQL `TranscriptionChunk` records request chunk intervals and text; `AudioFile.TranscriptSegmentsJson` stores segments. These are reusable for source lineage and migration compatibility, but are not enough to drive sentence-level playback on OpenAI fallback transcripts.
7. There is no existing HTTP/UI framework in `app`/requirements. The Phase 2 prototype uses the Python standard-library threaded server, bound to loopback, and a dependency-free HTML/JavaScript client. This avoids adding a framework solely for a local prototype; choose a supported/authenticated hosting architecture before employee deployment.

## Proposed architecture

1. **Read-only audio analysis:** FFmpeg/ffprobe decode the original MP3 to a temporary PCM stream/file. A local VAD emits speech intervals in original-file seconds. Never rewrite the source MP3.
2. **Candidate segmentation and alignment:** split VAD intervals into playable regions, targeting 5–15 seconds with flexible longer turns. Associate transcript text only when source timing or a human-approved alignment supports it. For existing untimed whole-record transcripts, mark candidate text as `unassigned`/`alignment_required`; don't claim word alignment from VAD alone.
3. **Review service (Phase 2 prototype):** `app/vad_review.py` loads only manifests from `workspace/vad_prototype`, validates them and limits MP3 access to paths beneath the configured Vault recording root. It provides loopback-only HTTP endpoints, single-byte-range streaming, validated playback windows, recording navigation metadata, and a sidecar adjustment store. Adjustments preserve original timestamps and never modify the manifest.
4. **Review client (Phase 2 prototype):** `app/vad_review.html` provides Plant/Month/Recording selectors, a scrollable 244-segment list, playback controls, context, previous/next navigation, and manual boundary adjustment. Transcript state is always “Not aligned”; it invents no text. Verification, approval, transcript correction and publishing are not implemented.
5. **Versioned persistence and export:** preserve immutable raw transcript/source hash; append review revisions and approval events. Render reviewed TXT/PDF into a separate review/versioned location, never overwrite original or standard Vault outputs. Approval must not mutate the source recording or raw transcript.

### Offline-first timing/alignment strategy

- Prototype local VAD (candidate: Silero VAD) on decoded audio. Evaluate thresholds, padding, minimum speech/silence duration and Thai factory noise before selecting defaults.
- Merge adjacent detected speech spans when the gap is short; split long continuous spans at conservative pauses where possible, retaining adjustable boundaries and long-segment support.
- VAD detects speech activity, not words or sentence content. Without word timestamps, automatic transcript-to-audio matching is uncertain. Use optional local alignment only if a suitable offline model is separately evaluated; do not make faster-whisper a required dependency for the first prototype.
- Failure state: keep audio regions and transcript text separately addressable; permit manual association, text correction and timing adjustment. Surface `alignment_status` and confidence. Never force generated/incorrect transcript into plausible timed segments.
- Keep all times as seconds relative to the original MP3; chunk offset + local offset can be used if any later processing still occurs on chunks.

## Proposed data model

Prefer a separate review schema/sidecar for the prototype, then decide SQL Server migration only after user workflow validation.

- **Recording reference:** stable recording ID/source path, Plant, Month, filename, duration, source fingerprint (and optionally SHA-256), raw transcript reference/version. Source path and content remain read-only.
- **VAD run:** recording ID, algorithm/model/version, configuration/thresholds, created time, status, source fingerprint. Supports reproducibility and reruns without replacing human edits.
- **Review segment:** stable segment ID, recording ID, ordering, start/end seconds, VAD source interval(s), raw text snapshot, current proposed/corrected text, alignment status/confidence, verified state/user/time, revision number. Validate start >= 0, end > start, chronological ordering and duration <= recording duration; permit overlapping ranges only when explicitly justified.
- **Review revision/event:** segment ID, previous/new text and boundaries, actor, time, reason, event type (`edited`, `verified`, `submitted`, `approved`, `rejected`), reviewer/approver and comments. Append-only audit history; corrections must not replace raw source text.
- **Export record:** review version, approval state, generated TXT/PDF paths, hashes, created time and source/review revision identifiers. Unique versioned name/path; no write to standard output names.

Existing chunk rows may be linked as provenance (`chunk_index`, `start_seconds`, `end_seconds`) but must not be reinterpreted as speech segments. Existing `TranscriptSegment` can support a serialized first implementation; add distinct alignment/review status and immutable correction/audit fields rather than overloading `review_required`.

## UI workflow

1. Browse Plant → Month → Recording, using the existing recording hierarchy and metadata.
2. Open a recording and see timeline/segment list, transcript source/status, VAD/alignment state and review progress.
3. Select a segment; play only its original-MP3 range, replay, slow down, or include preceding/following context. Adjust start/end and re-preview.
4. Edit transcript text while retaining a visible raw transcript/source reference. Mark verified only after listening; uncertainty/unassigned transcript remains clearly indicated.
5. Submit the recording's correction set for approval. Approver accepts/rejects with audit events; submitter cannot silently publish.
6. On approval, generate a new reviewed TXT/PDF version in the review output area, record its lineage and leave all existing artifacts untouched.

## Alternatives and recommendation

| Option | Benefits | Tradeoffs |
|---|---|---|
| Local VAD + manual alignment/correction | No Azure use; modest first scope; human handles poor transcripts | VAD does not align words; labor to associate text for existing untimed transcripts |
| Azure Speech phrase timestamps | Existing adapter and `TranscriptSegment` timestamps/speakers; phrase-level structure immediately | Azure call/network/config required; not offline-first and may not be same transcript as stored OpenAI output |
| Faster-Whisper transcription + timestamps | Offline transcript and alignment can be co-produced; placeholder class signals future intent | Model/runtime/download/CPU/GPU footprint; new transcription may differ and adds setup/testing complexity |
| Forced alignment against existing transcript | Potentially word/phrase timings while preserving text | Requires model/language quality assessment; severely incorrect transcript can produce misleading forced alignment |

**Recommendation:** begin with local VAD plus manual correction/timing adjustment and explicit alignment confidence. For review of the existing transcript without new Azure requests, unassigned text should be permitted; do not claim reliable sentence-level text/audio mapping until an offline aligner is validated. Compare optional offline alignment on a small, approved local fixture set in a later phase.

## Dependencies and exact likely files

Likely application files after approval (not changed in this investigation):

- `app/main.py` — unchanged; Phase 2 has a separate local CLI entry point in `app/vad_review.py`.
- `app/vad_review.py`, `app/vad_review.html` — isolated local Phase 2 review server and interface.
- `tests/test_vad_review.py` — review API, path validation, byte ranges, playback bounds and adjustments.
- `app/transcript.py` — extend validated segment/review data model or add separate review models.
- `app/file_manager.py` / `app/workspace_paths.py` — resolve read-only recordings and separate versioned review output locations.
- New `app/audio_review.py` (or equivalent) — VAD, interval merging, clip-range validation and audio-range serving support.
- New `app/review_repository.py` or `app/database.py` — persistence/API, chosen after deciding sidecar vs SQL.
- `app/pdf_export.py` and/or `app/meeting_outputs.py` — render reviewed exports only to new version paths.
- `requirements.txt` — only if the selected local VAD/UI dependency is adopted; pin/test versions and assess model/license/runtime footprint.
- `migrations/` — only after SQL Server persistence is approved; no DB change is part of the prototype.
- Focused additions to `tests/` for interval validation, timestamp offsets, alignment failure, immutable originals, revision/audit behavior, and collision-proof export paths.

Current code references: `app/transcriber.py`, `app/azure_speech.py`, `app/transcript.py`, `app/auto_scan.py`, `app/database.py`, `migrations/001_initial_schema.sql`, `migrations/002_phase9_transcripts.sql`, `app/production_batch.py`, `app/file_manager.py`, `app/config.py`, `requirements.txt`.

Expected first-phase dependencies: FFmpeg/ffprobe are already required for current transcription. A local VAD package/model is new and must be tested offline. A browser playback UI may need a web framework and authentication; none is declared in the current requirements. Avoid adding both VAD and offline transcription engines before the prototype establishes need.

## Risks and rollback

- **False speech boundaries / noise:** compare VAD intervals with manual sample labels; expose thresholds and editable boundaries; never auto-verify.
- **Misleading text alignment:** keep text-to-audio mapping confidence/status and an explicit manual-association path.
- **MP3 seek accuracy:** validate range playback around boundaries against source; decode/seek via FFmpeg and account for codec frame/seek precision. Do not create permanent edited source clips.
- **Privacy/access:** retain recordings locally/in existing controlled storage; determine employee authentication, authorization and audit retention before deployment.
- **Concurrent edits / lost corrections:** optimistic revision checks and append-only events.
- **Output collision/data loss:** write versioned review artifacts to a separate root, verify nonexistence/unique version before atomic publish; tests must prove existing files remain byte-identical.
- **Dependency/model footprint:** test install, offline availability, CPU performance, supported Windows environment and licensing before selecting VAD.
- **Rollback:** disable/remove the new review route/feature and leave all raw records, original files and standard outputs untouched. Keep review data in its own schema/sidecar with explicit versioning; do not change current pipeline behavior. A SQL migration, if later needed, must be additive and independently reviewed.

## Acceptance criteria

- Browse recordings by Plant/Month/Recording without writing to source folders.
- Each displayed range has valid original-audio start/end timestamps and playback targets that exact range; replay, slow speed and adjacent context work.
- Typical candidate ranges are 5–15 seconds where pauses permit; long speech remains playable and boundaries can be adjusted.
- VAD output is deterministic for fixed model/config/source; source fingerprint mismatch invalidates stale analysis.
- Poor/no transcript alignment is visibly flagged and can be manually repaired; no false confidence or automatic verified state.
- Raw transcript and original audio are preserved byte-for-byte; correction history records actor, time and previous values.
- Verification, submission, approval and rejection are distinct states with an audit trail.
- Reviewed TXT/PDF have unique new version paths; integration tests prove existing Vault TXT/PDF/audio are not overwritten.
- Initial prototype runs without Azure requests and without SQL Server mutation.
- Existing transcription tests and behavior remain unchanged; no production feature flag is enabled.

## Phases and first implementation phase

1. **Offline feasibility prototype (implemented and measured):** `app/vad_prototype.py` uses WebRTC VAD on FFmpeg-decoded mono 16 kHz PCM and writes a non-overwriting JSON manifest under `D:\AI\MorningTalk\workspace\vad_prototype`. It records frame activity, region/split metadata, padding, duration statistics, `not_aligned` status, and null transcript text. A CLI previews a range from the original MP3 with optional context and playback-rate adjustment. The requested recording produced 235 speech regions and 244 review segments; six focused tests pass using deterministic fake frame decisions. Full measurements and limitations are in `docs/trainning/PROGRESS.md`.
2. **Local review browser (implemented):** add safe manifest loading, loopback HTTP Range streaming, Plant/Month/Recording navigation, segment controls and workspace-only timestamp adjustments. Alignment and text editing are deliberately excluded.
3. **Employee UI and approval:** add browsing/editing/review queue, identity/authorization, audit trail and concurrency checks.
4. **Reviewed export:** generate versioned TXT/PDF in separate review location, approval-gated and collision-safe.
5. **Optional alignment / production integration:** evaluate offline alignment; only then propose additive persistence migration and controlled integration, preserving the existing pipeline as-is by default.

**Approval gate:** Phase 1 and Phase 2 local prototypes are complete. Phase 3 (employee identity, transcript correction, verification and approval), production behavior, and any UI deployment beyond loopback each require approval before work begins.

The standard-library server is a deliberate prototype-only choice because FastAPI/Uvicorn are not project dependencies. Production deployment requires a separate hosting and authentication decision.

## Phase 3 — transcript alignment feasibility assessment (pending approval)

### Evidence inspected

- The Azure OpenAI adapter in `app/transcriber.py` posts chunk audio and parses only the response's `text` property. It does not request a timestamped response format or preserve word/phrase timestamps. `ChunkProgress` and database chunk rows record each request's audio start/end and full chunk text; these are request ranges, not speech/word locations. The final overlap-merged transcript is returned as plain text.
- The optional `AzureSpeechDiarizer` in `app/azure_speech.py` parses a `phrases` array containing `text`, `offsetMilliseconds`, `durationMilliseconds`, and optional speaker. This yields phrase-level, not word-level, timings when that path is successfully used. It is an Azure-dependent alternate provider, not evidence that the OpenAI text transcript has timing.
- The target recording's linked Phase 9 artifact is `workspace/previews/live-test-20261107-resume-v3/meeting-20261107-0817.phase9.json`. It is linked from the existing Phase 10 curated record, and its SHA-256 and source fingerprint match that record. Its `raw_transcript` has 8,267 characters, its `cleaned_transcript` has 8,358, and its `segments` array has one object with both `start_seconds` and `end_seconds` null. `diarization_status` is `unavailable`; its one transcript segment is not one of the 219 optimized VAD segments.
- The target Vault TXT contains 8,358 characters in nine lines and no timestamp tokens. The full-transcript PDF has three pages; extracted text contains one time-like token (`08:17`, matching the recording identifier) and no timestamp/timecode labels. This is not segment timing. These are rendered text outputs, not alignment sources.
- A read-only structural scan of 25 existing `*.phase9.json` artifacts under workspace found 25 segment objects and zero with non-null start/end times. Some carry `diarization_status: "available"`, but that status alone is not timing evidence; the saved timing fields are null. A representative production-batch `*.raw.json` stores a single `raw_transcript` string plus source/config fingerprints and status, with no word or phrase timing array.
- The target recording's Phase 1/optimized manifests contain VAD intervals and source segment lineage only. Their `not_aligned` status and null transcript fields remain correct.

### Feasibility conclusion and limitations

The existing target transcript cannot be reliably mapped to the 219 optimized VAD segments from saved data. Chunk boundaries do not identify which words were spoken where; VAD identifies speech-like activity, not transcript text; and distributing characters or sentences proportionally would create false timing confidence. The PDF/TXT layout does not add timestamps.

An offline forced-alignment experiment is a possible separate avenue because it uses the existing transcript as input rather than requesting a new transcript. Its Thai acoustic-model availability, compatibility, license, model download/size requirements, runtime, and performance on this recording have not been verified. Forced alignment can assign plausible times to incorrect or missing words, so it must retain explicit provenance and an uncertainty/unaligned outcome; it must not be presented as transcript accuracy. If no approved local model is available, human-assisted manual text-to-audio association is the safe fallback.

### Proposed implementation after approval

1. **Phase 3A — read-only alignment feasibility prototype:** Load a transcript artifact only after validating its source filename/fingerprint and content hash. Preserve the raw transcript unchanged. Evaluate a candidate offline aligner on a small, locally approved sample with manually listened boundary references. Write only a separate workspace candidate artifact containing source hashes, algorithm/model/version/configuration, text-span or token offsets, original-audio start/end times, and a calibrated/qualified confidence or `uncertain`/`unaligned` status. Never call Azure, SQL Server, or distribute text proportionally. Obtain separate approval before installing dependencies or downloading model weights.
2. **Phase 3B — correction UI and local sidecar (separately approved):** Keep the Azure transcript read-only and visible alongside each segment only when the alignment record supports that association; otherwise show a clear unaligned state and let the employee listen without fabricated per-segment text. Keep a separate editable Corrected Transcript field and Save Correction action. Store corrections separately from all transcript/VAD sources, with source transcript hash, mode, segment ID, source segment IDs, current timestamps, correction text, edit time, and review status.
3. Keep correction identity scoped by `(recording_id, mode, segment_id)`. Preserve Original/Optimized lineage, but never copy a correction from an original segment to an optimized merged group automatically. Any reassociation must be an explicit, reviewable action.
4. Validate that all alignment times are finite, ordered, within source duration, and linked to the exact source fingerprint; handle gaps, overlaps, partial words, Thai Unicode, empty/malformed alignment, severe transcript errors and model failures by keeping segments unaligned. Preserve source text byte-for-byte and make correction persistence atomic and reloadable.

### Proposed Phase 3 acceptance checks

- The target recording's current artifacts display transcript text only as unaligned until a separately validated alignment exists.
- Word/phrase association includes alignment provenance and honest confidence; low-confidence or malformed results remain visibly unaligned.
- Corrections preserve original transcript content, support Thai Unicode through save/reload, and record timestamps, source IDs, mode, edit time and review state.
- Original and Optimized correction stores are independent, including merged-group lineage. No silent transfer or remapping occurs.
- Tests cover timestamp validity, transcript hash/source preservation, uncertain/missing alignment, persistence/reload, Thai text, mode isolation, and Original/Optimized source relationships.
- No production transcription behavior, Azure request, SQL write, Vault artifact, or original manifest is modified.

### Likely files after approval

- `app/vad_review.py` and `app/vad_review.html` — read-only transcript/alignment display first; correction APIs and controls only in the separately approved UI stage.
- New `app/vad_transcript_alignment.py` — offline aligner adapter, source-hash validation, timestamp/confidence validation, and workspace-only candidate output; exact dependency choice is pending feasibility results.
- New `app/vad_review_corrections.py` (or a small extension to the existing review catalog) — mode-scoped, atomic correction sidecars that do not alter source transcript or VAD files.
- `tests/test_vad_review.py` plus focused alignment/correction tests — missing and uncertain alignment, timestamp bounds, Thai Unicode, source preservation, persistence/reload, and mode/lineage isolation.
- `docs/trainning/README.md`, `docs/trainning/IMPLEMENTATION_PLAN.md`, and `docs/trainning/PROGRESS.md` — record approved decisions, measured model results, and remaining limitations.
- `requirements.txt` only if an aligner is selected and its dependency installation is approved. No SQL migration is proposed for this local prototype.

**Current gate:** This is an assessment and plan only. Do not start Phase 3A alignment, add correction storage, or change the UI until the user reviews and approves the assessment and any required dependency/model acquisition.

## Phase 3B — adaptive Azure transcription chunking

The local chunk planner consumes the existing optimized VAD manifest and groups consecutive segments chronologically. It targets 30–45 seconds, chooses a natural silence boundary when possible, includes the audio between grouped segments, and never divides an optimized VAD segment. If a safe non-overlapping audio boundary is unavailable in the target range, it can emit a shorter chunk; a source segment longer than the cap remains whole and is reported as an exception.

Each chunk records exact original-recording `start_ms`/`end_ms`, duration, optimized segment IDs, original VAD source IDs and source intervals, a separate null `transcript_text`, and the configured Azure OpenAI deployment/API-version metadata. It explicitly labels timestamps as chunk-only. A later Azure response must be stored per chunk with the same chunk ID and time range; neither chunk timing nor chunk transcript text implies word/sentence alignment or alignment to the previously saved transcript.

The planner writes a non-overwriting JSON plan under `workspace/vad_prototype`. FFmpeg clip export is a separate explicit command; clips must resolve beneath that workspace and existing outputs are never overwritten. The current dry run does not export audio or call Azure. This step does not modify the 180-second/10-second production chunker, production transcription settings, source MP3, VAD manifests, SQL Server, or transcript files.

### Phase 3B acceptance and operational checks

- Every optimized segment and original source ID is assigned exactly once, in chronological order; every core speech interval is covered, with no overlapping chunk audio windows.
- Chunk boundaries are within the original recording; reported duration equals `end_ms - start_ms`. Out-of-target chunks include a reason.
- Summary reports chunk duration distribution, target exceptions, speech-core coverage, recording-audio coverage, inter-chunk gaps and overlaps.
- The Azure-compatible response slot remains per chunk and is null/not-requested until a separately approved request. Secrets are never serialized to the plan.
- FFmpeg output is workspace-only, and source audio/optimized and original VAD manifests remain unchanged.
- Existing production 180-second/10-second overlap behavior remains untouched. Rollback consists of ignoring/removing the isolated planner and its workspace artifacts; no production setting or data migration is involved.

Implementation files: `app/vad_transcription_chunks.py`, `tests/test_vad_transcription_chunks.py`, and the workspace plan. FFmpeg is an existing project tool; no package/model installation is required. Measured results are recorded in `docs/trainning/PROGRESS.md`.
