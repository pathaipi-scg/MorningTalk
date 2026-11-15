# Sentence-Level Audio Review

## Purpose and status

This directory records the investigation and phased design for employee review of transcription segments. Phase 1 provides an isolated offline VAD prototype, and Phase 2 provides a local-only browser review prototype. Production transcription integration and the employee approval workflow are **not implemented**. This document distinguishes verified existing behavior, implemented prototype behavior, and future proposals.

## Verified current architecture

- `AzureOpenAITranscriber` in `app/transcriber.py` probes duration with `ffprobe`, then uses FFmpeg to resample/convert the original MP3 into mono, 16 kHz MP3 chunks. Chunk duration and overlap come from configuration and default to 180 and 10 seconds. The offsets are calculated as `index * (duration - overlap)`.
- FFmpeg only performs cutting and conversion here. There is no VAD in the inspected code. `FasterWhisperTranscriber` is an unimplemented placeholder.
- The Azure OpenAI adapter sends Thai audio to `/openai/deployments/{deployment}/audio/transcriptions` with the configured API version, or `/openai/v1/audio/transcriptions` with deployment as `model` for `v1`. It reads only JSON `text`; it does not request or parse timestamp-bearing output.
- Overlap is reconciled textually by matching a significant-character suffix/prefix (minimum four characters) using `SequenceMatcher`. If no match is established, both texts are retained, separated by a newline. Chunk progress metadata is recorded but per-word/per-phrase timing is not.
- The optional `AzureSpeechDiarizer` uses the Speech `transcriptions:transcribe` endpoint with diarization and parses phrase text, offset and duration in milliseconds, plus speaker. These are timestamped phrase-level segments, not word timings. It is an Azure-dependent path, not VAD. In `app/auto_scan.py`, it is tried before the OpenAI chunked path; when unavailable, the fallback OpenAI transcript becomes one untimed segment.
- `TranscriptSegment` already supports nullable start/end seconds, speaker, raw/cleaned text and review flags. `TranscriptDocument` separates raw, normalized and cleaned transcript representations. SQL persistence stores document JSON/text on `AudioFile`; the existing `TranscriptionChunk` table stores chunk index, start/end, status and text.
- `app/production_batch.py` is a separate resumable, manifest/staging-based production workflow. It estimates chunk count from the configured duration/overlap and records/reuses transcription attempts; it does not currently add word timing to the transcript.
- Output mapping derives Plant/Month from the `Record File\<Plant>\<MonthYY>` hierarchy and writes under the parallel `Transcript` tree. Existing standard TXT/PDF outputs may be overwritten by the current production path; reviewed versions must therefore use a separate, versioned review output path and never reuse those destinations.
- The inspected application modules expose no FastAPI/Flask routes or browser UI. The repository has an application/CLI style Python architecture with SQL repository and file-based artifacts; a review web UI would be a new surface.

## Phase 2 local review prototype

Start with `python -m app.vad_review` and open `http://127.0.0.1:8765/`. The page loads the Phase 1 manifests from `workspace/vad_prototype` and groups recordings by Plant / Month / Recording. It offers segment navigation, play/pause/replay, 0.75x/1x/1.25x speed, up to 60 seconds of context, and editable millisecond boundaries. Timing edits are stored in a separate `*.adjustments.json` file in the workspace; original manifest timestamps and null transcript text remain unchanged.

Audio is streamed from the original MP3 using bounded HTTP byte ranges; no preview copies are generated. The backend accepts only recordings whose manifest source resolves beneath `D:\KM\Vault\MorningTalk\Record File`, exposes no filesystem static root, validates all segment timestamps, and binds only to a loopback address. Do not bind the prototype to a LAN address. The browser labels transcript state “Not aligned”; no transcript text is inferred.

The project has no FastAPI/Uvicorn dependency installed or declared. This isolated prototype therefore uses only the Python standard-library threaded HTTP server rather than adding a web framework. It is intended for local evaluation, not employee production use; it does not include authentication, approval workflow, multi-user conflict resolution, or deployment hardening.

## Recommended approach

Start offline-first and preserve the original source MP3 and the existing raw transcript as immutable inputs. Use local VAD to propose speech regions, then map existing transcript text into editable, reviewable segments. Do not treat VAD as transcript alignment: VAD can say *when speech-like audio occurs*, but cannot reliably decide which words belong to each region. For existing plain text transcripts, alignment is inherently approximate; provide a low-confidence/unassigned state rather than silently presenting guessed sentence timing as verified.

The simplest reliable first implementation is a local review dataset and single-recording prototype before production SQL integration: extract VAD regions from a read-only MP3, create editable candidate segments with audio ranges and alignment status, then manually correct/adjust segments in a UI. A segment should typically target 5–15 seconds, but not split a continuous utterance solely to meet a hard cap. Allow boundary edits and adjacent-context playback. Keep original transcript and source hash beside every review revision.

For VAD, evaluate a local, pinned implementation (Silero VAD is a reasonable candidate) against representative Thai factory recordings. Its runtime/model footprint and license/deployment constraints need confirmation before adoption. FFmpeg can decode/resample/extract clips but is not itself the VAD detector. Avoid making a new transcription request necessary for prototype testing.

## Phase 1 prototype

`python -m app.vad_prototype analyze` analyzes the default recording, writes a new JSON manifest under `workspace/vad_prototype`, and prints measured segment statistics. An alternate local recording may be supplied as the positional argument. The output writer refuses to overwrite an existing manifest. No clips or modified MP3 files are written.

Preview a segment with `python -m app.vad_prototype preview <manifest.json> <segment_id>`. Optional `--context-before`, `--context-after`, and `--rate` arguments play adjacent audio from the original MP3 using FFplay. FFmpeg, ffprobe, and ffplay must be installed locally. Segments carry `alignment_status: "not_aligned"` and `transcript_text: null`; VAD is not transcript alignment.

The prototype uses WebRTC VAD over FFmpeg-decoded mono 16 kHz PCM. It uses 30 ms frames, configurable VAD aggressiveness, minimum speech duration, silence merge threshold, target/maximum segment duration, and padding. WebRTC returns binary frame decisions, not probabilities, sentences, speaker labels, or word boundaries. Background machinery and overlapping speakers may cause false positives/negatives; the aggressiveness setting trades missed speech against noise activation. The VAD must not be used to claim transcription accuracy.

Dependency: `webrtcvad-wheels>=2.0.14,<3` (provides the `webrtcvad` import). The package is available in the project `.venv`. The requested recording has been analyzed and its manifest is in `workspace/vad_prototype`; measured results and boundary limitations are recorded in [PROGRESS.md](./PROGRESS.md).

## Data model and workflows

See [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) for the proposed records, UI flow, dependencies, risks, acceptance criteria and phased plan.

## Phase 3 alignment feasibility assessment

The Phase 3 assessment found no reliable transcript timestamps in the persisted artifacts inspected for the target recording. The Azure OpenAI path stores text only; the optional Azure Speech path can produce phrase timestamps, but those timestamps are absent from this recording's saved transcript. The existing 219-segment optimized VAD manifest therefore cannot be treated as aligned to the transcript. No alignment or correction UI/storage has been implemented. See the Phase 3 assessment and proposed approval gates in [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) and measured artifact findings in [PROGRESS.md](./PROGRESS.md).

## Phase 3B adaptive transcription chunk plan

The local `app.vad_transcription_chunks` prototype groups consecutive optimized VAD segments into 30–45-second Azure transcription chunks without splitting segments. Its dry-run manifest preserves optimized and original segment lineage, exact recording-relative timestamps, and per-chunk transcript placeholders. It records the configured Azure deployment metadata but makes no Azure request; chunk timestamps are not word- or sentence-level alignment. FFmpeg clip export is a separate explicit command and is restricted to workspace paths. The current dry run and measured coverage/gap results are in [PROGRESS.md](./PROGRESS.md).

## Scope boundary

Phases 1 and 2, and the Phase 3 feasibility assessment, did not call Azure, access SQL Server, change production flags, alter existing source audio/transcripts/PDFs/Vault files, or modify the production transcription pipeline. Phase 3 alignment and correction work remains gated on review and approval.
