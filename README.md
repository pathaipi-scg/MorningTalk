# MorningTalk

MorningTalk discovers meeting MP3 files recursively and can create a Thai Unicode meeting-summary PDF and a separate full-transcript PDF beside each source MP3. Summary claims are evidence-linked internally and remain review-required unless transcript cleanup and analysis validation succeed. The application does not create persistent TXT files.

## Setup

Use the project virtual environment and install `requirements.txt`. Keep `.env` local; the application loads it from the project root using `python-dotenv`. `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_DEPLOYMENT` remain dedicated to GPT text processing. Transcription uses the separate `AZURE_TRANSCRIPTION_ENDPOINT`, `AZURE_TRANSCRIPTION_API_KEY`, `AZURE_TRANSCRIPTION_DEPLOYMENT`, and `AZURE_TRANSCRIPTION_API_VERSION` variables. Supported API contracts are `2025-03-01-preview`, `2024-02-01`, and `v1`; the first two use the deployment-specific path, while `v1` uses `/openai/v1/audio/transcriptions` and sends the deployment in the multipart `model` field.

Configuration can be checked locally by importing `get_azure_config()` and calling `validate_transcription()`; this checks the endpoint, required transcription settings, API version, and FFmpeg tools without making a network request or displaying credentials. A deployment’s availability and compatibility cannot be confirmed without Azure portal access or an Azure request.

## Use

```powershell
.\.venv\Scripts\python.exe -m app.main --dry-run
.\.venv\Scripts\python.exe -m app.main --file "D:\path\meeting-20261104-0815.mp3" --input-dir "D:\path"
```

An existing PDF is skipped. Use `--force` to intentionally regenerate it. PDF creation uses a temporary file in the source directory and atomically publishes the finished document. Exclusive per-source lock files prevent concurrent runs from processing the same MP3. Source MP3 files are read-only to the application.

Audio is segmented across the complete timeline into 180-second chunks with ten seconds of overlap by default. Configure `MORNINGTALK_CHUNK_DURATION_SECONDS`, `MORNINGTALK_CHUNK_OVERLAP_SECONDS`, and `MORNINGTALK_QUALITY_WARNING_CHARS_PER_MINUTE` in `.env` if needed. Chunks are sent and assembled chronologically; only verified matching text at boundaries is deduplicated. Empty or failed chunks fail the file rather than being silently omitted. Logs include chunk durations, status, and character counts, never transcript text. A transcript below the conservative default of ten characters per minute triggers a quality warning, not an automatic rejection. Chunk sizing is checked against the API request limit. Install FFmpeg and ffprobe and ensure they are available on `PATH`. The transcription request sends MP3 as multipart data with a Thai language hint and relies on the endpoint's default JSON response. A future adapter may implement offline faster-whisper through the `Transcriber` interface.

## Tests

Run `.\.venv\Scripts\python.exe -m pytest`.

## MSSQL scanner

Install the requirements and configure the `SQL_*` application-login settings in `.env`. The login needs DML permissions on `dbo`; the application never runs schema DDL. The idempotent schema migration is `migrations/001_initial_schema.sql` and is database-guarded for `MorningTalk`. An administrator must review and apply it before database-backed scanning; no administrator credentials belong in the application configuration.

`python -m app.main scan-once` performs a local scan; add `--sync-db` to persist scan status. `python -m app.main monitor` syncs scan state to MSSQL every 300 seconds by default but never calls Azure unless `--process` is explicitly supplied. Add `--once` to run a single monitor cycle, or `--local-only` to avoid MSSQL. `python -m app.main status` reports persisted status counts. Monitor operational logs are written to `%LOCALAPPDATA%\MorningTalk\logs\monitor.log`.

`run_monitor.bat` and the disabled `MorningTalkMonitor.task.xml` are prepared for Windows Task Scheduler. The task starts at boot after a one-minute delay, runs under the least-privilege account specified in the XML, restarts after failures, and invokes the project's `.venv` Python in scan-only mode. Replace the account placeholder and confirm that the selected account can read `.env` and the input directory before importing. The XML does not create or enable a scheduled task. The monitor also holds a process-wide lock to reject concurrent instances.

After review, use an elevated PowerShell or Command Prompt from the project directory to install the task disabled:

```powershell
schtasks /Create /XML .\MorningTalkMonitor.task.xml /TN "\MorningTalk\Monitor" /F
```

Enable it only when ready to start monitoring:

```powershell
schtasks /Change /TN "\MorningTalk\Monitor" /ENABLE
```

Verify task configuration and its latest result:

```powershell
schtasks /Query /TN "\MorningTalk\Monitor" /V /FO LIST
```

Stop the running monitor and remove the task:

```powershell
schtasks /End /TN "\MorningTalk\Monitor"
schtasks /Delete /TN "\MorningTalk\Monitor" /F
```

Operational logs are available at `%LOCALAPPDATA%\MorningTalk\logs\monitor.log`.

## Phase 9 transcript preview

The optional Azure Speech Fast Transcription adapter requests Thai phrase timestamps and diarization in one full-file pass, keeping provider speaker IDs within a single recording instead of trying to match IDs from independent audio chunks. This requires a separate Azure Speech resource; the existing `gpt-4o-transcribe` deployment does not by itself establish diarization availability. Configure `AZURE_SPEECH_ENDPOINT` and `AZURE_SPEECH_KEY` in the process environment only after confirming the resource, region, and model support. `AZURE_SPEECH_API_VERSION` defaults to `2025-10-15`. The target recording is within the documented 2-hour and 300-MB fast-transcription limits. See Microsoft's [Fast Transcription](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/fast-transcription-create) and [diarization configuration](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/configure-language-identification-diarization) documentation.

The configured text deployment is read from `AZURE_OPENAI_DEPLOYMENT` and is used exactly as configured; no substitution between GPT-5.4 model IDs is performed. `AZURE_OPENAI_ENDPOINT` is authoritative: a complete supported request URL keeps its path and query, including `api-version`, and selects the matching Responses or Chat Completions contract. A resource root plus `AZURE_OPENAI_API_VERSION=v1` retains the v1 Chat Completions behavior; a dated version with a root retains the legacy deployment-specific Chat Completions behavior. A Responses URL is never rewritten or retried against another route. Microsoft’s current [Responses API guide](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/responses) says v1 is required for the latest Responses API features; verify compatibility of any dated preview URL with the resource before a billable run. This text API setting is independent of `AZURE_TRANSCRIPTION_API_VERSION`, which remains required by the separate `gpt-4o-transcribe` integration. The glossary can be overridden with `MORNINGTALK_TERMINOLOGY`. No `.env` values are printed or modified by the Phase 9 commands. Cleanup receives segment text and returns text only; the application retains speaker labels and timestamps separately, then rejects segment-count changes, substantial shortening, and altered numbers/codes. Validation diagnostics identify the failed rule and log only segment indices, character lengths, and number/code counts, never transcript or rejected output text. Summary facts must cite exact transcript segment quotes; unsupported quotes and numeric claims are rejected to a review-required placeholder. These checks do not prove semantic correctness. Any failed validation keeps the raw transcript and marks it for review. Token usage is recorded when Azure returns it; estimated cost is deliberately not calculated from hard-coded rates.

When `--process` is explicitly used after prerequisites are verified, each MP3 produces two independent adjacent outputs: `<basename>.pdf` (SARA-style summary) and `<basename>_Full.pdf` (full transcript). Every summary item must pass strict evidence validation; PDF output shows compact segment references rather than repeating transcript quotations. Unsupported/empty sections are omitted. Either existing output is preserved and tracked independently; only missing outputs are created. Migrations 002 and 003 are required before Phase 9 database-backed Azure processing. The application checks for these columns before making Azure requests and never applies DDL.

Generate a clearly marked local fixture preview without Azure or MSSQL:

```powershell
.\.venv\Scripts\python.exe -m app.main phase9-preview
```

The paired PDFs are written to a timestamped filename pair in `%LOCALAPPDATA%\MorningTalk\previews\`; fixture text and analysis are visibly marked as demonstrations. Existing preview PDFs are never overwritten.

Before the controlled live test, a DBA must review and apply the additive, idempotent, transactional `migrations/002_phase9_transcripts.sql` and `migrations/003_meeting_summary_outputs.sql` migrations on SQL Server 2025 only. Together, they add separate raw/cleaned transcript, analysis, segment JSON, output paths/statuses, review, and token-usage columns; the application does not apply DDL. Follow [the Phase 9 DBA runbook](./migrations/PHASE9_DBA_EXECUTION.md). Confirm Speech resource support and the exact GPT deployment/API version first. The live test is restricted to `meeting-20261107-0817.mp3`, writes separate preview artifacts using the production basenames, does not overwrite adjacent PDFs, and requires explicit Azure billing and MSSQL write flags:

```powershell
.\.venv\Scripts\python.exe -m app.main phase9-live-test --confirm-azure-billing --confirm-mssql-write --preview-dir "$env:LOCALAPPDATA\MorningTalk\previews\live-test-20261107"
```

Before invoking a new controlled live test, verify the deployment, endpoint, API version, migrations, and billing authorization. A prior test completed transcription but received HTTP 404 during GPT cleanup; use `phase9-resume` to reuse its saved transcript rather than rerunning audio transcription. The reference structure/content comparison is documented in [references/phase9_sara_comparison.md](./references/phase9_sara_comparison.md); its PDF conclusions are not used as analysis input or truth labels.

The narrowly scoped `phase9-resume` command resumes only `AudioFileId=4` from its saved raw transcript and segment JSON. It verifies the SQL source fingerprint against the current MP3, never invokes audio transcription, requires both Azure-billing and MSSQL-write confirmations, and refuses an existing versioned preview directory. Use a new directory for each authorized attempt:

```powershell
.\.venv\Scripts\python.exe -m app.main phase9-resume --input-dir "D:\KM\Vault\MorningTalk\Record File" --confirm-azure-billing --confirm-mssql-write --resume-preview-dir "$env:LOCALAPPDATA\MorningTalk\previews\live-test-20261107-resume-v1"
```

For an offline layout demonstration from a previously validated cleaned transcript, use `phase10-summary-demo`. It extracts deterministic exact passages from the source transcript, labels the PDF as a layout demonstration (not a validated summary), verifies the source fingerprint, and refuses to overwrite an existing preview:

```powershell
.\.venv\Scripts\python.exe -m app.main phase10-summary-demo `
  --phase9-json "C:\Users\g\AppData\Local\MorningTalk\previews\live-test-20261107-resume-v3\meeting-20261107-0817.phase9.json" `
  --summary-source "D:\KM\Vault\MorningTalk\Record File\CB\Nov26\meeting-20261107-0817.mp3" `
  --summary-output-dir "C:\Users\g\AppData\Local\MorningTalk\previews\sara-style-phase10-v1"
```

For a separately authorized Meeting Analysis using an existing validated Phase 9 transcript, `phase10-analysis-resume` verifies the transcript and source fingerprint, endpoint/deployment, and isolated output directory before making exactly one GPT request. It does not call transcription, cleanup, or SQL, disables HTTP retries and redirects, and writes an analysis JSON with source offsets. V2 offset citations are kept `review_required` until a person verifies their semantic support; no summary PDF is generated automatically for those results. Each attempt requires a new preview directory:

```powershell
.\.venv\Scripts\python.exe -m app.main phase10-analysis-resume `
  --phase9-json "C:\Users\g\AppData\Local\MorningTalk\previews\live-test-20261107-resume-v3\meeting-20261107-0817.phase9.json" `
  --summary-source "D:\KM\Vault\MorningTalk\Record File\CB\Nov26\meeting-20261107-0817.mp3" `
  --summary-output-dir "C:\Users\g\AppData\Local\MorningTalk\previews\sara-style-phase10-live-v1" `
  --confirm-azure-billing
```

The V2 evidence rules, legacy compatibility, and semantic-review limitation are documented in [the Phase 10.5 resolver specification](./references/SaraAI/analysis/phase10_5_evidence_resolver_v2.md).

This demonstration does not call Azure, does not create a validated Meeting Analysis result, and must not be used as a production summary.

Database PDF statuses are `review_required` when the associated transcript or summary is not validated; summary and full-transcript statuses are tracked independently. Azure HTTP error logs include only status, structured error code, a generic sanitized message, request path, deployment name, and request ID when available. They never include response bodies, API keys, or transcript text.

## Phase 10.7 three-recording pilot

The pilot selects the shortest eligible matched recording in each of Sep26, Oct26, and Nov26 using `ffprobe`, verifies the MP3 and SARA reference pair, and checks for existing validated transcripts and approved MorningTalk summaries. Its dry run is offline and writes the selected files, fingerprints, chunk/request estimates, configuration readiness, and current stage state to the Phase 10.7 manifest and report:

```powershell
.\.venv\Scripts\python.exe -m app.main phase10-pilot-batch --dry-run
```

Stage artifacts are written only under `%LOCALAPPDATA%\MorningTalk\previews\phase10_7_pilot_batch`; neither production MP3s nor adjacent PDFs are written. The runner persists each request attempt before dispatch, does not automatically retry, and stops on an uncertain outcome. A retry requires explicit reconciliation:

```powershell
.\.venv\Scripts\python.exe -m app.main phase10-pilot-batch `
  --pilot-reconcile-attempt "<attempt-id>" --pilot-confirm-retry
```

Only after separate authorization for billable Azure processing should an operator invoke `--pilot-execute --confirm-azure-billing`; set `--pilot-max-requests` to cap each run. Phase 10.7 does not approve analysis or generate an approved PDF. Use the Phase 10.6 `phase10-review` workflow for item-level human review and approval.

## Phase 10.6 offline evaluation and review

`phase10-batch-evaluate` pairs the inventory entries with files on disk, validates existing transcript and analysis artifacts offline, and writes CSV and Markdown status reports. It does not call Azure, transcription, cleanup, or SQL:

```powershell
.\.venv\Scripts\python.exe -m app.main phase10-batch-evaluate
```

The default reports are written to `references\SaraAI\analysis\phase10_6_batch_evaluation.csv` and `.md`. A SARA PDF is only a comparison reference; it does not establish that a MorningTalk transcript or summary is valid.

Use `phase10-review` only with a substantive analysis artifact linked to its validated Phase 9 transcript and source MP3. The first invocation initializes the local review record and displays each item with its resolved transcript evidence:

```powershell
$analysis = "C:\path\to\meeting.phase10.json"
$transcript = "C:\path\to\meeting.phase9.json"
$source = "D:\path\to\meeting.mp3"
.\.venv\Scripts\python.exe -m app.main phase10-review `
  --analysis-json $analysis --transcript-json $transcript `
  --review-source $source --reviewer "Reviewer Name"
```

Make one item-level decision per invocation. `--confirm-review` enables an interactive typed confirmation (`APPROVE`, `REJECT`, or `EDIT`); rejection requires `--review-reason`. Edits support `--edit-text`, `--edit-owner`, and `--edit-deadline`, are revalidated against the unchanged evidence references, and return the item to pending until separately approved or rejected. Review state is stored separately from the original analysis with source hashes and a hash-chained audit trail.

Only when all items have been explicitly decided and the required overview, topics, and key issues are approved can a PDF be published. Choose a new directory below `%LOCALAPPDATA%\MorningTalk\previews`; the command prompts for `GENERATE APPROVED PDF` and refuses existing output directories:

```powershell
.\.venv\Scripts\python.exe -m app.main phase10-review `
  --analysis-json $analysis --transcript-json $transcript `
  --review-source $source --reviewer "Reviewer Name" `
  --generate-approved-pdf --approved-pdf-output-dir "$env:LOCALAPPDATA\MorningTalk\previews\approved-meeting" `
  --confirm-review
```
