# MorningTalk

MorningTalk discovers meeting MP3 files recursively and creates a Thai Unicode PDF beside each source MP3. PDFs contain the complete transcript; the application does not summarize or translate and does not create persistent TXT files.

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

`run_monitor.bat` and the disabled `MorningTalkMonitor.task.xml` are prepared for Windows Task Scheduler. Replace the account placeholder and review the service identity before importing; the XML does not create or enable a scheduled task. The monitor also holds a process-wide lock to reject concurrent instances.
