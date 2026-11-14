"""Fixture preview and explicitly authorized Phase 9 live-test workflow."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import uuid

from app.azure_speech import AzureSpeechDiarizer, DiarizationUnavailable
from app.config import (
    DEFAULT_INPUT_DIR,
    get_azure_config,
    get_azure_openai_api_version,
    get_sql_config,
)
from app.database import SavedTranscriptRecord, SqlRepository, connect_sql
from app.file_manager import (
    discover_mp3_files,
    pdf_path_for_source,
    source_fingerprint,
    source_processing_lock,
)
from app.factory_context import (
    normalize_transcript_segments,
    normalization_enabled,
)
from app.meeting_analysis import (
    AnalysisItem,
    EvidenceReference,
    GPTMeetingAnalyzer,
    MeetingAnalysis,
    MeetingAnalysisError,
)
from app.meeting_outputs import MeetingPdfOutputs, meeting_pdf_paths, write_meeting_pdfs
from app.scanner import is_file_stable
from app.transcriber import AzureOpenAITranscriber
from app.transcript import (
    TranscriptDocument,
    TranscriptQualityFinding,
    TranscriptSegment,
    mark_for_review,
    validate_segment_times,
)
from app.transcript_cleanup import TranscriptCleanup
from app.workspace_paths import DEFAULT_PREVIEW_ROOT

logger = logging.getLogger(__name__)
TARGET_RECORDING = "meeting-20261107-0817.mp3"
PREVIEW_DIRECTORY = DEFAULT_PREVIEW_ROOT
RESUME_AUDIO_FILE_ID = 4
RESUME_PREVIEW_DIRECTORY = PREVIEW_DIRECTORY / "live-test-20261107-resume-v1"


def locate_controlled_target(input_dir: Path) -> Path:
    matches = [
        path for path in discover_mp3_files(input_dir)
        if path.name.casefold() == TARGET_RECORDING.casefold()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {TARGET_RECORDING} under the input root; found {len(matches)}."
        )
    if not is_file_stable(matches[0]):
        raise RuntimeError("Controlled-test recording is still changing; no processing started.")
    return matches[0]


def fixture_document() -> TranscriptDocument:
    return TranscriptDocument(
        segments=(
            TranscriptSegment(80.0, 87.0, "Speaker A", "เช้านี้ตรวจ Encoder ที่เปลี่ยนเมื่อคืนก่อนใช่ไหมครับ"),
            TranscriptSegment(87.0, 95.0, "Speaker B", "ใช่ครับ จะตรวจชุด Stepping และปรับตั้งอีกครั้ง"),
            TranscriptSegment(95.0, 103.0, "Speaker C", "อะไหล่ Lot 123 อยู่ในสต็อกหรือยัง"),
        ),
        diarization_status="fixture",
        cleanup_status="fixture",
    )


def fixture_analysis() -> MeetingAnalysis:
    return MeetingAnalysis(
        overview=AnalysisItem(
            "ทบทวนการตรวจ Encoder และแผนตรวจชุด Stepping",
            (EvidenceReference(1, "เช้านี้ตรวจ Encoder ที่เปลี่ยนเมื่อคืนก่อนใช่ไหมครับ"),),
        ),
        classifications=(
            AnalysisItem(
                "เครื่องจักรและการผลิต",
                (EvidenceReference(2, "ใช่ครับ จะตรวจชุด Stepping และปรับตั้งอีกครั้ง"),),
            ),
        ),
        key_issues=(
            AnalysisItem(
                "มีการหารือเรื่อง Encoder และชุด Stepping",
                (EvidenceReference(1, "เช้านี้ตรวจ Encoder ที่เปลี่ยนเมื่อคืนก่อนใช่ไหมครับ"),),
            ),
        ),
        actions=(
            AnalysisItem(
                "ตรวจชุด Stepping และปรับตั้ง",
                (EvidenceReference(2, "ใช่ครับ จะตรวจชุด Stepping และปรับตั้งอีกครั้ง"),),
            ),
        ),
        dependencies=(),
        schedules=(),
        risks=(),
        status="fixture",
    )


def run_fixture_preview(output_dir: Path = PREVIEW_DIRECTORY) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    source = Path(f"phase9-fixture-preview-{stamp}.mp3")
    summary_path, full_path = meeting_pdf_paths(source, output_dir)
    counter = 1
    while summary_path.exists() or full_path.exists():
        source = Path(f"phase9-fixture-preview-{stamp}-{counter}.mp3")
        summary_path, full_path = meeting_pdf_paths(source, output_dir)
        counter += 1
    outputs = write_meeting_pdfs(
        source,
        "Fixture site",
        fixture_document(),
        fixture_analysis(),
        output_dir=output_dir,
        fixture_preview=True,
    )
    logger.info(
        "Fixture preview PDFs created; summary=%s full_transcript=%s",
        outputs.summary.path,
        outputs.full_transcript.path,
    )
    return outputs.summary.path


def _write_document_json(
    path: Path,
    document: TranscriptDocument,
    analysis: MeetingAnalysis,
    outputs: MeetingPdfOutputs,
    *,
    source_fingerprint_value: str | None = None,
) -> None:
    payload = {
        "raw_transcript": document.raw_transcript,
        "normalized_transcript": document.normalized_transcript,
        "cleaned_transcript": document.cleaned_transcript,
        "normalization_records": [
            asdict(item) for item in document.normalization_records
        ],
        "normalization_review_findings": list(
            document.normalization_review_findings
        ),
        "diarization_status": document.diarization_status,
        "cleanup_status": document.cleanup_status,
        "review_reason": document.review_reason,
        "input_tokens": document.input_tokens,
        "output_tokens": document.output_tokens,
        "segments": [asdict(segment) for segment in document.segments],
        "meeting_analysis": analysis.to_dict(),
        "pdf_outputs": {
            "summary": {
                "path": str(outputs.summary.path),
                "status": outputs.summary.status,
                "error_class": outputs.summary.error_class,
            },
            "full_transcript": {
                "path": str(outputs.full_transcript.path),
                "status": outputs.full_transcript.status,
                "error_class": outputs.full_transcript.error_class,
            },
        },
    }
    if source_fingerprint_value is not None:
        payload["source_fingerprint"] = source_fingerprint_value
    with path.open("x", encoding="utf-8") as output_file:
        json.dump(
            payload,
            output_file,
            ensure_ascii=False,
            indent=2,
        )


def _run_controlled_live_test(
    *,
    source: Path,
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = PREVIEW_DIRECTORY,
    confirm_azure_billing: bool = False,
    confirm_mssql_write: bool = False,
) -> tuple[Path, Path, Path, TranscriptDocument, MeetingAnalysis, MeetingPdfOutputs]:
    if not confirm_azure_billing:
        raise ValueError("This operation requires explicit --confirm-azure-billing authorization.")
    if not confirm_mssql_write:
        raise ValueError("This operation requires explicit --confirm-mssql-write authorization.")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path, full_path = meeting_pdf_paths(source)
    json_path = output_dir / f"{source.stem}.phase9.json"
    use_terminology_pipeline = normalization_enabled()
    if json_path.exists():
        raise FileExistsError("Phase 9 JSON output already exists; choose a new output directory.")
    if use_terminology_pipeline and (
        output_dir / f"{source.stem}.terminology.json"
    ).exists():
        raise FileExistsError(
            "Terminology stage output already exists; choose a new output directory."
        )

    config = get_azure_config()
    cleanup_version = get_azure_openai_api_version()
    if not config.text_endpoint or not config.text_api_key or not config.text_deployment:
        raise ValueError("Azure OpenAI text cleanup configuration is incomplete.")
    connection = connect_sql(get_sql_config())
    repository = SqlRepository(connection)
    try:
        if not repository.phase9_schema_ready():
            raise RuntimeError(
                "Phase 9 MSSQL migrations 002 and 003 are not applied; no Azure request was made."
            )
        existing = {
            record.source_path.casefold(): record
            for record in repository.list_audio_files()
        }
        key = str(source.resolve()).casefold()
        record = existing.get(key)
        if record is None:
            stat = source.stat()
            upsert_result = repository.upsert_scan(
                source,
                source_fingerprint(source),
                stat.st_size,
                datetime.fromtimestamp(stat.st_mtime, timezone.utc),
                pdf_path_for_source(source),
                pdf_path_for_source(source).is_file(),
            )
            audio_file_id = upsert_result[0]
        else:
            audio_file_id = record.audio_file_id
    except Exception:
        connection.close()
        raise

    try:
        endpoint = os.getenv("AZURE_SPEECH_ENDPOINT", "").strip()
        speech_key = os.getenv("AZURE_SPEECH_KEY", "").strip()
        api_version = os.getenv("AZURE_SPEECH_API_VERSION", "2025-10-15").strip()
        segments: tuple[TranscriptSegment, ...]
        diarization_status: str
        try:
            if not endpoint or not speech_key:
                raise DiarizationUnavailable("Azure Speech endpoint or key is not configured.")
            segments = AzureSpeechDiarizer(
                endpoint,
                speech_key,
                api_version=api_version,
            ).transcribe(source)
            validate_segment_times(segments)
            diarization_status = "available"
        except DiarizationUnavailable as error:
            logger.warning(
                "Diarization unavailable; falling back to transcript without speaker labels; "
                "error_class=%s",
                type(error).__name__,
            )
            try:
                text = AzureOpenAITranscriber(config).transcribe(source)
            except Exception as transcription_error:
                logger.error(
                    "Raw transcript fallback failed; error_class=%s",
                    type(transcription_error).__name__,
                )
                raise RuntimeError(
                    "Diarization and transcript fallback both failed; no outputs were produced."
                ) from None
            segments = (TranscriptSegment(None, None, None, text),)
            diarization_status = "unavailable"

        raw_transcript = "\n".join(segment.raw_text for segment in segments)
        normalization_records = ()
        normalization_findings = ()
        normalized_transcript = raw_transcript
        cleanup_segments = segments
        if use_terminology_pipeline:
            normalization = normalize_transcript_segments(
                segments,
                recording=source.name,
            )
            cleanup_segments = normalization.segments
            normalized_transcript = normalization.normalized_transcript
            normalization_records = normalization.replacements
            normalization_findings = normalization.review_findings
            normalized_path = output_dir / f"{source.stem}.terminology.json"
            with normalized_path.open("x", encoding="utf-8") as normalized_file:
                json.dump(
                    {
                        "source_filename": source.name,
                        "source_fingerprint": source_fingerprint(source),
                        "raw_transcript": normalization.raw_transcript,
                        "normalized_transcript": normalization.normalized_transcript,
                        "normalization_records": [
                            asdict(item) for item in normalization.replacements
                        ],
                        "normalization_review_findings": list(
                            normalization.review_findings
                        ),
                    },
                    normalized_file,
                    ensure_ascii=False,
                    indent=2,
                )
                normalized_file.write("\n")

        try:
            if normalization_findings:
                document = mark_for_review(
                    cleanup_segments,
                    "TerminologyCandidatesRequireReview",
                    diarization_status=diarization_status,
                )
            else:
                cleaned = TranscriptCleanup(
                    config,
                    api_version=cleanup_version,
                ).clean(cleanup_segments)
                document = TranscriptDocument(
                    cleaned.segments,
                    diarization_status,
                    cleaned.cleanup_status,
                    cleaned.review_reason,
                    cleaned.input_tokens,
                    cleaned.output_tokens,
                )
        except Exception as error:
            logger.error("Transcript cleanup unavailable; error_class=%s", type(error).__name__)
            document = mark_for_review(
                cleanup_segments,
                "CleanupUnavailable",
                diarization_status=diarization_status,
            )
        if use_terminology_pipeline:
            document = TranscriptDocument(
                document.segments,
                document.diarization_status,
                document.cleanup_status,
                document.review_reason,
                document.input_tokens,
                document.output_tokens,
                raw_transcript_source=raw_transcript,
                normalized_transcript_source=normalized_transcript,
                normalization_records=normalization_records,
                normalization_review_findings=normalization_findings,
            )

        if document.cleanup_status != "completed":
            analysis = MeetingAnalysis.for_review("TranscriptCleanupRequiresReview")
        else:
            try:
                analysis = GPTMeetingAnalyzer(
                    config,
                    api_version=cleanup_version,
                    terminology=os.getenv("MORNINGTALK_TERMINOLOGY", ""),
                ).analyze(document.segments)
            except MeetingAnalysisError as error:
                logger.error("Meeting analysis unavailable; error_class=%s", type(error).__name__)
                analysis = MeetingAnalysis.for_review(type(error).__name__)

        outputs = write_meeting_pdfs(
            source,
            source.relative_to(input_dir).parts[0],
            document,
            analysis,
        )
        _write_document_json(
            json_path,
            document,
            analysis,
            outputs,
            source_fingerprint_value=source_fingerprint(source),
        )
        repository.save_phase9_outputs(audio_file_id, document, analysis, outputs)
        return summary_path, full_path, json_path, document, analysis, outputs
    finally:
        connection.close()


def _saved_transcript_document(record: SavedTranscriptRecord) -> TranscriptDocument:
    if not record.raw_transcript or not record.transcript_segments_json:
        raise ValueError("The selected audio record has no complete saved transcript.")
    try:
        segment_data = json.loads(record.transcript_segments_json)
    except (ValueError, TypeError) as error:
        raise ValueError("The saved transcript segments are invalid JSON.") from error
    if not isinstance(segment_data, list) or not segment_data:
        raise ValueError("The saved transcript has no segments.")

    segments = []
    for item in segment_data:
        if not isinstance(item, dict):
            raise ValueError("The saved transcript contains a malformed segment.")
        raw_text = item.get("raw_text")
        speaker_id = item.get("speaker_id")
        start = item.get("start_seconds")
        end = item.get("end_seconds")
        if (
            not isinstance(raw_text, str)
            or speaker_id is not None and not isinstance(speaker_id, str)
            or start is not None and (isinstance(start, bool) or not isinstance(start, (int, float)))
            or end is not None and (isinstance(end, bool) or not isinstance(end, (int, float)))
            or not isinstance(item.get("review_required"), bool)
        ):
            raise ValueError("The saved transcript contains invalid segment fields.")
        cleaned_text = item.get("cleaned_text")
        if cleaned_text is not None and not isinstance(cleaned_text, str):
            raise ValueError("The saved transcript contains invalid cleaned text.")
        raw_findings = item.get("quality_findings", [])
        if not isinstance(raw_findings, list):
            raise ValueError("The saved transcript contains invalid quality findings.")
        findings = []
        for finding in raw_findings:
            if (
                not isinstance(finding, dict)
                or not isinstance(finding.get("original_phrase"), str)
                or finding.get("proposed_correction") is not None
                and not isinstance(finding.get("proposed_correction"), str)
                or not isinstance(finding.get("surrounding_context"), str)
                or not isinstance(finding.get("uncertainty"), str)
                or finding.get("uncertainty") not in {"low", "medium", "high"}
                or finding.get("review_required") is not True
            ):
                raise ValueError("The saved transcript contains invalid quality findings.")
            findings.append(TranscriptQualityFinding(
                finding["original_phrase"],
                finding.get("proposed_correction"),
                finding["surrounding_context"],
                finding["uncertainty"],
                True,
            ))
        segments.append(TranscriptSegment(
            start,
            end,
            speaker_id,
            raw_text,
            cleaned_text,
            item["review_required"],
            tuple(findings),
        ))

    transcript_segments = tuple(segments)
    validate_segment_times(transcript_segments)
    if "\n".join(segment.raw_text for segment in transcript_segments) != record.raw_transcript:
        raise ValueError("Saved RawTranscript does not match TranscriptSegmentsJson.")
    return TranscriptDocument(
        transcript_segments,
        record.diarization_status or "unavailable",
        "pending",
    )


def run_controlled_transcript_resume(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = RESUME_PREVIEW_DIRECTORY,
    confirm_azure_billing: bool = False,
    confirm_mssql_write: bool = False,
) -> tuple[Path, Path, Path, TranscriptDocument, MeetingAnalysis, MeetingPdfOutputs]:
    if not confirm_azure_billing or not confirm_mssql_write:
        raise ValueError(
            "Resume requires explicit --confirm-azure-billing and --confirm-mssql-write authorization."
        )

    input_dir = input_dir.resolve()
    preview_root = PREVIEW_DIRECTORY.resolve()
    output_dir = output_dir.resolve()
    try:
        relative_output = output_dir.relative_to(preview_root)
    except ValueError as error:
        raise ValueError("Resume outputs must be inside the Phase 9 preview directory.") from error
    if not relative_output.parts:
        raise ValueError("Resume requires a new versioned subdirectory inside the preview directory.")
    if output_dir.exists():
        raise FileExistsError("Resume preview directory already exists; choose a new versioned directory.")

    config = get_azure_config()
    api_version = get_azure_openai_api_version()
    if not config.text_endpoint or not config.text_api_key or not config.text_deployment:
        raise ValueError("Azure OpenAI text cleanup configuration is incomplete.")

    connection = connect_sql(get_sql_config())
    repository = SqlRepository(connection)
    lease_token: uuid.UUID | None = None
    try:
        if not repository.phase9_schema_ready():
            raise RuntimeError("Phase 9 MSSQL migrations 002 and 003 are not applied.")
        record = repository.load_saved_transcript(RESUME_AUDIO_FILE_ID)
        if record is None:
            raise FileNotFoundError(f"AudioFileId={RESUME_AUDIO_FILE_ID} was not found.")
        if record.status != "review_required":
            raise RuntimeError("The selected AudioFile is not in review_required status.")

        source = locate_controlled_target(input_dir)
        source_path = Path(record.source_path).resolve()
        if source_path != source.resolve() or source.name.casefold() != TARGET_RECORDING.casefold():
            raise RuntimeError("AudioFileId=4 does not point to the controlled target recording.")
        saved_fingerprint = record.fingerprint.strip().casefold()
        current_fingerprint = source_fingerprint(source).casefold()
        if saved_fingerprint != current_fingerprint:
            raise RuntimeError("The source fingerprint differs from the saved transcription; resume refused.")

        document = _saved_transcript_document(record)
        previous_json = (
            PREVIEW_DIRECTORY / "live-test-20261107" / f"{source.stem}.phase9.json"
        )
        if previous_json.is_file():
            try:
                previous_payload = json.loads(previous_json.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise RuntimeError("The original Phase 9 JSON cannot be validated.") from error
            previous_segments = previous_payload.get("segments")
            segments_match = (
                isinstance(previous_segments, list)
                and len(previous_segments) == len(document.segments)
                and all(
                    isinstance(previous, dict)
                    and all(
                        key not in previous and key == "quality_findings"
                        or previous.get(key) == value
                        for key, value in asdict(current).items()
                    )
                    for previous, current in zip(
                        previous_segments, document.segments, strict=True
                    )
                )
            )
            if (
                previous_payload.get("raw_transcript") != record.raw_transcript
                or not segments_match
            ):
                raise RuntimeError("The original Phase 9 JSON does not match the saved SQL transcript.")

        summary_path, full_path = meeting_pdf_paths(source)
        json_path = output_dir / f"{source.stem}.phase9.json"
        use_terminology_pipeline = normalization_enabled()
        if json_path.exists():
            raise FileExistsError(
                "A resume metadata file already exists; choose a new versioned directory."
            )
        if use_terminology_pipeline and (
            output_dir / f"{source.stem}.terminology.json"
        ).exists():
            raise FileExistsError(
                "Terminology stage output already exists; choose a new versioned directory."
            )

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(exist_ok=False)
        with source_processing_lock(source):
            if source_fingerprint(source).casefold() != saved_fingerprint:
                raise RuntimeError("The source changed after transcript validation; resume refused.")
            candidate_lease_token = uuid.uuid4()
            attempt = repository.claim_file(
                RESUME_AUDIO_FILE_ID,
                candidate_lease_token,
                lease_seconds=7200,
                allow_review_required=True,
            )
            if attempt is None:
                raise RuntimeError("AudioFileId=4 could not be claimed for transcript resume.")
            lease_token = candidate_lease_token

            cleanup_segments = document.segments
            normalized_transcript = document.raw_transcript
            normalization_records = ()
            normalization_findings = ()
            if use_terminology_pipeline:
                normalization = normalize_transcript_segments(
                    document.segments,
                    recording=source.name,
                )
                if normalization.raw_transcript != document.raw_transcript:
                    raise RuntimeError(
                        "Saved SQL transcript segments do not match the immutable raw transcript."
                    )
                cleanup_segments = normalization.segments
                normalized_transcript = normalization.normalized_transcript
                normalization_records = normalization.replacements
                normalization_findings = normalization.review_findings
                with (
                    output_dir / f"{source.stem}.terminology.json"
                ).open("x", encoding="utf-8") as normalized_file:
                    json.dump(
                        {
                            "source_filename": source.name,
                            "source_fingerprint": saved_fingerprint,
                            "raw_transcript": normalization.raw_transcript,
                            "normalized_transcript": normalization.normalized_transcript,
                            "normalization_records": [
                                asdict(item) for item in normalization.replacements
                            ],
                            "normalization_review_findings": list(
                                normalization.review_findings
                            ),
                        },
                        normalized_file,
                        ensure_ascii=False,
                        indent=2,
                    )
                    normalized_file.write("\n")

            if normalization_findings:
                cleaned = mark_for_review(
                    cleanup_segments,
                    "TerminologyCandidatesRequireReview",
                    diarization_status=document.diarization_status,
                )
            else:
                cleaned = TranscriptCleanup(
                    config,
                    api_version=api_version,
                ).clean(cleanup_segments)
            transcript = TranscriptDocument(
                cleaned.segments,
                document.diarization_status,
                cleaned.cleanup_status,
                cleaned.review_reason,
                cleaned.input_tokens,
                cleaned.output_tokens,
                raw_transcript_source=(
                    document.raw_transcript if use_terminology_pipeline else None
                ),
                normalized_transcript_source=(
                    normalized_transcript if use_terminology_pipeline else None
                ),
                normalization_records=normalization_records,
                normalization_review_findings=normalization_findings,
            )
            if transcript.cleanup_status != "completed":
                analysis = MeetingAnalysis.for_review("TranscriptCleanupRequiresReview")
            else:
                try:
                    analysis = GPTMeetingAnalyzer(
                        config,
                        api_version=api_version,
                        terminology=os.getenv("MORNINGTALK_TERMINOLOGY", ""),
                    ).analyze(transcript.segments)
                except MeetingAnalysisError as error:
                    logger.error(
                        "Meeting analysis unavailable; error_class=%s",
                        type(error).__name__,
                    )
                    analysis = MeetingAnalysis.for_review(type(error).__name__)

            if source_fingerprint(source).casefold() != saved_fingerprint:
                raise RuntimeError("The source changed during transcript resume; outputs were not published.")
            if not repository.renew_lease(
                RESUME_AUDIO_FILE_ID, lease_token, lease_seconds=7200
            ):
                raise RuntimeError("The database processing claim was lost during transcript resume.")

            outputs = write_meeting_pdfs(
                source,
                source.relative_to(input_dir).parts[0],
                transcript,
                analysis,
            )
            _write_document_json(
                json_path,
                transcript,
                analysis,
                outputs,
                source_fingerprint_value=saved_fingerprint,
            )
            repository.save_phase9_outputs(
                RESUME_AUDIO_FILE_ID, transcript, analysis, outputs
            )

            duration = AzureOpenAITranscriber._probe_duration(source)
            low_quality = len(transcript.cleaned_transcript) / max(
                duration / 60, 1 / 60
            ) < config.quality_warning_chars_per_minute
            outputs_validated = all(
                result.status in {"created", "overwritten"}
                for result in (outputs.summary, outputs.full_transcript)
            )
            needs_review = (
                low_quality
                or transcript.cleanup_status != "completed"
                or analysis.status != "validated"
                or any(segment.review_required for segment in transcript.segments)
                or not outputs_validated
            )
            if not repository.complete_file(
                RESUME_AUDIO_FILE_ID,
                lease_token,
                outputs.summary.path,
                review_required=needs_review,
            ):
                raise RuntimeError("The database processing claim was lost before resume completion.")
            final_status = "review_required" if needs_review else "completed"
            repository.log_event(
                RESUME_AUDIO_FILE_ID,
                "phase9_resume",
                final_status,
            )
            return summary_path, full_path, json_path, transcript, analysis, outputs
    except Exception as error:
        if lease_token is not None:
            repository.release_claim_for_review(
                RESUME_AUDIO_FILE_ID, lease_token, type(error).__name__
            )
            repository.log_event(
                RESUME_AUDIO_FILE_ID,
                "phase9_resume",
                "review_required",
                error_class=type(error).__name__,
            )
        raise
    finally:
        connection.close()


def run_controlled_live_test(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = PREVIEW_DIRECTORY,
    confirm_azure_billing: bool = False,
    confirm_mssql_write: bool = False,
) -> tuple[Path, Path, Path, TranscriptDocument, MeetingAnalysis, MeetingPdfOutputs]:
    if not confirm_azure_billing or not confirm_mssql_write:
        raise ValueError(
            "The controlled test requires explicit Azure billing and MSSQL write authorization."
        )
    source = locate_controlled_target(input_dir)
    with source_processing_lock(source):
        return _run_controlled_live_test(
            source=source,
            input_dir=input_dir,
            output_dir=output_dir,
            confirm_azure_billing=confirm_azure_billing,
            confirm_mssql_write=confirm_mssql_write,
        )
