"""Application and Azure transcription configuration."""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from urllib.parse import urlparse

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_INPUT_DIR = Path(r"D:\KM\Vault\MorningTalk\Record File")
SUPPORTED_TRANSCRIPTION_API_VERSIONS = {
    "2024-02-01",
    "2024-10-21",
    "2025-03-01-preview",
    "2025-04-01-preview",
    "v1",
}
DEFAULT_AZURE_OPENAI_API_VERSION = "v1"


def get_azure_openai_api_version() -> str:
    return os.getenv("AZURE_OPENAI_API_VERSION", "").strip() or DEFAULT_AZURE_OPENAI_API_VERSION


@dataclass(frozen=True)
class AzureConfig:
    text_endpoint: str
    text_api_key: str
    text_deployment: str
    transcription_endpoint: str
    transcription_api_key: str
    transcription_deployment: str
    transcription_api_version: str
    chunk_duration_seconds: int = 180
    chunk_overlap_seconds: int = 10
    quality_warning_chars_per_minute: float = 10.0

    def validate_transcription(self) -> None:
        missing = []
        if not self.transcription_endpoint:
            missing.append("AZURE_TRANSCRIPTION_ENDPOINT")
        if not self.transcription_api_key:
            missing.append("AZURE_TRANSCRIPTION_API_KEY")
        if not self.transcription_deployment:
            missing.append("AZURE_TRANSCRIPTION_DEPLOYMENT")
        if not self.transcription_api_version:
            missing.append("AZURE_TRANSCRIPTION_API_VERSION")

        parsed = urlparse(self.transcription_endpoint)
        if self.transcription_endpoint and (
            parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
        ):
            raise ValueError("AZURE_TRANSCRIPTION_ENDPOINT must be an HTTPS URL without embedded credentials.")
        if missing:
            raise ValueError("Missing Azure transcription configuration: " + ", ".join(missing))
        if self.transcription_api_version.casefold() not in SUPPORTED_TRANSCRIPTION_API_VERSIONS:
            supported = ", ".join(sorted(SUPPORTED_TRANSCRIPTION_API_VERSIONS))
            raise ValueError(f"Unsupported AZURE_TRANSCRIPTION_API_VERSION; supported values: {supported}.")
        if self.chunk_duration_seconds <= 0 or not 0 <= self.chunk_overlap_seconds < self.chunk_duration_seconds:
            raise ValueError("Audio chunk duration must be positive and overlap must be shorter than the duration.")
        if self.quality_warning_chars_per_minute < 0:
            raise ValueError("Transcript quality warning threshold cannot be negative.")

        missing_tools = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
        if missing_tools:
            raise RuntimeError(
                "FFmpeg tools are not accessible on PATH: " + ", ".join(missing_tools)
            )


@dataclass(frozen=True)
class SqlConfig:
    driver: str
    server: str
    database: str
    username: str
    password: str
    encrypt: bool
    trust_server_certificate: bool
    connection_timeout: int
    expected_database: str
    expected_login: str

    @classmethod
    def from_environment(cls) -> "SqlConfig":
        def boolean(name: str, default: bool) -> bool:
            value = os.getenv(name)
            return default if value is None else value.strip().casefold() in {"1", "true", "yes", "on"}

        return cls(
            driver=os.getenv("SQL_DRIVER", "").strip(),
            server=os.getenv("SQL_SERVER", "").strip(),
            database=os.getenv("SQL_DATABASE", "").strip(),
            username=os.getenv("SQL_USERNAME", "").strip(),
            password=os.getenv("SQL_PASSWORD", ""),
            encrypt=boolean("SQL_ENCRYPT", True),
            trust_server_certificate=boolean("SQL_TRUST_SERVER_CERTIFICATE", False),
            connection_timeout=int(os.getenv("SQL_CONNECTION_TIMEOUT", "15")),
            expected_database=os.getenv("SQL_EXPECTED_DATABASE", "").strip(),
            expected_login=os.getenv("SQL_EXPECTED_LOGIN", "").strip(),
        )

    def validate(self, installed_drivers: set[str] | None = None) -> None:
        missing = [
            name for name, value in (
                ("SQL_DRIVER", self.driver),
                ("SQL_SERVER", self.server),
                ("SQL_DATABASE", self.database),
                ("SQL_USERNAME", self.username),
                ("SQL_PASSWORD", self.password),
            ) if not value
        ]
        if missing:
            raise ValueError("Missing SQL configuration: " + ", ".join(missing))
        if self.connection_timeout <= 0:
            raise ValueError("SQL_CONNECTION_TIMEOUT must be positive.")
        if installed_drivers is not None and self.driver not in installed_drivers:
            raise ValueError("Configured SQL ODBC driver is not installed.")


def get_azure_config() -> AzureConfig:
    return AzureConfig(
        text_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/"),
        text_api_key=os.getenv("AZURE_OPENAI_API_KEY", "").strip(),
        text_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip(),
        transcription_endpoint=os.getenv("AZURE_TRANSCRIPTION_ENDPOINT", "").strip().rstrip("/"),
        transcription_api_key=os.getenv("AZURE_TRANSCRIPTION_API_KEY", "").strip(),
        transcription_deployment=os.getenv("AZURE_TRANSCRIPTION_DEPLOYMENT", "").strip(),
        transcription_api_version=os.getenv("AZURE_TRANSCRIPTION_API_VERSION", "").strip(),
        chunk_duration_seconds=int(os.getenv("MORNINGTALK_CHUNK_DURATION_SECONDS", "180")),
        chunk_overlap_seconds=int(os.getenv("MORNINGTALK_CHUNK_OVERLAP_SECONDS", "10")),
        quality_warning_chars_per_minute=float(
            os.getenv("MORNINGTALK_QUALITY_WARNING_CHARS_PER_MINUTE", "10")
        ),
    )


def get_sql_config() -> SqlConfig:
    return SqlConfig.from_environment()
