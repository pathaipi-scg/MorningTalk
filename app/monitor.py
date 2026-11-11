"""Operational monitor logging, singleton lock, and continuous scan loop."""

from contextlib import contextmanager
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import time
from typing import Callable, Iterator

logger = logging.getLogger(__name__)


class MonitorAlreadyRunning(RuntimeError):
    """Raised when another live monitor owns the singleton lock."""


def runtime_directory() -> Path:
    local_app_data = os.getenv("LOCALAPPDATA")
    base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    return base / "MorningTalk"


def configure_monitor_logging() -> Path:
    log_directory = runtime_directory() / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / "monitor.log"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        if (
            getattr(handler, "_morningtalk_file_handler", False)
            and Path(handler.baseFilename) != log_path.resolve()
        ):
            root_logger.removeHandler(handler)
            handler.close()
    if not any(getattr(handler, "_morningtalk_file_handler", False) for handler in root_logger.handlers):
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler._morningtalk_file_handler = True
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root_logger.addHandler(file_handler)
    if not any(getattr(handler, "_morningtalk_console_handler", False) for handler in root_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler._morningtalk_console_handler = True
        console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root_logger.addHandler(console_handler)
    return log_path


def _process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        return getattr(error, "winerror", None) != 87
    return True


@contextmanager
def monitor_instance_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    for _attempt in range(3):
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError as error:
            try:
                owner_pid = int(lock_path.read_text(encoding="ascii").strip())
            except (OSError, ValueError):
                try:
                    lock_age = time.time() - lock_path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if lock_age < 60:
                    raise MonitorAlreadyRunning(
                        "A monitor is currently acquiring its singleton lock."
                    ) from error
                owner_pid = 0
            if _process_is_running(owner_pid):
                raise MonitorAlreadyRunning("Another MorningTalk monitor is already running.") from error
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
    if descriptor is None:
        raise MonitorAlreadyRunning("Unable to acquire the MorningTalk monitor lock.")

    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as lock_file:
            lock_file.write(str(os.getpid()))
        yield
    finally:
        try:
            if lock_path.read_text(encoding="ascii").strip() == str(os.getpid()):
                lock_path.unlink()
        except FileNotFoundError:
            pass


def run_monitor(
    scan_cycle: Callable[..., int],
    input_dir: Path,
    *,
    interval_seconds: int = 300,
    local_only: bool = False,
    process_enabled: bool = False,
    once: bool = False,
) -> int:
    if interval_seconds <= 0:
        raise ValueError("Monitor interval must be positive.")
    log_path = configure_monitor_logging()
    lock_path = runtime_directory() / "monitor.lock"
    try:
        with monitor_instance_lock(lock_path):
            logger.info(
                "Monitor started; interval_seconds=%d scan_only=%s log_file=%s",
                interval_seconds, not process_enabled, log_path,
            )
            while True:
                try:
                    result = scan_cycle(
                        input_dir,
                        sync_db=(not local_only or process_enabled),
                        process_enabled=process_enabled,
                    )
                except Exception as error:
                    result = 2
                    logger.error("Monitor scan failed; error_class=%s", type(error).__name__)
                if result:
                    logger.warning("Monitor cycle returned status=%d; will retry next cycle.", result)
                else:
                    logger.info("Monitor cycle completed successfully.")
                if once:
                    return result
                time.sleep(interval_seconds)
    except MonitorAlreadyRunning:
        logger.error("Monitor lock is already owned by another running instance.")
        return 2
    except KeyboardInterrupt:
        logger.info("Monitor stopped by user.")
        return 0
