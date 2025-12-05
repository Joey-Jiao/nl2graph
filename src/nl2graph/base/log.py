import sys
from pathlib import Path
from datetime import datetime

from loguru import logger

from .configs import ConfigService


class LogService:
    def __init__(self, config: ConfigService):
        log_cfg = config.get("log", {})

        log_dir = log_cfg.get("log_dir", "outputs/logs")
        console_level = log_cfg.get("console_level", "INFO")
        file_level = log_cfg.get("file_level", "DEBUG")
        rotation = log_cfg.get("rotation", "200 MB")
        retention = log_cfg.get("retention", "14 days")

        logger.remove()

        # --------------------------------
        # Console sink
        # --------------------------------
        logger.add(
            sys.stderr,
            level=console_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{extra[name]}</cyan> - "
                "<level>{message}</level>"
            ),
            colorize=True,
        ),

        # --------------------------------
        # File sink
        # --------------------------------
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"app_{timestamp}.log"
        log_path = log_dir / filename
        logger.add(
            log_path,
            level=file_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "{extra[name]}:{line} - {message}"
            ),
            rotation=rotation,
            retention=retention,
            compression="zip",
            enqueue=False,
            backtrace=False,
            diagnose=False
        )

        self._logger = logger

    def get(self, name: str):
        return self._logger.bind(name=name)
