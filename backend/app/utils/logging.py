import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once for the whole backend app."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
