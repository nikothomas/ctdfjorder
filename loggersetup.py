import colorlog
import traceback
import logging

formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d) - %(name)s",
    datefmt="%H:%M",
    reset=True,
    log_colors={
        "DEBUG": "white",
        "INFO": "cyan",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
    secondary_log_colors={},
    style="%",
)


# Define a filter class to sanitize newlines
def setup_logging(verbosity):
    base_loglevel = 30
    verbosity = min(verbosity, 2)
    loglevel = base_loglevel - (verbosity * 10)
    logger = logging.getLogger("ctdfjorder")
    # Clear existing handlers if they exist
    if logger.hasHandlers():
        logger.handlers.clear()
    logging.basicConfig(level=loglevel)
    logger.setLevel(loglevel)
    console = colorlog.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(loglevel)

    file_log = logging.FileHandler("ctdfjorder.log")
    file_log.setLevel(loglevel)

    logger.addHandler(console)
    logger.addHandler(file_log)
    return logger
