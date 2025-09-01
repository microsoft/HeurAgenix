import os
import logging
from logging import Logger, StreamHandler, FileHandler, Formatter

def get_log(log_file: str) -> Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = StreamHandler()
    ch.setLevel(logging.INFO)
    fh = FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger