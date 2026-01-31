import logging
from functools import cache

def setup_logger(name="root"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(name)s:%(lineno)d] [%(levelname)s] %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

@cache
def get_logger(name):
    return setup_logger(name)
