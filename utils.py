import logging
import sys


def setup_logging():
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
