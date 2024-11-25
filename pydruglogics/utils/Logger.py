import logging
import sys


class Logger:

    @classmethod
    def set_logger(cls, level=logging.INFO, external_lib_off=True):
        logger = logging.getLogger()

        if logger.hasHandlers():
            logger.handlers.clear()

        logger.setLevel(level)

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stdout_handler.setFormatter(stdout_formatter)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stderr_handler.setFormatter(stderr_formatter)

        logger.addHandler(stdout_handler)
        logger.addHandler(stderr_handler)

        if external_lib_off:
            for module_name, module in logging.Logger.manager.loggerDict.items():
                if isinstance(module, logging.Logger):
                    module.setLevel(logging.INFO)

Logger.set_logger(level=logging.INFO)
