# logger_config.py
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class CustomLogger:
    def __init__(self, name='main', log_file='logs/main.log', max_bytes=5*1024*1024, backup_count=5):
        logfolder = Path(log_file).parent
        if not logfolder.exists():
            logfolder.mkdir(parents=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Check if handlers already exist to prevent adding them multiple times
        if not self.logger.hasHandlers():
            # Create a file handler that logs debug and higher level messages
            file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
            file_handler.setLevel(logging.DEBUG)

            # Create a console handler for higher level logs (e.g., warnings, errors)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)  # Set to DEBUG to see all messages on the console

            # Create a logging format
            formatter = logging.Formatter('%(asctime)s | %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # Add the handlers to the logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def __getattr__(self, attr):
        return getattr(self.logger, attr)

# Instantiate and configure the custom logger
logger = CustomLogger().logger
