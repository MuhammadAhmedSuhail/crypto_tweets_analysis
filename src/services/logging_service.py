import os
import logging
from logging.handlers import TimedRotatingFileHandler


class LoggingService:
    def __init__(self, name='my_logger', log_dir="../logs", log_file='app.log'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Define the log directory and file name
        self.log_directory = log_dir
        self.log_file = os.path.join(self.log_directory, log_file)

        # Create the log directory if it doesn't exist
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)

        # Create a timed rotating file handler
        handler = TimedRotatingFileHandler(self.log_file, when='D', interval=1, backupCount=7)
        handler.setLevel(logging.DEBUG)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(handler)