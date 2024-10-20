import logging
import os
from datetime import datetime
import torch
from torchinfo import summary


class Logger:
    def __init__(self, log_dir, log_level=logging.INFO):
        # Create a logger object
        self.logger = logging.getLogger(__name__)

        # Set the logger level
        self.logger.setLevel(log_level)

        # Create a log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Generate a log file name based on current date and time
        self.log_file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
        log_path = os.path.join(log_dir, self.log_file_name)

        # Create a file handler and set the level
        self.file_handler = logging.FileHandler(log_path)
        self.file_handler.setLevel(log_level)

        # # Create a console handler and set the level
        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(log_level)

        # Create a formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(formatter)
        # console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(self.file_handler)
        # self.logger.addHandler(console_handler)

    def close_log(self):
        # Remove the file handler from the logger
        self.logger.removeHandler(self.file_handler)
        # Close the file handler
        self.file_handler.close()

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    def log_model_info(self, model, input_size, device):
        # Create a string representation of the model summary
        model_info = summary(model, input_size=input_size, verbose=2,
                             col_names=("input_size", "output_size", "num_params"), device=device)
        # Log the model information
        self.logger.info(f"Model Information:\n{model_info}\n")
