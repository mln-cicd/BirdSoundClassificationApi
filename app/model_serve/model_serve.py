"""Model Server Module.

This module provides the ModelServer class for loading and running the bird sound
detection model. It includes methods for loading the model, running detection on audio
files, and getting the classification results.

"""

import logging

from src.models.run_detection_cpu import (
    load_model,
    run_detection,
)
from src.visualization.visu import (
    merge_images,
    visualise_model_out,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

WEIGHTS_PATH = "models/detr_noneg_100q_bs20_r50dc5"
TEST_FILE_PATH = "inference/Turdus_merlula.wav"


class ModelServer:
    """
    ModelServer class for loading and running the bird sound detection model.
    """

    def __init__(self, weights_path, bird_dict) -> None:
        """Initialize the ModelServer.

        Args:
            weights_path (str): Path to the model weights file.
            bird_dict (dict): Dictionary mapping bird names to their corresponding IDs.

        """
        self.weights_path = weights_path
        logger.info("Weights path: %s", self.weights_path)

        self.bird_dict = bird_dict
        self.bird_dict["Non bird sound"] = 0
        self.reverse_bird_dict = {id: bird_name for bird_name, id in self.bird_dict.items()}
        logger.info("Reversed birds dict: %s", self.reverse_bird_dict)

        self.model = None
        self.config = None
        self.model_loaded = False

    def load(self) -> None:
        """
        Load the model.
        """
        logger.info("Loading model...")
        self.model, self.config = load_model(self.weights_path)
        logger.info("Model loaded successfully")
        self.model_loaded = True

    def run_detection(self, file_path, return_spectrogram=False):
        """Run detection on the given audio file.

        Args:
            file_path (str): Path to the audio file.
            return_spectrogram (bool): Whether to return the spectrogram. Default is False.

        Returns:
            tuple: A tuple containing the following elements:
                - fp: The processed audio file.
                - outputs: The model outputs.
                - spectrogram: The spectrogram of the audio file (if return_spectrogram is True).

        """
        spectrogram = None

        if not self.model_loaded:
            self.load()

        logger.info("Starting run_detection on %s...", file_path.split("/")[-1])
        fp, outputs, spectrogram = run_detection(
            self.model, self.config, file_path, return_spectrogram=return_spectrogram
        )
        logger.info("[fp]: \n%s\n\n", fp)
        self.detection_ready = True

        return fp, outputs, spectrogram

    def get_classification(self, file_path, return_spectrogram=False):
        """Get the classification results for the given audio file.

        Args:
            file_path (str): Path to the audio file.
            return_spectrogram (bool): Whether to return the spectrogram. Default is False.

        Returns:
            dict: A dictionary containing the classification results.

        """
        fp, outputs, spectrogram = self.run_detection(file_path, return_spectrogram)

        class_bbox = merge_images(fp, outputs, self.config.num_classes)
        output = {
            self.reverse_bird_dict[idx]: {
                key: value.cpu().numpy().tolist() for key, value in class_bbox[str(idx)].items()
            }
            for idx in range(1, len(class_bbox) + 1)
            if len(class_bbox[str(idx)]["bbox_coord"]) > 0
        }

        logger.info("[output]: \n%s", output)
        if return_spectrogram:
            visualise_model_out(output, fp, spectrogram, self.reverse_bird_dict)
            # TODO: enregistrer le spectrogram
        return output


# TODO ensure bird name is given as output
# TODO unload model method
