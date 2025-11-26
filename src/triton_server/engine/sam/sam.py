import os
from pathlib import Path
import json

import torch
import torchvision
import hydra
from hydra.core.global_hydra import GlobalHydra
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from ...configs import settings


class SAMEngine:
    def __init__(
        self,
        model_name: str = settings.sam_settings.MODEL_NAME,
        device: str = settings.sam_settings.DEVICE,
        color_map: str = settings.sam_settings.COLOR_MAP,
    ):
        checkpoint_path = os.path.join(
            settings.sam_settings.CHECKPOINT_DIR_NAME,
            f"{model_name}.pt",
        )

        self.color_map = color_map

        self._reconfigure_hydra(config_dir=settings.sam_settings.CONFIG_DIR_NAME)

        self.mask_generator = SAM2AutomaticMaskGenerator(
            build_sam2(
                model_name, checkpoint_path, device=device, apply_postprocessing=False
            )
        )

    def predict(self, image_arr: np.ndarray, font_scale: float = 0.8) -> np.ndarray:
        """
        Predict masks for a batch of images.

        Args:
            image_arr: Input images as numpy array. Can be:
                - Single image: (H, W, C)
                - Batch of images: (B, H, W, C)
            font_scale: Font size scale factor for mask numbers (default: 0.8)

        Returns:
            Result images with masks overlaid. Same shape as input.
        """
        # Handle single image case by adding batch dimension
        if len(image_arr.shape) == 3:
            image_arr = np.expand_dims(image_arr, axis=0)

        # Generate masks for all images in the batch
        batch_masks = self.generate_masks(image_arr)

        # Add masks to images
        result_images = self.add_masks_to_image(
            image_arr, batch_masks, self.color_map, alpha=0.6, font_scale=font_scale
        )

        return result_images

    def generate_masks(self, image_arr: np.ndarray) -> list[list[dict]]:
        """
        Generate masks for a batch of images.

        Args:
            image_arr: Batch of images as numpy array (B, H, W, C)

        Returns:
            List of lists, where each inner list contains mask dictionaries for one image
        """
        batch_size = image_arr.shape[0]
        batch_masks = []

        with torch.inference_mode():
            for i in range(batch_size):
                # Process each image individually
                single_image = image_arr[i]
                masks = self.mask_generator.generate(single_image)
                batch_masks.append(masks)

        return batch_masks

    def add_masks_to_image(
        self,
        image_arr: np.ndarray,
        masks: list[list[dict]],
        color_map: str,
        alpha: float = 0.6,
        font_scale: float = 0.8,
    ) -> np.ndarray:
        """
        Add masks to a batch of image arrays by overlaying colored masks with numbered labels.

        Args:
            image_arr: Input images as numpy array (B, H, W, C)
            masks: List of lists, where each inner list contains mask dictionaries for one image
            color_map: Colormap name for mask colors
            alpha: Transparency factor
            font_scale: Font size scale factor for mask numbers (default: 0.8)
        Returns:
            Image array with masks overlaid and numbered (B, H, W, C)
        """
        batch_size = image_arr.shape[0]
        result_images = np.zeros_like(image_arr)

        for batch_idx in range(batch_size):
            # Process each image in the batch
            single_image = image_arr[batch_idx]
            single_masks = masks[batch_idx]

            # Create a copy of the original image
            result_image = single_image.copy()

            # Generate colors for each mask
            colors = cm.get_cmap(color_map)(np.linspace(0, 1, len(single_masks)))

            for i, mask_data in enumerate(single_masks):
                segmentation = mask_data["segmentation"]

                # Convert boolean mask to uint8
                mask = segmentation.astype(np.uint8)

                # Create colored mask
                color = (colors[i][:3] * 255).astype(np.uint8)

                # Apply mask with transparency directly to the original image
                for c in range(3):  # For each color channel
                    result_image[:, :, c] = np.where(
                        mask == 1,
                        (1 - alpha) * result_image[:, :, c] + alpha * color[c],
                        result_image[:, :, c],
                    )

            # Add numbered labels to each mask
            result_image = self._add_mask_numbers(
                result_image, single_masks, font_scale
            )
            result_images[batch_idx] = result_image

        return result_images

    @staticmethod
    def _add_mask_numbers(
        image_arr: np.ndarray, masks: list[dict], font_scale: float = 0.1
    ) -> np.ndarray:
        """
        Add numbered labels to the center of each mask.

        Args:
            image_arr: Image array with masks overlaid
            masks: List of mask dictionaries
            font_scale: Font size scale factor (default: 0.3)

        Returns:
            Image array with numbered labels
        """
        result_image = image_arr.copy()

        for i, mask_data in enumerate(masks):
            segmentation = mask_data["segmentation"]

            # Find the center of the mask
            mask_coords = np.where(segmentation)
            if len(mask_coords[0]) == 0:  # Skip empty masks
                continue

            center_y = int(np.mean(mask_coords[0]))
            center_x = int(np.mean(mask_coords[1]))

            # Prepare text
            text = str(i + 1)  # Numbers start from 1

            # Font settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_thickness = max(
                1, int(font_scale * 2)
            )  # Scale thickness with font size

            # Get text size for positioning
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, font_thickness
            )

            # Calculate text position (centered)
            text_x = center_x - text_width // 2
            text_y = center_y + text_height // 2

            # Draw text (black) directly on the image
            cv2.putText(
                result_image,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 0),  # Black text
                font_thickness,
                cv2.LINE_AA,
            )

        return result_image

    @staticmethod
    def save_image(image_arr: np.ndarray, output_path: str) -> None:
        """
        Save a numpy array as an image to a specific location.

        Args:
            image_arr: Image array as numpy array (H, W, C) in RGB format
            output_path: Path where to save the image (including filename and extension)
        """
        # Ensure the output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert RGB to BGR for OpenCV saving
        if len(image_arr.shape) == 3 and image_arr.shape[2] == 3:
            # Assume RGB format, convert to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_arr

        # Save the image
        cv2.imwrite(output_path, image_bgr)
        print(f"Image saved to: {output_path}")

    @staticmethod
    def _reconfigure_hydra(config_dir: str):
        # Clear any existing Hydra instance
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        # Initialize Hydra with the config directory (relative path)
        hydra.initialize(config_path=config_dir, version_base=None)
