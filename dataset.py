import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from torch.utils.data import Dataset as BaseDataset
import random
import cv2

class Dataset(BaseDataset):
    def __init__(self, images_dir, logos_dir, augmentation=None, artificial_len=None, dilate=0):
        """
        Simplified Dataset for applying only logo watermarks.

        Args:
            images_dir: Directory containing the training images
            logos_dir: Directory containing the watermark logos
            augmentation: Optional augmentation pipeline
            artificial_len: Optional artificial length for the dataset
            dilate: Optional dilation size for the mask
        """
        # Setup image paths - recursively find all images
        self.images_fps = []
        for root, _, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    self.images_fps.append(os.path.join(root, file))

        if not self.images_fps:
            raise ValueError(f"No valid images found in {images_dir}")

        if artificial_len:
            self.artificial_len=artificial_len
        else: self.artificial_len = None

        self.dilate = dilate

        self.augmentation = augmentation
        self.logos_dir = logos_dir  # Store directory path instead of loaded images

        # Validate that we have watermark logos
        self.logo_files = [f for f in os.listdir(logos_dir) if f.lower().endswith('.png')]
        if not self.logo_files:
            raise ValueError(f"No valid PNG logos found in {logos_dir}")

        # Hardcode watermark type to 'logos'
        self.watermark_types = ['logos']

        # Keep blend modes (can be simplified later if needed)
        self.blend_modes = ['normal', 'multiply', 'overlay']
        # Default to equal blend mode probabilities
        self.blend_mode_probs = {mode: 1.0/len(self.blend_modes) for mode in self.blend_modes}

    def _blend_watermark(self, image, watermark, alpha, blend_mode=None):
        """
        Blend watermark into image using specified blend mode.

        Args:
            image: Background image array (H, W, C)
            watermark: Watermark image array (H, W, C)
            alpha: Alpha channel of watermark (H, W, 1)
            blend_mode: Blend mode to use, if None a random one is selected

        Returns:
            Blended image array
        """
        if blend_mode is None:
            blend_mode = random.choices(
                self.blend_modes,
                weights=[self.blend_mode_probs[mode] for mode in self.blend_modes],
                k=1
            )[0]

        # Ensure alpha is in range [0, 1]
        alpha = np.clip(alpha, 0, 1)

        # Extract RGB channels
        img_rgb = image[:, :, :3]
        wm_rgb = watermark[:, :, :3]

        # Apply blend mode
        if blend_mode == 'normal':
            # Standard alpha compositing
            blended = img_rgb * (1 - alpha) + wm_rgb * alpha

        elif blend_mode == 'multiply':
            # Multiply blend mode
            blended = img_rgb * (1 - alpha) + (img_rgb * wm_rgb) * alpha

        elif blend_mode == 'overlay':
             # Overlay blend mode
             mask_blend = img_rgb < 0.5
             blended = np.zeros_like(img_rgb)
             blended[mask_blend] = (2 * img_rgb * wm_rgb)[mask_blend]
             blended[~mask_blend] = (1 - 2 * (1 - img_rgb) * (1 - wm_rgb))[~mask_blend]
             blended = img_rgb * (1 - alpha) + blended * alpha

        else:
            # Default to normal blend if mode is somehow invalid
            blended = img_rgb * (1 - alpha) + wm_rgb * alpha

        # Ensure result is in valid range
        blended = np.clip(blended, 0, 1)

        # Combine with alpha channel
        result = np.copy(image)
        result[:, :, :3] = blended

        return result

    def _apply_watermarks(self, image):
        """Apply logo watermarks to image and return watermarked image and mask."""
        # Convert to PIL Image if necessary
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))

        # Convert to RGBA for watermarking
        image = image.convert('RGBA')
        image_array = np.array(image).astype(np.float32) / 255.0

        # Create blank mask
        mask = np.zeros(image.size[::-1], dtype=np.float32)

        # Directly apply 'logos' watermark type
        watermark_type = 'logos' # Hardcoded

        # --- Logic specifically for 'logos' type ---
        # Decide number of watermarks
        num_wm = random.randint(1, 5)

        # Calculate image diagonal for scaling
        image_diagonal = (image.width ** 2 + image.height ** 2) ** 0.5

        # Apply watermarks
        for _ in range(num_wm):
            # Load watermark on demand
            logo_file = random.choice(self.logo_files)
            wm = Image.open(os.path.join(self.logos_dir, logo_file))
            if wm.mode != 'RGBA':
                wm = wm.convert('RGBA')

            # Add random rotation before scaling
            rotation = random.uniform(0, 360) if random.random() < 0.5 else 0  # 50% chance of rotation
            if rotation != 0:
                wm = wm.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)

            # Calculate watermark diagonal
            wm_diagonal = (wm.width ** 2 + wm.height ** 2) ** 0.5

            # Calculate scale
            # Adjusted scale slightly based on original code
            min_scale_factor = 0.05
            max_scale_factor = 0.70
            # Prevent division by zero if wm_diagonal is 0
            if wm_diagonal > 1e-6:
                 min_scale = min_scale_factor * image_diagonal / wm_diagonal
                 max_scale = max_scale_factor * image_diagonal / wm_diagonal
                 scale = random.uniform(min_scale, max_scale)
            else:
                 scale = random.uniform(0.1, 0.5) # Default scale if diagonal is zero

            opacity = random.uniform(0.3, 1.0)

            # Calculate position
            new_width = max(1, int(wm.width * scale))
            new_height = max(1, int(wm.height * scale))
            new_size = (new_width, new_height)

            # Ensure position allows some part of the logo to be visible
            min_x = -new_width + 10 # Allow partial off-screen placement but ensure some overlap
            min_y = -new_height + 10
            max_x = image.width - 10
            max_y = image.height - 10

            # Handle cases where max < min (e.g., very large logo on small image)
            if max_x < min_x: max_x = min_x
            if max_y < min_y: max_y = min_y

            pos_x = random.randint(min_x, max_x)
            pos_y = random.randint(min_y, max_y)

            # Resize watermark
            wm = wm.resize(new_size, Image.Resampling.LANCZOS)
            wm_array = np.array(wm).astype(np.float32) / 255.0

            # Apply opacity
            wm_array[:, :, 3] = wm_array[:, :, 3] * opacity

            # Calculate valid regions for blending
            start_y = max(0, pos_y)
            start_x = max(0, pos_x)
            end_y = min(image_array.shape[0], pos_y + wm_array.shape[0])
            end_x = min(image_array.shape[1], pos_x + wm_array.shape[1])

            wm_start_y = max(0, -pos_y)
            wm_start_x = max(0, -pos_x)
            wm_end_y = start_y - pos_y + (end_y - start_y) # Adjusted calculation
            wm_end_x = start_x - pos_x + (end_x - start_x) # Adjusted calculation

            # Ensure calculated slice dimensions are positive
            if end_y > start_y and end_x > start_x and wm_end_y > wm_start_y and wm_end_x > wm_start_x:
                # Get alpha channel for the valid region of the watermark
                alpha = wm_array[wm_start_y:wm_end_y, wm_start_x:wm_end_x, 3:4]

                # Apply blending
                image_array[start_y:end_y, start_x:end_x] = self._blend_watermark(
                    image_array[start_y:end_y, start_x:end_x],
                    wm_array[wm_start_y:wm_end_y, wm_start_x:wm_end_x],
                    alpha,
                    blend_mode=None  # Random blend mode for each logo
                )

                # Update mask - using full opacity regardless of watermark opacity
                mask[start_y:end_y, start_x:end_x] = np.maximum(
                    mask[start_y:end_y, start_x:end_x],
                    (wm_array[wm_start_y:wm_end_y, wm_start_x:wm_end_x, 3] > 0).astype(float) # Check if alpha > 0
                )
        # --- End of 'logos' logic ---


        # Convert back to RGB
        image_array = np.clip(image_array, 0, 1)
        image_array = (image_array * 255).astype(np.uint8)
        image = Image.fromarray(image_array).convert('RGB')
        return np.array(image) / 255.0, mask

    def __getitem__(self, i):
        # Load image using the correct index for self.images_fps
        idx = i % len(self.images_fps) # Handle artificial length
        image = np.array(Image.open(self.images_fps[idx]).convert('RGB'))
        image = image.astype('float32') / 255.0

        # Apply watermarks first
        watermarked_image, mask = self._apply_watermarks(image)

        # Ensure proper dtype for watermarked_image and mask
        watermarked_image = watermarked_image.astype(np.float32)  # Convert to float32
        mask = mask.astype(np.float32)
        if self.dilate > 0:
            kernel = np.ones((self.dilate, self.dilate), np.uint8)
            # Ensure mask is uint8 for OpenCV dilation
            mask_uint8 = (mask * 255).astype(np.uint8)
            dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)
            mask = (dilated_mask / 255.0).astype(np.float32) # Convert back to float32

        mask = np.expand_dims(mask, axis=-1)

        # Apply augmentations to both image and mask if specified
        if self.augmentation:
            augmented = self.augmentation(image=watermarked_image, mask=mask)
            watermarked_image = augmented['image']
            mask = augmented['mask']

        # Ensure image is in (H, W, C) format before transpose
        if len(watermarked_image.shape) == 2:
            watermarked_image = np.expand_dims(watermarked_image, axis=-1)

        # Transpose to (C, H, W) format for PyTorch
        return watermarked_image.transpose(2, 0, 1), mask.transpose(2, 0, 1)

    def __len__(self):
        if self.artificial_len:
            return self.artificial_len
        return len(self.images_fps)