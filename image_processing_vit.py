# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for ViT."""

import sys
import os
import torch

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))
from char_splitting.char_splitting import Character_Splitting

from typing import Dict, List, Optional, Union

import numpy as np
import random
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from transformers.image_transforms import convert_to_rgb, resize, to_channel_dimension_format
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from transformers.utils import TensorType, filter_out_non_signature_kwargs, logging


logger = logging.get_logger(__name__)


class ViTImageProcessor(BaseImageProcessor):
    r"""
    Constructs a ViT image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `(size["height"],
            size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*):
            Whether to convert the image to RGB.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 224, "width": 224}
        size = get_size_dict(size)
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.do_convert_rgb = do_convert_rgb

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        size = get_size_dict(size)
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        output_size = (size["height"], size["width"])
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        do_convert_rgb: Optional[bool] = None,
    ):
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Dictionary in the format `{"height": h, "width": w}` specifying the size of the output image after
                resizing.
            resample (`PILImageResampling` filter, *optional*, defaults to `self.resample`):
                `PILImageResampling` filter to use if resizing the image e.g. `PILImageResampling.BILINEAR`. Only has
                an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use if `do_normalize` is set to `True`.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
        """


        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        resample = resample if resample is not None else self.resample
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        size = size if size is not None else self.size
        size_dict = get_size_dict(size)

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]
        
        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        if do_resize:
            images = [
                self.resize(image=image, size=size_dict, resample=resample, input_data_format=input_data_format)
                for image in images
            ]

        if do_rescale:
            images = [
                self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                for image in images
            ]

        if do_normalize:
            images = [
                self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
                for image in images
            ]

        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]

        data = {"pixel_values": images}
        for i in images:
            print(i.shape)
        return BatchFeature(data=data, tensor_type=return_tensors)

class CharImageProcessor(BaseImageProcessor):
    r"""
    Constructs a ViT image processor.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        min_area_character = 10,
        is_concatenate_character = False,
        character_size = [64,64],
        max_text_length = 25,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 224, "width": 224}
        size = get_size_dict(size)
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.do_convert_rgb = do_convert_rgb
        self.min_area_character = min_area_character
        self.char_splitter = Character_Splitting(self.min_area_character)
        self.is_concatenate_character = is_concatenate_character
        self.character_size = character_size
        self.max_text_length = max_text_length
        
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        size = get_size_dict(size)
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        output_size = (size["height"], size["width"])
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
    

    def get_char_images(self,image = None):
        if image is None:
            raise ValueError("Image input cannot be None")
        
        _, height, width = image.shape  # Lấy chiều cao và chiều rộng của ảnh
        
        cropped_images = []
        


        # for _ in range(5):  # Cắt 5 ảnh
        #     crop_h = random.randint(height // 4, height)  # Chiều cao ngẫu nhiên từ 1/4 đến full height
        #     crop_w = random.randint(width // 4, width//2)  # Chiều rộng ngẫu nhiên từ 1/4 đến full width

        #     # Chọn ngẫu nhiên điểm bắt đầu để crop
        #     start_h = random.randint(0, height - crop_h)
        #     start_w = random.randint(0, width - crop_w)

        #     cropped_image = image[:, start_h:start_h + crop_h, start_w:start_w + crop_w]
        #     cropped_images.append(cropped_image)
        # cropped_images.append(image[:,:,0:38])
        # cropped_images.append(image[:,:,40:50])
        # cropped_images.append(image[:,:,50:75])
        # cropped_images.append(image[:,:,75:95])
        # cropped_images.append(image[:,:,95:135])
    
        # return cropped_images
        boxes = self.char_splitter.extract_characters(image)
        img_for_cropping = np.transpose(image, (1, 2, 0))
        cropped_characters = []
        for (x1, y1, x2, y2) in boxes:
            crop = img_for_cropping[y1:y2, x1:x2]
            crop = np.transpose(crop, (2,0,1))
            cropped_characters.append(crop)

        return cropped_characters

        # return [np.random.rand(3, 10,30),np.random.rand(3, 15,20), np.random.rand(3, 20,40), np.random.rand(3, 16,48),np.random.rand(3, 27,48) ]
    
    def _padding_square_image(self, image, value=255):
        """
        Pads the smaller dimension of the image to make it square.
        
        Parameters:
        image (numpy.ndarray): Input image of shape (C, H, W)
        value (int or float): Padding value to use.
        
        Returns:
        numpy.ndarray: Square padded image.
        """
        C, H, W = image.shape
        
        if H == W:
            return image  # Already square
        
        # Determine padding sizes
        if H > W:
            pad_left = (H - W) // 2
            pad_right = (H - W) - pad_left
            padding = ((0, 0), (0, 0), (pad_left, pad_right))
        else:
            pad_top = (W - H) // 2
            pad_bottom = (W - H) - pad_top
            padding = ((0, 0), (pad_top, pad_bottom), (0, 0))
        
        # Apply padding
        padded_image = np.pad(image, padding, mode='constant', constant_values=value)
        
        return padded_image

    def padding_images(
        self,
        images,
        value = 255
    ):
        padded_images = []
        for image in images:
            padded_images.append(self._padding_square_image(image,value=value))
        return padded_images
    
    def concatenate_image(
        self, 
        images, 
        image_size = None,
        padding_value = 255,
        ):
        # Resize all images to char_image_size
        
        
        # Create a blank (white) image with the specified size (image_size)
        blank_image = np.ones((3, image_size[0], image_size[1]))*padding_value  # (Channels, Height, Width), white image
        current_x = 0
        # Iterate over the resized images and place them into the blank image
        for img in images:
            # Check if there is enough space in the blank image to place the resized image
            if current_x + img.shape[2] > image_size[1]:
                raise ValueError("Not enough width in image_size to fit all resized images.")
            
            # Insert the resized image into the blank image at the current position
            blank_image[:, :, current_x:current_x + img.shape[2]] = img #np.transpose(img, (1, 2, 0))  # Convert (Channels, Height, Width) -> (Height, Width, Channels)
            
            # Update the x position to the right after the current image
            current_x += img.shape[2]
    
        return blank_image
    def create_char_images(
        self,
        image = None,
        char_image_size =  [16,16], 
        image_size = [16,96],
        resample = None,
        input_data_format = None,
        padding_value = 255
        ):
        print(image.shape)
        char_images = self.get_char_images(image)

        padded_images = self.padding_images(char_images, value=padding_value)
        resized_images = [
            self.resize(image=image, size=char_image_size, resample=resample, input_data_format=input_data_format)
            for image in padded_images
        ]
        res = self.concatenate_image(resized_images,image_size = image_size)
        return res
        # return res, padded_images, char_images
    def format_input(
        self,
        images,
    ):
        converted_images = []
        for img in images:
            if img.ndim != 3:
                raise ValueError(f"Each image must be 3-dimensional (channels, height, width) or (height, width, channels) but found {img.ndim}.")
            if img.shape[0] == 3:
                converted_images.append(img)
            elif img.shape[-1] == 3:
                # Transpose from (height, width, channels) to (channels, height, width).
                img_converted = np.transpose(img, (2, 0, 1))
                converted_images.append(img_converted)
            else:
                raise ValueError("Image does not contain exactly 3 channels.")
        return converted_images
    @filter_out_non_signature_kwargs(extra = ["is_concatenate_character"])
    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        do_convert_rgb: Optional[bool] = None,
        is_concatenate_character = None,
        character_size = None,
        max_text_length = None,
    ):
        
        
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        resample = resample if resample is not None else self.resample
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        character_size = character_size if character_size is not None else self.character_size
        max_text_length = max_text_length if max_text_length is not None else self.max_text_length
        is_concatenate_character = is_concatenate_character if is_concatenate_character is not None else self.is_concatenate_character

        size = size if size is not None else self.size
        size_dict = get_size_dict(size)

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        images = self.format_input(images)

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])
        
        #----------------------------------------------------------------

        
        # if do_resize:
        #     images = [
        #         self.resize(image=image, size=size_dict, resample=resample, input_data_format=input_data_format)
        #         for image in images
        #     ]
        image_size = [character_size[0], character_size[1]*max_text_length]
        char_images = [
            self.create_char_images(image=image, image_size = image_size,char_image_size = character_size , resample=resample, input_data_format=input_data_format)\
            for image in images
        ]
        if do_rescale:
            images = [
                self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                for image in char_images
            ]

        if do_normalize:
            images = [
                self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
                for image in images
            ]

        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]

        B,S = len(images),max_text_length
        C,H,W = images[0].shape
        # print(is_concatenate_character)
        if not is_concatenate_character:
            images = torch.Tensor(images)
            images = images.reshape([B,C,H,S,W//S])
            images = images.permute(0, 3, 1, 2, 4)
            images = images.contiguous().view(B * S, C,H,W//S)
        
        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)
__all__ = ["ViTImageProcessor", "CharImageProcessor"]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import Image
    import cv2
    image_processor = CharImageProcessor(do_normalize=True, do_rescale=True, do_resize=True, image_mean=[0.5,0.5,0.5],
                                    image_std = [0.5,0.5,0.5], resample=2, size=384,
                                    )
    # res = image_processor.create_char_images(np.random.rand(3, 488,512))
    # image = Image.open("/home/app/ocr/kientdt/CharOCR/test_images/GSK3.png").convert("RGB")
    image = cv2.imread("/home/app/ocr/kientdt/CharOCR/test_images/GSK3.png")
    image = np.transpose(image, (2,0,1))
    device = 'cuda'
    pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
    print(pixel_values.shape)