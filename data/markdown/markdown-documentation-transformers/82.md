# Image Processor

An image processor is in charge of preparing input features for vision models and post processing their outputs. This includes transformations such as resizing, normalization, and conversion to PyTorch, TensorFlow, Flax and Numpy tensors. It may also include model specific post-processing such as converting logits to segmentation masks.

## ImageProcessingMixin

### class transformers.ImageProcessingMixin

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_processing_utils.py#L68)

( \*\*kwargs )

This is an image processor mixin used to provide saving/loading functionality for sequential and image feature extractors.

#### from\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_processing_utils.py#L92)

( pretrained\_model\_name\_or\_path: typing.Union\[str, os.PathLike\]cache\_dir: typing.Union\[str, os.PathLike, NoneType\] = Noneforce\_download: bool = Falselocal\_files\_only: bool = Falsetoken: typing.Union\[bool, str, NoneType\] = Nonerevision: str = 'main'\*\*kwargs )

Instantiate a type of [ImageProcessingMixin](/docs/transformers/v4.34.0/en/main_classes/image_processor#transformers.ImageProcessingMixin) from an image processor.

Examples:

```

image_processor = CLIPImageProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)  
image_processor = CLIPImageProcessor.from_pretrained(
    "./test/saved_model/"
)  
image_processor = CLIPImageProcessor.from_pretrained("./test/saved_model/preprocessor_config.json")
image_processor = CLIPImageProcessor.from_pretrained(
    "openai/clip-vit-base-patch32", do_normalize=False, foo=False
)
assert image_processor.do_normalize is False
image_processor, unused_kwargs = CLIPImageProcessor.from_pretrained(
    "openai/clip-vit-base-patch32", do_normalize=False, foo=False, return_unused_kwargs=True
)
assert image_processor.do_normalize is False
assert unused_kwargs == {"foo": False}
```

#### save\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_processing_utils.py#L206)

( save\_directory: typing.Union\[str, os.PathLike\]push\_to\_hub: bool = False\*\*kwargs )

Parameters

-   **save\_directory** (`str` or `os.PathLike`) — Directory where the image processor JSON file will be saved (will be created if it does not exist).
-   **push\_to\_hub** (`bool`, _optional_, defaults to `False`) — Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with `repo_id` (will default to the name of `save_directory` in your namespace).
-   **kwargs** (`Dict[str, Any]`, _optional_) — Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/v4.34.0/en/main_classes/processors#transformers.ProcessorMixin.push_to_hub) method.

Save an image processor object to the directory `save_directory`, so that it can be re-loaded using the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/image_processor#transformers.ImageProcessingMixin.from_pretrained) class method.

## BatchFeature

### class transformers.BatchFeature

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/feature_extraction_utils.py#L61)

( data: typing.Union\[typing.Dict\[str, typing.Any\], NoneType\] = Nonetensor\_type: typing.Union\[NoneType, str, transformers.utils.generic.TensorType\] = None )

Parameters

-   **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call**/pad methods (‘input\_values’, ‘attention\_mask’, etc.).
-   **tensor\_type** (`Union[None, str, TensorType]`, _optional_) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at initialization.

Holds the output of the [pad()](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor.pad) and feature extractor specific `__call__` methods.

This class is derived from a python dictionary and can be used as a dictionary.

#### convert\_to\_tensors

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/feature_extraction_utils.py#L115)

( tensor\_type: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None )

Parameters

-   **tensor\_type** (`str` or [TensorType](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.TensorType), _optional_) — The type of tensors to use. If `str`, should be one of the values of the enum [TensorType](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.TensorType). If `None`, no modification is done.

Convert the inner content to tensors.

#### to

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/feature_extraction_utils.py#L188)

( \*args\*\*kwargs ) → [BatchFeature](/docs/transformers/v4.34.0/en/main_classes/image_processor#transformers.BatchFeature)

Parameters

-   **args** (`Tuple`) — Will be passed to the `to(...)` function of the tensors.
-   **kwargs** (`Dict`, _optional_) — Will be passed to the `to(...)` function of the tensors.

The same instance after modification.

Send all values to device by calling `v.to(*args, **kwargs)` (PyTorch only). This should support casting in different `dtypes` and sending the `BatchFeature` to a different `device`.

## BaseImageProcessor

### class transformers.image\_processing\_utils.BaseImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_processing_utils.py#L540)

( \*\*kwargs )

#### center\_crop

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_processing_utils.py#L620)

( image: ndarraysize: typing.Dict\[str, int\]data\_format: typing.Union\[transformers.image\_utils.ChannelDimension, str, NoneType\] = Noneinput\_data\_format: typing.Union\[transformers.image\_utils.ChannelDimension, str, NoneType\] = None\*\*kwargs )

Center crop an image to `(size["height"], size["width"])`. If the input size is smaller than `crop_size` along any edge, the image is padded with 0’s and then center cropped.

#### normalize

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_processing_utils.py#L583)

( image: ndarraymean: typing.Union\[float, typing.Iterable\[float\]\]std: typing.Union\[float, typing.Iterable\[float\]\]data\_format: typing.Union\[transformers.image\_utils.ChannelDimension, str, NoneType\] = Noneinput\_data\_format: typing.Union\[transformers.image\_utils.ChannelDimension, str, NoneType\] = None\*\*kwargs ) → `np.ndarray`

Normalize an image. image = (image - image\_mean) / image\_std.

#### rescale

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_processing_utils.py#L551)

( image: ndarrayscale: floatdata\_format: typing.Union\[transformers.image\_utils.ChannelDimension, str, NoneType\] = Noneinput\_data\_format: typing.Union\[transformers.image\_utils.ChannelDimension, str, NoneType\] = None\*\*kwargs ) → `np.ndarray`

Rescale an image by a scale factor. image = image \* scale.