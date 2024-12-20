# Feature Extractor

A feature extractor is in charge of preparing input features for audio or vision models. This includes feature extraction from sequences, _e.g._, pre-processing audio files to Log-Mel Spectrogram features, feature extraction from images _e.g._ cropping image image files, but also padding, normalization, and conversion to Numpy, PyTorch, and TensorFlow tensors.

## FeatureExtractionMixin

This is a feature extraction mixin used to provide saving/loading functionality for sequential and image feature extractors.

( pretrained\_model\_name\_or\_path: typing.Union\[str, os.PathLike\]cache\_dir: typing.Union\[str, os.PathLike, NoneType\] = Noneforce\_download: bool = Falselocal\_files\_only: bool = Falsetoken: typing.Union\[bool, str, NoneType\] = Nonerevision: str = 'main'\*\*kwargs )

Instantiate a type of [FeatureExtractionMixin](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin) from a feature extractor, _e.g._ a derived class of [SequenceFeatureExtractor](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor).

Examples:

```

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)  
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "./test/saved_model/"
)  
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("./test/saved_model/preprocessor_config.json")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/wav2vec2-base-960h", return_attention_mask=False, foo=False
)
assert feature_extractor.return_attention_mask is False
feature_extractor, unused_kwargs = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/wav2vec2-base-960h", return_attention_mask=False, foo=False, return_unused_kwargs=True
)
assert feature_extractor.return_attention_mask is False
assert unused_kwargs == {"foo": False}
```

( save\_directory: typing.Union\[str, os.PathLike\]push\_to\_hub: bool = False\*\*kwargs )

Parameters

-   **save\_directory** (`str` or `os.PathLike`) — Directory where the feature extractor JSON file will be saved (will be created if it does not exist).
-   **push\_to\_hub** (`bool`, _optional_, defaults to `False`) — Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with `repo_id` (will default to the name of `save_directory` in your namespace).
-   **kwargs** (`Dict[str, Any]`, _optional_) — Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/v4.34.0/en/main_classes/processors#transformers.ProcessorMixin.push_to_hub) method.

Save a feature\_extractor object to the directory `save_directory`, so that it can be re-loaded using the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained) class method.

## SequenceFeatureExtractor

( feature\_size: intsampling\_rate: intpadding\_value: float\*\*kwargs )

Parameters

-   **feature\_size** (`int`) — The feature dimension of the extracted features.
-   **sampling\_rate** (`int`) — The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
-   **padding\_value** (`float`) — The value that is used to fill the padding values / vectors.

This is a general feature extraction class for speech recognition.

( processed\_features: typing.Union\[transformers.feature\_extraction\_utils.BatchFeature, typing.List\[transformers.feature\_extraction\_utils.BatchFeature\], typing.Dict\[str, transformers.feature\_extraction\_utils.BatchFeature\], typing.Dict\[str, typing.List\[transformers.feature\_extraction\_utils.BatchFeature\]\], typing.List\[typing.Dict\[str, transformers.feature\_extraction\_utils.BatchFeature\]\]\]padding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = Truemax\_length: typing.Optional\[int\] = Nonetruncation: bool = Falsepad\_to\_multiple\_of: typing.Optional\[int\] = Nonereturn\_attention\_mask: typing.Optional\[bool\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None )

Pad input values / input vectors or a batch of input values / input vectors up to predefined length or to the max sequence length in the batch.

Padding side (left/right) padding values are defined at the feature extractor level (with `self.padding_side`, `self.padding_value`)

If the `processed_features` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of PyTorch tensors, you will lose the specific device of your tensors however.

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

## ImageFeatureExtractionMixin

Mixin that contain utilities for preparing image features.

( imagesize ) → new\_image

Parameters

-   **image** (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor` of shape (n\_channels, height, width) or (height, width, n\_channels)) — The image to resize.
-   **size** (`int` or `Tuple[int, int]`) — The size to which crop the image.

A center cropped `PIL.Image.Image` or `np.ndarray` or `torch.Tensor` of shape: (n\_channels, height, width).

Crops `image` to the given size using a center crop. Note that if the image is too small to be cropped to the size given, it will be padded (so the returned result has the size asked).

( image )

Parameters

-   **image** (`PIL.Image.Image`) — The image to convert.

Converts `PIL.Image.Image` to RGB format.

#### expand\_dims

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_utils.py#L421)

( image )

Parameters

-   **image** (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`) — The image to expand.

Expands 2-dimensional `image` to 3 dimensions.

( image )

Parameters

-   **image** (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`) — The image whose color channels to flip. If `np.ndarray` or `torch.Tensor`, the channel dimension should be first.

Flips the channel order of `image` from RGB to BGR, or vice versa. Note that this will trigger a conversion of `image` to a NumPy array if it’s a PIL Image.

( imagemeanstdrescale = False )

Parameters

-   **image** (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`) — The image to normalize.
-   **mean** (`List[float]` or `np.ndarray` or `torch.Tensor`) — The mean (per channel) to use for normalization.
-   **std** (`List[float]` or `np.ndarray` or `torch.Tensor`) — The standard deviation (per channel) to use for normalization.
-   **rescale** (`bool`, _optional_, defaults to `False`) — Whether or not to rescale the image to be between 0 and 1. If a PIL image is provided, scaling will happen automatically.

Normalizes `image` with `mean` and `std`. Note that this will trigger a conversion of `image` to a NumPy array if it’s a PIL Image.

( image: ndarrayscale: typing.Union\[float, int\] )

Rescale a numpy image by scale amount

( imagesizeresample = Nonedefault\_to\_square = Truemax\_size = None ) → image

Resizes `image`. Enforces conversion of input to PIL.Image.

( imageangleresample = Noneexpand = 0center = Nonetranslate = Nonefillcolor = None ) → image

Parameters

-   **image** (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`) — The image to rotate. If `np.ndarray` or `torch.Tensor`, will be converted to `PIL.Image.Image` before rotating.

A rotated `PIL.Image.Image`.

Returns a rotated copy of `image`. This method returns a copy of `image`, rotated the given number of degrees counter clockwise around its centre.

( imagerescale = Nonechannel\_first = True )

Parameters

-   **image** (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`) — The image to convert to a NumPy array.
-   **rescale** (`bool`, _optional_) — Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.). Will default to `True` if the image is a PIL Image or an array/tensor of integers, `False` otherwise.
-   **channel\_first** (`bool`, _optional_, defaults to `True`) — Whether or not to permute the dimensions of the image to put the channel dimension first.

Converts `image` to a numpy array. Optionally rescales it and puts the channel dimension as the first dimension.

( imagerescale = None )

Parameters

-   **image** (`PIL.Image.Image` or `numpy.ndarray` or `torch.Tensor`) — The image to convert to the PIL Image format.
-   **rescale** (`bool`, _optional_) — Whether or not to apply the scaling factor (to make pixel values integers between 0 and 255). Will default to `True` if the image type is a floating type, `False` otherwise.

Converts `image` to a PIL Image. Optionally rescales it and puts the channel dimension back as the last axis if needed.