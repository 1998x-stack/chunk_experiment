# Utilities for Image Processors

This page lists all the utility functions used by the image processors, mainly the functional transformations used to process the images.

Most of those are only useful if you are studying the code of the image processors in the library.

## Image Transformations

#### transformers.image\_transforms.center\_crop

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_transforms.py#L402)

( image: ndarray size: typing.Tuple\[int, int\] data\_format: typing.Union\[transformers.image\_utils.ChannelDimension, str, NoneType\] = None input\_data\_format: typing.Union\[transformers.image\_utils.ChannelDimension, str, NoneType\] = None return\_numpy: typing.Optional\[bool\] = None ) → `np.ndarray`

Parameters

-   **image** (`np.ndarray`) — The image to crop.
-   **size** (`Tuple[int, int]`) — The target size for the cropped image.
-   **data\_format** (`str` or `ChannelDimension`, _optional_) — The channel dimension format for the output image. Can be one of:
    
    -   `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
    -   `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format. If unset, will use the inferred format of the input image.
    
-   **input\_data\_format** (`str` or `ChannelDimension`, _optional_) — The channel dimension format for the input image. Can be one of:
    
    -   `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
    -   `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format. If unset, will use the inferred format of the input image.
    
-   **return\_numpy** (`bool`, _optional_) — Whether or not to return the cropped image as a numpy array. Used for backwards compatibility with the previous ImageFeatureExtractionMixin method.
    
    -   Unset: will return the same type as the input image.
    -   `True`: will return a numpy array.
    -   `False`: will return a `PIL.Image.Image` object.
    

The cropped image.

Crops the `image` to the specified `size` using a center crop. Note that if the image is too small to be cropped to the size given, it will be padded (so the returned result will always be of size `size`).

#### transformers.image\_transforms.center\_to\_corners\_format

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_transforms.py#L532)

( bboxes\_center: TensorType )

Converts bounding boxes from center format to corners format.

center format: contains the coordinate for the center of the box and its width, height dimensions (center\_x, center\_y, width, height) corners format: contains the coodinates for the top-left and bottom-right corners of the box (top\_left\_x, top\_left\_y, bottom\_right\_x, bottom\_right\_y)

#### transformers.image\_transforms.corners\_to\_center\_format

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_transforms.py#L592)

( bboxes\_corners: TensorType )

Converts bounding boxes from corners format to center format.

corners format: contains the coodinates for the top-left and bottom-right corners of the box (top\_left\_x, top\_left\_y, bottom\_right\_x, bottom\_right\_y) center format: contains the coordinate for the center of the box and its the width, height dimensions (center\_x, center\_y, width, height)

#### transformers.image\_transforms.id\_to\_rgb

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_transforms.py#L626)

( id\_map )

Converts unique ID to RGB color.

#### transformers.image\_transforms.normalize

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_transforms.py#L347)

( image: ndarray mean: typing.Union\[float, typing.Iterable\[float\]\] std: typing.Union\[float, typing.Iterable\[float\]\] data\_format: typing.Optional\[transformers.image\_utils.ChannelDimension\] = None input\_data\_format: typing.Union\[transformers.image\_utils.ChannelDimension, str, NoneType\] = None )

Parameters

-   **image** (`np.ndarray`) — The image to normalize.
-   **mean** (`float` or `Iterable[float]`) — The mean to use for normalization.
-   **std** (`float` or `Iterable[float]`) — The standard deviation to use for normalization.
-   **data\_format** (`ChannelDimension`, _optional_) — The channel dimension format of the output image. If unset, will use the inferred format from the input.
-   **input\_data\_format** (`ChannelDimension`, _optional_) — The channel dimension format of the input image. If unset, will use the inferred format from the input.

Normalizes `image` using the mean and standard deviation specified by `mean` and `std`.

image = (image - mean) / std

#### transformers.image\_transforms.pad

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_transforms.py#L656)

( image: ndarray padding: typing.Union\[int, typing.Tuple\[int, int\], typing.Iterable\[typing.Tuple\[int, int\]\]\] mode: PaddingMode = <PaddingMode.CONSTANT: 'constant'> constant\_values: typing.Union\[float, typing.Iterable\[float\]\] = 0.0 data\_format: typing.Union\[transformers.image\_utils.ChannelDimension, str, NoneType\] = None input\_data\_format: typing.Union\[transformers.image\_utils.ChannelDimension, str, NoneType\] = None ) → `np.ndarray`

Parameters

-   **image** (`np.ndarray`) — The image to pad.
-   **padding** (`int` or `Tuple[int, int]` or `Iterable[Tuple[int, int]]`) — Padding to apply to the edges of the height, width axes. Can be one of three formats:
    
    -   `((before_height, after_height), (before_width, after_width))` unique pad widths for each axis.
    -   `((before, after),)` yields same before and after pad for height and width.
    -   `(pad,)` or int is a shortcut for before = after = pad width for all axes.
    
-   **mode** (`PaddingMode`) — The padding mode to use. Can be one of:
    
    -   `"constant"`: pads with a constant value.
    -   `"reflect"`: pads with the reflection of the vector mirrored on the first and last values of the vector along each axis.
    -   `"replicate"`: pads with the replication of the last value on the edge of the array along each axis.
    -   `"symmetric"`: pads with the reflection of the vector mirrored along the edge of the array.
    
-   **constant\_values** (`float` or `Iterable[float]`, _optional_) — The value to use for the padding if `mode` is `"constant"`.
-   **data\_format** (`str` or `ChannelDimension`, _optional_) — The channel dimension format for the output image. Can be one of:
    
    -   `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
    -   `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format. If unset, will use same as the input image.
    
-   **input\_data\_format** (`str` or `ChannelDimension`, _optional_) — The channel dimension format for the input image. Can be one of:
    
    -   `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
    -   `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format. If unset, will use the inferred format of the input image.
    

The padded image.

Pads the `image` with the specified (height, width) `padding` and `mode`.

#### transformers.image\_transforms.rgb\_to\_id

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_transforms.py#L615)

( color )

Converts RGB color to unique ID.

#### transformers.image\_transforms.rescale

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_transforms.py#L92)

( image: ndarray scale: float data\_format: typing.Optional\[transformers.image\_utils.ChannelDimension\] = None dtype: dtype = <class 'numpy.float32'> input\_data\_format: typing.Union\[transformers.image\_utils.ChannelDimension, str, NoneType\] = None ) → `np.ndarray`

Parameters

-   **image** (`np.ndarray`) — The image to rescale.
-   **scale** (`float`) — The scale to use for rescaling the image.
-   **data\_format** (`ChannelDimension`, _optional_) — The channel dimension format of the image. If not provided, it will be the same as the input image.
-   **dtype** (`np.dtype`, _optional_, defaults to `np.float32`) — The dtype of the output image. Defaults to `np.float32`. Used for backwards compatibility with feature extractors.
-   **input\_data\_format** (`ChannelDimension`, _optional_) — The channel dimension format of the input image. If not provided, it will be inferred from the input image.

The rescaled image.

Rescales `image` by `scale`.

#### transformers.image\_transforms.resize

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_transforms.py#L276)

( image size: typing.Tuple\[int, int\] resample: PILImageResampling = None reducing\_gap: typing.Optional\[int\] = None data\_format: typing.Optional\[transformers.image\_utils.ChannelDimension\] = None return\_numpy: bool = True input\_data\_format: typing.Union\[transformers.image\_utils.ChannelDimension, str, NoneType\] = None ) → `np.ndarray`

Parameters

-   **image** (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`) — The image to resize.
-   **size** (`Tuple[int, int]`) — The size to use for resizing the image.
-   **resample** (`int`, _optional_, defaults to `PILImageResampling.BILINEAR`) — The filter to user for resampling.
-   **reducing\_gap** (`int`, _optional_) — Apply optimization by resizing the image in two steps. The bigger `reducing_gap`, the closer the result to the fair resampling. See corresponding Pillow documentation for more details.
-   **data\_format** (`ChannelDimension`, _optional_) — The channel dimension format of the output image. If unset, will use the inferred format from the input.
-   **return\_numpy** (`bool`, _optional_, defaults to `True`) — Whether or not to return the resized image as a numpy array. If False a `PIL.Image.Image` object is returned.
-   **input\_data\_format** (`ChannelDimension`, _optional_) — The channel dimension format of the input image. If unset, will use the inferred format from the input.

The resized image.

Resizes `image` to `(height, width)` specified by `size` using the PIL library.

#### transformers.image\_transforms.to\_pil\_image

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_transforms.py#L157)

( image: typing.Union\[numpy.ndarray, ForwardRef('PIL.Image.Image'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor'), ForwardRef('jnp.ndarray')\] do\_rescale: typing.Optional\[bool\] = None input\_data\_format: typing.Union\[transformers.image\_utils.ChannelDimension, str, NoneType\] = None ) → `PIL.Image.Image`

Parameters

-   **image** (`PIL.Image.Image` or `numpy.ndarray` or `torch.Tensor` or `tf.Tensor`) — The image to convert to the `PIL.Image` format.
-   **do\_rescale** (`bool`, _optional_) — Whether or not to apply the scaling factor (to make pixel values integers between 0 and 255). Will default to `True` if the image type is a floating type and casting to `int` would result in a loss of precision, and `False` otherwise.
-   **input\_data\_format** (`ChannelDimension`, _optional_) — The channel dimension format of the input image. If unset, will use the inferred format from the input.

The converted image.

Converts `image` to a PIL Image. Optionally rescales it and puts the channel dimension back as the last axis if needed.

## ImageProcessingMixin

### class transformers.ImageProcessingMixin

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_processing_utils.py#L68)

( \*\*kwargs )

This is an image processor mixin used to provide saving/loading functionality for sequential and image feature extractors.

#### fetch\_images

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_processing_utils.py#L517)

( image\_url\_or\_urls: typing.Union\[str, typing.List\[str\]\] )

Convert a single or a list of urls into the corresponding `PIL.Image` objects.

If a single url is passed, the return value will be a single object. If a list is passed a list of objects is returned.

#### from\_dict

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_processing_utils.py#L380)

( image\_processor\_dict: typing.Dict\[str, typing.Any\] \*\*kwargs ) → [ImageProcessingMixin](/docs/transformers/v4.34.0/en/main_classes/image_processor#transformers.ImageProcessingMixin)

Parameters

-   **image\_processor\_dict** (`Dict[str, Any]`) — Dictionary that will be used to instantiate the image processor object. Such a dictionary can be retrieved from a pretrained checkpoint by leveraging the [to\_dict()](/docs/transformers/v4.34.0/en/internal/image_processing_utils#transformers.ImageProcessingMixin.to_dict) method.
-   **kwargs** (`Dict[str, Any]`) — Additional parameters from which to initialize the image processor object.

The image processor object instantiated from those parameters.

Instantiates a type of [ImageProcessingMixin](/docs/transformers/v4.34.0/en/main_classes/image_processor#transformers.ImageProcessingMixin) from a Python dictionary of parameters.

#### from\_json\_file

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_processing_utils.py#L437)

( json\_file: typing.Union\[str, os.PathLike\] ) → A image processor of type [ImageProcessingMixin](/docs/transformers/v4.34.0/en/main_classes/image_processor#transformers.ImageProcessingMixin)

Parameters

-   **json\_file** (`str` or `os.PathLike`) — Path to the JSON file containing the parameters.

Returns

A image processor of type [ImageProcessingMixin](/docs/transformers/v4.34.0/en/main_classes/image_processor#transformers.ImageProcessingMixin)

The image\_processor object instantiated from that JSON file.

Instantiates a image processor of type [ImageProcessingMixin](/docs/transformers/v4.34.0/en/main_classes/image_processor#transformers.ImageProcessingMixin) from the path to a JSON file of parameters.

#### from\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_processing_utils.py#L92)

( pretrained\_model\_name\_or\_path: typing.Union\[str, os.PathLike\] cache\_dir: typing.Union\[str, os.PathLike, NoneType\] = None force\_download: bool = False local\_files\_only: bool = False token: typing.Union\[bool, str, NoneType\] = None revision: str = 'main' \*\*kwargs )

Parameters

-   **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) — This can be either:
    
    -   a string, the _model id_ of a pretrained image\_processor hosted inside a model repo on huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
    -   a path to a _directory_ containing a image processor file saved using the [save\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/image_processor#transformers.ImageProcessingMixin.save_pretrained) method, e.g., `./my_model_directory/`.
    -   a path or url to a saved image processor JSON _file_, e.g., `./my_model_directory/preprocessor_config.json`.
    
-   **cache\_dir** (`str` or `os.PathLike`, _optional_) — Path to a directory in which a downloaded pretrained model image processor should be cached if the standard cache should not be used.
-   **force\_download** (`bool`, _optional_, defaults to `False`) — Whether or not to force to (re-)download the image processor files and override the cached versions if they exist.
-   **resume\_download** (`bool`, _optional_, defaults to `False`) — Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
-   **proxies** (`Dict[str, str]`, _optional_) — A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
-   **token** (`str` or `bool`, _optional_) — The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
-   **revision** (`str`, _optional_, defaults to `"main"`) — The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git.

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

#### get\_image\_processor\_dict

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_processing_utils.py#L266)

( pretrained\_model\_name\_or\_path: typing.Union\[str, os.PathLike\] \*\*kwargs ) → `Tuple[Dict, Dict]`

Parameters

-   **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) — The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
-   **subfolder** (`str`, _optional_, defaults to `""`) — In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can specify the folder name here.

Returns

`Tuple[Dict, Dict]`

The dictionary(ies) that will be used to instantiate the image processor object.

From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a image processor of type `~image_processor_utils.ImageProcessingMixin` using `from_dict`.

#### push\_to\_hub

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/hub.py#L786)

( repo\_id: str use\_temp\_dir: typing.Optional\[bool\] = None commit\_message: typing.Optional\[str\] = None private: typing.Optional\[bool\] = None token: typing.Union\[bool, str, NoneType\] = None max\_shard\_size: typing.Union\[int, str, NoneType\] = '10GB' create\_pr: bool = False safe\_serialization: bool = False revision: str = None \*\*deprecated\_kwargs )

Parameters

-   **repo\_id** (`str`) — The name of the repository you want to push your image processor to. It should contain your organization name when pushing to a given organization.
-   **use\_temp\_dir** (`bool`, _optional_) — Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub. Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
-   **commit\_message** (`str`, _optional_) — Message to commit while pushing. Will default to `"Upload image processor"`.
-   **private** (`bool`, _optional_) — Whether or not the repository created should be private.
-   **token** (`bool` or `str`, _optional_) — The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url` is not specified.
-   **max\_shard\_size** (`int` or `str`, _optional_, defaults to `"10GB"`) — Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
-   **create\_pr** (`bool`, _optional_, defaults to `False`) — Whether or not to create a PR with the uploaded files or directly commit.
-   **safe\_serialization** (`bool`, _optional_, defaults to `False`) — Whether or not to convert the model weights in safetensors format for safer serialization.
-   **revision** (`str`, _optional_) — Branch to push the uploaded files to.

Upload the image processor file to the 🤗 Model Hub.

Examples:

```
from transformers import AutoImageProcessor

image processor = AutoImageProcessor.from_pretrained("bert-base-cased")


image processor.push_to_hub("my-finetuned-bert")


image processor.push_to_hub("huggingface/my-finetuned-bert")
```

#### register\_for\_auto\_class

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_processing_utils.py#L491)

( auto\_class = 'AutoImageProcessor' )

Parameters

-   **auto\_class** (`str` or `type`, _optional_, defaults to `"AutoImageProcessor "`) — The auto class to register this new image processor with.

Register this class with a given auto class. This should only be used for custom image processors as the ones in the library are already mapped with `AutoImageProcessor` .

This API is experimental and may have some slight breaking changes in the next releases.

#### save\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_processing_utils.py#L206)

( save\_directory: typing.Union\[str, os.PathLike\] push\_to\_hub: bool = False \*\*kwargs )

Parameters

-   **save\_directory** (`str` or `os.PathLike`) — Directory where the image processor JSON file will be saved (will be created if it does not exist).
-   **push\_to\_hub** (`bool`, _optional_, defaults to `False`) — Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with `repo_id` (will default to the name of `save_directory` in your namespace).
-   **kwargs** (`Dict[str, Any]`, _optional_) — Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/v4.34.0/en/main_classes/processors#transformers.ProcessorMixin.push_to_hub) method.

Save an image processor object to the directory `save_directory`, so that it can be re-loaded using the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/image_processor#transformers.ImageProcessingMixin.from_pretrained) class method.

#### to\_dict

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_processing_utils.py#L425)

( ) → `Dict[str, Any]`

Dictionary of all the attributes that make up this image processor instance.

Serializes this instance to a Python dictionary.

#### to\_json\_file

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_processing_utils.py#L477)

( json\_file\_path: typing.Union\[str, os.PathLike\] )

Parameters

-   **json\_file\_path** (`str` or `os.PathLike`) — Path to the JSON file in which this image\_processor instance’s parameters will be saved.

Save this instance to a JSON file.

#### to\_json\_string

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/image_processing_utils.py#L456)

( ) → `str`

String containing all the attributes that make up this feature\_extractor instance in JSON format.

Serializes this instance to a JSON string.