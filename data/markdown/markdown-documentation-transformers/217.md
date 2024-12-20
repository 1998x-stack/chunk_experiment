# MobileNet V1

## Overview

The MobileNet model was proposed in [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) by Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam.

The abstract from the paper is the following:

_We present a class of efficient models called MobileNets for mobile and embedded vision applications. MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks. We introduce two simple global hyper-parameters that efficiently trade off between latency and accuracy. These hyper-parameters allow the model builder to choose the right sized model for their application based on the constraints of the problem. We present extensive experiments on resource and accuracy tradeoffs and show strong performance compared to other popular models on ImageNet classification. We then demonstrate the effectiveness of MobileNets across a wide range of applications and use cases including object detection, finegrain classification, face attributes and large scale geo-localization._

Tips:

-   The checkpoints are named **mobilenet\_v1\__depth_\__size_**, for example **mobilenet\_v1\_1.0\_224**, where **1.0** is the depth multiplier (sometimes also referred to as “alpha” or the width multiplier) and **224** is the resolution of the input images the model was trained on.
    
-   Even though the checkpoint is trained on images of specific size, the model will work on images of any size. The smallest supported image size is 32x32.
    
-   One can use [MobileNetV1ImageProcessor](/docs/transformers/v4.34.0/en/model_doc/mobilenet_v1#transformers.MobileNetV1ImageProcessor) to prepare images for the model.
    
-   The available image classification checkpoints are pre-trained on [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k) (also referred to as ILSVRC 2012, a collection of 1.3 million images and 1,000 classes). However, the model predicts 1001 classes: the 1000 classes from ImageNet plus an extra “background” class (index 0).
    
-   The original TensorFlow checkpoints use different padding rules than PyTorch, requiring the model to determine the padding amount at inference time, since this depends on the input image size. To use native PyTorch padding behavior, create a [MobileNetV1Config](/docs/transformers/v4.34.0/en/model_doc/mobilenet_v1#transformers.MobileNetV1Config) with `tf_padding = False`.
    

Unsupported features:

-   The [MobileNetV1Model](/docs/transformers/v4.34.0/en/model_doc/mobilenet_v1#transformers.MobileNetV1Model) outputs a globally pooled version of the last hidden state. In the original model it is possible to use a 7x7 average pooling layer with stride 2 instead of global pooling. For larger inputs, this gives a pooled output that is larger than 1x1 pixel. The HuggingFace implementation does not support this.
    
-   It is currently not possible to specify an `output_stride`. For smaller output strides, the original model invokes dilated convolution to prevent the spatial resolution from being reduced further. The output stride of the HuggingFace model is always 32.
    
-   The original TensorFlow checkpoints include quantized models. We do not support these models as they include additional “FakeQuantization” operations to unquantize the weights.
    
-   It’s common to extract the output from the pointwise layers at indices 5, 11, 12, 13 for downstream purposes. Using `output_hidden_states=True` returns the output from all intermediate layers. There is currently no way to limit this to specific layers.
    

This model was contributed by [matthijs](https://huggingface.co/Matthijs). The original code and weights can be found [here](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with MobileNetV1.

-   [MobileNetV1ForImageClassification](/docs/transformers/v4.34.0/en/model_doc/mobilenet_v1#transformers.MobileNetV1ForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
-   See also: [Image classification task guide](../tasks/image_classification)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## MobileNetV1Config

### class transformers.MobileNetV1Config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilenet_v1/configuration_mobilenet_v1.py#L36)

( num\_channels = 3 image\_size = 224 depth\_multiplier = 1.0 min\_depth = 8 hidden\_act = 'relu6' tf\_padding = True classifier\_dropout\_prob = 0.999 initializer\_range = 0.02 layer\_norm\_eps = 0.001 \*\*kwargs )

Parameters

-   **num\_channels** (`int`, _optional_, defaults to 3) — The number of input channels.
-   **image\_size** (`int`, _optional_, defaults to 224) — The size (resolution) of each image.
-   **depth\_multiplier** (`float`, _optional_, defaults to 1.0) — Shrinks or expands the number of channels in each layer. Default is 1.0, which starts the network with 32 channels. This is sometimes also called “alpha” or “width multiplier”.
-   **min\_depth** (`int`, _optional_, defaults to 8) — All layers will have at least this many channels.
-   **hidden\_act** (`str` or `function`, _optional_, defaults to `"relu6"`) — The non-linear activation function (function or string) in the Transformer encoder and convolution layers.
-   **tf\_padding** (`bool`, `optional`, defaults to `True`) — Whether to use TensorFlow padding rules on the convolution layers.
-   **classifier\_dropout\_prob** (`float`, _optional_, defaults to 0.999) — The dropout ratio for attached classifiers.
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 0.001) — The epsilon used by the layer normalization layers.

This is the configuration class to store the configuration of a [MobileNetV1Model](/docs/transformers/v4.34.0/en/model_doc/mobilenet_v1#transformers.MobileNetV1Model). It is used to instantiate a MobileNetV1 model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the MobileNetV1 [google/mobilenet\_v1\_1.0\_224](https://huggingface.co/google/mobilenet_v1_1.0_224) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import MobileNetV1Config, MobileNetV1Model

>>> 
>>> configuration = MobileNetV1Config()

>>> 
>>> model = MobileNetV1Model(configuration)

>>> 
>>> configuration = model.config
```

## MobileNetV1FeatureExtractor

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\] do\_resize: typing.Optional\[bool\] = None size: typing.Dict\[str, int\] = None resample: Resampling = None do\_center\_crop: bool = None crop\_size: typing.Dict\[str, int\] = None do\_rescale: typing.Optional\[bool\] = None rescale\_factor: typing.Optional\[float\] = None do\_normalize: typing.Optional\[bool\] = None image\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = None image\_std: typing.Union\[float, typing.List\[float\], NoneType\] = None return\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension\] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union\[transformers.image\_utils.ChannelDimension, str, NoneType\] = None \*\*kwargs )

Parameters

-   **images** (`ImageInput`) — Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.
-   **do\_resize** (`bool`, _optional_, defaults to `self.do_resize`) — Whether to resize the image.
-   **size** (`Dict[str, int]`, _optional_, defaults to `self.size`) — Size of the image after resizing. Shortest edge of the image is resized to size\[“shortest\_edge”\], with the longest edge resized to keep the input aspect ratio.
-   **resample** (`PILImageResampling` filter, _optional_, defaults to `self.resample`) — `PILImageResampling` filter to use if resizing the image e.g. `PILImageResampling.BILINEAR`. Only has an effect if `do_resize` is set to `True`.
-   **do\_center\_crop** (`bool`, _optional_, defaults to `self.do_center_crop`) — Whether to center crop the image.
-   **crop\_size** (`Dict[str, int]`, _optional_, defaults to `self.crop_size`) — Size of the center crop. Only has an effect if `do_center_crop` is set to `True`.
-   **do\_rescale** (`bool`, _optional_, defaults to `self.do_rescale`) — Whether to rescale the image values between \[0 - 1\].
-   **rescale\_factor** (`float`, _optional_, defaults to `self.rescale_factor`) — Rescale factor to rescale the image by if `do_rescale` is set to `True`.
-   **do\_normalize** (`bool`, _optional_, defaults to `self.do_normalize`) — Whether to normalize the image.
-   **image\_mean** (`float` or `List[float]`, _optional_, defaults to `self.image_mean`) — Image mean to use if `do_normalize` is set to `True`.
-   **image\_std** (`float` or `List[float]`, _optional_, defaults to `self.image_std`) — Image standard deviation to use if `do_normalize` is set to `True`.
-   **return\_tensors** (`str` or `TensorType`, _optional_) — The type of tensors to return. Can be one of:
    
    -   Unset: Return a list of `np.ndarray`.
    -   `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
    -   `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
    -   `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
    -   `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
    
-   **data\_format** (`ChannelDimension` or `str`, _optional_, defaults to `ChannelDimension.FIRST`) — The channel dimension format for the output image. Can be one of:
    
    -   `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
    -   `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
    -   Unset: Use the channel dimension format of the input image.
    
-   **input\_data\_format** (`ChannelDimension` or `str`, _optional_) — The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of:
    
    -   `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
    -   `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
    -   `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
    

Preprocess an image or batch of images.

## MobileNetV1ImageProcessor

### class transformers.MobileNetV1ImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilenet_v1/image_processing_mobilenet_v1.py#L45)

( do\_resize: bool = True size: typing.Union\[typing.Dict\[str, int\], NoneType\] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_center\_crop: bool = True crop\_size: typing.Dict\[str, int\] = None do\_rescale: bool = True rescale\_factor: typing.Union\[int, float\] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = None image\_std: typing.Union\[float, typing.List\[float\], NoneType\] = None \*\*kwargs )

Parameters

-   **do\_resize** (`bool`, _optional_, defaults to `True`) — Whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden by `do_resize` in the `preprocess` method.
-   **size** (`Dict[str, int]` _optional_, defaults to `{"shortest_edge" -- 256}`): Size of the image after resizing. The shortest edge of the image is resized to size\[“shortest\_edge”\], with the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess` method.
-   **resample** (`PILImageResampling`, _optional_, defaults to `PILImageResampling.BILINEAR`) — Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the `preprocess` method.
-   **do\_center\_crop** (`bool`, _optional_, defaults to `True`) — Whether to center crop the image. If the input size is smaller than `crop_size` along any edge, the image is padded with 0’s and then center cropped. Can be overridden by the `do_center_crop` parameter in the `preprocess` method.
-   **crop\_size** (`Dict[str, int]`, _optional_, defaults to `{"height" -- 224, "width": 224}`): Desired output size when applying center-cropping. Only has an effect if `do_center_crop` is set to `True`. Can be overridden by the `crop_size` parameter in the `preprocess` method.
-   **do\_rescale** (`bool`, _optional_, defaults to `True`) — Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale` parameter in the `preprocess` method.
-   **rescale\_factor** (`int` or `float`, _optional_, defaults to `1/255`) — Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the `preprocess` method. do\_normalize — Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
-   **image\_mean** (`float` or `List[float]`, _optional_, defaults to `IMAGENET_STANDARD_MEAN`) — Mean to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
-   **image\_std** (`float` or `List[float]`, _optional_, defaults to `IMAGENET_STANDARD_STD`) — Standard deviation to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.

Constructs a MobileNetV1 image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilenet_v1/image_processing_mobilenet_v1.py#L157)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\] do\_resize: typing.Optional\[bool\] = None size: typing.Dict\[str, int\] = None resample: Resampling = None do\_center\_crop: bool = None crop\_size: typing.Dict\[str, int\] = None do\_rescale: typing.Optional\[bool\] = None rescale\_factor: typing.Optional\[float\] = None do\_normalize: typing.Optional\[bool\] = None image\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = None image\_std: typing.Union\[float, typing.List\[float\], NoneType\] = None return\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension\] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union\[transformers.image\_utils.ChannelDimension, str, NoneType\] = None \*\*kwargs )

Parameters

-   **images** (`ImageInput`) — Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.
-   **do\_resize** (`bool`, _optional_, defaults to `self.do_resize`) — Whether to resize the image.
-   **size** (`Dict[str, int]`, _optional_, defaults to `self.size`) — Size of the image after resizing. Shortest edge of the image is resized to size\[“shortest\_edge”\], with the longest edge resized to keep the input aspect ratio.
-   **resample** (`PILImageResampling` filter, _optional_, defaults to `self.resample`) — `PILImageResampling` filter to use if resizing the image e.g. `PILImageResampling.BILINEAR`. Only has an effect if `do_resize` is set to `True`.
-   **do\_center\_crop** (`bool`, _optional_, defaults to `self.do_center_crop`) — Whether to center crop the image.
-   **crop\_size** (`Dict[str, int]`, _optional_, defaults to `self.crop_size`) — Size of the center crop. Only has an effect if `do_center_crop` is set to `True`.
-   **do\_rescale** (`bool`, _optional_, defaults to `self.do_rescale`) — Whether to rescale the image values between \[0 - 1\].
-   **rescale\_factor** (`float`, _optional_, defaults to `self.rescale_factor`) — Rescale factor to rescale the image by if `do_rescale` is set to `True`.
-   **do\_normalize** (`bool`, _optional_, defaults to `self.do_normalize`) — Whether to normalize the image.
-   **image\_mean** (`float` or `List[float]`, _optional_, defaults to `self.image_mean`) — Image mean to use if `do_normalize` is set to `True`.
-   **image\_std** (`float` or `List[float]`, _optional_, defaults to `self.image_std`) — Image standard deviation to use if `do_normalize` is set to `True`.
-   **return\_tensors** (`str` or `TensorType`, _optional_) — The type of tensors to return. Can be one of:
    
    -   Unset: Return a list of `np.ndarray`.
    -   `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
    -   `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
    -   `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
    -   `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
    
-   **data\_format** (`ChannelDimension` or `str`, _optional_, defaults to `ChannelDimension.FIRST`) — The channel dimension format for the output image. Can be one of:
    
    -   `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
    -   `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
    -   Unset: Use the channel dimension format of the input image.
    
-   **input\_data\_format** (`ChannelDimension` or `str`, _optional_) — The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of:
    
    -   `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
    -   `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
    -   `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
    

Preprocess an image or batch of images.

## MobileNetV1Model

### class transformers.MobileNetV1Model

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilenet_v1/modeling_mobilenet_v1.py#L301)

( config: MobileNetV1Config add\_pooling\_layer: bool = True )

Parameters

-   **config** ([MobileNetV1Config](/docs/transformers/v4.34.0/en/model_doc/mobilenet_v1#transformers.MobileNetV1Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare MobileNetV1 model outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilenet_v1/modeling_mobilenet_v1.py#L355)

( pixel\_values: typing.Optional\[torch.Tensor\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [MobileNetV1ImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([MobileNetV1Config](/docs/transformers/v4.34.0/en/model_doc/mobilenet_v1#transformers.MobileNetV1Config)) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state after a pooling operation on the spatial dimensions.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    

The [MobileNetV1Model](/docs/transformers/v4.34.0/en/model_doc/mobilenet_v1#transformers.MobileNetV1Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, MobileNetV1Model
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("google/mobilenet_v1_1.0_224")
>>> model = MobileNetV1Model.from_pretrained("google/mobilenet_v1_1.0_224")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 1024, 7, 7]
```

## MobileNetV1ForImageClassification

### class transformers.MobileNetV1ForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilenet_v1/modeling_mobilenet_v1.py#L411)

( config: MobileNetV1Config )

Parameters

-   **config** ([MobileNetV1Config](/docs/transformers/v4.34.0/en/model_doc/mobilenet_v1#transformers.MobileNetV1Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

MobileNetV1 model with an image classification head on top (a linear layer on top of the pooled features), e.g. for ImageNet.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilenet_v1/modeling_mobilenet_v1.py#L427)

( pixel\_values: typing.Optional\[torch.Tensor\] = None output\_hidden\_states: typing.Optional\[bool\] = None labels: typing.Optional\[torch.Tensor\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)`

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [MobileNetV1ImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss). If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

A [transformers.modeling\_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([MobileNetV1Config](/docs/transformers/v4.34.0/en/model_doc/mobilenet_v1#transformers.MobileNetV1Config)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at the output of each stage.

The [MobileNetV1ForImageClassification](/docs/transformers/v4.34.0/en/model_doc/mobilenet_v1#transformers.MobileNetV1ForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, MobileNetV1ForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("google/mobilenet_v1_1.0_224")
>>> model = MobileNetV1ForImageClassification.from_pretrained("google/mobilenet_v1_1.0_224")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
tabby, tabby cat
```