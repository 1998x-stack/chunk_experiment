# Pyramid Vision Transformer (PVT)

## Overview

The PVT model was proposed in [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/abs/2102.12122) by Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao. The PVT is a type of vision transformer that utilizes a pyramid structure to make it an effective backbone for dense prediction tasks. Specifically it allows for more fine-grained inputs (4 x 4 pixels per patch) to be used, while simultaneously shrinking the sequence length of the Transformer as it deepens - reducing the computational cost. Additionally, a spatial-reduction attention (SRA) layer is used to further reduce the resource consumption when learning high-resolution features.

The abstract from the paper is the following:

_Although convolutional neural networks (CNNs) have achieved great success in computer vision, this work investigates a simpler, convolution-free backbone network useful for many dense prediction tasks. Unlike the recently proposed Vision Transformer (ViT) that was designed for image classification specifically, we introduce the Pyramid Vision Transformer (PVT), which overcomes the difficulties of porting Transformer to various dense prediction tasks. PVT has several merits compared to current state of the arts. Different from ViT that typically yields low resolution outputs and incurs high computational and memory costs, PVT not only can be trained on dense partitions of an image to achieve high output resolution, which is important for dense prediction, but also uses a progressive shrinking pyramid to reduce the computations of large feature maps. PVT inherits the advantages of both CNN and Transformer, making it a unified backbone for various vision tasks without convolutions, where it can be used as a direct replacement for CNN backbones. We validate PVT through extensive experiments, showing that it boosts the performance of many downstream tasks, including object detection, instance and semantic segmentation. For example, with a comparable number of parameters, PVT+RetinaNet achieves 40.4 AP on the COCO dataset, surpassing ResNet50+RetinNet (36.3 AP) by 4.1 absolute AP (see Figure 2). We hope that PVT could serve as an alternative and useful backbone for pixel-level predictions and facilitate future research._

This model was contributed by \[Xrenya\](<[https://huggingface.co/Xrenya](https://huggingface.co/Xrenya)). The original code can be found [here](https://github.com/whai362/PVT).

-   PVTv1 on ImageNet-1K

| **Model variant** | **Size** | **Acc@1** | **Params (M)** |
| --- | --- | --- | --- |
| PVT-Tiny | 224 | 75.1 | 13.2 |
| PVT-Small | 224 | 79.8 | 24.5 |
| PVT-Medium | 224 | 81.2 | 44.2 |
| PVT-Large | 224 | 81.7 | 61.4 |

## PvtConfig

### class transformers.PvtConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pvt/configuration_pvt.py#L37)

( image\_size: int = 224 num\_channels: int = 3 num\_encoder\_blocks: int = 4 depths: typing.List\[int\] = \[2, 2, 2, 2\] sequence\_reduction\_ratios: typing.List\[int\] = \[8, 4, 2, 1\] hidden\_sizes: typing.List\[int\] = \[64, 128, 320, 512\] patch\_sizes: typing.List\[int\] = \[4, 2, 2, 2\] strides: typing.List\[int\] = \[4, 2, 2, 2\] num\_attention\_heads: typing.List\[int\] = \[1, 2, 5, 8\] mlp\_ratios: typing.List\[int\] = \[8, 8, 4, 4\] hidden\_act: typing.Mapping\[str, typing.Callable\] = 'gelu' hidden\_dropout\_prob: float = 0.0 attention\_probs\_dropout\_prob: float = 0.0 initializer\_range: float = 0.02 drop\_path\_rate: float = 0.0 layer\_norm\_eps: float = 1e-06 qkv\_bias: bool = True num\_labels: int = 1000 \*\*kwargs )

Parameters

-   **image\_size** (`int`, _optional_, defaults to 224) — The input image size
-   **num\_channels** (`int`, _optional_, defaults to 3) — The number of input channels.
-   **num\_encoder\_blocks** (`[int]`, _optional_., defaults to 4) — The number of encoder blocks (i.e. stages in the Mix Transformer encoder).
-   **depths** (`List[int]`, _optional_, defaults to `[2, 2, 2, 2]`) — The number of layers in each encoder block.
-   **sequence\_reduction\_ratios** (`List[int]`, _optional_, defaults to `[8, 4, 2, 1]`) — Sequence reduction ratios in each encoder block.
-   **hidden\_sizes** (`List[int]`, _optional_, defaults to `[64, 128, 320, 512]`) — Dimension of each of the encoder blocks.
-   **patch\_sizes** (`List[int]`, _optional_, defaults to `[4, 2, 2, 2]`) — Patch size before each encoder block.
-   **strides** (`List[int]`, _optional_, defaults to `[4, 2, 2, 2]`) — Stride before each encoder block.
-   **num\_attention\_heads** (`List[int]`, _optional_, defaults to `[1, 2, 5, 8]`) — Number of attention heads for each attention layer in each block of the Transformer encoder.
-   **mlp\_ratios** (`List[int]`, _optional_, defaults to `[8, 8, 4, 4]`) — Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the encoder blocks.
-   **hidden\_act** (`str` or `function`, _optional_, defaults to `"gelu"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
-   **hidden\_dropout\_prob** (`float`, _optional_, defaults to 0.0) — The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
-   **attention\_probs\_dropout\_prob** (`float`, _optional_, defaults to 0.0) — The dropout ratio for the attention probabilities.
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **drop\_path\_rate** (`float`, _optional_, defaults to 0.0) — The dropout probability for stochastic depth, used in the blocks of the Transformer encoder.
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-6) — The epsilon used by the layer normalization layers.
-   **qkv\_bias** (`bool`, _optional_, defaults to `True`) — Whether or not a learnable bias should be added to the queries, keys and values.
-   **num\_labels** (‘int’, _optional_, defaults to 1000) — The number of classes.

This is the configuration class to store the configuration of a [PvtModel](/docs/transformers/v4.34.0/en/model_doc/pvt#transformers.PvtModel). It is used to instantiate an Pvt model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Pvt [Xrenya/pvt-tiny-224](https://huggingface.co/Xrenya/pvt-tiny-224) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import PvtModel, PvtConfig

>>> 
>>> configuration = PvtConfig()

>>> 
>>> model = PvtModel(configuration)

>>> 
>>> configuration = model.config
```

## PvtImageProcessor

### class transformers.PvtImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pvt/image_processing_pvt.py#L41)

( do\_resize: bool = True size: typing.Union\[typing.Dict\[str, int\], NoneType\] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_rescale: bool = True rescale\_factor: typing.Union\[int, float\] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = None image\_std: typing.Union\[float, typing.List\[float\], NoneType\] = None \*\*kwargs )

Parameters

-   **do\_resize** (`bool`, _optional_, defaults to `True`) — Whether to resize the image’s (height, width) dimensions to the specified `(size["height"], size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
-   **size** (`dict`, _optional_, defaults to `{"height" -- 224, "width": 224}`): Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess` method.
-   **resample** (`PILImageResampling`, _optional_, defaults to `PILImageResampling.BILINEAR`) — Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the `preprocess` method.
-   **do\_rescale** (`bool`, _optional_, defaults to `True`) — Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale` parameter in the `preprocess` method.
-   **rescale\_factor** (`int` or `float`, _optional_, defaults to `1/255`) — Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the `preprocess` method.
-   **do\_normalize** (`bool`, _optional_, defaults to `True) -- Whether to normalize the image. Can be overridden by the` do\_normalize`parameter in the`preprocess\` method.
-   **image\_mean** (`float` or `List[float]`, _optional_, defaults to `IMAGENET_DEFAULT_MEAN`) — Mean to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
-   **image\_std** (`float` or `List[float]`, _optional_, defaults to `IMAGENET_DEFAULT_STD`) — Standard deviation to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.

Constructs a PVT image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pvt/image_processing_pvt.py#L147)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\] do\_resize: typing.Optional\[bool\] = None size: typing.Dict\[str, int\] = None resample: Resampling = None do\_rescale: typing.Optional\[bool\] = None rescale\_factor: typing.Optional\[float\] = None do\_normalize: typing.Optional\[bool\] = None image\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = None image\_std: typing.Union\[float, typing.List\[float\], NoneType\] = None return\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension\] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None \*\*kwargs )

Parameters

-   **images** (`ImageInput`) — Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.
-   **do\_resize** (`bool`, _optional_, defaults to `self.do_resize`) — Whether to resize the image.
-   **size** (`Dict[str, int]`, _optional_, defaults to `self.size`) — Dictionary in the format `{"height": h, "width": w}` specifying the size of the output image after resizing.
-   **resample** (`PILImageResampling` filter, _optional_, defaults to `self.resample`) — `PILImageResampling` filter to use if resizing the image e.g. `PILImageResampling.BILINEAR`. Only has an effect if `do_resize` is set to `True`.
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

## PvtForImageClassification

### class transformers.PvtForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pvt/modeling_pvt.py#L593)

( config: PvtConfig )

Parameters

-   **config** ([~PvtConfig](/docs/transformers/v4.34.0/en/model_doc/pvt#transformers.PvtConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Pvt Model transformer with an image classification head on top (a linear layer on top of the final hidden state of the \[CLS\] token) e.g. for ImageNet.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pvt/modeling_pvt.py#L608)

( pixel\_values: typing.Optional\[torch.Tensor\] labels: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [PvtImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

A [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([PvtConfig](/docs/transformers/v4.34.0/en/model_doc/pvt#transformers.PvtConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the model at the output of each stage.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [PvtForImageClassification](/docs/transformers/v4.34.0/en/model_doc/pvt#transformers.PvtForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, PvtForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("Zetatech/pvt-tiny-224")
>>> model = PvtForImageClassification.from_pretrained("Zetatech/pvt-tiny-224")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
tabby, tabby cat
```

## PvtModel

### class transformers.PvtModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pvt/modeling_pvt.py#L528)

( config: PvtConfig )

Parameters

-   **config** ([~PvtConfig](/docs/transformers/v4.34.0/en/model_doc/pvt#transformers.PvtConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Pvt encoder outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pvt/modeling_pvt.py#L547)

( pixel\_values: FloatTensor output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [PvtImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([PvtConfig](/docs/transformers/v4.34.0/en/model_doc/pvt#transformers.PvtConfig)) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [PvtModel](/docs/transformers/v4.34.0/en/model_doc/pvt#transformers.PvtModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, PvtModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("Zetatech/pvt-tiny-224")
>>> model = PvtModel.from_pretrained("Zetatech/pvt-tiny-224")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 50, 512]
```