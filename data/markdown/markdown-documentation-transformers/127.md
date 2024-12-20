# Conditional DETR

## Overview

The Conditional DETR model was proposed in [Conditional DETR for Fast Training Convergence](https://arxiv.org/abs/2108.06152) by Depu Meng, Xiaokang Chen, Zejia Fan, Gang Zeng, Houqiang Li, Yuhui Yuan, Lei Sun, Jingdong Wang. Conditional DETR presents a conditional cross-attention mechanism for fast DETR training. Conditional DETR converges 6.7× to 10× faster than DETR.

The abstract from the paper is the following:

_The recently-developed DETR approach applies the transformer encoder and decoder architecture to object detection and achieves promising performance. In this paper, we handle the critical issue, slow training convergence, and present a conditional cross-attention mechanism for fast DETR training. Our approach is motivated by that the cross-attention in DETR relies highly on the content embeddings for localizing the four extremities and predicting the box, which increases the need for high-quality content embeddings and thus the training difficulty. Our approach, named conditional DETR, learns a conditional spatial query from the decoder embedding for decoder multi-head cross-attention. The benefit is that through the conditional spatial query, each cross-attention head is able to attend to a band containing a distinct region, e.g., one object extremity or a region inside the object box. This narrows down the spatial range for localizing the distinct regions for object classification and box regression, thus relaxing the dependence on the content embeddings and easing the training. Empirical results show that conditional DETR converges 6.7× faster for the backbones R50 and R101 and 10× faster for stronger backbones DC5-R50 and DC5-R101. Code is available at [https://github.com/Atten4Vis/ConditionalDETR](https://github.com/Atten4Vis/ConditionalDETR)._

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/conditional_detr_curve.jpg) Conditional DETR shows much faster convergence compared to the original DETR. Taken from the [original paper](https://arxiv.org/abs/2108.06152).

This model was contributed by [DepuMeng](https://huggingface.co/DepuMeng). The original code can be found [here](https://github.com/Atten4Vis/ConditionalDETR).

## Documentation resources

-   [Object detection task guide](../tasks/object_detection)

## ConditionalDetrConfig

### class transformers.ConditionalDetrConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/conditional_detr/configuration_conditional_detr.py#L36)

( use\_timm\_backbone = Truebackbone\_config = Nonenum\_channels = 3num\_queries = 300encoder\_layers = 6encoder\_ffn\_dim = 2048encoder\_attention\_heads = 8decoder\_layers = 6decoder\_ffn\_dim = 2048decoder\_attention\_heads = 8encoder\_layerdrop = 0.0decoder\_layerdrop = 0.0is\_encoder\_decoder = Trueactivation\_function = 'relu'd\_model = 256dropout = 0.1attention\_dropout = 0.0activation\_dropout = 0.0init\_std = 0.02init\_xavier\_std = 1.0auxiliary\_loss = Falseposition\_embedding\_type = 'sine'backbone = 'resnet50'use\_pretrained\_backbone = Truedilation = Falseclass\_cost = 2bbox\_cost = 5giou\_cost = 2mask\_loss\_coefficient = 1dice\_loss\_coefficient = 1cls\_loss\_coefficient = 2bbox\_loss\_coefficient = 5giou\_loss\_coefficient = 2focal\_alpha = 0.25\*\*kwargs )

This is the configuration class to store the configuration of a [ConditionalDetrModel](/docs/transformers/v4.34.0/en/model_doc/conditional_detr#transformers.ConditionalDetrModel). It is used to instantiate a Conditional DETR model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Conditional DETR [microsoft/conditional-detr-resnet-50](https://huggingface.co/microsoft/conditional-detr-resnet-50) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import ConditionalDetrConfig, ConditionalDetrModel

>>> 
>>> configuration = ConditionalDetrConfig()

>>> 
>>> model = ConditionalDetrModel(configuration)

>>> 
>>> configuration = model.config
```

## ConditionalDetrImageProcessor

### class transformers.ConditionalDetrImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/conditional_detr/image_processing_conditional_detr.py#L763)

( format: typing.Union\[str, transformers.models.conditional\_detr.image\_processing\_conditional\_detr.AnnotionFormat\] = <AnnotionFormat.COCO\_DETECTION: 'coco\_detection'>do\_resize: bool = Truesize: typing.Dict\[str, int\] = Noneresample: Resampling = <Resampling.BILINEAR: 2>do\_rescale: bool = Truerescale\_factor: typing.Union\[int, float\] = 0.00392156862745098do\_normalize: bool = Trueimage\_mean: typing.Union\[float, typing.List\[float\]\] = Noneimage\_std: typing.Union\[float, typing.List\[float\]\] = Nonedo\_pad: bool = True\*\*kwargs )

Constructs a Conditional Detr image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/conditional_detr/image_processing_conditional_detr.py#L1104)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]annotations: typing.Union\[typing.Dict\[str, typing.Union\[int, str, typing.List\[typing.Dict\]\]\], typing.List\[typing.Dict\[str, typing.Union\[int, str, typing.List\[typing.Dict\]\]\]\], NoneType\] = Nonereturn\_segmentation\_masks: bool = Nonemasks\_path: typing.Union\[str, pathlib.Path, NoneType\] = Nonedo\_resize: typing.Optional\[bool\] = Nonesize: typing.Union\[typing.Dict\[str, int\], NoneType\] = Noneresample = Nonedo\_rescale: typing.Optional\[bool\] = Nonerescale\_factor: typing.Union\[int, float, NoneType\] = Nonedo\_normalize: typing.Optional\[bool\] = Noneimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonedo\_pad: typing.Optional\[bool\] = Noneformat: typing.Union\[str, transformers.models.conditional\_detr.image\_processing\_conditional\_detr.AnnotionFormat, NoneType\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension\] = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs )

Preprocess an image or a batch of images so that it can be used by the model.

#### post\_process\_object\_detection

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/conditional_detr/image_processing_conditional_detr.py#L1396)

( outputsthreshold: float = 0.5target\_sizes: typing.Union\[transformers.utils.generic.TensorType, typing.List\[typing.Tuple\]\] = Nonetop\_k: int = 100 ) → `List[Dict]`

Parameters

-   **outputs** (`DetrObjectDetectionOutput`) — Raw outputs of the model.
-   **threshold** (`float`, _optional_) — Score threshold to keep object detection predictions.
-   **target\_sizes** (`torch.Tensor` or `List[Tuple[int, int]]`, _optional_) — Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size (height, width) of each image in the batch. If left to None, predictions will not be resized.
-   **top\_k** (`int`, _optional_, defaults to 100) — Keep only top k bounding boxes before filtering by thresholding.

A list of dictionaries, each dictionary containing the scores, labels and boxes for an image in the batch as predicted by the model.

Converts the raw output of [ConditionalDetrForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/conditional_detr#transformers.ConditionalDetrForObjectDetection) into final bounding boxes in (top\_left\_x, top\_left\_y, bottom\_right\_x, bottom\_right\_y) format. Only supports PyTorch.

#### post\_process\_instance\_segmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/conditional_detr/image_processing_conditional_detr.py#L1504)

( outputsthreshold: float = 0.5mask\_threshold: float = 0.5overlap\_mask\_area\_threshold: float = 0.8target\_sizes: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], NoneType\] = Nonereturn\_coco\_annotation: typing.Optional\[bool\] = False ) → `List[Dict]`

Converts the output of [ConditionalDetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation) into instance segmentation predictions. Only supports PyTorch.

#### post\_process\_semantic\_segmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/conditional_detr/image_processing_conditional_detr.py#L1455)

( outputstarget\_sizes: typing.List\[typing.Tuple\[int, int\]\] = None ) → `List[torch.Tensor]`

Parameters

-   **outputs** ([ConditionalDetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation)) — Raw outputs of the model.
-   **target\_sizes** (`List[Tuple[int, int]]`, _optional_) — A list of tuples (`Tuple[int, int]`) containing the target size (height, width) of each image in the batch. If unset, predictions will not be resized.

Returns

`List[torch.Tensor]`

A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [ConditionalDetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation) into semantic segmentation maps. Only supports PyTorch.

#### post\_process\_panoptic\_segmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/conditional_detr/image_processing_conditional_detr.py#L1589)

( outputsthreshold: float = 0.5mask\_threshold: float = 0.5overlap\_mask\_area\_threshold: float = 0.8label\_ids\_to\_fuse: typing.Optional\[typing.Set\[int\]\] = Nonetarget\_sizes: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], NoneType\] = None ) → `List[Dict]`

Converts the output of [ConditionalDetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation) into image panoptic segmentation predictions. Only supports PyTorch.

## ConditionalDetrFeatureExtractor

Preprocess an image or a batch of images.

( outputsthreshold: float = 0.5target\_sizes: typing.Union\[transformers.utils.generic.TensorType, typing.List\[typing.Tuple\]\] = Nonetop\_k: int = 100 ) → `List[Dict]`

Parameters

-   **outputs** (`DetrObjectDetectionOutput`) — Raw outputs of the model.
-   **threshold** (`float`, _optional_) — Score threshold to keep object detection predictions.
-   **target\_sizes** (`torch.Tensor` or `List[Tuple[int, int]]`, _optional_) — Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size (height, width) of each image in the batch. If left to None, predictions will not be resized.
-   **top\_k** (`int`, _optional_, defaults to 100) — Keep only top k bounding boxes before filtering by thresholding.

A list of dictionaries, each dictionary containing the scores, labels and boxes for an image in the batch as predicted by the model.

Converts the raw output of [ConditionalDetrForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/conditional_detr#transformers.ConditionalDetrForObjectDetection) into final bounding boxes in (top\_left\_x, top\_left\_y, bottom\_right\_x, bottom\_right\_y) format. Only supports PyTorch.

( outputsthreshold: float = 0.5mask\_threshold: float = 0.5overlap\_mask\_area\_threshold: float = 0.8target\_sizes: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], NoneType\] = Nonereturn\_coco\_annotation: typing.Optional\[bool\] = False ) → `List[Dict]`

Converts the output of [ConditionalDetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation) into instance segmentation predictions. Only supports PyTorch.

( outputstarget\_sizes: typing.List\[typing.Tuple\[int, int\]\] = None ) → `List[torch.Tensor]`

Parameters

-   **outputs** ([ConditionalDetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation)) — Raw outputs of the model.
-   **target\_sizes** (`List[Tuple[int, int]]`, _optional_) — A list of tuples (`Tuple[int, int]`) containing the target size (height, width) of each image in the batch. If unset, predictions will not be resized.

A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [ConditionalDetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation) into semantic segmentation maps. Only supports PyTorch.

( outputsthreshold: float = 0.5mask\_threshold: float = 0.5overlap\_mask\_area\_threshold: float = 0.8label\_ids\_to\_fuse: typing.Optional\[typing.Set\[int\]\] = Nonetarget\_sizes: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], NoneType\] = None ) → `List[Dict]`

Converts the output of [ConditionalDetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation) into image panoptic segmentation predictions. Only supports PyTorch.

## ConditionalDetrModel

### class transformers.ConditionalDetrModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/conditional_detr/modeling_conditional_detr.py#L1604)

( config: ConditionalDetrConfig )

Parameters

-   **config** ([ConditionalDetrConfig](/docs/transformers/v4.34.0/en/model_doc/conditional_detr#transformers.ConditionalDetrConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Conditional DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/conditional_detr/modeling_conditional_detr.py#L1638)

( pixel\_values: FloatTensorpixel\_mask: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Noneencoder\_outputs: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.conditional_detr.modeling_conditional_detr.ConditionalDetrModelOutput` or `tuple(torch.FloatTensor)`

The [ConditionalDetrModel](/docs/transformers/v4.34.0/en/model_doc/conditional_detr#transformers.ConditionalDetrModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, AutoModel
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50")
>>> model = AutoModel.from_pretrained("microsoft/conditional-detr-resnet-50")

>>> 
>>> inputs = image_processor(images=image, return_tensors="pt")

>>> 
>>> outputs = model(**inputs)

>>> 
>>> 
>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 300, 256]
```

## ConditionalDetrForObjectDetection

### class transformers.ConditionalDetrForObjectDetection

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/conditional_detr/modeling_conditional_detr.py#L1773)

( config: ConditionalDetrConfig )

Parameters

-   **config** ([ConditionalDetrConfig](/docs/transformers/v4.34.0/en/model_doc/conditional_detr#transformers.ConditionalDetrConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

CONDITIONAL\_DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on top, for tasks such as COCO detection.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/conditional_detr/modeling_conditional_detr.py#L1799)

( pixel\_values: FloatTensorpixel\_mask: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Noneencoder\_outputs: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[typing.List\[dict\]\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.conditional_detr.modeling_conditional_detr.ConditionalDetrObjectDetectionOutput` or `tuple(torch.FloatTensor)`

The [ConditionalDetrForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/conditional_detr#transformers.ConditionalDetrForObjectDetection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, AutoModelForObjectDetection
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50")
>>> model = AutoModelForObjectDetection.from_pretrained("microsoft/conditional-detr-resnet-50")

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)

>>> 
>>> target_sizes = torch.tensor([image.size[::-1]])
>>> results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[
...     0
... ]
>>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
...     box = [round(i, 2) for i in box.tolist()]
...     print(
...         f"Detected {model.config.id2label[label.item()]} with confidence "
...         f"{round(score.item(), 3)} at location {box}"
...     )
Detected remote with confidence 0.833 at location [38.31, 72.1, 177.63, 118.45]
Detected cat with confidence 0.831 at location [9.2, 51.38, 321.13, 469.0]
Detected cat with confidence 0.804 at location [340.3, 16.85, 642.93, 370.95]
Detected remote with confidence 0.683 at location [334.48, 73.49, 366.37, 190.01]
Detected couch with confidence 0.535 at location [0.52, 1.19, 640.35, 475.1]
```

## ConditionalDetrForSegmentation

### class transformers.ConditionalDetrForSegmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/conditional_detr/modeling_conditional_detr.py#L1961)

( config: ConditionalDetrConfig )

Parameters

-   **config** ([ConditionalDetrConfig](/docs/transformers/v4.34.0/en/model_doc/conditional_detr#transformers.ConditionalDetrConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

CONDITIONAL\_DETR Model (consisting of a backbone and encoder-decoder Transformer) with a segmentation head on top, for tasks such as COCO panoptic.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/conditional_detr/modeling_conditional_detr.py#L1983)

( pixel\_values: FloatTensorpixel\_mask: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Noneencoder\_outputs: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[typing.List\[dict\]\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.conditional_detr.modeling_conditional_detr.ConditionalDetrSegmentationOutput` or `tuple(torch.FloatTensor)`

The [ConditionalDetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import io
>>> import requests
>>> from PIL import Image
>>> import torch
>>> import numpy

>>> from transformers import (
...     AutoImageProcessor,
...     ConditionalDetrConfig,
...     ConditionalDetrForSegmentation,
... )
>>> from transformers.image_transforms import rgb_to_id

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50")

>>> 
>>> config = ConditionalDetrConfig()
>>> model = ConditionalDetrForSegmentation(config)

>>> 
>>> inputs = image_processor(images=image, return_tensors="pt")

>>> 
>>> outputs = model(**inputs)

>>> 
>>> 
>>> result = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=[(300, 500)])
>>> 
>>> panoptic_seg = result[0]["segmentation"]
>>> 
>>> panoptic_segments_info = result[0]["segments_info"]
```