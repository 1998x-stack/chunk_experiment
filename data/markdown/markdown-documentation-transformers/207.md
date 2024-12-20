# Mask2Former

## Overview

The Mask2Former model was proposed in [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527) by Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, Rohit Girdhar. Mask2Former is a unified framework for panoptic, instance and semantic segmentation and features significant performance and efficiency improvements over [MaskFormer](maskformer).

The abstract from the paper is the following:

_Image segmentation groups pixels with different semantics, e.g., category or instance membership. Each choice of semantics defines a task. While only the semantics of each task differ, current research focuses on designing specialized architectures for each task. We present Masked-attention Mask Transformer (Mask2Former), a new architecture capable of addressing any image segmentation task (panoptic, instance or semantic). Its key components include masked attention, which extracts localized features by constraining cross-attention within predicted mask regions. In addition to reducing the research effort by at least three times, it outperforms the best specialized architectures by a significant margin on four popular datasets. Most notably, Mask2Former sets a new state-of-the-art for panoptic segmentation (57.8 PQ on COCO), instance segmentation (50.1 AP on COCO) and semantic segmentation (57.7 mIoU on ADE20K)._

Tips:

-   Mask2Former uses the same preprocessing and postprocessing steps as [MaskFormer](maskformer). Use [Mask2FormerImageProcessor](/docs/transformers/v4.34.0/en/model_doc/mask2former#transformers.Mask2FormerImageProcessor) or [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor) to prepare images and optional targets for the model.
-   To get the final segmentation, depending on the task, you can call [post\_process\_semantic\_segmentation()](/docs/transformers/v4.34.0/en/model_doc/mask2former#transformers.Mask2FormerImageProcessor.post_process_semantic_segmentation) or [post\_process\_instance\_segmentation()](/docs/transformers/v4.34.0/en/model_doc/mask2former#transformers.Mask2FormerImageProcessor.post_process_instance_segmentation) or [post\_process\_panoptic\_segmentation()](/docs/transformers/v4.34.0/en/model_doc/mask2former#transformers.Mask2FormerImageProcessor.post_process_panoptic_segmentation). All three tasks can be solved using [Mask2FormerForUniversalSegmentation](/docs/transformers/v4.34.0/en/model_doc/mask2former#transformers.Mask2FormerForUniversalSegmentation) output, panoptic segmentation accepts an optional `label_ids_to_fuse` argument to fuse instances of the target object/s (e.g. sky) together.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/mask2former_architecture.jpg) Mask2Former architecture. Taken from the [original paper.](https://arxiv.org/abs/2112.01527)

This model was contributed by [Shivalika Singh](https://huggingface.co/shivi) and [Alara Dirik](https://huggingface.co/adirik). The original code can be found [here](https://github.com/facebookresearch/Mask2Former).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with Mask2Former.

-   Demo notebooks regarding inference + fine-tuning Mask2Former on custom data can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Mask2Former).

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we will review it. The resource should ideally demonstrate something new instead of duplicating an existing resource.

## MaskFormer specific outputs

### class transformers.models.mask2former.modeling\_mask2former.Mask2FormerModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mask2former/modeling_mask2former.py#L146)

( encoder\_last\_hidden\_state: FloatTensor = Nonepixel\_decoder\_last\_hidden\_state: FloatTensor = Nonetransformer\_decoder\_last\_hidden\_state: FloatTensor = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonepixel\_decoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonetransformer\_decoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonetransformer\_decoder\_intermediate\_states: typing.Tuple\[torch.FloatTensor\] = Nonemasks\_queries\_logits: typing.Tuple\[torch.FloatTensor\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Class for outputs of [Mask2FormerModel](/docs/transformers/v4.34.0/en/model_doc/mask2former#transformers.Mask2FormerModel). This class returns all the needed hidden states to compute the logits.

### class transformers.models.mask2former.modeling\_mask2former.Mask2FormerForUniversalSegmentationOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mask2former/modeling_mask2former.py#L192)

( loss: typing.Optional\[torch.FloatTensor\] = Noneclass\_queries\_logits: FloatTensor = Nonemasks\_queries\_logits: FloatTensor = Noneauxiliary\_logits: typing.Union\[typing.List\[typing.Dict\[str, torch.FloatTensor\]\], NoneType\] = Noneencoder\_last\_hidden\_state: FloatTensor = Nonepixel\_decoder\_last\_hidden\_state: FloatTensor = Nonetransformer\_decoder\_last\_hidden\_state: FloatTensor = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonepixel\_decoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonetransformer\_decoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Class for outputs of `Mask2FormerForUniversalSegmentationOutput`.

This output can be directly passed to [post\_process\_semantic\_segmentation()](/docs/transformers/v4.34.0/en/model_doc/mask2former#transformers.Mask2FormerImageProcessor.post_process_semantic_segmentation) or [post\_process\_instance\_segmentation()](/docs/transformers/v4.34.0/en/model_doc/mask2former#transformers.Mask2FormerImageProcessor.post_process_instance_segmentation) or [post\_process\_panoptic\_segmentation()](/docs/transformers/v4.34.0/en/model_doc/mask2former#transformers.Mask2FormerImageProcessor.post_process_panoptic_segmentation) to compute final segmentation maps. Please, see \[\`~Mask2FormerImageProcessor\] for details regarding usage.

## Mask2FormerConfig

### class transformers.Mask2FormerConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mask2former/configuration_mask2former.py#L33)

( backbone\_config: typing.Optional\[typing.Dict\] = Nonefeature\_size: int = 256mask\_feature\_size: int = 256hidden\_dim: int = 256encoder\_feedforward\_dim: int = 1024activation\_function: str = 'relu'encoder\_layers: int = 6decoder\_layers: int = 10num\_attention\_heads: int = 8dropout: float = 0.0dim\_feedforward: int = 2048pre\_norm: bool = Falseenforce\_input\_projection: bool = Falsecommon\_stride: int = 4ignore\_value: int = 255num\_queries: int = 100no\_object\_weight: float = 0.1class\_weight: float = 2.0mask\_weight: float = 5.0dice\_weight: float = 5.0train\_num\_points: int = 12544oversample\_ratio: float = 3.0importance\_sample\_ratio: float = 0.75init\_std: float = 0.02init\_xavier\_std: float = 1.0use\_auxiliary\_loss: bool = Truefeature\_strides: typing.List\[int\] = \[4, 8, 16, 32\]output\_auxiliary\_logits: bool = None\*\*kwargs )

This is the configuration class to store the configuration of a [Mask2FormerModel](/docs/transformers/v4.34.0/en/model_doc/mask2former#transformers.Mask2FormerModel). It is used to instantiate a Mask2Former model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Mask2Former [facebook/mask2former-swin-small-coco-instance](https://huggingface.co/facebook/mask2former-swin-small-coco-instance) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Currently, Mask2Former only supports the [Swin Transformer](swin) as backbone.

Examples:

```
>>> from transformers import Mask2FormerConfig, Mask2FormerModel

>>> 
>>> configuration = Mask2FormerConfig()

>>> 
>>> model = Mask2FormerModel(configuration)

>>> 
>>> configuration = model.config
```

Instantiate a [Mask2FormerConfig](/docs/transformers/v4.34.0/en/model_doc/mask2former#transformers.Mask2FormerConfig) (or a derived class) from a pre-trained backbone model configuration.

## Mask2FormerModel

### class transformers.Mask2FormerModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mask2former/modeling_mask2former.py#L2211)

( config: Mask2FormerConfig )

Parameters

-   **config** ([Mask2FormerConfig](/docs/transformers/v4.34.0/en/model_doc/mask2former#transformers.Mask2FormerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Mask2Former Model outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

The [Mask2FormerModel](/docs/transformers/v4.34.0/en/model_doc/mask2former#transformers.Mask2FormerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import torch
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoImageProcessor, Mask2FormerModel

>>> 
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> 
>>> image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")
>>> model = Mask2FormerModel.from_pretrained("facebook/mask2former-swin-small-coco-instance")
>>> inputs = image_processor(image, return_tensors="pt")

>>> 
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> 
>>> print(outputs.transformer_decoder_last_hidden_state.shape)
torch.Size([1, 100, 256])
```

## Mask2FormerForUniversalSegmentation

### class transformers.Mask2FormerForUniversalSegmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mask2former/modeling_mask2former.py#L2315)

( config: Mask2FormerConfig )

Parameters

-   **config** ([Mask2FormerConfig](/docs/transformers/v4.34.0/en/model_doc/mask2former#transformers.Mask2FormerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Mask2Former Model with heads on top for instance/semantic/panoptic segmentation. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mask2former/modeling_mask2former.py#L2368)

( pixel\_values: Tensormask\_labels: typing.Optional\[typing.List\[torch.Tensor\]\] = Noneclass\_labels: typing.Optional\[typing.List\[torch.Tensor\]\] = Nonepixel\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneoutput\_auxiliary\_logits: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.models.mask2former.modeling\_mask2former.Mask2FormerForUniversalSegmentationOutput](/docs/transformers/v4.34.0/en/model_doc/mask2former#transformers.models.mask2former.modeling_mask2former.Mask2FormerForUniversalSegmentationOutput) or `tuple(torch.FloatTensor)`

The [Mask2FormerForUniversalSegmentation](/docs/transformers/v4.34.0/en/model_doc/mask2former#transformers.Mask2FormerForUniversalSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

Instance segmentation example:

```
>>> from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
>>> from PIL import Image
>>> import requests
>>> import torch

>>> 
>>> image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")
>>> model = Mask2FormerForUniversalSegmentation.from_pretrained(
...     "facebook/mask2former-swin-small-coco-instance"
... )

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> 
>>> 
>>> class_queries_logits = outputs.class_queries_logits
>>> masks_queries_logits = outputs.masks_queries_logits

>>> 
>>> pred_instance_map = image_processor.post_process_semantic_segmentation(
...     outputs, target_sizes=[image.size[::-1]]
... )[0]
>>> print(pred_instance_map.shape)
torch.Size([480, 640])
```

Semantic segmentation example:

```
>>> from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
>>> from PIL import Image
>>> import requests
>>> import torch

>>> 
>>> image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
>>> model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic")

>>> url = (
...     "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
... )
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> 
>>> 
>>> class_queries_logits = outputs.class_queries_logits
>>> masks_queries_logits = outputs.masks_queries_logits

>>> 
>>> pred_semantic_map = image_processor.post_process_semantic_segmentation(
...     outputs, target_sizes=[image.size[::-1]]
... )[0]
>>> print(pred_semantic_map.shape)
torch.Size([512, 683])
```

Panoptic segmentation example:

```
>>> from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
>>> from PIL import Image
>>> import requests
>>> import torch

>>> 
>>> image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-panoptic")
>>> model = Mask2FormerForUniversalSegmentation.from_pretrained(
...     "facebook/mask2former-swin-small-cityscapes-panoptic"
... )

>>> url = "https://cdn-media.huggingface.co/Inference-API/Sample-results-on-the-Cityscapes-dataset-The-above-images-show-how-our-method-can-handle.png"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> 
>>> 
>>> class_queries_logits = outputs.class_queries_logits
>>> masks_queries_logits = outputs.masks_queries_logits

>>> 
>>> pred_panoptic_map = image_processor.post_process_panoptic_segmentation(
...     outputs, target_sizes=[image.size[::-1]]
... )[0]["segmentation"]
>>> print(pred_panoptic_map.shape)
torch.Size([338, 676])
```

## Mask2FormerImageProcessor

### class transformers.Mask2FormerImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mask2former/image_processing_mask2former.py#L343)

( do\_resize: bool = Truesize: typing.Dict\[str, int\] = Nonesize\_divisor: int = 32resample: Resampling = <Resampling.BILINEAR: 2>do\_rescale: bool = Truerescale\_factor: float = 0.00392156862745098do\_normalize: bool = Trueimage\_mean: typing.Union\[float, typing.List\[float\]\] = Noneimage\_std: typing.Union\[float, typing.List\[float\]\] = Noneignore\_index: typing.Optional\[int\] = Nonereduce\_labels: bool = False\*\*kwargs )

Constructs a Mask2Former image processor. The image processor can be used to prepare image(s) and optional targets for the model.

This image processor inherits from `BaseImageProcessor` which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mask2former/image_processing_mask2former.py#L671)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]segmentation\_maps: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\], NoneType\] = Noneinstance\_id\_to\_semantic\_id: typing.Union\[typing.Dict\[int, int\], NoneType\] = Nonedo\_resize: typing.Optional\[bool\] = Nonesize: typing.Union\[typing.Dict\[str, int\], NoneType\] = Nonesize\_divisor: typing.Optional\[int\] = Noneresample: Resampling = Nonedo\_rescale: typing.Optional\[bool\] = Nonerescale\_factor: typing.Optional\[float\] = Nonedo\_normalize: typing.Optional\[bool\] = Noneimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Noneignore\_index: typing.Optional\[int\] = Nonereduce\_labels: typing.Optional\[bool\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension\] = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs )

#### encode\_inputs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mask2former/image_processing_mask2former.py#L859)

( pixel\_values\_list: typing.List\[typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]\]segmentation\_maps: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\] = Noneinstance\_id\_to\_semantic\_id: typing.Union\[typing.List\[typing.Dict\[int, int\]\], typing.Dict\[int, int\], NoneType\] = Noneignore\_index: typing.Optional\[int\] = Nonereduce\_labels: bool = Falsereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Noneinput\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None ) → [BatchFeature](/docs/transformers/v4.34.0/en/main_classes/image_processor#transformers.BatchFeature)

Pad images up to the largest image in a batch and create a corresponding `pixel_mask`.

Mask2Former addresses semantic segmentation with a mask classification paradigm, thus input segmentation maps will be converted to lists of binary masks and their respective labels. Let’s see an example, assuming `segmentation_maps = [[2,6,7,9]]`, the output will contain `mask_labels = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]` (four binary masks) and `class_labels = [2,6,7,9]`, the labels for each mask.

#### post\_process\_semantic\_segmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mask2former/image_processing_mask2former.py#L962)

( outputstarget\_sizes: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], NoneType\] = None ) → `List[torch.Tensor]`

Parameters

-   **outputs** ([Mask2FormerForUniversalSegmentation](/docs/transformers/v4.34.0/en/model_doc/mask2former#transformers.Mask2FormerForUniversalSegmentation)) — Raw outputs of the model.
-   **target\_sizes** (`List[Tuple[int, int]]`, _optional_) — List of length (batch\_size), where each list item (`Tuple[int, int]]`) corresponds to the requested final size (height, width) of each prediction. If left to None, predictions will not be resized.

Returns

`List[torch.Tensor]`

A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [Mask2FormerForUniversalSegmentation](/docs/transformers/v4.34.0/en/model_doc/mask2former#transformers.Mask2FormerForUniversalSegmentation) into semantic segmentation maps. Only supports PyTorch.

#### post\_process\_instance\_segmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mask2former/image_processing_mask2former.py#L1017)

( outputsthreshold: float = 0.5mask\_threshold: float = 0.5overlap\_mask\_area\_threshold: float = 0.8target\_sizes: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], NoneType\] = Nonereturn\_coco\_annotation: typing.Optional\[bool\] = Falsereturn\_binary\_maps: typing.Optional\[bool\] = False ) → `List[Dict]`

Converts the output of `Mask2FormerForUniversalSegmentationOutput` into instance segmentation predictions. Only supports PyTorch.

#### post\_process\_panoptic\_segmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mask2former/image_processing_mask2former.py#L1136)

( outputsthreshold: float = 0.5mask\_threshold: float = 0.5overlap\_mask\_area\_threshold: float = 0.8label\_ids\_to\_fuse: typing.Optional\[typing.Set\[int\]\] = Nonetarget\_sizes: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], NoneType\] = None ) → `List[Dict]`

Converts the output of `Mask2FormerForUniversalSegmentationOutput` into image panoptic segmentation predictions. Only supports PyTorch.