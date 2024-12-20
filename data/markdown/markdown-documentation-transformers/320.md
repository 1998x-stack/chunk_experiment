# YOLOS

## Overview

The YOLOS model was proposed in [You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection](https://arxiv.org/abs/2106.00666) by Yuxin Fang, Bencheng Liao, Xinggang Wang, Jiemin Fang, Jiyang Qi, Rui Wu, Jianwei Niu, Wenyu Liu. YOLOS proposes to just leverage the plain [Vision Transformer (ViT)](vit) for object detection, inspired by DETR. It turns out that a base-sized encoder-only Transformer can also achieve 42 AP on COCO, similar to DETR and much more complex frameworks such as Faster R-CNN.

The abstract from the paper is the following:

_Can Transformer perform 2D object- and region-level recognition from a pure sequence-to-sequence perspective with minimal knowledge about the 2D spatial structure? To answer this question, we present You Only Look at One Sequence (YOLOS), a series of object detection models based on the vanilla Vision Transformer with the fewest possible modifications, region priors, as well as inductive biases of the target task. We find that YOLOS pre-trained on the mid-sized ImageNet-1k dataset only can already achieve quite competitive performance on the challenging COCO object detection benchmark, e.g., YOLOS-Base directly adopted from BERT-Base architecture can obtain 42.0 box AP on COCO val. We also discuss the impacts as well as limitations of current pre-train schemes and model scaling strategies for Transformer in vision through YOLOS._

Tips:

-   One can use [YolosImageProcessor](/docs/transformers/v4.34.0/en/model_doc/yolos#transformers.YolosImageProcessor) for preparing images (and optional targets) for the model. Contrary to [DETR](detr), YOLOS doesn’t require a `pixel_mask` to be created.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/yolos_architecture.png) YOLOS architecture. Taken from the [original paper](https://arxiv.org/abs/2106.00666).

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/hustvl/YOLOS).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with YOLOS.

-   All example notebooks illustrating inference + fine-tuning [YolosForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/yolos#transformers.YolosForObjectDetection) on a custom dataset can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/YOLOS).
-   See also: [Object detection task guide](../tasks/object_detection)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## YolosConfig

### class transformers.YolosConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/yolos/configuration_yolos.py#L35)

( hidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout\_prob = 0.0attention\_probs\_dropout\_prob = 0.0initializer\_range = 0.02layer\_norm\_eps = 1e-12image\_size = \[512, 864\]patch\_size = 16num\_channels = 3qkv\_bias = Truenum\_detection\_tokens = 100use\_mid\_position\_embeddings = Trueauxiliary\_loss = Falseclass\_cost = 1bbox\_cost = 5giou\_cost = 2bbox\_loss\_coefficient = 5giou\_loss\_coefficient = 2eos\_coefficient = 0.1\*\*kwargs )

This is the configuration class to store the configuration of a [YolosModel](/docs/transformers/v4.34.0/en/model_doc/yolos#transformers.YolosModel). It is used to instantiate a YOLOS model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the YOLOS [hustvl/yolos-base](https://huggingface.co/hustvl/yolos-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import YolosConfig, YolosModel

>>> 
>>> configuration = YolosConfig()

>>> 
>>> model = YolosModel(configuration)

>>> 
>>> configuration = model.config
```

## YolosImageProcessor

### class transformers.YolosImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/yolos/image_processing_yolos.py#L673)

( format: typing.Union\[str, transformers.models.yolos.image\_processing\_yolos.AnnotionFormat\] = <AnnotionFormat.COCO\_DETECTION: 'coco\_detection'>do\_resize: bool = Truesize: typing.Dict\[str, int\] = Noneresample: Resampling = <Resampling.BILINEAR: 2>do\_rescale: bool = Truerescale\_factor: typing.Union\[int, float\] = 0.00392156862745098do\_normalize: bool = Trueimage\_mean: typing.Union\[float, typing.List\[float\]\] = Noneimage\_std: typing.Union\[float, typing.List\[float\]\] = Nonedo\_pad: bool = True\*\*kwargs )

Constructs a Detr image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/yolos/image_processing_yolos.py#L1011)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]annotations: typing.Union\[typing.Dict\[str, typing.Union\[int, str, typing.List\[typing.Dict\]\]\], typing.List\[typing.Dict\[str, typing.Union\[int, str, typing.List\[typing.Dict\]\]\]\], NoneType\] = Nonereturn\_segmentation\_masks: bool = Nonemasks\_path: typing.Union\[str, pathlib.Path, NoneType\] = Nonedo\_resize: typing.Optional\[bool\] = Nonesize: typing.Union\[typing.Dict\[str, int\], NoneType\] = Noneresample = Nonedo\_rescale: typing.Optional\[bool\] = Nonerescale\_factor: typing.Union\[int, float, NoneType\] = Nonedo\_normalize: typing.Optional\[bool\] = Noneimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonedo\_pad: typing.Optional\[bool\] = Noneformat: typing.Union\[str, transformers.models.yolos.image\_processing\_yolos.AnnotionFormat, NoneType\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension\] = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs )

Preprocess an image or a batch of images so that it can be used by the model.

#### pad

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/yolos/image_processing_yolos.py#L956)

( images: typing.List\[numpy.ndarray\]constant\_values: typing.Union\[float, typing.Iterable\[float\]\] = 0return\_pixel\_mask: bool = Falsereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: typing.Optional\[transformers.image\_utils.ChannelDimension\] = Noneinput\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None )

Pads a batch of images to the bottom and right of the image with zeros to the size of largest height and width in the batch and optionally returns their corresponding pixel mask.

#### post\_process\_object\_detection

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/yolos/image_processing_yolos.py#L1294)

( outputsthreshold: float = 0.5target\_sizes: typing.Union\[transformers.utils.generic.TensorType, typing.List\[typing.Tuple\]\] = None ) → `List[Dict]`

Parameters

-   **outputs** (`YolosObjectDetectionOutput`) — Raw outputs of the model.
-   **threshold** (`float`, _optional_) — Score threshold to keep object detection predictions.
-   **target\_sizes** (`torch.Tensor` or `List[Tuple[int, int]]`, _optional_) — Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size `(height, width)` of each image in the batch. If unset, predictions will not be resized.

A list of dictionaries, each dictionary containing the scores, labels and boxes for an image in the batch as predicted by the model.

Converts the raw output of [YolosForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/yolos#transformers.YolosForObjectDetection) into final bounding boxes in (top\_left\_x, top\_left\_y, bottom\_right\_x, bottom\_right\_y) format. Only supports PyTorch.

## YolosFeatureExtractor

Preprocess an image or a batch of images.

( images: typing.List\[numpy.ndarray\]constant\_values: typing.Union\[float, typing.Iterable\[float\]\] = 0return\_pixel\_mask: bool = Falsereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: typing.Optional\[transformers.image\_utils.ChannelDimension\] = Noneinput\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None )

Pads a batch of images to the bottom and right of the image with zeros to the size of largest height and width in the batch and optionally returns their corresponding pixel mask.

( outputsthreshold: float = 0.5target\_sizes: typing.Union\[transformers.utils.generic.TensorType, typing.List\[typing.Tuple\]\] = None ) → `List[Dict]`

Parameters

-   **outputs** (`YolosObjectDetectionOutput`) — Raw outputs of the model.
-   **threshold** (`float`, _optional_) — Score threshold to keep object detection predictions.
-   **target\_sizes** (`torch.Tensor` or `List[Tuple[int, int]]`, _optional_) — Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size `(height, width)` of each image in the batch. If unset, predictions will not be resized.

A list of dictionaries, each dictionary containing the scores, labels and boxes for an image in the batch as predicted by the model.

Converts the raw output of [YolosForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/yolos#transformers.YolosForObjectDetection) into final bounding boxes in (top\_left\_x, top\_left\_y, bottom\_right\_x, bottom\_right\_y) format. Only supports PyTorch.

## YolosModel

### class transformers.YolosModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/yolos/modeling_yolos.py#L597)

( config: YolosConfigadd\_pooling\_layer: bool = True )

Parameters

-   **config** ([YolosConfig](/docs/transformers/v4.34.0/en/model_doc/yolos#transformers.YolosConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare YOLOS Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/yolos/modeling_yolos.py#L625)

( pixel\_values: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

The [YolosModel](/docs/transformers/v4.34.0/en/model_doc/yolos#transformers.YolosModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, YolosModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-small")
>>> model = YolosModel.from_pretrained("hustvl/yolos-small")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 3401, 384]
```

## YolosForObjectDetection

### class transformers.YolosForObjectDetection

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/yolos/modeling_yolos.py#L705)

( config: YolosConfig )

Parameters

-   **config** ([YolosConfig](/docs/transformers/v4.34.0/en/model_doc/yolos#transformers.YolosConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

YOLOS Model (consisting of a ViT encoder) with object detection heads on top, for tasks such as COCO detection.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/yolos/modeling_yolos.py#L732)

( pixel\_values: FloatTensorlabels: typing.Optional\[typing.List\[typing.Dict\]\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.yolos.modeling_yolos.YolosObjectDetectionOutput` or `tuple(torch.FloatTensor)`

The [YolosForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/yolos#transformers.YolosForObjectDetection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, AutoModelForObjectDetection
>>> import torch
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
>>> model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

>>> inputs = image_processor(images=image, return_tensors="pt")
>>> outputs = model(**inputs)

>>> 
>>> target_sizes = torch.tensor([image.size[::-1]])
>>> results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
...     0
... ]

>>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
...     box = [round(i, 2) for i in box.tolist()]
...     print(
...         f"Detected {model.config.id2label[label.item()]} with confidence "
...         f"{round(score.item(), 3)} at location {box}"
...     )
Detected remote with confidence 0.994 at location [46.96, 72.61, 181.02, 119.73]
Detected remote with confidence 0.975 at location [340.66, 79.19, 372.59, 192.65]
Detected cat with confidence 0.984 at location [12.27, 54.25, 319.42, 470.99]
Detected remote with confidence 0.922 at location [41.66, 71.96, 178.7, 120.33]
Detected cat with confidence 0.914 at location [342.34, 21.48, 638.64, 372.46]
```