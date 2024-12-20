# Deformable DETR

## Overview

The Deformable DETR model was proposed in [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159) by Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, Jifeng Dai. Deformable DETR mitigates the slow convergence issues and limited feature spatial resolution of the original [DETR](detr) by leveraging a new deformable attention module which only attends to a small set of key sampling points around a reference.

The abstract from the paper is the following:

_DETR has been recently proposed to eliminate the need for many hand-designed components in object detection while demonstrating good performance. However, it suffers from slow convergence and limited feature spatial resolution, due to the limitation of Transformer attention modules in processing image feature maps. To mitigate these issues, we proposed Deformable DETR, whose attention modules only attend to a small set of key sampling points around a reference. Deformable DETR can achieve better performance than DETR (especially on small objects) with 10 times less training epochs. Extensive experiments on the COCO benchmark demonstrate the effectiveness of our approach._

Tips:

-   One can use [DeformableDetrImageProcessor](/docs/transformers/v4.34.0/en/model_doc/deformable_detr#transformers.DeformableDetrImageProcessor) to prepare images (and optional targets) for the model.
-   Training Deformable DETR is equivalent to training the original [DETR](detr) model. See the [resources](#resources) section below for demo notebooks.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/deformable_detr_architecture.png) Deformable DETR architecture. Taken from the [original paper](https://arxiv.org/abs/2010.04159).

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/fundamentalvision/Deformable-DETR).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with Deformable DETR.

-   Demo notebooks regarding inference + fine-tuning on a custom dataset for [DeformableDetrForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/deformable_detr#transformers.DeformableDetrForObjectDetection) can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Deformable-DETR).
-   See also: [Object detection task guide](../tasks/object_detection).

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## DeformableDetrImageProcessor

### class transformers.DeformableDetrImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deformable_detr/image_processing_deformable_detr.py#L761)

( format: typing.Union\[str, transformers.models.deformable\_detr.image\_processing\_deformable\_detr.AnnotionFormat\] = <AnnotionFormat.COCO\_DETECTION: 'coco\_detection'>do\_resize: bool = Truesize: typing.Dict\[str, int\] = Noneresample: Resampling = <Resampling.BILINEAR: 2>do\_rescale: bool = Truerescale\_factor: typing.Union\[int, float\] = 0.00392156862745098do\_normalize: bool = Trueimage\_mean: typing.Union\[float, typing.List\[float\]\] = Noneimage\_std: typing.Union\[float, typing.List\[float\]\] = Nonedo\_pad: bool = True\*\*kwargs )

Constructs a Deformable DETR image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deformable_detr/image_processing_deformable_detr.py#L1102)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]annotations: typing.Union\[typing.Dict\[str, typing.Union\[int, str, typing.List\[typing.Dict\]\]\], typing.List\[typing.Dict\[str, typing.Union\[int, str, typing.List\[typing.Dict\]\]\]\], NoneType\] = Nonereturn\_segmentation\_masks: bool = Nonemasks\_path: typing.Union\[str, pathlib.Path, NoneType\] = Nonedo\_resize: typing.Optional\[bool\] = Nonesize: typing.Union\[typing.Dict\[str, int\], NoneType\] = Noneresample = Nonedo\_rescale: typing.Optional\[bool\] = Nonerescale\_factor: typing.Union\[int, float, NoneType\] = Nonedo\_normalize: typing.Optional\[bool\] = Noneimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonedo\_pad: typing.Optional\[bool\] = Noneformat: typing.Union\[str, transformers.models.deformable\_detr.image\_processing\_deformable\_detr.AnnotionFormat, NoneType\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension\] = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs )

Preprocess an image or a batch of images so that it can be used by the model.

#### post\_process\_object\_detection

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deformable_detr/image_processing_deformable_detr.py#L1393)

( outputsthreshold: float = 0.5target\_sizes: typing.Union\[transformers.utils.generic.TensorType, typing.List\[typing.Tuple\]\] = Nonetop\_k: int = 100 ) → `List[Dict]`

Parameters

-   **outputs** (`DetrObjectDetectionOutput`) — Raw outputs of the model.
-   **threshold** (`float`, _optional_) — Score threshold to keep object detection predictions.
-   **target\_sizes** (`torch.Tensor` or `List[Tuple[int, int]]`, _optional_) — Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size (height, width) of each image in the batch. If left to None, predictions will not be resized.
-   **top\_k** (`int`, _optional_, defaults to 100) — Keep only top k bounding boxes before filtering by thresholding.

A list of dictionaries, each dictionary containing the scores, labels and boxes for an image in the batch as predicted by the model.

Converts the raw output of [DeformableDetrForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/deformable_detr#transformers.DeformableDetrForObjectDetection) into final bounding boxes in (top\_left\_x, top\_left\_y, bottom\_right\_x, bottom\_right\_y) format. Only supports PyTorch.

## DeformableDetrFeatureExtractor

Preprocess an image or a batch of images.

( outputsthreshold: float = 0.5target\_sizes: typing.Union\[transformers.utils.generic.TensorType, typing.List\[typing.Tuple\]\] = Nonetop\_k: int = 100 ) → `List[Dict]`

Parameters

-   **outputs** (`DetrObjectDetectionOutput`) — Raw outputs of the model.
-   **threshold** (`float`, _optional_) — Score threshold to keep object detection predictions.
-   **target\_sizes** (`torch.Tensor` or `List[Tuple[int, int]]`, _optional_) — Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size (height, width) of each image in the batch. If left to None, predictions will not be resized.
-   **top\_k** (`int`, _optional_, defaults to 100) — Keep only top k bounding boxes before filtering by thresholding.

A list of dictionaries, each dictionary containing the scores, labels and boxes for an image in the batch as predicted by the model.

Converts the raw output of [DeformableDetrForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/deformable_detr#transformers.DeformableDetrForObjectDetection) into final bounding boxes in (top\_left\_x, top\_left\_y, bottom\_right\_x, bottom\_right\_y) format. Only supports PyTorch.

## DeformableDetrConfig

### class transformers.DeformableDetrConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deformable_detr/configuration_deformable_detr.py#L30)

( use\_timm\_backbone = Truebackbone\_config = Nonenum\_channels = 3num\_queries = 300max\_position\_embeddings = 1024encoder\_layers = 6encoder\_ffn\_dim = 1024encoder\_attention\_heads = 8decoder\_layers = 6decoder\_ffn\_dim = 1024decoder\_attention\_heads = 8encoder\_layerdrop = 0.0is\_encoder\_decoder = Trueactivation\_function = 'relu'd\_model = 256dropout = 0.1attention\_dropout = 0.0activation\_dropout = 0.0init\_std = 0.02init\_xavier\_std = 1.0return\_intermediate = Trueauxiliary\_loss = Falseposition\_embedding\_type = 'sine'backbone = 'resnet50'use\_pretrained\_backbone = Truedilation = Falsenum\_feature\_levels = 4encoder\_n\_points = 4decoder\_n\_points = 4two\_stage = Falsetwo\_stage\_num\_proposals = 300with\_box\_refine = Falseclass\_cost = 1bbox\_cost = 5giou\_cost = 2mask\_loss\_coefficient = 1dice\_loss\_coefficient = 1bbox\_loss\_coefficient = 5giou\_loss\_coefficient = 2eos\_coefficient = 0.1focal\_alpha = 0.25disable\_custom\_kernels = False\*\*kwargs )

This is the configuration class to store the configuration of a [DeformableDetrModel](/docs/transformers/v4.34.0/en/model_doc/deformable_detr#transformers.DeformableDetrModel). It is used to instantiate a Deformable DETR model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Deformable DETR [SenseTime/deformable-detr](https://huggingface.co/SenseTime/deformable-detr) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import DeformableDetrConfig, DeformableDetrModel

>>> 
>>> configuration = DeformableDetrConfig()

>>> 
>>> model = DeformableDetrModel(configuration)

>>> 
>>> configuration = model.config
```

## DeformableDetrModel

### class transformers.DeformableDetrModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deformable_detr/modeling_deformable_detr.py#L1475)

( config: DeformableDetrConfig )

Parameters

-   **config** ([DeformableDetrConfig](/docs/transformers/v4.34.0/en/model_doc/deformable_detr#transformers.DeformableDetrConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Deformable DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deformable_detr/modeling_deformable_detr.py#L1624)

( pixel\_values: FloatTensorpixel\_mask: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Noneencoder\_outputs: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrModelOutput` or `tuple(torch.FloatTensor)`

The [DeformableDetrModel](/docs/transformers/v4.34.0/en/model_doc/deformable_detr#transformers.DeformableDetrModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, DeformableDetrModel
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
>>> model = DeformableDetrModel.from_pretrained("SenseTime/deformable-detr")

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 300, 256]
```

## DeformableDetrForObjectDetection

### class transformers.DeformableDetrForObjectDetection

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deformable_detr/modeling_deformable_detr.py#L1832)

( config: DeformableDetrConfig )

Parameters

-   **config** ([DeformableDetrConfig](/docs/transformers/v4.34.0/en/model_doc/deformable_detr#transformers.DeformableDetrConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Deformable DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on top, for tasks such as COCO detection.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deformable_detr/modeling_deformable_detr.py#L1884)

( pixel\_values: FloatTensorpixel\_mask: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Noneencoder\_outputs: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[typing.List\[dict\]\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrObjectDetectionOutput` or `tuple(torch.FloatTensor)`

The [DeformableDetrForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/deformable_detr#transformers.DeformableDetrForObjectDetection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
>>> model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")

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
Detected cat with confidence 0.8 at location [16.5, 52.84, 318.25, 470.78]
Detected cat with confidence 0.789 at location [342.19, 24.3, 640.02, 372.25]
Detected remote with confidence 0.633 at location [40.79, 72.78, 176.76, 117.25]
```