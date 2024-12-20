# DETA

## Overview

The DETA model was proposed in [NMS Strikes Back](https://arxiv.org/abs/2212.06137) by Jeffrey Ouyang-Zhang, Jang Hyun Cho, Xingyi Zhou, Philipp Krähenbühl. DETA (short for Detection Transformers with Assignment) improves [Deformable DETR](deformable_detr) by replacing the one-to-one bipartite Hungarian matching loss with one-to-many label assignments used in traditional detectors with non-maximum suppression (NMS). This leads to significant gains of up to 2.5 mAP.

The abstract from the paper is the following:

_Detection Transformer (DETR) directly transforms queries to unique objects by using one-to-one bipartite matching during training and enables end-to-end object detection. Recently, these models have surpassed traditional detectors on COCO with undeniable elegance. However, they differ from traditional detectors in multiple designs, including model architecture and training schedules, and thus the effectiveness of one-to-one matching is not fully understood. In this work, we conduct a strict comparison between the one-to-one Hungarian matching in DETRs and the one-to-many label assignments in traditional detectors with non-maximum supervision (NMS). Surprisingly, we observe one-to-many assignments with NMS consistently outperform standard one-to-one matching under the same setting, with a significant gain of up to 2.5 mAP. Our detector that trains Deformable-DETR with traditional IoU-based label assignment achieved 50.2 COCO mAP within 12 epochs (1x schedule) with ResNet50 backbone, outperforming all existing traditional or transformer-based detectors in this setting. On multiple datasets, schedules, and architectures, we consistently show bipartite matching is unnecessary for performant detection transformers. Furthermore, we attribute the success of detection transformers to their expressive transformer architecture._

Tips:

-   One can use [DetaImageProcessor](/docs/transformers/v4.34.0/en/model_doc/deta#transformers.DetaImageProcessor) to prepare images and optional targets for the model.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/deta_architecture.jpg) DETA overview. Taken from the [original paper](https://arxiv.org/abs/2212.06137).

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/jozhang97/DETA).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with DETA.

-   Demo notebooks for DETA can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETA).
-   See also: [Object detection task guide](../tasks/object_detection)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## DetaConfig

### class transformers.DetaConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deta/configuration_deta.py#L30)

( backbone\_config = Nonenum\_queries = 900max\_position\_embeddings = 2048encoder\_layers = 6encoder\_ffn\_dim = 2048encoder\_attention\_heads = 8decoder\_layers = 6decoder\_ffn\_dim = 1024decoder\_attention\_heads = 8encoder\_layerdrop = 0.0is\_encoder\_decoder = Trueactivation\_function = 'relu'd\_model = 256dropout = 0.1attention\_dropout = 0.0activation\_dropout = 0.0init\_std = 0.02init\_xavier\_std = 1.0return\_intermediate = Trueauxiliary\_loss = Falseposition\_embedding\_type = 'sine'num\_feature\_levels = 5encoder\_n\_points = 4decoder\_n\_points = 4two\_stage = Truetwo\_stage\_num\_proposals = 300with\_box\_refine = Trueassign\_first\_stage = Trueclass\_cost = 1bbox\_cost = 5giou\_cost = 2mask\_loss\_coefficient = 1dice\_loss\_coefficient = 1bbox\_loss\_coefficient = 5giou\_loss\_coefficient = 2eos\_coefficient = 0.1focal\_alpha = 0.25\*\*kwargs )

This is the configuration class to store the configuration of a [DetaModel](/docs/transformers/v4.34.0/en/model_doc/deta#transformers.DetaModel). It is used to instantiate a DETA model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the DETA [SenseTime/deformable-detr](https://huggingface.co/SenseTime/deformable-detr) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import DetaConfig, DetaModel

>>> 
>>> configuration = DetaConfig()

>>> 
>>> model = DetaModel(configuration)

>>> 
>>> configuration = model.config
```

## DetaImageProcessor

### class transformers.DetaImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deta/image_processing_deta.py#L468)

( format: typing.Union\[str, transformers.models.deta.image\_processing\_deta.AnnotionFormat\] = <AnnotionFormat.COCO\_DETECTION: 'coco\_detection'>do\_resize: bool = Truesize: typing.Dict\[str, int\] = Noneresample: Resampling = <Resampling.BILINEAR: 2>do\_rescale: bool = Truerescale\_factor: typing.Union\[int, float\] = 0.00392156862745098do\_normalize: bool = Trueimage\_mean: typing.Union\[float, typing.List\[float\]\] = Noneimage\_std: typing.Union\[float, typing.List\[float\]\] = Nonedo\_pad: bool = True\*\*kwargs )

Constructs a Deformable DETR image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deta/image_processing_deta.py#L774)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]annotations: typing.Union\[typing.List\[typing.Dict\], typing.List\[typing.List\[typing.Dict\]\], NoneType\] = Nonereturn\_segmentation\_masks: bool = Nonemasks\_path: typing.Union\[str, pathlib.Path, NoneType\] = Nonedo\_resize: typing.Optional\[bool\] = Nonesize: typing.Union\[typing.Dict\[str, int\], NoneType\] = Noneresample = Nonedo\_rescale: typing.Optional\[bool\] = Nonerescale\_factor: typing.Union\[int, float, NoneType\] = Nonedo\_normalize: typing.Optional\[bool\] = Noneimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonedo\_pad: typing.Optional\[bool\] = Noneformat: typing.Union\[str, transformers.models.deta.image\_processing\_deta.AnnotionFormat, NoneType\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension\] = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs )

Preprocess an image or a batch of images so that it can be used by the model.

#### post\_process\_object\_detection

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deta/image_processing_deta.py#L1011)

( outputsthreshold: float = 0.5target\_sizes: typing.Union\[transformers.utils.generic.TensorType, typing.List\[typing.Tuple\]\] = Nonenms\_threshold: float = 0.7 ) → `List[Dict]`

Parameters

-   **outputs** (`DetrObjectDetectionOutput`) — Raw outputs of the model.
-   **threshold** (`float`, _optional_, defaults to 0.5) — Score threshold to keep object detection predictions.
-   **target\_sizes** (`torch.Tensor` or `List[Tuple[int, int]]`, _optional_) — Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size (height, width) of each image in the batch. If left to None, predictions will not be resized.
-   **nms\_threshold** (`float`, _optional_, defaults to 0.7) — NMS threshold.

A list of dictionaries, each dictionary containing the scores, labels and boxes for an image in the batch as predicted by the model.

Converts the output of [DetaForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/deta#transformers.DetaForObjectDetection) into final bounding boxes in (top\_left\_x, top\_left\_y, bottom\_right\_x, bottom\_right\_y) format. Only supports PyTorch.

## DetaModel

### class transformers.DetaModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deta/modeling_deta.py#L1367)

( config: DetaConfig )

Parameters

-   **config** ([DetaConfig](/docs/transformers/v4.34.0/en/model_doc/deta#transformers.DetaConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare DETA Model (consisting of a backbone and encoder-decoder Transformer) outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deta/modeling_deta.py#L1532)

( pixel\_values: FloatTensorpixel\_mask: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Noneencoder\_outputs: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.deta.modeling_deta.DetaModelOutput` or `tuple(torch.FloatTensor)`

The [DetaModel](/docs/transformers/v4.34.0/en/model_doc/deta#transformers.DetaModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, DetaModel
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("jozhang97/deta-swin-large-o365")
>>> model = DetaModel.from_pretrained("jozhang97/deta-swin-large-o365", two_stage=False)

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 900, 256]
```

## DetaForObjectDetection

### class transformers.DetaForObjectDetection

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deta/modeling_deta.py#L1784)

( config: DetaConfig )

Parameters

-   **config** ([DetaConfig](/docs/transformers/v4.34.0/en/model_doc/deta#transformers.DetaConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

DETA Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on top, for tasks such as COCO detection.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deta/modeling_deta.py#L1837)

( pixel\_values: FloatTensorpixel\_mask: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Noneencoder\_outputs: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[typing.List\[dict\]\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.deta.modeling_deta.DetaObjectDetectionOutput` or `tuple(torch.FloatTensor)`

The [DetaForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/deta#transformers.DetaForObjectDetection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, DetaForObjectDetection
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("jozhang97/deta-swin-large")
>>> model = DetaForObjectDetection.from_pretrained("jozhang97/deta-swin-large")

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
Detected cat with confidence 0.683 at location [345.85, 23.68, 639.86, 372.83]
Detected cat with confidence 0.683 at location [8.8, 52.49, 316.93, 473.45]
Detected remote with confidence 0.568 at location [40.02, 73.75, 175.96, 117.33]
Detected remote with confidence 0.546 at location [333.68, 77.13, 370.12, 187.51]
```