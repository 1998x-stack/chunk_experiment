# OneFormer

## Overview

The OneFormer model was proposed in [OneFormer: One Transformer to Rule Universal Image Segmentation](https://arxiv.org/abs/2211.06220) by Jitesh Jain, Jiachen Li, MangTik Chiu, Ali Hassani, Nikita Orlov, Humphrey Shi. OneFormer is a universal image segmentation framework that can be trained on a single panoptic dataset to perform semantic, instance, and panoptic segmentation tasks. OneFormer uses a task token to condition the model on the task in focus, making the architecture task-guided for training, and task-dynamic for inference.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/oneformer_teaser.png)

The abstract from the paper is the following:

_Universal Image Segmentation is not a new concept. Past attempts to unify image segmentation in the last decades include scene parsing, panoptic segmentation, and, more recently, new panoptic architectures. However, such panoptic architectures do not truly unify image segmentation because they need to be trained individually on the semantic, instance, or panoptic segmentation to achieve the best performance. Ideally, a truly universal framework should be trained only once and achieve SOTA performance across all three image segmentation tasks. To that end, we propose OneFormer, a universal image segmentation framework that unifies segmentation with a multi-task train-once design. We first propose a task-conditioned joint training strategy that enables training on ground truths of each domain (semantic, instance, and panoptic segmentation) within a single multi-task training process. Secondly, we introduce a task token to condition our model on the task at hand, making our model task-dynamic to support multi-task training and inference. Thirdly, we propose using a query-text contrastive loss during training to establish better inter-task and inter-class distinctions. Notably, our single OneFormer model outperforms specialized Mask2Former models across all three segmentation tasks on ADE20k, CityScapes, and COCO, despite the latter being trained on each of the three tasks individually with three times the resources. With new ConvNeXt and DiNAT backbones, we observe even more performance improvement. We believe OneFormer is a significant step towards making image segmentation more universal and accessible._

Tips:

-   OneFormer requires two inputs during inference: _image_ and _task token_.
-   During training, OneFormer only uses panoptic annotations.
-   If you want to train the model in a distributed environment across multiple nodes, then one should update the `get_num_masks` function inside in the `OneFormerLoss` class of `modeling_oneformer.py`. When training on multiple nodes, this should be set to the average number of target masks across all nodes, as can be seen in the original implementation [here](https://github.com/SHI-Labs/OneFormer/blob/33ebb56ed34f970a30ae103e786c0cb64c653d9a/oneformer/modeling/criterion.py#L287).
-   One can use [OneFormerProcessor](/docs/transformers/v4.34.0/en/model_doc/oneformer#transformers.OneFormerProcessor) to prepare input images and task inputs for the model and optional targets for the model. `OneformerProcessor` wraps [OneFormerImageProcessor](/docs/transformers/v4.34.0/en/model_doc/oneformer#transformers.OneFormerImageProcessor) and [CLIPTokenizer](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPTokenizer) into a single instance to both prepare the images and encode the task inputs.
-   To get the final segmentation, depending on the task, you can call [post\_process\_semantic\_segmentation()](/docs/transformers/v4.34.0/en/model_doc/oneformer#transformers.OneFormerProcessor.post_process_semantic_segmentation) or [post\_process\_instance\_segmentation()](/docs/transformers/v4.34.0/en/model_doc/oneformer#transformers.OneFormerImageProcessor.post_process_instance_segmentation) or [post\_process\_panoptic\_segmentation()](/docs/transformers/v4.34.0/en/model_doc/oneformer#transformers.OneFormerImageProcessor.post_process_panoptic_segmentation). All three tasks can be solved using [OneFormerForUniversalSegmentation](/docs/transformers/v4.34.0/en/model_doc/oneformer#transformers.OneFormerForUniversalSegmentation) output, panoptic segmentation accepts an optional `label_ids_to_fuse` argument to fuse instances of the target object/s (e.g. sky) together.

The figure below illustrates the architecture of OneFormer. Taken from the [original paper](https://arxiv.org/abs/2211.06220).

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/oneformer_architecture.png)

This model was contributed by [Jitesh Jain](https://huggingface.co/praeclarumjj3). The original code can be found [here](https://github.com/SHI-Labs/OneFormer).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with OneFormer.

-   Demo notebooks regarding inference + fine-tuning on custom data can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/OneFormer).

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we will review it. The resource should ideally demonstrate something new instead of duplicating an existing resource.

## OneFormer specific outputs

### class transformers.models.oneformer.modeling\_oneformer.OneFormerModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/oneformer/modeling_oneformer.py#L804)

( encoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonepixel\_decoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonetransformer\_decoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = Nonetransformer\_decoder\_object\_queries: FloatTensor = Nonetransformer\_decoder\_contrastive\_queries: typing.Optional\[torch.FloatTensor\] = Nonetransformer\_decoder\_mask\_predictions: FloatTensor = Nonetransformer\_decoder\_class\_predictions: FloatTensor = Nonetransformer\_decoder\_auxiliary\_predictions: typing.Union\[typing.Tuple\[typing.Dict\[str, torch.FloatTensor\]\], NoneType\] = Nonetext\_queries: typing.Optional\[torch.FloatTensor\] = Nonetask\_token: FloatTensor = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Class for outputs of [OneFormerModel](/docs/transformers/v4.34.0/en/model_doc/oneformer#transformers.OneFormerModel). This class returns all the needed hidden states to compute the logits.

### class transformers.models.oneformer.modeling\_oneformer.OneFormerForUniversalSegmentationOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/oneformer/modeling_oneformer.py#L854)

( loss: typing.Optional\[torch.FloatTensor\] = Noneclass\_queries\_logits: FloatTensor = Nonemasks\_queries\_logits: FloatTensor = Noneauxiliary\_predictions: typing.List\[typing.Dict\[str, torch.FloatTensor\]\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonepixel\_decoder\_hidden\_states: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonetransformer\_decoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = Nonetransformer\_decoder\_object\_queries: FloatTensor = Nonetransformer\_decoder\_contrastive\_queries: typing.Optional\[torch.FloatTensor\] = Nonetransformer\_decoder\_mask\_predictions: FloatTensor = Nonetransformer\_decoder\_class\_predictions: FloatTensor = Nonetransformer\_decoder\_auxiliary\_predictions: typing.Union\[typing.List\[typing.Dict\[str, torch.FloatTensor\]\], NoneType\] = Nonetext\_queries: typing.Optional\[torch.FloatTensor\] = Nonetask\_token: FloatTensor = Noneattentions: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = None )

Class for outputs of `OneFormerForUniversalSegmentationOutput`.

This output can be directly passed to [post\_process\_semantic\_segmentation()](/docs/transformers/v4.34.0/en/model_doc/oneformer#transformers.OneFormerImageProcessor.post_process_semantic_segmentation) or [post\_process\_instance\_segmentation()](/docs/transformers/v4.34.0/en/model_doc/oneformer#transformers.OneFormerImageProcessor.post_process_instance_segmentation) or [post\_process\_panoptic\_segmentation()](/docs/transformers/v4.34.0/en/model_doc/oneformer#transformers.OneFormerImageProcessor.post_process_panoptic_segmentation) depending on the task. Please, see \[\`~OneFormerImageProcessor\] for details regarding usage.

## OneFormerConfig

### class transformers.OneFormerConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/oneformer/configuration_oneformer.py#L33)

( backbone\_config: typing.Optional\[typing.Dict\] = Noneignore\_value: int = 255num\_queries: int = 150no\_object\_weight: int = 0.1class\_weight: float = 2.0mask\_weight: float = 5.0dice\_weight: float = 5.0contrastive\_weight: float = 0.5contrastive\_temperature: float = 0.07train\_num\_points: int = 12544oversample\_ratio: float = 3.0importance\_sample\_ratio: float = 0.75init\_std: float = 0.02init\_xavier\_std: float = 1.0layer\_norm\_eps: float = 1e-05is\_training: bool = Falseuse\_auxiliary\_loss: bool = Trueoutput\_auxiliary\_logits: bool = Truestrides: typing.Optional\[list\] = \[4, 8, 16, 32\]task\_seq\_len: int = 77text\_encoder\_width: int = 256text\_encoder\_context\_length: int = 77text\_encoder\_num\_layers: int = 6text\_encoder\_vocab\_size: int = 49408text\_encoder\_proj\_layers: int = 2text\_encoder\_n\_ctx: int = 16conv\_dim: int = 256mask\_dim: int = 256hidden\_dim: int = 256encoder\_feedforward\_dim: int = 1024norm: str = 'GN'encoder\_layers: int = 6decoder\_layers: int = 10use\_task\_norm: bool = Truenum\_attention\_heads: int = 8dropout: float = 0.1dim\_feedforward: int = 2048pre\_norm: bool = Falseenforce\_input\_proj: bool = Falsequery\_dec\_layers: int = 2common\_stride: int = 4\*\*kwargs )

This is the configuration class to store the configuration of a [OneFormerModel](/docs/transformers/v4.34.0/en/model_doc/oneformer#transformers.OneFormerModel). It is used to instantiate a OneFormer model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the OneFormer [shi-labs/oneformer\_ade20k\_swin\_tiny](https://huggingface.co/shi-labs/oneformer_ade20k_swin_tiny) architecture trained on [ADE20k-150](https://huggingface.co/datasets/scene_parse_150).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import OneFormerConfig, OneFormerModel

>>> 
>>> configuration = OneFormerConfig()
>>> 
>>> model = OneFormerModel(configuration)
>>> 
>>> configuration = model.config
```

## OneFormerImageProcessor

### class transformers.OneFormerImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/oneformer/image_processing_oneformer.py#L348)

( do\_resize: bool = Truesize: typing.Dict\[str, int\] = Noneresample: Resampling = <Resampling.BILINEAR: 2>do\_rescale: bool = Truerescale\_factor: float = 0.00392156862745098do\_normalize: bool = Trueimage\_mean: typing.Union\[float, typing.List\[float\]\] = Noneimage\_std: typing.Union\[float, typing.List\[float\]\] = Noneignore\_index: typing.Optional\[int\] = Nonedo\_reduce\_labels: bool = Falserepo\_path: str = 'shi-labs/oneformer\_demo'class\_info\_file: str = Nonenum\_text: typing.Optional\[int\] = None\*\*kwargs )

Constructs a OneFormer image processor. The image processor can be used to prepare image(s), task input(s) and optional text inputs and targets for the model.

This image processor inherits from `BaseImageProcessor` which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/oneformer/image_processing_oneformer.py#L636)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]task\_inputs: typing.Optional\[typing.List\[str\]\] = Nonesegmentation\_maps: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\], NoneType\] = Noneinstance\_id\_to\_semantic\_id: typing.Union\[typing.Dict\[int, int\], NoneType\] = Nonedo\_resize: typing.Optional\[bool\] = Nonesize: typing.Union\[typing.Dict\[str, int\], NoneType\] = Noneresample: Resampling = Nonedo\_rescale: typing.Optional\[bool\] = Nonerescale\_factor: typing.Optional\[float\] = Nonedo\_normalize: typing.Optional\[bool\] = Noneimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Noneignore\_index: typing.Optional\[int\] = Nonedo\_reduce\_labels: typing.Optional\[bool\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension\] = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs )

#### encode\_inputs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/oneformer/image_processing_oneformer.py#L934)

( pixel\_values\_list: typing.List\[typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]\]task\_inputs: typing.List\[str\]segmentation\_maps: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\] = Noneinstance\_id\_to\_semantic\_id: typing.Union\[typing.List\[typing.Dict\[int, int\]\], typing.Dict\[int, int\], NoneType\] = Noneignore\_index: typing.Optional\[int\] = Nonereduce\_labels: bool = Falsereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Noneinput\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None ) → [BatchFeature](/docs/transformers/v4.34.0/en/main_classes/image_processor#transformers.BatchFeature)

Pad images up to the largest image in a batch and create a corresponding `pixel_mask`.

OneFormer addresses semantic segmentation with a mask classification paradigm, thus input segmentation maps will be converted to lists of binary masks and their respective labels. Let’s see an example, assuming `segmentation_maps = [[2,6,7,9]]`, the output will contain `mask_labels = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]` (four binary masks) and `class_labels = [2,6,7,9]`, the labels for each mask.

#### post\_process\_semantic\_segmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/oneformer/image_processing_oneformer.py#L1069)

( outputstarget\_sizes: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], NoneType\] = None ) → `List[torch.Tensor]`

Parameters

-   **outputs** ([MaskFormerForInstanceSegmentation](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation)) — Raw outputs of the model.
-   **target\_sizes** (`List[Tuple[int, int]]`, _optional_) — List of length (batch\_size), where each list item (`Tuple[int, int]]`) corresponds to the requested final size (height, width) of each prediction. If left to None, predictions will not be resized.

Returns

`List[torch.Tensor]`

A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [MaskFormerForInstanceSegmentation](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation) into semantic segmentation maps. Only supports PyTorch.

#### post\_process\_instance\_segmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/oneformer/image_processing_oneformer.py#L1119)

( outputstask\_type: str = 'instance'is\_demo: bool = Truethreshold: float = 0.5mask\_threshold: float = 0.5overlap\_mask\_area\_threshold: float = 0.8target\_sizes: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], NoneType\] = Nonereturn\_coco\_annotation: typing.Optional\[bool\] = False ) → `List[Dict]`

Converts the output of `OneFormerForUniversalSegmentationOutput` into image instance segmentation predictions. Only supports PyTorch.

#### post\_process\_panoptic\_segmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/oneformer/image_processing_oneformer.py#L1238)

( outputsthreshold: float = 0.5mask\_threshold: float = 0.5overlap\_mask\_area\_threshold: float = 0.8label\_ids\_to\_fuse: typing.Optional\[typing.Set\[int\]\] = Nonetarget\_sizes: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], NoneType\] = None ) → `List[Dict]`

Converts the output of `MaskFormerForInstanceSegmentationOutput` into image panoptic segmentation predictions. Only supports PyTorch.

## OneFormerProcessor

### class transformers.OneFormerProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/oneformer/processing_oneformer.py#L29)

( image\_processor = Nonetokenizer = Nonemax\_seq\_length: int = 77task\_seq\_length: int = 77\*\*kwargs )

Parameters

-   **image\_processor** ([OneFormerImageProcessor](/docs/transformers/v4.34.0/en/model_doc/oneformer#transformers.OneFormerImageProcessor)) — The image processor is a required input.
-   **tokenizer** (\[`CLIPTokenizer`, `CLIPTokenizerFast`\]) — The tokenizer is a required input.
-   **max\_seq\_len** (`int`, _optional_, defaults to 77)) — Sequence length for input text list.
-   **task\_seq\_len** (`int`, _optional_, defaults to 77) — Sequence length for input task token.

Constructs an OneFormer processor which wraps [OneFormerImageProcessor](/docs/transformers/v4.34.0/en/model_doc/oneformer#transformers.OneFormerImageProcessor) and [CLIPTokenizer](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPTokenizer)/[CLIPTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPTokenizerFast) into a single processor that inherits both the image processor and tokenizer functionalities.

#### encode\_inputs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/oneformer/processing_oneformer.py#L145)

( images = Nonetask\_inputs = Nonesegmentation\_maps = None\*\*kwargs )

This method forwards all its arguments to [OneFormerImageProcessor.encode\_inputs()](/docs/transformers/v4.34.0/en/model_doc/oneformer#transformers.OneFormerImageProcessor.encode_inputs) and then tokenizes the task\_inputs. Please refer to the docstring of this method for more information.

## OneFormerModel

### class transformers.OneFormerModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/oneformer/modeling_oneformer.py#L2902)

( config: OneFormerConfig )

Parameters

-   **config** ([OneFormerConfig](/docs/transformers/v4.34.0/en/model_doc/oneformer#transformers.OneFormerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare OneFormer Model outputting raw hidden-states without any specific head on top. This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/oneformer/modeling_oneformer.py#L2919)

( pixel\_values: Tensortask\_inputs: Tensortext\_inputs: typing.Optional\[torch.Tensor\] = Nonepixel\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.models.oneformer.modeling\_oneformer.OneFormerModelOutput](/docs/transformers/v4.34.0/en/model_doc/oneformer#transformers.models.oneformer.modeling_oneformer.OneFormerModelOutput) or `tuple(torch.FloatTensor)`

The [OneFormerModel](/docs/transformers/v4.34.0/en/model_doc/oneformer#transformers.OneFormerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from PIL import Image
>>> import requests
>>> from transformers import OneFormerProcessor, OneFormerModel

>>> 
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> 
>>> processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
>>> model = OneFormerModel.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
>>> inputs = processor(image, ["semantic"], return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> mask_predictions = outputs.transformer_decoder_mask_predictions
>>> class_predictions = outputs.transformer_decoder_class_predictions

>>> f"👉 Mask Predictions Shape: {list(mask_predictions.shape)}, Class Predictions Shape: {list(class_predictions.shape)}"
'👉 Mask Predictions Shape: [1, 150, 128, 171], Class Predictions Shape: [1, 150, 151]'
```

## OneFormerForUniversalSegmentation

### class transformers.OneFormerForUniversalSegmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/oneformer/modeling_oneformer.py#L3031)

( config: OneFormerConfig )

Parameters

-   **config** ([OneFormerConfig](/docs/transformers/v4.34.0/en/model_doc/oneformer#transformers.OneFormerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

OneFormer Model for instance, semantic and panoptic image segmentation. This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/oneformer/modeling_oneformer.py#L3098)

( pixel\_values: Tensortask\_inputs: Tensortext\_inputs: typing.Optional\[torch.Tensor\] = Nonemask\_labels: typing.Optional\[typing.List\[torch.Tensor\]\] = Noneclass\_labels: typing.Optional\[typing.List\[torch.Tensor\]\] = Nonepixel\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_auxiliary\_logits: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.models.oneformer.modeling\_oneformer.OneFormerForUniversalSegmentationOutput](/docs/transformers/v4.34.0/en/model_doc/oneformer#transformers.models.oneformer.modeling_oneformer.OneFormerForUniversalSegmentationOutput) or `tuple(torch.FloatTensor)`

The [OneFormerForUniversalSegmentation](/docs/transformers/v4.34.0/en/model_doc/oneformer#transformers.OneFormerForUniversalSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

Universal segmentation example:

```
>>> from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
>>> from PIL import Image
>>> import requests
>>> import torch

>>> 
>>> processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
>>> model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")

>>> url = (
...     "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
... )
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> 
>>> inputs = processor(image, ["semantic"], return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)
>>> 
>>> 
>>> class_queries_logits = outputs.class_queries_logits
>>> masks_queries_logits = outputs.masks_queries_logits

>>> 
>>> predicted_semantic_map = processor.post_process_semantic_segmentation(
...     outputs, target_sizes=[image.size[::-1]]
... )[0]
>>> f"👉 Semantic Predictions Shape: {list(predicted_semantic_map.shape)}"
'👉 Semantic Predictions Shape: [512, 683]'

>>> 
>>> inputs = processor(image, ["instance"], return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)
>>> 
>>> 
>>> class_queries_logits = outputs.class_queries_logits
>>> masks_queries_logits = outputs.masks_queries_logits

>>> 
>>> predicted_instance_map = processor.post_process_instance_segmentation(
...     outputs, target_sizes=[image.size[::-1]]
... )[0]["segmentation"]
>>> f"👉 Instance Predictions Shape: {list(predicted_instance_map.shape)}"
'👉 Instance Predictions Shape: [512, 683]'

>>> 
>>> inputs = processor(image, ["panoptic"], return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)
>>> 
>>> 
>>> class_queries_logits = outputs.class_queries_logits
>>> masks_queries_logits = outputs.masks_queries_logits

>>> 
>>> predicted_panoptic_map = processor.post_process_panoptic_segmentation(
...     outputs, target_sizes=[image.size[::-1]]
... )[0]["segmentation"]
>>> f"👉 Panoptic Predictions Shape: {list(predicted_panoptic_map.shape)}"
'👉 Panoptic Predictions Shape: [512, 683]'
```