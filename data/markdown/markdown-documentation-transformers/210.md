# MaskFormer

This is a recently introduced model so the API hasn’t been tested extensively. There may be some bugs or slight breaking changes to fix it in the future. If you see something strange, file a [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title).

## Overview

The MaskFormer model was proposed in [Per-Pixel Classification is Not All You Need for Semantic Segmentation](https://arxiv.org/abs/2107.06278) by Bowen Cheng, Alexander G. Schwing, Alexander Kirillov. MaskFormer addresses semantic segmentation with a mask classification paradigm instead of performing classic pixel-level classification.

The abstract from the paper is the following:

_Modern approaches typically formulate semantic segmentation as a per-pixel classification task, while instance-level segmentation is handled with an alternative mask classification. Our key insight: mask classification is sufficiently general to solve both semantic- and instance-level segmentation tasks in a unified manner using the exact same model, loss, and training procedure. Following this observation, we propose MaskFormer, a simple mask classification model which predicts a set of binary masks, each associated with a single global class label prediction. Overall, the proposed mask classification-based method simplifies the landscape of effective approaches to semantic and panoptic segmentation tasks and shows excellent empirical results. In particular, we observe that MaskFormer outperforms per-pixel classification baselines when the number of classes is large. Our mask classification-based method outperforms both current state-of-the-art semantic (55.6 mIoU on ADE20K) and panoptic segmentation (52.7 PQ on COCO) models._

Tips:

-   MaskFormer’s Transformer decoder is identical to the decoder of [DETR](detr). During training, the authors of DETR did find it helpful to use auxiliary losses in the decoder, especially to help the model output the correct number of objects of each class. If you set the parameter `use_auxilary_loss` of [MaskFormerConfig](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerConfig) to `True`, then prediction feedforward neural networks and Hungarian losses are added after each decoder layer (with the FFNs sharing parameters).
-   If you want to train the model in a distributed environment across multiple nodes, then one should update the `get_num_masks` function inside in the `MaskFormerLoss` class of `modeling_maskformer.py`. When training on multiple nodes, this should be set to the average number of target masks across all nodes, as can be seen in the original implementation [here](https://github.com/facebookresearch/MaskFormer/blob/da3e60d85fdeedcb31476b5edd7d328826ce56cc/mask_former/modeling/criterion.py#L169).
-   One can use [MaskFormerImageProcessor](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerImageProcessor) to prepare images for the model and optional targets for the model.
-   To get the final segmentation, depending on the task, you can call [post\_process\_semantic\_segmentation()](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerFeatureExtractor.post_process_semantic_segmentation) or [post\_process\_panoptic\_segmentation()](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerFeatureExtractor.post_process_panoptic_segmentation). Both tasks can be solved using [MaskFormerForInstanceSegmentation](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation) output, panoptic segmentation accepts an optional `label_ids_to_fuse` argument to fuse instances of the target object/s (e.g. sky) together.

The figure below illustrates the architecture of MaskFormer. Taken from the [original paper](https://arxiv.org/abs/2107.06278).

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/maskformer_architecture.png)

This model was contributed by [francesco](https://huggingface.co/francesco). The original code can be found [here](https://github.com/facebookresearch/MaskFormer).

## Resources

-   All notebooks that illustrate inference as well as fine-tuning on custom data with MaskFormer can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/MaskFormer).

## MaskFormer specific outputs

### class transformers.models.maskformer.modeling\_maskformer.MaskFormerModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/maskformer/modeling_maskformer.py#L146)

( encoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Nonepixel\_decoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Nonetransformer\_decoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonepixel\_decoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonetransformer\_decoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Class for outputs of [MaskFormerModel](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerModel). This class returns all the needed hidden states to compute the logits.

### class transformers.models.maskformer.modeling\_maskformer.MaskFormerForInstanceSegmentationOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/maskformer/modeling_maskformer.py#L189)

( loss: typing.Optional\[torch.FloatTensor\] = Noneclass\_queries\_logits: FloatTensor = Nonemasks\_queries\_logits: FloatTensor = Noneauxiliary\_logits: FloatTensor = Noneencoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Nonepixel\_decoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Nonetransformer\_decoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonepixel\_decoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonetransformer\_decoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Class for outputs of [MaskFormerForInstanceSegmentation](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation).

This output can be directly passed to [post\_process\_semantic\_segmentation()](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerFeatureExtractor.post_process_semantic_segmentation) or or [post\_process\_instance\_segmentation()](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerFeatureExtractor.post_process_instance_segmentation) or [post\_process\_panoptic\_segmentation()](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerFeatureExtractor.post_process_panoptic_segmentation) depending on the task. Please, see \[\`~MaskFormerImageProcessor\] for details regarding usage.

## MaskFormerConfig

### class transformers.MaskFormerConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/maskformer/configuration_maskformer.py#L35)

( fpn\_feature\_size: int = 256mask\_feature\_size: int = 256no\_object\_weight: float = 0.1use\_auxiliary\_loss: bool = Falsebackbone\_config: typing.Optional\[typing.Dict\] = Nonedecoder\_config: typing.Optional\[typing.Dict\] = Noneinit\_std: float = 0.02init\_xavier\_std: float = 1.0dice\_weight: float = 1.0cross\_entropy\_weight: float = 1.0mask\_weight: float = 20.0output\_auxiliary\_logits: typing.Optional\[bool\] = None\*\*kwargs )

This is the configuration class to store the configuration of a [MaskFormerModel](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerModel). It is used to instantiate a MaskFormer model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the MaskFormer [facebook/maskformer-swin-base-ade](https://huggingface.co/facebook/maskformer-swin-base-ade) architecture trained on [ADE20k-150](https://huggingface.co/datasets/scene_parse_150).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Currently, MaskFormer only supports the [Swin Transformer](swin) as backbone.

Examples:

```
>>> from transformers import MaskFormerConfig, MaskFormerModel

>>> 
>>> configuration = MaskFormerConfig()

>>> 
>>> model = MaskFormerModel(configuration)

>>> 
>>> configuration = model.config
```

#### from\_backbone\_and\_decoder\_configs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/maskformer/configuration_maskformer.py#L181)

( backbone\_config: PretrainedConfigdecoder\_config: PretrainedConfig\*\*kwargs ) → [MaskFormerConfig](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerConfig)

Parameters

-   **backbone\_config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The backbone configuration.
-   **decoder\_config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The transformer decoder configuration to use.

An instance of a configuration object

Instantiate a [MaskFormerConfig](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerConfig) (or a derived class) from a pre-trained backbone model configuration and DETR model configuration.

## MaskFormerImageProcessor

### class transformers.MaskFormerImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/maskformer/image_processing_maskformer.py#L345)

( do\_resize: bool = Truesize: typing.Dict\[str, int\] = Nonesize\_divisor: int = 32resample: Resampling = <Resampling.BILINEAR: 2>do\_rescale: bool = Truerescale\_factor: float = 0.00392156862745098do\_normalize: bool = Trueimage\_mean: typing.Union\[float, typing.List\[float\]\] = Noneimage\_std: typing.Union\[float, typing.List\[float\]\] = Noneignore\_index: typing.Optional\[int\] = Nonedo\_reduce\_labels: bool = False\*\*kwargs )

Constructs a MaskFormer image processor. The image processor can be used to prepare image(s) and optional targets for the model.

This image processor inherits from `BaseImageProcessor` which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/maskformer/image_processing_maskformer.py#L678)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]segmentation\_maps: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\], NoneType\] = Noneinstance\_id\_to\_semantic\_id: typing.Union\[typing.Dict\[int, int\], NoneType\] = Nonedo\_resize: typing.Optional\[bool\] = Nonesize: typing.Union\[typing.Dict\[str, int\], NoneType\] = Nonesize\_divisor: typing.Optional\[int\] = Noneresample: Resampling = Nonedo\_rescale: typing.Optional\[bool\] = Nonerescale\_factor: typing.Optional\[float\] = Nonedo\_normalize: typing.Optional\[bool\] = Noneimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Noneignore\_index: typing.Optional\[int\] = Nonedo\_reduce\_labels: typing.Optional\[bool\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension\] = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs )

#### encode\_inputs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/maskformer/image_processing_maskformer.py#L876)

( pixel\_values\_list: typing.List\[typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]\]segmentation\_maps: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\] = Noneinstance\_id\_to\_semantic\_id: typing.Union\[typing.List\[typing.Dict\[int, int\]\], typing.Dict\[int, int\], NoneType\] = Noneignore\_index: typing.Optional\[int\] = Nonereduce\_labels: bool = Falsereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Noneinput\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None ) → [BatchFeature](/docs/transformers/v4.34.0/en/main_classes/image_processor#transformers.BatchFeature)

Pad images up to the largest image in a batch and create a corresponding `pixel_mask`.

MaskFormer addresses semantic segmentation with a mask classification paradigm, thus input segmentation maps will be converted to lists of binary masks and their respective labels. Let’s see an example, assuming `segmentation_maps = [[2,6,7,9]]`, the output will contain `mask_labels = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]` (four binary masks) and `class_labels = [2,6,7,9]`, the labels for each mask.

#### post\_process\_semantic\_segmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/maskformer/image_processing_maskformer.py#L1030)

( outputstarget\_sizes: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], NoneType\] = None ) → `List[torch.Tensor]`

Parameters

-   **outputs** ([MaskFormerForInstanceSegmentation](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation)) — Raw outputs of the model.
-   **target\_sizes** (`List[Tuple[int, int]]`, _optional_) — List of length (batch\_size), where each list item (`Tuple[int, int]]`) corresponds to the requested final size (height, width) of each prediction. If left to None, predictions will not be resized.

Returns

`List[torch.Tensor]`

A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [MaskFormerForInstanceSegmentation](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation) into semantic segmentation maps. Only supports PyTorch.

#### post\_process\_instance\_segmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/maskformer/image_processing_maskformer.py#L1080)

( outputsthreshold: float = 0.5mask\_threshold: float = 0.5overlap\_mask\_area\_threshold: float = 0.8target\_sizes: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], NoneType\] = Nonereturn\_coco\_annotation: typing.Optional\[bool\] = Falsereturn\_binary\_maps: typing.Optional\[bool\] = False ) → `List[Dict]`

Converts the output of `MaskFormerForInstanceSegmentationOutput` into instance segmentation predictions. Only supports PyTorch.

#### post\_process\_panoptic\_segmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/maskformer/image_processing_maskformer.py#L1194)

( outputsthreshold: float = 0.5mask\_threshold: float = 0.5overlap\_mask\_area\_threshold: float = 0.8label\_ids\_to\_fuse: typing.Optional\[typing.Set\[int\]\] = Nonetarget\_sizes: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], NoneType\] = None ) → `List[Dict]`

Converts the output of `MaskFormerForInstanceSegmentationOutput` into image panoptic segmentation predictions. Only supports PyTorch.

## MaskFormerFeatureExtractor

( imagessegmentation\_maps = None\*\*kwargs )

( pixel\_values\_list: typing.List\[typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]\]segmentation\_maps: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\] = Noneinstance\_id\_to\_semantic\_id: typing.Union\[typing.List\[typing.Dict\[int, int\]\], typing.Dict\[int, int\], NoneType\] = Noneignore\_index: typing.Optional\[int\] = Nonereduce\_labels: bool = Falsereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Noneinput\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None ) → [BatchFeature](/docs/transformers/v4.34.0/en/main_classes/image_processor#transformers.BatchFeature)

Pad images up to the largest image in a batch and create a corresponding `pixel_mask`.

MaskFormer addresses semantic segmentation with a mask classification paradigm, thus input segmentation maps will be converted to lists of binary masks and their respective labels. Let’s see an example, assuming `segmentation_maps = [[2,6,7,9]]`, the output will contain `mask_labels = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]` (four binary masks) and `class_labels = [2,6,7,9]`, the labels for each mask.

( outputstarget\_sizes: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], NoneType\] = None ) → `List[torch.Tensor]`

Parameters

-   **outputs** ([MaskFormerForInstanceSegmentation](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation)) — Raw outputs of the model.
-   **target\_sizes** (`List[Tuple[int, int]]`, _optional_) — List of length (batch\_size), where each list item (`Tuple[int, int]]`) corresponds to the requested final size (height, width) of each prediction. If left to None, predictions will not be resized.

A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [MaskFormerForInstanceSegmentation](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation) into semantic segmentation maps. Only supports PyTorch.

( outputsthreshold: float = 0.5mask\_threshold: float = 0.5overlap\_mask\_area\_threshold: float = 0.8target\_sizes: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], NoneType\] = Nonereturn\_coco\_annotation: typing.Optional\[bool\] = Falsereturn\_binary\_maps: typing.Optional\[bool\] = False ) → `List[Dict]`

Converts the output of `MaskFormerForInstanceSegmentationOutput` into instance segmentation predictions. Only supports PyTorch.

( outputsthreshold: float = 0.5mask\_threshold: float = 0.5overlap\_mask\_area\_threshold: float = 0.8label\_ids\_to\_fuse: typing.Optional\[typing.Set\[int\]\] = Nonetarget\_sizes: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], NoneType\] = None ) → `List[Dict]`

Converts the output of `MaskFormerForInstanceSegmentationOutput` into image panoptic segmentation predictions. Only supports PyTorch.

## MaskFormerModel

### class transformers.MaskFormerModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/maskformer/modeling_maskformer.py#L1633)

( config: MaskFormerConfig )

Parameters

-   **config** ([MaskFormerConfig](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare MaskFormer Model outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

The [MaskFormerModel](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, MaskFormerModel
>>> from PIL import Image
>>> import requests

>>> 
>>> image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-ade")
>>> model = MaskFormerModel.from_pretrained("facebook/maskformer-swin-base-ade")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = image_processor(image, return_tensors="pt")

>>> 
>>> outputs = model(**inputs)

>>> 
>>> transformer_decoder_last_hidden_state = outputs.transformer_decoder_last_hidden_state
>>> list(transformer_decoder_last_hidden_state.shape)
[1, 100, 256]
```

## MaskFormerForInstanceSegmentation

### class transformers.MaskFormerForInstanceSegmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/maskformer/modeling_maskformer.py#L1732)

( config: MaskFormerConfig )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/maskformer/modeling_maskformer.py#L1827)

( pixel\_values: Tensormask\_labels: typing.Optional\[typing.List\[torch.Tensor\]\] = Noneclass\_labels: typing.Optional\[typing.List\[torch.Tensor\]\] = Nonepixel\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_auxiliary\_logits: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.models.maskformer.modeling\_maskformer.MaskFormerForInstanceSegmentationOutput](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput) or `tuple(torch.FloatTensor)`

The [MaskFormerForInstanceSegmentation](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

Semantic segmentation example:

```
>>> from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation
>>> from PIL import Image
>>> import requests

>>> 
>>> image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-ade")
>>> model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade")

>>> url = (
...     "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
... )
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = image_processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> 
>>> 
>>> class_queries_logits = outputs.class_queries_logits
>>> masks_queries_logits = outputs.masks_queries_logits

>>> 
>>> predicted_semantic_map = image_processor.post_process_semantic_segmentation(
...     outputs, target_sizes=[image.size[::-1]]
... )[0]

>>> 
>>> list(predicted_semantic_map.shape)
[512, 683]
```

Panoptic segmentation example:

```
>>> from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation
>>> from PIL import Image
>>> import requests

>>> 
>>> image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
>>> model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = image_processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> 
>>> 
>>> class_queries_logits = outputs.class_queries_logits
>>> masks_queries_logits = outputs.masks_queries_logits

>>> 
>>> result = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

>>> 
>>> predicted_panoptic_map = result["segmentation"]
>>> list(predicted_panoptic_map.shape)
[480, 640]
```