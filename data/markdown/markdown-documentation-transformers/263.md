# SAM

## Overview

SAM (Segment Anything Model) was proposed in [Segment Anything](https://arxiv.org/pdf/2304.02643v1.pdf) by Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alex Berg, Wan-Yen Lo, Piotr Dollar, Ross Girshick.

The model can be used to predict segmentation masks of any object of interest given an input image.

![example image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-output.png)

The abstract from the paper is the following:

_We introduce the Segment Anything (SA) project: a new task, model, and dataset for image segmentation. Using our efficient model in a data collection loop, we built the largest segmentation dataset to date (by far), with over 1 billion masks on 11M licensed and privacy respecting images. The model is designed and trained to be promptable, so it can transfer zero-shot to new image distributions and tasks. We evaluate its capabilities on numerous tasks and find that its zero-shot performance is impressive — often competitive with or even superior to prior fully supervised results. We are releasing the Segment Anything Model (SAM) and corresponding dataset (SA-1B) of 1B masks and 11M images at [https://segment-anything.com](https://segment-anything.com/) to foster research into foundation models for computer vision._

Tips:

-   The model predicts binary masks that states the presence or not of the object of interest given an image.
-   The model predicts much better results if input 2D points and/or input bounding boxes are provided
-   You can prompt multiple points for the same image, and predict a single mask.
-   Fine-tuning the model is not supported yet
-   According to the paper, textual input should be also supported. However, at this time of writing this seems to be not supported according to [the official repository](https://github.com/facebookresearch/segment-anything/issues/4#issuecomment-1497626844).

This model was contributed by [ybelkada](https://huggingface.co/ybelkada) and [ArthurZ](https://huggingface.co/ArthurZ). The original code can be found [here](https://github.com/facebookresearch/segment-anything).

Below is an example on how to run mask generation given an image and a 2D point:

```
import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
input_points = [[[450, 600]]]  

inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores
```

Resources:

-   [Demo notebook](https://github.com/huggingface/notebooks/blob/main/examples/segment_anything.ipynb) for using the model.
-   [Demo notebook](https://github.com/huggingface/notebooks/blob/main/examples/automatic_mask_generation.ipynb) for using the automatic mask generation pipeline.
-   [Demo notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Run_inference_with_MedSAM_using_HuggingFace_Transformers.ipynb) for inference with MedSAM, a fine-tuned version of SAM on the medical domain.
-   [Demo notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb) for fine-tuning the model on custom data.

## SamConfig

### class transformers.SamConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sam/configuration_sam.py#L237)

( vision\_config = Noneprompt\_encoder\_config = Nonemask\_decoder\_config = Noneinitializer\_range = 0.02\*\*kwargs )

Parameters

-   **vision\_config** (Union\[`dict`, `SamVisionConfig`\], _optional_) — Dictionary of configuration options used to initialize [SamVisionConfig](/docs/transformers/v4.34.0/en/model_doc/sam#transformers.SamVisionConfig).
-   **prompt\_encoder\_config** (Union\[`dict`, `SamPromptEncoderConfig`\], _optional_) — Dictionary of configuration options used to initialize [SamPromptEncoderConfig](/docs/transformers/v4.34.0/en/model_doc/sam#transformers.SamPromptEncoderConfig).
-   **mask\_decoder\_config** (Union\[`dict`, `SamMaskDecoderConfig`\], _optional_) — Dictionary of configuration options used to initialize [SamMaskDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/sam#transformers.SamMaskDecoderConfig).
-   **kwargs** (_optional_) — Dictionary of keyword arguments.

[SamConfig](/docs/transformers/v4.34.0/en/model_doc/sam#transformers.SamConfig) is the configuration class to store the configuration of a [SamModel](/docs/transformers/v4.34.0/en/model_doc/sam#transformers.SamModel). It is used to instantiate a SAM model according to the specified arguments, defining the vision model, prompt-encoder model and mask decoder configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the SAM-ViT-H [facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import (
...     SamVisionConfig,
...     SamPromptEncoderConfig,
...     SamMaskDecoderConfig,
...     SamModel,
... )

>>> 
>>> configuration = SamConfig()

>>> 
>>> model = SamModel(configuration)

>>> 
>>> configuration = model.config

>>> 

>>> 
>>> vision_config = SamVisionConfig()
>>> prompt_encoder_config = SamPromptEncoderConfig()
>>> mask_decoder_config = SamMaskDecoderConfig()

>>> config = SamConfig(vision_config, prompt_encoder_config, mask_decoder_config)
```

## SamVisionConfig

### class transformers.SamVisionConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sam/configuration_sam.py#L139)

( hidden\_size = 768output\_channels = 256num\_hidden\_layers = 12num\_attention\_heads = 12num\_channels = 3image\_size = 1024patch\_size = 16hidden\_act = 'gelu'layer\_norm\_eps = 1e-06attention\_dropout = 0.0initializer\_range = 1e-10qkv\_bias = Truemlp\_ratio = 4.0use\_abs\_pos = Trueuse\_rel\_pos = Truewindow\_size = 14global\_attn\_indexes = \[2, 5, 8, 11\]num\_pos\_feats = 128mlp\_dim = None\*\*kwargs )

This is the configuration class to store the configuration of a `SamVisionModel`. It is used to instantiate a SAM vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration defaults will yield a similar configuration to that of the SAM ViT-h [facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## SamMaskDecoderConfig

### class transformers.SamMaskDecoderConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sam/configuration_sam.py#L78)

( hidden\_size = 256hidden\_act = 'relu'mlp\_dim = 2048num\_hidden\_layers = 2num\_attention\_heads = 8attention\_downsample\_rate = 2num\_multimask\_outputs = 3iou\_head\_depth = 3iou\_head\_hidden\_dim = 256layer\_norm\_eps = 1e-06\*\*kwargs )

This is the configuration class to store the configuration of a `SamMaskDecoder`. It is used to instantiate a SAM mask decoder to the specified arguments, defining the model architecture. Instantiating a configuration defaults will yield a similar configuration to that of the SAM-vit-h [facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## SamPromptEncoderConfig

### class transformers.SamPromptEncoderConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sam/configuration_sam.py#L31)

( hidden\_size = 256image\_size = 1024patch\_size = 16mask\_input\_channels = 16num\_point\_embeddings = 4hidden\_act = 'gelu'layer\_norm\_eps = 1e-06\*\*kwargs )

Parameters

-   **hidden\_size** (`int`, _optional_, defaults to 256) — Dimensionality of the hidden states.
-   **image\_size** (`int`, _optional_, defaults to 1024) — The expected output resolution of the image.
-   **patch\_size** (`int`, _optional_, defaults to 16) — The size (resolution) of each patch.
-   **mask\_input\_channels** (`int`, _optional_, defaults to 16) — The number of channels to be fed to the `MaskDecoder` module.
-   **num\_point\_embeddings** (`int`, _optional_, defaults to 4) — The number of point embeddings to be used.
-   **hidden\_act** (`str`, _optional_, defaults to `"gelu"`) — The non-linear activation function in the encoder and pooler.

This is the configuration class to store the configuration of a `SamPromptEncoder`. The `SamPromptEncoder` module is used to encode the input 2D points and bounding boxes. Instantiating a configuration defaults will yield a similar configuration to that of the SAM-vit-h [facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## SamProcessor

### class transformers.SamProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sam/processing_sam.py#L35)

( image\_processor )

Parameters

-   **image\_processor** (`SamImageProcessor`) — An instance of [SamImageProcessor](/docs/transformers/v4.34.0/en/model_doc/sam#transformers.SamImageProcessor). The image processor is a required input.

Constructs a SAM processor which wraps a SAM image processor and an 2D points & Bounding boxes processor into a single processor.

[SamProcessor](/docs/transformers/v4.34.0/en/model_doc/sam#transformers.SamProcessor) offers all the functionalities of [SamImageProcessor](/docs/transformers/v4.34.0/en/model_doc/sam#transformers.SamImageProcessor). See the docstring of [**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for more information.

## SamImageProcessor

### class transformers.SamImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sam/image_processing_sam.py#L64)

( do\_resize: bool = Truesize: typing.Dict\[str, int\] = Noneresample: Resampling = <Resampling.BILINEAR: 2>do\_rescale: bool = Truerescale\_factor: typing.Union\[int, float\] = 0.00392156862745098do\_normalize: bool = Trueimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonedo\_pad: bool = Truepad\_size: int = Nonedo\_convert\_rgb: bool = True\*\*kwargs )

Constructs a SAM image processor.

#### filter\_masks

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sam/image_processing_sam.py#L631)

( masksiou\_scoresoriginal\_sizecropped\_box\_imagepred\_iou\_thresh = 0.88stability\_score\_thresh = 0.95mask\_threshold = 0stability\_score\_offset = 1return\_tensors = 'pt' )

Parameters

-   **masks** (`Union[torch.Tensor, tf.Tensor]`) — Input masks.
-   **iou\_scores** (`Union[torch.Tensor, tf.Tensor]`) — List of IoU scores.
-   **original\_size** (`Tuple[int,int]`) — Size of the orginal image.
-   **cropped\_box\_image** (`np.array`) — The cropped image.
-   **pred\_iou\_thresh** (`float`, _optional_, defaults to 0.88) — The threshold for the iou scores.
-   **stability\_score\_thresh** (`float`, _optional_, defaults to 0.95) — The threshold for the stability score.
-   **mask\_threshold** (`float`, _optional_, defaults to 0) — The threshold for the predicted masks.
-   **stability\_score\_offset** (`float`, _optional_, defaults to 1) — The offset for the stability score used in the `_compute_stability_score` method.
-   **return\_tensors** (`str`, _optional_, defaults to `pt`) — If `pt`, returns `torch.Tensor`. If `tf`, returns `tf.Tensor`.

Filters the predicted masks by selecting only the ones that meets several criteria. The first criterion being that the iou scores needs to be greater than `pred_iou_thresh`. The second criterion is that the stability score needs to be greater than `stability_score_thresh`. The method also converts the predicted masks to bounding boxes and pad the predicted masks if necessary.

#### generate\_crop\_boxes

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sam/image_processing_sam.py#L566)

( imagetarget\_sizecrop\_n\_layers: int = 0overlap\_ratio: float = 0.3413333333333333points\_per\_crop: typing.Optional\[int\] = 32crop\_n\_points\_downscale\_factor: typing.Optional\[typing.List\[int\]\] = 1device: typing.Optional\[ForwardRef('torch.device')\] = Noneinput\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = Nonereturn\_tensors: str = 'pt' )

Generates a list of crop boxes of different sizes. Each layer has (2**i)**2 boxes for the ith layer.

#### pad\_image

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sam/image_processing_sam.py#L142)

( image: ndarraypad\_size: typing.Dict\[str, int\]data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = Noneinput\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs )

Parameters

-   **image** (`np.ndarray`) — Image to pad.
-   **pad\_size** (`Dict[str, int]`) — Size of the output image after padding.
-   **data\_format** (`str` or `ChannelDimension`, _optional_) — The data format of the image. Can be either “channels\_first” or “channels\_last”. If `None`, the `data_format` of the `image` will be used.
-   **input\_data\_format** (`str` or `ChannelDimension`, _optional_) — The channel dimension format of the input image. If not provided, it will be inferred.

Pad an image to `(pad_size["height"], pad_size["width"])` with zeros to the right and bottom.

#### post\_process\_for\_mask\_generation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sam/image_processing_sam.py#L543)

( all\_masksall\_scoresall\_boxescrops\_nms\_threshreturn\_tensors = 'pt' )

Parameters

-   **all\_masks** (`Union[List[torch.Tensor], List[tf.Tensor]]`) — List of all predicted segmentation masks
-   **all\_scores** (`Union[List[torch.Tensor], List[tf.Tensor]]`) — List of all predicted iou scores
-   **all\_boxes** (`Union[List[torch.Tensor], List[tf.Tensor]]`) — List of all bounding boxes of the predicted masks
-   **crops\_nms\_thresh** (`float`) — Threshold for NMS (Non Maximum Suppression) algorithm.
-   **return\_tensors** (`str`, _optional_, defaults to `pt`) — If `pt`, returns `torch.Tensor`. If `tf`, returns `tf.Tensor`.

Post processes mask that are generated by calling the Non Maximum Suppression algorithm on the predicted masks.

#### post\_process\_masks

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sam/image_processing_sam.py#L399)

( masksoriginal\_sizesreshaped\_input\_sizesmask\_threshold = 0.0binarize = Truepad\_size = Nonereturn\_tensors = 'pt' ) → (`Union[torch.Tensor, tf.Tensor]`)

Remove padding and upscale masks to the original image size.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sam/image_processing_sam.py#L239)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]do\_resize: typing.Optional\[bool\] = Nonesize: typing.Union\[typing.Dict\[str, int\], NoneType\] = Noneresample: typing.Optional\[ForwardRef('PILImageResampling')\] = Nonedo\_rescale: typing.Optional\[bool\] = Nonerescale\_factor: typing.Union\[int, float, NoneType\] = Nonedo\_normalize: typing.Optional\[bool\] = Noneimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonedo\_pad: typing.Optional\[bool\] = Nonepad\_size: typing.Union\[typing.Dict\[str, int\], NoneType\] = Nonedo\_convert\_rgb: bool = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs )

Preprocess an image or batch of images.

#### resize

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sam/image_processing_sam.py#L190)

( image: ndarraysize: typing.Dict\[str, int\]resample: Resampling = <Resampling.BICUBIC: 3>data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = Noneinput\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs ) → `np.ndarray`

Resize an image to `(size["height"], size["width"])`.

## SamModel

### class transformers.SamModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sam/modeling_sam.py#L1192)

( config )

Parameters

-   **config** ([SamConfig](/docs/transformers/v4.34.0/en/model_doc/sam#transformers.SamConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Segment Anything Model (SAM) for generating segmentation masks, given an input image and optional 2D location and bounding boxes. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sam/modeling_sam.py#L1285)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Noneinput\_points: typing.Optional\[torch.FloatTensor\] = Noneinput\_labels: typing.Optional\[torch.LongTensor\] = Noneinput\_boxes: typing.Optional\[torch.FloatTensor\] = Noneinput\_masks: typing.Optional\[torch.LongTensor\] = Noneimage\_embeddings: typing.Optional\[torch.FloatTensor\] = Nonemultimask\_output: bool = Trueattention\_similarity: typing.Optional\[torch.FloatTensor\] = Nonetarget\_embedding: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None\*\*kwargs )

The [SamModel](/docs/transformers/v4.34.0/en/model_doc/sam#transformers.SamModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## TFSamModel

### class transformers.TFSamModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sam/modeling_tf_sam.py#L1237)

( \*args\*\*kwargs )

Parameters

-   **config** ([SamConfig](/docs/transformers/v4.34.0/en/model_doc/sam#transformers.SamConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

Segment Anything Model (SAM) for generating segmentation masks, given an input image and optional 2D location and bounding boxes. This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a TensorFlow [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TensorFlow Model and refer to the TensorFlow documentation for all matter related to general usage and behavior.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sam/modeling_tf_sam.py#L1327)

( pixel\_values: TFModelInputType | None = Noneinput\_points: tf.Tensor | None = Noneinput\_labels: tf.Tensor | None = Noneinput\_boxes: tf.Tensor | None = Noneinput\_masks: tf.Tensor | None = Noneimage\_embeddings: tf.Tensor | None = Nonemultimask\_output: bool = Trueoutput\_attentions: bool | None = Noneoutput\_hidden\_states: bool | None = Nonereturn\_dict: bool | None = Nonetraining: bool = False\*\*kwargs )

The [TFSamModel](/docs/transformers/v4.34.0/en/model_doc/sam#transformers.TFSamModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.