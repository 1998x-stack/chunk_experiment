# OWL-ViT

## Overview

The OWL-ViT (short for Vision Transformer for Open-World Localization) was proposed in [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230) by Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf, and Neil Houlsby. OWL-ViT is an open-vocabulary object detection network trained on a variety of (image, text) pairs. It can be used to query an image with one or multiple text queries to search for and detect target objects described in text.

The abstract from the paper is the following:

_Combining simple architectures with large-scale pre-training has led to massive improvements in image classification. For object detection, pre-training and scaling approaches are less well established, especially in the long-tailed and open-vocabulary setting, where training data is relatively scarce. In this paper, we propose a strong recipe for transferring image-text models to open-vocabulary object detection. We use a standard Vision Transformer architecture with minimal modifications, contrastive image-text pre-training, and end-to-end detection fine-tuning. Our analysis of the scaling properties of this setup shows that increasing image-level pre-training and model size yield consistent improvements on the downstream detection task. We provide the adaptation strategies and regularizations needed to attain very strong performance on zero-shot text-conditioned and one-shot image-conditioned object detection. Code and models are available on GitHub._

## Usage

OWL-ViT is a zero-shot text-conditioned object detection model. OWL-ViT uses [CLIP](clip) as its multi-modal backbone, with a ViT-like Transformer to get visual features and a causal language model to get the text features. To use CLIP for detection, OWL-ViT removes the final token pooling layer of the vision model and attaches a lightweight classification and box head to each transformer output token. Open-vocabulary classification is enabled by replacing the fixed classification layer weights with the class-name embeddings obtained from the text model. The authors first train CLIP from scratch and fine-tune it end-to-end with the classification and box heads on standard detection datasets using a bipartite matching loss. One or multiple text queries per image can be used to perform zero-shot text-conditioned object detection.

[OwlViTImageProcessor](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTImageProcessor) can be used to resize (or rescale) and normalize images for the model and [CLIPTokenizer](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPTokenizer) is used to encode the text. [OwlViTProcessor](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTProcessor) wraps [OwlViTImageProcessor](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTImageProcessor) and [CLIPTokenizer](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPTokenizer) into a single instance to both encode the text and prepare the images. The following example shows how to perform object detection using [OwlViTProcessor](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTProcessor) and [OwlViTForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTForObjectDetection).

```
>>> import requests
>>> from PIL import Image
>>> import torch

>>> from transformers import OwlViTProcessor, OwlViTForObjectDetection

>>> processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
>>> model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> texts = [["a photo of a cat", "a photo of a dog"]]
>>> inputs = processor(text=texts, images=image, return_tensors="pt")
>>> outputs = model(**inputs)

>>> 
>>> target_sizes = torch.Tensor([image.size[::-1]])
>>> 
>>> results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
>>> i = 0  
>>> text = texts[i]
>>> boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
>>> for box, score, label in zip(boxes, scores, labels):
...     box = [round(i, 2) for i in box.tolist()]
...     print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
Detected a photo of a cat with confidence 0.707 at location [324.97, 20.44, 640.58, 373.29]
Detected a photo of a cat with confidence 0.717 at location [1.46, 55.26, 315.55, 472.17]
```

This model was contributed by [adirik](https://huggingface.co/adirik). The original code can be found [here](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit).

## OwlViTConfig

### class transformers.OwlViTConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/owlvit/configuration_owlvit.py#L251)

( text\_config = Nonevision\_config = Noneprojection\_dim = 512logit\_scale\_init\_value = 2.6592return\_dict = True\*\*kwargs )

Parameters

-   **text\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [OwlViTTextConfig](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTTextConfig).
-   **vision\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [OwlViTVisionConfig](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTVisionConfig).
-   **projection\_dim** (`int`, _optional_, defaults to 512) — Dimensionality of text and vision projection layers.
-   **logit\_scale\_init\_value** (`float`, _optional_, defaults to 2.6592) — The inital value of the _logit\_scale_ parameter. Default is used as per the original OWL-ViT implementation.
-   **kwargs** (_optional_) — Dictionary of keyword arguments.

[OwlViTConfig](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTConfig) is the configuration class to store the configuration of an [OwlViTModel](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTModel). It is used to instantiate an OWL-ViT model according to the specified arguments, defining the text model and vision model configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the OWL-ViT [google/owlvit-base-patch32](https://huggingface.co/google/owlvit-base-patch32) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

#### from\_text\_vision\_configs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/owlvit/configuration_owlvit.py#L318)

( text\_config: typing.Dictvision\_config: typing.Dict\*\*kwargs ) → [OwlViTConfig](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTConfig)

An instance of a configuration object

Instantiate a [OwlViTConfig](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTConfig) (or a derived class) from owlvit text model configuration and owlvit vision model configuration.

## OwlViTTextConfig

### class transformers.OwlViTTextConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/owlvit/configuration_owlvit.py#L40)

( vocab\_size = 49408hidden\_size = 512intermediate\_size = 2048num\_hidden\_layers = 12num\_attention\_heads = 8max\_position\_embeddings = 16hidden\_act = 'quick\_gelu'layer\_norm\_eps = 1e-05attention\_dropout = 0.0initializer\_range = 0.02initializer\_factor = 1.0pad\_token\_id = 0bos\_token\_id = 49406eos\_token\_id = 49407\*\*kwargs )

This is the configuration class to store the configuration of an [OwlViTTextModel](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTTextModel). It is used to instantiate an OwlViT text encoder according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the OwlViT [google/owlvit-base-patch32](https://huggingface.co/google/owlvit-base-patch32) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import OwlViTTextConfig, OwlViTTextModel

>>> 
>>> configuration = OwlViTTextConfig()

>>> 
>>> model = OwlViTTextModel(configuration)

>>> 
>>> configuration = model.config
```

## OwlViTVisionConfig

### class transformers.OwlViTVisionConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/owlvit/configuration_owlvit.py#L146)

( hidden\_size = 768intermediate\_size = 3072num\_hidden\_layers = 12num\_attention\_heads = 12num\_channels = 3image\_size = 768patch\_size = 32hidden\_act = 'quick\_gelu'layer\_norm\_eps = 1e-05attention\_dropout = 0.0initializer\_range = 0.02initializer\_factor = 1.0\*\*kwargs )

This is the configuration class to store the configuration of an [OwlViTVisionModel](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTVisionModel). It is used to instantiate an OWL-ViT image encoder according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the OWL-ViT [google/owlvit-base-patch32](https://huggingface.co/google/owlvit-base-patch32) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import OwlViTVisionConfig, OwlViTVisionModel

>>> 
>>> configuration = OwlViTVisionConfig()

>>> 
>>> model = OwlViTVisionModel(configuration)

>>> 
>>> configuration = model.config
```

## OwlViTImageProcessor

### class transformers.OwlViTImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/owlvit/image_processing_owlvit.py#L91)

( do\_resize = Truesize = Noneresample = <Resampling.BICUBIC: 3>do\_center\_crop = Falsecrop\_size = Nonedo\_rescale = Truerescale\_factor = 0.00392156862745098do\_normalize = Trueimage\_mean = Noneimage\_std = None\*\*kwargs )

Constructs an OWL-ViT image processor.

This image processor inherits from [ImageProcessingMixin](/docs/transformers/v4.34.0/en/main_classes/image_processor#transformers.ImageProcessingMixin) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/owlvit/image_processing_owlvit.py#L269)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]do\_resize: typing.Optional\[bool\] = Nonesize: typing.Union\[typing.Dict\[str, int\], NoneType\] = Noneresample: Resampling = Nonedo\_center\_crop: typing.Optional\[bool\] = Nonecrop\_size: typing.Union\[typing.Dict\[str, int\], NoneType\] = Nonedo\_rescale: typing.Optional\[bool\] = Nonerescale\_factor: typing.Optional\[float\] = Nonedo\_normalize: typing.Optional\[bool\] = Noneimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension\] = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs )

Prepares an image or batch of images for the model.

#### post\_process\_object\_detection

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/owlvit/image_processing_owlvit.py#L458)

( outputsthreshold: float = 0.1target\_sizes: typing.Union\[transformers.utils.generic.TensorType, typing.List\[typing.Tuple\]\] = None ) → `List[Dict]`

Parameters

-   **outputs** (`OwlViTObjectDetectionOutput`) — Raw outputs of the model.
-   **threshold** (`float`, _optional_) — Score threshold to keep object detection predictions.
-   **target\_sizes** (`torch.Tensor` or `List[Tuple[int, int]]`, _optional_) — Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size `(height, width)` of each image in the batch. If unset, predictions will not be resized.

A list of dictionaries, each dictionary containing the scores, labels and boxes for an image in the batch as predicted by the model.

Converts the raw output of [OwlViTForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTForObjectDetection) into final bounding boxes in (top\_left\_x, top\_left\_y, bottom\_right\_x, bottom\_right\_y) format.

## OwlViTFeatureExtractor

Preprocess an image or a batch of images.

( outputstarget\_sizes ) → `List[Dict]`

Parameters

-   **outputs** (`OwlViTObjectDetectionOutput`) — Raw outputs of the model.
-   **target\_sizes** (`torch.Tensor` of shape `(batch_size, 2)`) — Tensor containing the size (h, w) of each image of the batch. For evaluation, this must be the original image size (before any data augmentation). For visualization, this should be the image size after data augment, but before padding.

A list of dictionaries, each dictionary containing the scores, labels and boxes for an image in the batch as predicted by the model.

Converts the raw output of [OwlViTForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTForObjectDetection) into final bounding boxes in (top\_left\_x, top\_left\_y, bottom\_right\_x, bottom\_right\_y) format.

## OwlViTProcessor

### class transformers.OwlViTProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/owlvit/processing_owlvit.py#L29)

( image\_processor = Nonetokenizer = None\*\*kwargs )

Parameters

-   **image\_processor** ([OwlViTImageProcessor](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTImageProcessor)) — The image processor is a required input.
-   **tokenizer** (\[`CLIPTokenizer`, `CLIPTokenizerFast`\]) — The tokenizer is a required input.

Constructs an OWL-ViT processor which wraps [OwlViTImageProcessor](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTImageProcessor) and [CLIPTokenizer](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPTokenizer)/[CLIPTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPTokenizerFast) into a single processor that interits both the image processor and tokenizer functionalities. See the `__call__()` and [decode()](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTProcessor.decode) for more information.

This method forwards all its arguments to CLIPTokenizerFast’s [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.batch_decode). Please refer to the docstring of this method for more information.

This method forwards all its arguments to CLIPTokenizerFast’s [decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.decode). Please refer to the docstring of this method for more information.

#### post\_process\_image\_guided\_detection

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/owlvit/processing_owlvit.py#L189)

( \*args\*\*kwargs )

This method forwards all its arguments to `OwlViTImageProcessor.post_process_one_shot_object_detection`. Please refer to the docstring of this method for more information.

## OwlViTModel

### class transformers.OwlViTModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/owlvit/modeling_owlvit.py#L1038)

( config: OwlViTConfig )

Parameters

-   **This** model is a PyTorch \[torch.nn.Module\](https — //pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
-   **as** a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and — behavior. — config ([OwlViTConfig](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTConfig)): Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/owlvit/modeling_owlvit.py#L1153)

( input\_ids: typing.Optional\[torch.LongTensor\] = Nonepixel\_values: typing.Optional\[torch.FloatTensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonereturn\_loss: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_base\_image\_embeds: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.owlvit.modeling_owlvit.OwlViTOutput` or `tuple(torch.FloatTensor)`

The [OwlViTModel](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, OwlViTModel

>>> model = OwlViTModel.from_pretrained("google/owlvit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = processor(text=[["a photo of a cat", "a photo of a dog"]], images=image, return_tensors="pt")
>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  
>>> probs = logits_per_image.softmax(dim=1)  
```

#### get\_text\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/owlvit/modeling_owlvit.py#L1073)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The [OwlViTModel](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoProcessor, OwlViTModel

>>> model = OwlViTModel.from_pretrained("google/owlvit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
>>> inputs = processor(
...     text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]], return_tensors="pt"
... )
>>> text_features = model.get_text_features(**inputs)
```

#### get\_image\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/owlvit/modeling_owlvit.py#L1108)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The image embeddings obtained by applying the projection layer to the pooled output of [OwlViTVisionModel](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTVisionModel).

The [OwlViTModel](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, OwlViTModel

>>> model = OwlViTModel.from_pretrained("google/owlvit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = processor(images=image, return_tensors="pt")
>>> image_features = model.get_image_features(**inputs)
```

## OwlViTTextModel

### class transformers.OwlViTTextModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/owlvit/modeling_owlvit.py#L877)

( config: OwlViTTextConfig )

The [OwlViTTextModel](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoProcessor, OwlViTTextModel

>>> model = OwlViTTextModel.from_pretrained("google/owlvit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
>>> inputs = processor(
...     text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]], return_tensors="pt"
... )
>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  
```

## OwlViTVisionModel

### class transformers.OwlViTVisionModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/owlvit/modeling_owlvit.py#L987)

( config: OwlViTVisionConfig )

The [OwlViTVisionModel](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, OwlViTVisionModel

>>> model = OwlViTVisionModel.from_pretrained("google/owlvit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  
```

## OwlViTForObjectDetection

### class transformers.OwlViTForObjectDetection

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/owlvit/modeling_owlvit.py#L1320)

( config: OwlViTConfig )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/owlvit/modeling_owlvit.py#L1622)

( input\_ids: Tensorpixel\_values: FloatTensorattention\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.owlvit.modeling_owlvit.OwlViTObjectDetectionOutput` or `tuple(torch.FloatTensor)`

The [OwlViTForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTForObjectDetection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import requests
>>> from PIL import Image
>>> import torch
>>> from transformers import AutoProcessor, OwlViTForObjectDetection

>>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
>>> model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> texts = [["a photo of a cat", "a photo of a dog"]]
>>> inputs = processor(text=texts, images=image, return_tensors="pt")
>>> outputs = model(**inputs)

>>> 
>>> target_sizes = torch.Tensor([image.size[::-1]])
>>> 
>>> results = processor.post_process_object_detection(
...     outputs=outputs, threshold=0.1, target_sizes=target_sizes
... )

>>> i = 0  
>>> text = texts[i]
>>> boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

>>> for box, score, label in zip(boxes, scores, labels):
...     box = [round(i, 2) for i in box.tolist()]
...     print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
Detected a photo of a cat with confidence 0.707 at location [324.97, 20.44, 640.58, 373.29]
Detected a photo of a cat with confidence 0.717 at location [1.46, 55.26, 315.55, 472.17]
```

#### image\_guided\_detection

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/owlvit/modeling_owlvit.py#L1527)

( pixel\_values: FloatTensorquery\_pixel\_values: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.owlvit.modeling_owlvit.OwlViTImageGuidedObjectDetectionOutput` or `tuple(torch.FloatTensor)`

The [OwlViTForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTForObjectDetection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import requests
>>> from PIL import Image
>>> import torch
>>> from transformers import AutoProcessor, OwlViTForObjectDetection

>>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch16")
>>> model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16")
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> query_url = "http://images.cocodataset.org/val2017/000000001675.jpg"
>>> query_image = Image.open(requests.get(query_url, stream=True).raw)
>>> inputs = processor(images=image, query_images=query_image, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model.image_guided_detection(**inputs)
>>> 
>>> target_sizes = torch.Tensor([image.size[::-1]])
>>> 
>>> results = processor.post_process_image_guided_detection(
...     outputs=outputs, threshold=0.6, nms_threshold=0.3, target_sizes=target_sizes
... )
>>> i = 0  
>>> boxes, scores = results[i]["boxes"], results[i]["scores"]
>>> for box, score in zip(boxes, scores):
...     box = [round(i, 2) for i in box.tolist()]
...     print(f"Detected similar object with confidence {round(score.item(), 3)} at location {box}")
Detected similar object with confidence 0.856 at location [10.94, 50.4, 315.8, 471.39]
Detected similar object with confidence 1.0 at location [334.84, 25.33, 636.16, 374.71]
```