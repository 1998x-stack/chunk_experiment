# CLIPSeg

## Overview

The CLIPSeg model was proposed in [Image Segmentation Using Text and Image Prompts](https://arxiv.org/abs/2112.10003) by Timo Lüddecke and Alexander Ecker. CLIPSeg adds a minimal decoder on top of a frozen [CLIP](clip) model for zero- and one-shot image segmentation.

The abstract from the paper is the following:

_Image segmentation is usually addressed by training a model for a fixed set of object classes. Incorporating additional classes or more complex queries later is expensive as it requires re-training the model on a dataset that encompasses these expressions. Here we propose a system that can generate image segmentations based on arbitrary prompts at test time. A prompt can be either a text or an image. This approach enables us to create a unified model (trained once) for three common segmentation tasks, which come with distinct challenges: referring expression segmentation, zero-shot segmentation and one-shot segmentation. We build upon the CLIP model as a backbone which we extend with a transformer-based decoder that enables dense prediction. After training on an extended version of the PhraseCut dataset, our system generates a binary segmentation map for an image based on a free-text prompt or on an additional image expressing the query. We analyze different variants of the latter image-based prompts in detail. This novel hybrid input allows for dynamic adaptation not only to the three segmentation tasks mentioned above, but to any binary segmentation task where a text or image query can be formulated. Finally, we find our system to adapt well to generalized queries involving affordances or properties_

Tips:

-   [CLIPSegForImageSegmentation](/docs/transformers/v4.34.0/en/model_doc/clipseg#transformers.CLIPSegForImageSegmentation) adds a decoder on top of [CLIPSegModel](/docs/transformers/v4.34.0/en/model_doc/clipseg#transformers.CLIPSegModel). The latter is identical to [CLIPModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPModel).
-   [CLIPSegForImageSegmentation](/docs/transformers/v4.34.0/en/model_doc/clipseg#transformers.CLIPSegForImageSegmentation) can generate image segmentations based on arbitrary prompts at test time. A prompt can be either a text (provided to the model as `input_ids`) or an image (provided to the model as `conditional_pixel_values`). One can also provide custom conditional embeddings (provided to the model as `conditional_embeddings`).

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/clipseg_architecture.png) CLIPSeg overview. Taken from the [original paper.](https://arxiv.org/abs/2112.10003)

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/timojl/clipseg).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with CLIPSeg. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

-   A notebook that illustrates [zero-shot image segmentation with CLIPSeg](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/CLIPSeg/Zero_shot_image_segmentation_with_CLIPSeg.ipynb).

## CLIPSegConfig

### class transformers.CLIPSegConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clipseg/configuration_clipseg.py#L239)

( text\_config = Nonevision\_config = Noneprojection\_dim = 512logit\_scale\_init\_value = 2.6592extract\_layers = \[3, 6, 9\]reduce\_dim = 64decoder\_num\_attention\_heads = 4decoder\_attention\_dropout = 0.0decoder\_hidden\_act = 'quick\_gelu'decoder\_intermediate\_size = 2048conditional\_layer = 0use\_complex\_transposed\_convolution = False\*\*kwargs )

[CLIPSegConfig](/docs/transformers/v4.34.0/en/model_doc/clipseg#transformers.CLIPSegConfig) is the configuration class to store the configuration of a [CLIPSegModel](/docs/transformers/v4.34.0/en/model_doc/clipseg#transformers.CLIPSegModel). It is used to instantiate a CLIPSeg model according to the specified arguments, defining the text model and vision model configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the CLIPSeg [CIDAS/clipseg-rd64](https://huggingface.co/CIDAS/clipseg-rd64) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import CLIPSegConfig, CLIPSegModel

>>> 
>>> configuration = CLIPSegConfig()

>>> 
>>> model = CLIPSegModel(configuration)

>>> 
>>> configuration = model.config

>>> 

>>> 
>>> config_text = CLIPSegTextConfig()
>>> config_vision = CLIPSegVisionConfig()

>>> config = CLIPSegConfig.from_text_vision_configs(config_text, config_vision)
```

#### from\_text\_vision\_configs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clipseg/configuration_clipseg.py#L414)

( text\_config: CLIPSegTextConfigvision\_config: CLIPSegVisionConfig\*\*kwargs ) → [CLIPSegConfig](/docs/transformers/v4.34.0/en/model_doc/clipseg#transformers.CLIPSegConfig)

An instance of a configuration object

Instantiate a [CLIPSegConfig](/docs/transformers/v4.34.0/en/model_doc/clipseg#transformers.CLIPSegConfig) (or a derived class) from clipseg text model configuration and clipseg vision model configuration.

## CLIPSegTextConfig

### class transformers.CLIPSegTextConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clipseg/configuration_clipseg.py#L31)

( vocab\_size = 49408hidden\_size = 512intermediate\_size = 2048num\_hidden\_layers = 12num\_attention\_heads = 8max\_position\_embeddings = 77hidden\_act = 'quick\_gelu'layer\_norm\_eps = 1e-05attention\_dropout = 0.0initializer\_range = 0.02initializer\_factor = 1.0pad\_token\_id = 1bos\_token\_id = 49406eos\_token\_id = 49407\*\*kwargs )

This is the configuration class to store the configuration of a [CLIPSegModel](/docs/transformers/v4.34.0/en/model_doc/clipseg#transformers.CLIPSegModel). It is used to instantiate an CLIPSeg model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the CLIPSeg [CIDAS/clipseg-rd64](https://huggingface.co/CIDAS/clipseg-rd64) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import CLIPSegTextConfig, CLIPSegTextModel

>>> 
>>> configuration = CLIPSegTextConfig()

>>> 
>>> model = CLIPSegTextModel(configuration)

>>> 
>>> configuration = model.config
```

## CLIPSegVisionConfig

### class transformers.CLIPSegVisionConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clipseg/configuration_clipseg.py#L136)

( hidden\_size = 768intermediate\_size = 3072num\_hidden\_layers = 12num\_attention\_heads = 12num\_channels = 3image\_size = 224patch\_size = 32hidden\_act = 'quick\_gelu'layer\_norm\_eps = 1e-05attention\_dropout = 0.0initializer\_range = 0.02initializer\_factor = 1.0\*\*kwargs )

This is the configuration class to store the configuration of a [CLIPSegModel](/docs/transformers/v4.34.0/en/model_doc/clipseg#transformers.CLIPSegModel). It is used to instantiate an CLIPSeg model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the CLIPSeg [CIDAS/clipseg-rd64](https://huggingface.co/CIDAS/clipseg-rd64) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import CLIPSegVisionConfig, CLIPSegVisionModel

>>> 
>>> configuration = CLIPSegVisionConfig()

>>> 
>>> model = CLIPSegVisionModel(configuration)

>>> 
>>> configuration = model.config
```

## CLIPSegProcessor

### class transformers.CLIPSegProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clipseg/processing_clipseg.py#L25)

( image\_processor = Nonetokenizer = None\*\*kwargs )

Parameters

-   **image\_processor** ([ViTImageProcessor](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTImageProcessor)) — The image processor is a required input.
-   **tokenizer** ([CLIPTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPTokenizerFast)) — The tokenizer is a required input.

Constructs a CLIPSeg processor which wraps a CLIPSeg image processor and a CLIP tokenizer into a single processor.

[CLIPSegProcessor](/docs/transformers/v4.34.0/en/model_doc/clipseg#transformers.CLIPSegProcessor) offers all the functionalities of [ViTImageProcessor](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTImageProcessor) and [CLIPTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPTokenizerFast). See the `__call__()` and [decode()](/docs/transformers/v4.34.0/en/model_doc/clipseg#transformers.CLIPSegProcessor.decode) for more information.

This method forwards all its arguments to CLIPTokenizerFast’s [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.batch_decode). Please refer to the docstring of this method for more information.

This method forwards all its arguments to CLIPTokenizerFast’s [decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.decode). Please refer to the docstring of this method for more information.

## CLIPSegModel

### class transformers.CLIPSegModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clipseg/modeling_clipseg.py#L968)

( config: CLIPSegConfig )

Parameters

-   **config** ([CLIPSegConfig](/docs/transformers/v4.34.0/en/model_doc/clipseg#transformers.CLIPSegConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clipseg/modeling_clipseg.py#L1099)

( input\_ids: typing.Optional\[torch.LongTensor\] = Nonepixel\_values: typing.Optional\[torch.FloatTensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonereturn\_loss: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.clipseg.modeling_clipseg.CLIPSegOutput` or `tuple(torch.FloatTensor)`

The [CLIPSegModel](/docs/transformers/v4.34.0/en/model_doc/clipseg#transformers.CLIPSegModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, CLIPSegModel

>>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
>>> model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(
...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
... )

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  
>>> probs = logits_per_image.softmax(dim=1)  
```

#### get\_text\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clipseg/modeling_clipseg.py#L1003)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The [CLIPSegModel](/docs/transformers/v4.34.0/en/model_doc/clipseg#transformers.CLIPSegModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, CLIPSegModel

>>> tokenizer = AutoTokenizer.from_pretrained("CIDAS/clipseg-rd64-refined")
>>> model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
>>> text_features = model.get_text_features(**inputs)
```

#### get\_image\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clipseg/modeling_clipseg.py#L1050)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The image embeddings obtained by applying the projection layer to the pooled output of [CLIPSegVisionModel](/docs/transformers/v4.34.0/en/model_doc/clipseg#transformers.CLIPSegVisionModel).

The [CLIPSegModel](/docs/transformers/v4.34.0/en/model_doc/clipseg#transformers.CLIPSegModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, CLIPSegModel

>>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
>>> model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> image_features = model.get_image_features(**inputs)
```

## CLIPSegTextModel

### class transformers.CLIPSegTextModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clipseg/modeling_clipseg.py#L800)

( config: CLIPSegTextConfig )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clipseg/modeling_clipseg.py#L817)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

The [CLIPSegTextModel](/docs/transformers/v4.34.0/en/model_doc/clipseg#transformers.CLIPSegTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, CLIPSegTextModel

>>> tokenizer = AutoTokenizer.from_pretrained("CIDAS/clipseg-rd64-refined")
>>> model = CLIPSegTextModel.from_pretrained("CIDAS/clipseg-rd64-refined")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  
```

## CLIPSegVisionModel

### class transformers.CLIPSegVisionModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clipseg/modeling_clipseg.py#L915)

( config: CLIPSegVisionConfig )

The [CLIPSegVisionModel](/docs/transformers/v4.34.0/en/model_doc/clipseg#transformers.CLIPSegVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, CLIPSegVisionModel

>>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
>>> model = CLIPSegVisionModel.from_pretrained("CIDAS/clipseg-rd64-refined")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  
```

## CLIPSegForImageSegmentation

### class transformers.CLIPSegForImageSegmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clipseg/modeling_clipseg.py#L1356)

( config: CLIPSegConfig )

Parameters

-   **config** ([CLIPSegConfig](/docs/transformers/v4.34.0/en/model_doc/clipseg#transformers.CLIPSegConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

CLIPSeg model with a Transformer-based decoder on top for zero-shot and one-shot image segmentation.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clipseg/modeling_clipseg.py#L1401)

( input\_ids: typing.Optional\[torch.FloatTensor\] = Nonepixel\_values: typing.Optional\[torch.FloatTensor\] = Noneconditional\_pixel\_values: typing.Optional\[torch.FloatTensor\] = Noneconditional\_embeddings: typing.Optional\[torch.FloatTensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.clipseg.modeling_clipseg.CLIPSegImageSegmentationOutput` or `tuple(torch.FloatTensor)`

The [CLIPSegForImageSegmentation](/docs/transformers/v4.34.0/en/model_doc/clipseg#transformers.CLIPSegForImageSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoProcessor, CLIPSegForImageSegmentation
>>> from PIL import Image
>>> import requests

>>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
>>> model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> texts = ["a cat", "a remote", "a blanket"]
>>> inputs = processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt")

>>> outputs = model(**inputs)

>>> logits = outputs.logits
>>> print(logits.shape)
torch.Size([3, 352, 352])
```