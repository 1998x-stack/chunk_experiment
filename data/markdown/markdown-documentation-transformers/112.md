# BLIP

## Overview

The BLIP model was proposed in [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086) by Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi.

BLIP is a model that is able to perform various multi-modal tasks including

-   Visual Question Answering
-   Image-Text retrieval (Image-text matching)
-   Image Captioning

The abstract from the paper is the following:

_Vision-Language Pre-training (VLP) has advanced the performance for many vision-language tasks. However, most existing pre-trained models only excel in either understanding-based tasks or generation-based tasks. Furthermore, performance improvement has been largely achieved by scaling up the dataset with noisy image-text pairs collected from the web, which is a suboptimal source of supervision. In this paper, we propose BLIP, a new VLP framework which transfers flexibly to both vision-language understanding and generation tasks. BLIP effectively utilizes the noisy web data by bootstrapping the captions, where a captioner generates synthetic captions and a filter removes the noisy ones. We achieve state-of-the-art results on a wide range of vision-language tasks, such as image-text retrieval (+2.7% in average recall@1), image captioning (+2.8% in CIDEr), and VQA (+1.6% in VQA score). BLIP also demonstrates strong generalization ability when directly transferred to videolanguage tasks in a zero-shot manner. Code, models, and datasets are released._

![BLIP.gif](https://cdn-uploads.huggingface.co/production/uploads/1670928184033-62441d1d9fdefb55a0b7d12c.gif)

This model was contributed by [ybelkada](https://huggingface.co/ybelkada). The original code can be found [here](https://github.com/salesforce/BLIP).

## Resources

-   [Jupyter notebook](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_blip.ipynb) on how to fine-tune BLIP for image captioning on a custom dataset

## BlipConfig

### class transformers.BlipConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/configuration_blip.py#L279)

( text\_config = Nonevision\_config = Noneprojection\_dim = 512logit\_scale\_init\_value = 2.6592image\_text\_hidden\_size = 256\*\*kwargs )

Parameters

-   **text\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [BlipTextConfig](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipTextConfig).
-   **vision\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [BlipVisionConfig](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipVisionConfig).
-   **projection\_dim** (`int`, _optional_, defaults to 512) — Dimentionality of text and vision projection layers.
-   **logit\_scale\_init\_value** (`float`, _optional_, defaults to 2.6592) — The inital value of the _logit\_scale_ paramter. Default is used as per the original BLIP implementation.
-   **image\_text\_hidden\_size** (`int`, _optional_, defaults to 768) — Dimentionality of the hidden state of the image-text fusion layer.
-   **kwargs** (_optional_) — Dictionary of keyword arguments.

[BlipConfig](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipConfig) is the configuration class to store the configuration of a [BlipModel](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipModel). It is used to instantiate a BLIP model according to the specified arguments, defining the text model and vision model configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the BLIP-base [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import BlipConfig, BlipModel

>>> 
>>> configuration = BlipConfig()

>>> 
>>> model = BlipModel(configuration)

>>> 
>>> configuration = model.config

>>> 

>>> 
>>> config_text = BlipTextConfig()
>>> config_vision = BlipVisionConfig()

>>> config = BlipConfig.from_text_vision_configs(config_text, config_vision)
```

#### from\_text\_vision\_configs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/configuration_blip.py#L358)

( text\_config: BlipTextConfigvision\_config: BlipVisionConfig\*\*kwargs ) → [BlipConfig](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipConfig)

An instance of a configuration object

Instantiate a [BlipConfig](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipConfig) (or a derived class) from blip text model configuration and blip vision model configuration.

## BlipTextConfig

### class transformers.BlipTextConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/configuration_blip.py#L46)

( vocab\_size = 30524hidden\_size = 768encoder\_hidden\_size = 768intermediate\_size = 3072projection\_dim = 768num\_hidden\_layers = 12num\_attention\_heads = 8max\_position\_embeddings = 512hidden\_act = 'gelu'layer\_norm\_eps = 1e-12hidden\_dropout\_prob = 0.0attention\_probs\_dropout\_prob = 0.0initializer\_range = 0.02bos\_token\_id = 30522eos\_token\_id = 2pad\_token\_id = 0sep\_token\_id = 102is\_decoder = Trueuse\_cache = True\*\*kwargs )

This is the configuration class to store the configuration of a [BlipTextModel](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipTextModel). It is used to instantiate a BLIP text model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the `BlipText` used by the [base architectures](https://huggingface.co/Salesforce/blip-vqa-base).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import BlipTextConfig, BlipTextModel

>>> 
>>> configuration = BlipTextConfig()

>>> 
>>> model = BlipTextModel(configuration)

>>> 
>>> configuration = model.config
```

## BlipVisionConfig

### class transformers.BlipVisionConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/configuration_blip.py#L180)

( hidden\_size = 768intermediate\_size = 3072projection\_dim = 512num\_hidden\_layers = 12num\_attention\_heads = 12image\_size = 384patch\_size = 16hidden\_act = 'gelu'layer\_norm\_eps = 1e-05attention\_dropout = 0.0initializer\_range = 1e-10\*\*kwargs )

This is the configuration class to store the configuration of a [BlipVisionModel](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipVisionModel). It is used to instantiate a BLIP vision model according to the specified arguments, defining the model architecture. Instantiating a configuration defaults will yield a similar configuration to that of the Blip-base [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import BlipVisionConfig, BlipVisionModel

>>> 
>>> configuration = BlipVisionConfig()

>>> 
>>> model = BlipVisionModel(configuration)

>>> 
>>> configuration = model.config
```

## BlipProcessor

### class transformers.BlipProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/processing_blip.py#L27)

( image\_processortokenizer )

Parameters

-   **image\_processor** (`BlipImageProcessor`) — An instance of [BlipImageProcessor](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipImageProcessor). The image processor is a required input.
-   **tokenizer** (`BertTokenizerFast`) — An instance of \[‘BertTokenizerFast\`\]. The tokenizer is a required input.

Constructs a BLIP processor which wraps a BERT tokenizer and BLIP image processor into a single processor.

[BlipProcessor](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipProcessor) offers all the functionalities of [BlipImageProcessor](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipImageProcessor) and [BertTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizerFast). See the docstring of `__call__()` and [decode()](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipProcessor.decode) for more information.

This method forwards all its arguments to BertTokenizerFast’s [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.batch_decode). Please refer to the docstring of this method for more information.

This method forwards all its arguments to BertTokenizerFast’s [decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.decode). Please refer to the docstring of this method for more information.

## BlipImageProcessor

### class transformers.BlipImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/image_processing_blip.py#L45)

( do\_resize: bool = Truesize: typing.Dict\[str, int\] = Noneresample: Resampling = <Resampling.BICUBIC: 3>do\_rescale: bool = Truerescale\_factor: typing.Union\[int, float\] = 0.00392156862745098do\_normalize: bool = Trueimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonedo\_convert\_rgb: bool = True\*\*kwargs )

Constructs a BLIP image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/image_processing_blip.py#L158)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]do\_resize: typing.Optional\[bool\] = Nonesize: typing.Union\[typing.Dict\[str, int\], NoneType\] = Noneresample: Resampling = Nonedo\_rescale: typing.Optional\[bool\] = Nonerescale\_factor: typing.Optional\[float\] = Nonedo\_normalize: typing.Optional\[bool\] = Noneimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedo\_convert\_rgb: bool = Nonedata\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs )

Preprocess an image or batch of images.

## BlipModel

### class transformers.BlipModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_blip.py#L726)

( config: BlipConfig )

Parameters

-   **config** ([BlipConfig](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_blip.py#L836)

( input\_ids: typing.Optional\[torch.LongTensor\] = Nonepixel\_values: typing.Optional\[torch.FloatTensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonereturn\_loss: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.blip.modeling_blip.BlipOutput` or `tuple(torch.FloatTensor)`

The [BlipModel](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, BlipModel

>>> model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
>>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_blip.py#L761)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The [BlipModel](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoProcessor, BlipModel

>>> model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
>>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
>>> text_features = model.get_text_features(**inputs)
```

#### get\_image\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_blip.py#L799)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using [BlipImageProcessor](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipImageProcessor). See [BlipImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The image embeddings obtained by applying the projection layer to the pooled output of [BlipVisionModel](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipVisionModel).

The [BlipModel](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, BlipModel

>>> model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
>>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> image_features = model.get_image_features(**inputs)
```

## BlipTextModel

### class transformers.BlipTextModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_blip_text.py#L572)

( configadd\_pooling\_layer = True )

The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of cross-attention is added between the self-attention layers, following the architecture described in [Attention is all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin. argument and `is_decoder` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_blip_text.py#L671)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Noneencoder\_embeds: typing.Optional\[torch.Tensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.Tensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Noneis\_decoder: typing.Optional\[bool\] = False )

encoder\_hidden\_states (`torch.FloatTensor`, _optional_): Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model is configured as a decoder. encoder\_attention\_mask (`torch.FloatTensor`, _optional_): Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

-   1 for tokens that are **not masked**,
-   0 for tokens that are **masked**. past\_key\_values (`tuple(tuple(torch.FloatTensor))`, _optional_): Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`. use\_cache (`bool`, _optional_): If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

## BlipVisionModel

### class transformers.BlipVisionModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_blip.py#L659)

( config: BlipVisionConfig )

The [BlipVisionModel](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## BlipForConditionalGeneration

### class transformers.BlipForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_blip.py#L939)

( config: BlipConfig )

Parameters

-   **config** ([BlipConfig](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

BLIP Model for image captioning. The model consists of a vision encoder and a text decoder. One can optionally pass `input_ids` to the model, which serve as a text prompt, to make the text decoder continue the prompt. Otherwise, the decoder starts generating text from the \[BOS\] (beginning-of-sequence) token. will start generating the caption from the text input. If no text input is provided, the decoder will start with the \[BOS\] token only.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_blip.py#L960)

( pixel\_values: FloatTensorinput\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.blip.modeling_blip.BlipForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)`

The [BlipForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, BlipForConditionalGeneration

>>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
>>> model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> text = "A picture of"

>>> inputs = processor(images=image, text=text, return_tensors="pt")

>>> outputs = model(**inputs)
```

## BlipForImageTextRetrieval

### class transformers.BlipForImageTextRetrieval

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_blip.py#L1333)

( config: BlipConfig )

Parameters

-   **config** ([BlipConfig](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

BLIP Model with a vision and text projector, and a classification head on top. The model is used in the context of image-text retrieval. Given an image and a text, the model returns the probability of the text being relevant to the image.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_blip.py#L1369)

( input\_ids: LongTensorpixel\_values: FloatTensoruse\_itm\_head: typing.Optional\[bool\] = Trueattention\_mask: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.blip.modeling_blip.BlipTextVisionModelOutput` or `tuple(torch.FloatTensor)`

The [BlipForImageTextRetrieval](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipForImageTextRetrieval) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, BlipForImageTextRetrieval

>>> model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
>>> processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> text = "an image of a cat"

>>> inputs = processor(images=image, text=text, return_tensors="pt")
>>> outputs = model(**inputs)
```

## BlipForQuestionAnswering

### class transformers.BlipForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_blip.py#L1111)

( config: BlipConfig )

Parameters

-   **config** ([BlipConfig](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

BLIP Model for visual question answering. The model consists of a vision encoder, a text encoder as well as a text decoder. The vision encoder will encode the input image, the text encoder will encode the input question together with the encoding of the image, and the text decoder will output the answer to the question.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_blip.py#L1133)

( input\_ids: LongTensorpixel\_values: FloatTensordecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.blip.modeling_blip.BlipTextVisionModelOutput` or `tuple(torch.FloatTensor)`

The [BlipForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, BlipForQuestionAnswering

>>> model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
>>> processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> 
>>> text = "How many cats are in the picture?"
>>> label = "2"
>>> inputs = processor(images=image, text=text, return_tensors="pt")
>>> labels = processor(text=label, return_tensors="pt").input_ids

>>> inputs["labels"] = labels
>>> outputs = model(**inputs)
>>> loss = outputs.loss
>>> loss.backward()

>>> 
>>> text = "How many cats are in the picture?"
>>> inputs = processor(images=image, text=text, return_tensors="pt")
>>> outputs = model.generate(**inputs)
>>> print(processor.decode(outputs[0], skip_special_tokens=True))
2
```

## TFBlipModel

### class transformers.TFBlipModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_tf_blip.py#L853)

( \*args\*\*kwargs )

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_tf_blip.py#L871)

( input\_ids: tf.Tensor | None = Nonepixel\_values: tf.Tensor | None = Noneattention\_mask: tf.Tensor | None = Noneposition\_ids: tf.Tensor | None = Nonereturn\_loss: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: Optional\[bool\] = None ) → `transformers.models.blip.modeling_tf_blip.TFBlipOutput` or `tuple(tf.Tensor)`

The [TFBlipModel](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.TFBlipModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, TFBlipModel

>>> model = TFBlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
>>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(
...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="tf", padding=True
... )

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  
>>> probs = tf.nn.softmax(logits_per_image, axis=1)  
```

#### get\_text\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_tf_blip.py#L923)

( input\_ids: tf.Tensor | None = Noneattention\_mask: tf.Tensor | None = Noneposition\_ids: tf.Tensor | None = Nonereturn\_dict: Optional\[bool\] = None ) → text\_features (`tf.Tensor` of shape `(batch_size, output_dim`)

The [TFBlipModel](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.TFBlipModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoProcessor, TFBlipModel

>>> model = TFBlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
>>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")
>>> text_features = model.get_text_features(**inputs)
```

#### get\_image\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_tf_blip.py#L961)

( pixel\_values: tf.Tensor | None = Nonereturn\_dict: Optional\[bool\] = None ) → image\_features (`tf.Tensor` of shape `(batch_size, output_dim`)

Parameters

-   **pixel\_values** (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using [BlipImageProcessor](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipImageProcessor). See [BlipImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

image\_features (`tf.Tensor` of shape `(batch_size, output_dim`)

The image embeddings obtained by applying the projection layer to the pooled output of [TFBlipVisionModel](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.TFBlipVisionModel).

The [TFBlipModel](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.TFBlipModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, TFBlipModel

>>> model = TFBlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
>>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="tf")

>>> image_features = model.get_image_features(**inputs)
```

## TFBlipTextModel

### class transformers.TFBlipTextModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_tf_blip_text.py#L580)

( \*args\*\*kwargs )

The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of cross-attention is added between the self-attention layers, following the architecture described in [Attention is all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin. argument and `is_decoder` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_tf_blip_text.py#L668)

( input\_ids: TFModelInputType | None = Noneattention\_mask: tf.Tensor | None = Noneposition\_ids: tf.Tensor | None = Nonehead\_mask: tf.Tensor | None = Noneinputs\_embeds: tf.Tensor | None = Noneencoder\_embeds: tf.Tensor | None = Noneencoder\_hidden\_states: tf.Tensor | None = Noneencoder\_attention\_mask: tf.Tensor | None = Nonepast\_key\_values: Tuple\[Tuple\[tf.Tensor\]\] | None = Noneuse\_cache: bool | None = Noneoutput\_attentions: bool | None = Noneoutput\_hidden\_states: bool | None = Nonereturn\_dict: bool | None = Noneis\_decoder: bool = Falsetraining: bool = False )

The [TFBlipTextModel](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.TFBlipTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## TFBlipVisionModel

### class transformers.TFBlipVisionModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_tf_blip.py#L649)

( \*args\*\*kwargs )

The [TFBlipVisionModel](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.TFBlipVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## TFBlipForConditionalGeneration

### class transformers.TFBlipForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_tf_blip.py#L1008)

( \*args\*\*kwargs )

Parameters

-   **config** ([BlipConfig](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

BLIP Model for image captioning. The model consists of a vision encoder and a text decoder. One can optionally pass `input_ids` to the model, which serve as a text prompt, to make the text decoder continue the prompt. Otherwise, the decoder starts generating text from the \[BOS\] (beginning-of-sequence) token. will start generating the caption from the text input. If no text input is provided, the decoder will start with the \[BOS\] token only.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_tf_blip.py#L1026)

( pixel\_values: tf.Tensorinput\_ids: tf.Tensor | None = Noneattention\_mask: tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonelabels: tf.Tensor | None = Nonereturn\_dict: Optional\[bool\] = Nonetraining: Optional\[bool\] = None ) → `transformers.models.blip.modeling_tf_blip.TFBlipForConditionalGenerationModelOutput` or `tuple(tf.Tensor)`

The [TFBlipForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.TFBlipForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, TFBlipForConditionalGeneration

>>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
>>> model = TFBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> text = "A picture of"

>>> inputs = processor(images=image, text=text, return_tensors="tf")

>>> outputs = model(**inputs)
```

## TFBlipForImageTextRetrieval

### class transformers.TFBlipForImageTextRetrieval

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_tf_blip.py#L1421)

( \*args\*\*kwargs )

Parameters

-   **config** ([BlipConfig](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

BLIP Model with a vision and text projector, and a classification head on top. The model is used in the context of image-text retrieval. Given an image and a text, the model returns the probability of the text being relevant to the image.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_tf_blip.py#L1464)

( input\_ids: tf.Tensorpixel\_values: tf.Tensor | None = Noneuse\_itm\_head: Optional\[bool\] = Trueattention\_mask: tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: Optional\[bool\] = None ) → `transformers.models.blip.modeling_tf_blip.TFBlipImageTextMatchingModelOutput` or `tuple(tf.Tensor)`

The [TFBlipForImageTextRetrieval](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.TFBlipForImageTextRetrieval) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, TFBlipForImageTextRetrieval

>>> model = TFBlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
>>> processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> text = "an image of a cat"

>>> inputs = processor(images=image, text=text, return_tensors="tf")
>>> outputs = model(**inputs)
```

## TFBlipForQuestionAnswering

### class transformers.TFBlipForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_tf_blip.py#L1180)

( \*args\*\*kwargs )

Parameters

-   **config** ([BlipConfig](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

BLIP Model for visual question answering. The model consists of a vision encoder, a text encoder as well as a text decoder. The vision encoder will encode the input image, the text encoder will encode the input question together with the encoding of the image, and the text decoder will output the answer to the question.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip/modeling_tf_blip.py#L1223)

( input\_ids: tf.Tensorpixel\_values: tf.Tensor | None = Nonedecoder\_input\_ids: tf.Tensor | None = Nonedecoder\_attention\_mask: tf.Tensor | None = Noneattention\_mask: tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonelabels: tf.Tensor | None = Nonereturn\_dict: Optional\[bool\] = Nonetraining: Optional\[bool\] = None ) → `transformers.models.blip.modeling_tf_blip.TFBlipTextVisionModelOutput` or `tuple(tf.Tensor)`

The [TFBlipForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.TFBlipForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, TFBlipForQuestionAnswering

>>> model = TFBlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
>>> processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> 
>>> text = "How many cats are in the picture?"
>>> label = "2"
>>> inputs = processor(images=image, text=text, return_tensors="tf")
>>> labels = processor(text=label, return_tensors="tf").input_ids

>>> inputs["labels"] = labels
>>> outputs = model(**inputs)
>>> loss = outputs.loss

>>> 
>>> text = "How many cats are in the picture?"
>>> inputs = processor(images=image, text=text, return_tensors="tf")
>>> outputs = model.generate(**inputs)
>>> print(processor.decode(outputs[0], skip_special_tokens=True))
2
```