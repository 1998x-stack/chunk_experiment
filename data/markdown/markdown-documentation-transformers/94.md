# AltCLIP

## Overview

The AltCLIP model was proposed in [AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities](https://arxiv.org/abs/2211.06679v2) by Zhongzhi Chen, Guang Liu, Bo-Wen Zhang, Fulong Ye, Qinghong Yang, Ledell Wu. AltCLIP (Altering the Language Encoder in CLIP) is a neural network trained on a variety of image-text and text-text pairs. By switching CLIP’s text encoder with a pretrained multilingual text encoder XLM-R, we could obtain very close performances with CLIP on almost all tasks, and extended original CLIP’s capabilities such as multilingual understanding.

The abstract from the paper is the following:

_In this work, we present a conceptually simple and effective method to train a strong bilingual multimodal representation model. Starting from the pretrained multimodal representation model CLIP released by OpenAI, we switched its text encoder with a pretrained multilingual text encoder XLM-R, and aligned both languages and image representations by a two-stage training schema consisting of teacher learning and contrastive learning. We validate our method through evaluations of a wide range of tasks. We set new state-of-the-art performances on a bunch of tasks including ImageNet-CN, Flicker30k- CN, and COCO-CN. Further, we obtain very close performances with CLIP on almost all tasks, suggesting that one can simply alter the text encoder in CLIP for extended capabilities such as multilingual understanding._

## Usage

The usage of AltCLIP is very similar to the CLIP. the difference between CLIP is the text encoder. Note that we use bidirectional attention instead of casual attention and we take the \[CLS\] token in XLM-R to represent text embedding.

AltCLIP is a multi-modal vision and language model. It can be used for image-text similarity and for zero-shot image classification. AltCLIP uses a ViT like transformer to get visual features and a bidirectional language model to get the text features. Both the text and visual features are then projected to a latent space with identical dimension. The dot product between the projected image and text features is then used as a similar score.

To feed images to the Transformer encoder, each image is split into a sequence of fixed-size non-overlapping patches, which are then linearly embedded. A \[CLS\] token is added to serve as representation of an entire image. The authors also add absolute position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder. The [CLIPImageProcessor](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPImageProcessor) can be used to resize (or rescale) and normalize images for the model.

The [AltCLIPProcessor](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPProcessor) wraps a [CLIPImageProcessor](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPImageProcessor) and a [XLMRobertaTokenizer](/docs/transformers/v4.34.0/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizer) into a single instance to both encode the text and prepare the images. The following example shows how to get the image-text similarity scores using [AltCLIPProcessor](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPProcessor) and [AltCLIPModel](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPModel).

```
>>> from PIL import Image
>>> import requests

>>> from transformers import AltCLIPModel, AltCLIPProcessor

>>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
>>> processor = AltCLIPProcessor.from_pretrained("BAAI/AltCLIP")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  
>>> probs = logits_per_image.softmax(dim=1)  
```

Tips:

This model is build on `CLIPModel`, so use it like a original CLIP.

This model was contributed by [jongjyh](https://huggingface.co/jongjyh).

## AltCLIPConfig

### class transformers.AltCLIPConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/altclip/configuration_altclip.py#L247)

( text\_config = None vision\_config = None projection\_dim = 768 logit\_scale\_init\_value = 2.6592 \*\*kwargs )

Parameters

-   **text\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [AltCLIPTextConfig](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPTextConfig).
-   **vision\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [AltCLIPVisionConfig](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPVisionConfig).
-   **projection\_dim** (`int`, _optional_, defaults to 512) — Dimentionality of text and vision projection layers.
-   **logit\_scale\_init\_value** (`float`, _optional_, defaults to 2.6592) — The inital value of the _logit\_scale_ paramter. Default is used as per the original CLIP implementation.
-   **kwargs** (_optional_) — Dictionary of keyword arguments.

This is the configuration class to store the configuration of a [AltCLIPModel](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPModel). It is used to instantiate an AltCLIP model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the AltCLIP [BAAI/AltCLIP](https://huggingface.co/BAAI/AltCLIP) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import AltCLIPConfig, AltCLIPModel

>>> 
>>> configuration = AltCLIPConfig()

>>> 
>>> model = AltCLIPModel(configuration)

>>> 
>>> configuration = model.config

>>> 

>>> 
>>> config_text = AltCLIPTextConfig()
>>> config_vision = AltCLIPVisionConfig()

>>> config = AltCLIPConfig.from_text_vision_configs(config_text, config_vision)
```

#### from\_text\_vision\_configs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/altclip/configuration_altclip.py#L382)

( text\_config: AltCLIPTextConfig vision\_config: AltCLIPVisionConfig \*\*kwargs ) → [AltCLIPConfig](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPConfig)

An instance of a configuration object

Instantiate a [AltCLIPConfig](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPConfig) (or a derived class) from altclip text model configuration and altclip vision model configuration.

## AltCLIPTextConfig

### class transformers.AltCLIPTextConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/altclip/configuration_altclip.py#L31)

( vocab\_size = 250002 hidden\_size = 1024 num\_hidden\_layers = 24 num\_attention\_heads = 16 intermediate\_size = 4096 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 514 type\_vocab\_size = 1 initializer\_range = 0.02 initializer\_factor = 0.02 layer\_norm\_eps = 1e-05 pad\_token\_id = 1 bos\_token\_id = 0 eos\_token\_id = 2 position\_embedding\_type = 'absolute' use\_cache = True project\_dim = 768 \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 250002) — Vocabulary size of the AltCLIP model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [AltCLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPTextModel).
-   **hidden\_size** (`int`, _optional_, defaults to 1024) — Dimensionality of the encoder layers and the pooler layer.
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 24) — Number of hidden layers in the Transformer encoder.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 16) — Number of attention heads for each attention layer in the Transformer encoder.
-   **intermediate\_size** (`int`, _optional_, defaults to 4096) — Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
-   **hidden\_act** (`str` or `Callable`, _optional_, defaults to `"gelu"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
-   **hidden\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
-   **attention\_probs\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout ratio for the attention probabilities.
-   **max\_position\_embeddings** (`int`, _optional_, defaults to 514) — The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
-   **type\_vocab\_size** (`int`, _optional_, defaults to 2) — The vocabulary size of the `token_type_ids` passed when calling [AltCLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPTextModel)
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-5) — The epsilon used by the layer normalization layers.
-   **position\_embedding\_type** (`str`, _optional_, defaults to `"absolute"`) — Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155). For more information on `"relative_key_query"`, please refer to _Method 4_ in [Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
-   **use\_cache** (`bool`, _optional_, defaults to `True`) — Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True`.
-   **project\_dim** (`int`, _optional_, defaults to 768) — The dimentions of the teacher model before the mapping layer.

This is the configuration class to store the configuration of a [AltCLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPTextModel). It is used to instantiate a AltCLIP text model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the AltCLIP [BAAI/AltCLIP](https://huggingface.co/BAAI/AltCLIP) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import AltCLIPTextModel, AltCLIPTextConfig

>>> 
>>> configuration = AltCLIPTextConfig()

>>> 
>>> model = AltCLIPTextModel(configuration)

>>> 
>>> configuration = model.config
```

## AltCLIPVisionConfig

### class transformers.AltCLIPVisionConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/altclip/configuration_altclip.py#L141)

( hidden\_size = 768 intermediate\_size = 3072 projection\_dim = 512 num\_hidden\_layers = 12 num\_attention\_heads = 12 num\_channels = 3 image\_size = 224 patch\_size = 32 hidden\_act = 'quick\_gelu' layer\_norm\_eps = 1e-05 attention\_dropout = 0.0 initializer\_range = 0.02 initializer\_factor = 1.0 \*\*kwargs )

Parameters

-   **hidden\_size** (`int`, _optional_, defaults to 768) — Dimensionality of the encoder layers and the pooler layer.
-   **intermediate\_size** (`int`, _optional_, defaults to 3072) — Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 12) — Number of hidden layers in the Transformer encoder.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 12) — Number of attention heads for each attention layer in the Transformer encoder.
-   **image\_size** (`int`, _optional_, defaults to 224) — The size (resolution) of each image.
-   **patch\_size** (`int`, _optional_, defaults to 32) — The size (resolution) of each patch.
-   **hidden\_act** (`str` or `function`, _optional_, defaults to `"quick_gelu"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` \``"quick_gelu"` are supported.
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-5) — The epsilon used by the layer normalization layers.
-   **attention\_dropout** (`float`, _optional_, defaults to 0.0) — The dropout ratio for the attention probabilities.
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **initializer\_factor** (\`float“, _optional_, defaults to 1) — A factor for initializing all weight matrices (should be kept to 1, used internally for initialization testing).

This is the configuration class to store the configuration of a [AltCLIPModel](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPModel). It is used to instantiate an AltCLIP model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the AltCLIP [BAAI/AltCLIP](https://huggingface.co/BAAI/AltCLIP) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import AltCLIPVisionConfig, AltCLIPVisionModel

>>> 
>>> configuration = AltCLIPVisionConfig()

>>> 
>>> model = AltCLIPVisionModel(configuration)

>>> 
>>> configuration = model.config
```

## AltCLIPProcessor

### class transformers.AltCLIPProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/altclip/processing_altclip.py#L24)

( image\_processor = None tokenizer = None \*\*kwargs )

Parameters

-   **image\_processor** ([CLIPImageProcessor](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPImageProcessor)) — The image processor is a required input.
-   **tokenizer** ([XLMRobertaTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizerFast)) — The tokenizer is a required input.

Constructs a AltCLIP processor which wraps a CLIP image processor and a XLM-Roberta tokenizer into a single processor.

[AltCLIPProcessor](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPProcessor) offers all the functionalities of [CLIPImageProcessor](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPImageProcessor) and [XLMRobertaTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizerFast). See the `__call__()` and [decode()](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPProcessor.decode) for more information.

This method forwards all its arguments to XLMRobertaTokenizerFast’s [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.batch_decode). Please refer to the docstring of this method for more information.

This method forwards all its arguments to XLMRobertaTokenizerFast’s [decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.decode). Please refer to the docstring of this method for more information.

## AltCLIPModel

### class transformers.AltCLIPModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/altclip/modeling_altclip.py#L1481)

( config: AltCLIPConfig )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/altclip/modeling_altclip.py#L1608)

( input\_ids: typing.Optional\[torch.LongTensor\] = None pixel\_values: typing.Optional\[torch.FloatTensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None token\_type\_ids: typing.Optional\[torch.Tensor\] = None return\_loss: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → `transformers.models.altclip.modeling_altclip.AltCLIPOutput` or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **return\_loss** (`bool`, _optional_) — Whether or not to return the contrastive loss.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.altclip.modeling_altclip.AltCLIPOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.altclip.modeling_altclip.AltCLIPOutput` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.altclip.configuration_altclip.AltCLIPConfig'>`) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `return_loss` is `True`) — Contrastive loss for image-text similarity.
-   **logits\_per\_image:(`torch.FloatTensor`** of shape `(image_batch_size, text_batch_size)`) — The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text similarity scores.
-   **logits\_per\_text:(`torch.FloatTensor`** of shape `(text_batch_size, image_batch_size)`) — The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image similarity scores.
-   **text\_embeds(`torch.FloatTensor`** of shape `(batch_size, output_dim`) — The text embeddings obtained by applying the projection layer to the pooled output of [AltCLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPTextModel).
-   **image\_embeds(`torch.FloatTensor`** of shape `(batch_size, output_dim`) — The image embeddings obtained by applying the projection layer to the pooled output of [AltCLIPVisionModel](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPVisionModel).
-   **text\_model\_output(`BaseModelOutputWithPooling`):** The output of the [AltCLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPTextModel).
-   **vision\_model\_output(`BaseModelOutputWithPooling`):** The output of the [AltCLIPVisionModel](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPVisionModel).

The [AltCLIPModel](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, AltCLIPModel

>>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
>>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")
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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/altclip/modeling_altclip.py#L1515)

( input\_ids: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.Tensor\] = None token\_type\_ids = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The text embeddings obtained by applying the projection layer to the pooled output of [AltCLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPTextModel).

The [AltCLIPModel](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoProcessor, AltCLIPModel

>>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
>>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")
>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
>>> text_features = model.get_text_features(**inputs)
```

#### get\_image\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/altclip/modeling_altclip.py#L1562)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The image embeddings obtained by applying the projection layer to the pooled output of [AltCLIPVisionModel](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPVisionModel).

The [AltCLIPModel](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, AltCLIPModel

>>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
>>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = processor(images=image, return_tensors="pt")
>>> image_features = model.get_image_features(**inputs)
```

## AltCLIPTextModel

### class transformers.AltCLIPTextModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/altclip/modeling_altclip.py#L1389)

( config )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/altclip/modeling_altclip.py#L1408)

( input\_ids: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None token\_type\_ids: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.Tensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None inputs\_embeds: typing.Optional\[torch.Tensor\] = None encoder\_hidden\_states: typing.Optional\[torch.Tensor\] = None encoder\_attention\_mask: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None ) → `transformers.modeling_outputs.BaseModelOutputWithPoolingAndProjection` or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.modeling_outputs.BaseModelOutputWithPoolingAndProjection` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.BaseModelOutputWithPoolingAndProjection` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.altclip.configuration_altclip.AltCLIPTextConfig'>`) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **projection\_state** (`tuple(torch.FloatTensor)`, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` of shape `(batch_size,config.project_dim)`.
    
    Text embeddings before the projection layer, used to mimic the last hidden state of the teacher encoder.
    

The [AltCLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoProcessor, AltCLIPTextModel

>>> model = AltCLIPTextModel.from_pretrained("BAAI/AltCLIP")
>>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")

>>> texts = ["it's a cat", "it's a dog"]

>>> inputs = processor(text=texts, padding=True, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  
```

## AltCLIPVisionModel

### class transformers.AltCLIPVisionModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/altclip/modeling_altclip.py#L1158)

( config: AltCLIPVisionConfig )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/altclip/modeling_altclip.py#L1171)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.altclip.configuration_altclip.AltCLIPVisionConfig'>`) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [AltCLIPVisionModel](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, AltCLIPVisionModel

>>> model = AltCLIPVisionModel.from_pretrained("BAAI/AltCLIP")
>>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  
```