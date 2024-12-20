# ALIGN

## Overview

The ALIGN model was proposed in [Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918) by Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig. ALIGN is a multi-modal vision and language model. It can be used for image-text similarity and for zero-shot image classification. ALIGN features a dual-encoder architecture with [EfficientNet](efficientnet) as its vision encoder and [BERT](bert) as its text encoder, and learns to align visual and text representations with contrastive learning. Unlike previous work, ALIGN leverages a massive noisy dataset and shows that the scale of the corpus can be used to achieve SOTA representations with a simple recipe.

The abstract from the paper is the following:

_Pre-trained representations are becoming crucial for many NLP and perception tasks. While representation learning in NLP has transitioned to training on raw text without human annotations, visual and vision-language representations still rely heavily on curated training datasets that are expensive or require expert knowledge. For vision applications, representations are mostly learned using datasets with explicit class labels such as ImageNet or OpenImages. For vision-language, popular datasets like Conceptual Captions, MSCOCO, or CLIP all involve a non-trivial data collection (and cleaning) process. This costly curation process limits the size of datasets and hence hinders the scaling of trained models. In this paper, we leverage a noisy dataset of over one billion image alt-text pairs, obtained without expensive filtering or post-processing steps in the Conceptual Captions dataset. A simple dual-encoder architecture learns to align visual and language representations of the image and text pairs using a contrastive loss. We show that the scale of our corpus can make up for its noise and leads to state-of-the-art representations even with such a simple learning scheme. Our visual representation achieves strong performance when transferred to classification tasks such as ImageNet and VTAB. The aligned visual and language representations enables zero-shot image classification and also set new state-of-the-art results on Flickr30K and MSCOCO image-text retrieval benchmarks, even when compared with more sophisticated cross-attention models. The representations also enable cross-modality search with complex text and text + image queries._

## Usage

ALIGN uses EfficientNet to get visual features and BERT to get the text features. Both the text and visual features are then projected to a latent space with identical dimension. The dot product between the projected image and text features is then used as a similarity score.

[AlignProcessor](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignProcessor) wraps [EfficientNetImageProcessor](/docs/transformers/v4.34.0/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor) and [BertTokenizer](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizer) into a single instance to both encode the text and preprocess the images. The following example shows how to get the image-text similarity scores using [AlignProcessor](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignProcessor) and [AlignModel](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignModel).

```
import requests
import torch
from PIL import Image
from transformers import AlignProcessor, AlignModel

processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
model = AlignModel.from_pretrained("kakaobrain/align-base")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
candidate_labels = ["an image of a cat", "an image of a dog"]

inputs = processor(text=candidate_labels, images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)


logits_per_image = outputs.logits_per_image


probs = logits_per_image.softmax(dim=1)
print(probs)
```

This model was contributed by [Alara Dirik](https://huggingface.co/adirik). The original code is not released, this implementation is based on the Kakao Brain implementation based on the original paper.

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with ALIGN.

-   A blog post on [ALIGN and the COYO-700M dataset](https://huggingface.co/blog/vit-align).
-   A zero-shot image classification [demo](https://huggingface.co/spaces/adirik/ALIGN-zero-shot-image-classification).
-   [Model card](https://huggingface.co/kakaobrain/align-base) of `kakaobrain/align-base` model.

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we will review it. The resource should ideally demonstrate something new instead of duplicating an existing resource.

## AlignConfig

### class transformers.AlignConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/align/configuration_align.py#L297)

( text\_config = None vision\_config = None projection\_dim = 640 temperature\_init\_value = 1.0 initializer\_range = 0.02 \*\*kwargs )

Parameters

-   **text\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [AlignTextConfig](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignTextConfig).
-   **vision\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [AlignVisionConfig](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignVisionConfig).
-   **projection\_dim** (`int`, _optional_, defaults to 640) — Dimentionality of text and vision projection layers.
-   **temperature\_init\_value** (`float`, _optional_, defaults to 1.0) — The inital value of the _temperature_ paramter. Default is used as per the original ALIGN implementation.
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **kwargs** (_optional_) — Dictionary of keyword arguments.

[AlignConfig](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignConfig) is the configuration class to store the configuration of a [AlignModel](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignModel). It is used to instantiate a ALIGN model according to the specified arguments, defining the text model and vision model configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the ALIGN [kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import AlignConfig, AlignModel

>>> 
>>> configuration = AlignConfig()

>>> 
>>> model = AlignModel(configuration)

>>> 
>>> configuration = model.config

>>> 
>>> from transformers import AlignTextConfig, AlignVisionConfig

>>> 
>>> config_text = AlignTextConfig()
>>> config_vision = AlignVisionConfig()

>>> config = AlignConfig.from_text_vision_configs(config_text, config_vision)
```

#### from\_text\_vision\_configs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/align/configuration_align.py#L373)

( text\_config: AlignTextConfig vision\_config: AlignVisionConfig \*\*kwargs ) → [AlignConfig](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignConfig)

An instance of a configuration object

Instantiate a [AlignConfig](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignConfig) (or a derived class) from align text model configuration and align vision model configuration.

## AlignTextConfig

### class transformers.AlignTextConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/align/configuration_align.py#L35)

( vocab\_size = 30522 hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 512 type\_vocab\_size = 2 initializer\_range = 0.02 layer\_norm\_eps = 1e-12 pad\_token\_id = 0 position\_embedding\_type = 'absolute' use\_cache = True \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 30522) — Vocabulary size of the Align Text model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [AlignTextModel](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignTextModel).
-   **hidden\_size** (`int`, _optional_, defaults to 768) — Dimensionality of the encoder layers and the pooler layer.
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 12) — Number of hidden layers in the Transformer encoder.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 12) — Number of attention heads for each attention layer in the Transformer encoder.
-   **intermediate\_size** (`int`, _optional_, defaults to 3072) — Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
-   **hidden\_act** (`str` or `Callable`, _optional_, defaults to `"gelu"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
-   **hidden\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
-   **attention\_probs\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout ratio for the attention probabilities.
-   **max\_position\_embeddings** (`int`, _optional_, defaults to 512) — The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
-   **type\_vocab\_size** (`int`, _optional_, defaults to 2) — The vocabulary size of the `token_type_ids` passed when calling [AlignTextModel](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignTextModel).
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-12) — The epsilon used by the layer normalization layers.
-   **position\_embedding\_type** (`str`, _optional_, defaults to `"absolute"`) — Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155). For more information on `"relative_key_query"`, please refer to _Method 4_ in [Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
-   **use\_cache** (`bool`, _optional_, defaults to `True`) — Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True`.
-   **pad\_token\_id** (`int`, _optional_, defaults to 0) — Padding token id.

This is the configuration class to store the configuration of a [AlignTextModel](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignTextModel). It is used to instantiate a ALIGN text encoder according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the text encoder of the ALIGN [kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base) architecture. The default values here are copied from BERT.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import AlignTextConfig, AlignTextModel

>>> 
>>> configuration = AlignTextConfig()

>>> 
>>> model = AlignTextModel(configuration)

>>> 
>>> configuration = model.config
```

## AlignVisionConfig

### class transformers.AlignVisionConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/align/configuration_align.py#L158)

( num\_channels: int = 3 image\_size: int = 600 width\_coefficient: float = 2.0 depth\_coefficient: float = 3.1 depth\_divisor: int = 8 kernel\_sizes: typing.List\[int\] = \[3, 3, 5, 3, 5, 5, 3\] in\_channels: typing.List\[int\] = \[32, 16, 24, 40, 80, 112, 192\] out\_channels: typing.List\[int\] = \[16, 24, 40, 80, 112, 192, 320\] depthwise\_padding: typing.List\[int\] = \[\] strides: typing.List\[int\] = \[1, 2, 2, 2, 1, 2, 1\] num\_block\_repeats: typing.List\[int\] = \[1, 2, 2, 3, 3, 4, 1\] expand\_ratios: typing.List\[int\] = \[1, 6, 6, 6, 6, 6, 6\] squeeze\_expansion\_ratio: float = 0.25 hidden\_act: str = 'swish' hidden\_dim: int = 2560 pooling\_type: str = 'mean' initializer\_range: float = 0.02 batch\_norm\_eps: float = 0.001 batch\_norm\_momentum: float = 0.99 drop\_connect\_rate: float = 0.2 \*\*kwargs )

Parameters

-   **num\_channels** (`int`, _optional_, defaults to 3) — The number of input channels.
-   **image\_size** (`int`, _optional_, defaults to 600) — The input image size.
-   **width\_coefficient** (`float`, _optional_, defaults to 2.0) — Scaling coefficient for network width at each stage.
-   **depth\_coefficient** (`float`, _optional_, defaults to 3.1) — Scaling coefficient for network depth at each stage.
-   **depth\_divisor** `int`, _optional_, defaults to 8) — A unit of network width.
-   **kernel\_sizes** (`List[int]`, _optional_, defaults to `[3, 3, 5, 3, 5, 5, 3]`) — List of kernel sizes to be used in each block.
-   **in\_channels** (`List[int]`, _optional_, defaults to `[32, 16, 24, 40, 80, 112, 192]`) — List of input channel sizes to be used in each block for convolutional layers.
-   **out\_channels** (`List[int]`, _optional_, defaults to `[16, 24, 40, 80, 112, 192, 320]`) — List of output channel sizes to be used in each block for convolutional layers.
-   **depthwise\_padding** (`List[int]`, _optional_, defaults to `[]`) — List of block indices with square padding.
-   **strides** (`List[int]`, _optional_, defaults to `[1, 2, 2, 2, 1, 2, 1]`) — List of stride sizes to be used in each block for convolutional layers.
-   **num\_block\_repeats** (`List[int]`, _optional_, defaults to `[1, 2, 2, 3, 3, 4, 1]`) — List of the number of times each block is to repeated.
-   **expand\_ratios** (`List[int]`, _optional_, defaults to `[1, 6, 6, 6, 6, 6, 6]`) — List of scaling coefficient of each block.
-   **squeeze\_expansion\_ratio** (`float`, _optional_, defaults to 0.25) — Squeeze expansion ratio.
-   **hidden\_act** (`str` or `function`, _optional_, defaults to `"silu"`) — The non-linear activation function (function or string) in each block. If string, `"gelu"`, `"relu"`, `"selu",` “gelu\_new”`,` “silu”`and`“mish”\` are supported.
-   **hiddem\_dim** (`int`, _optional_, defaults to 1280) — The hidden dimension of the layer before the classification head.
-   **pooling\_type** (`str` or `function`, _optional_, defaults to `"mean"`) — Type of final pooling to be applied before the dense classification head. Available options are \[`"mean"`, `"max"`\]
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **batch\_norm\_eps** (`float`, _optional_, defaults to 1e-3) — The epsilon used by the batch normalization layers.
-   **batch\_norm\_momentum** (`float`, _optional_, defaults to 0.99) — The momentum used by the batch normalization layers.
-   **drop\_connect\_rate** (`float`, _optional_, defaults to 0.2) — The drop rate for skip connections.

This is the configuration class to store the configuration of a [AlignVisionModel](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignVisionModel). It is used to instantiate a ALIGN vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the vision encoder of the ALIGN [kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base) architecture. The default values are copied from EfficientNet (efficientnet-b7)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import AlignVisionConfig, AlignVisionModel

>>> 
>>> configuration = AlignVisionConfig()

>>> 
>>> model = AlignVisionModel(configuration)

>>> 
>>> configuration = model.config
```

## AlignProcessor

### class transformers.AlignProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/align/processing_align.py#L24)

( image\_processor tokenizer )

Parameters

-   **image\_processor** ([EfficientNetImageProcessor](/docs/transformers/v4.34.0/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor)) — The image processor is a required input.
-   **tokenizer** (\[`BertTokenizer`, `BertTokenizerFast`\]) — The tokenizer is a required input.

Constructs an ALIGN processor which wraps [EfficientNetImageProcessor](/docs/transformers/v4.34.0/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor) and [BertTokenizer](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizer)/[BertTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizerFast) into a single processor that interits both the image processor and tokenizer functionalities. See the `__call__()` and [decode()](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTProcessor.decode) for more information.

This method forwards all its arguments to BertTokenizerFast’s [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.batch_decode). Please refer to the docstring of this method for more information.

This method forwards all its arguments to BertTokenizerFast’s [decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.decode). Please refer to the docstring of this method for more information.

## AlignModel

### class transformers.AlignModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/align/modeling_align.py#L1420)

( config: AlignConfig )

Parameters

-   **config** ([AlignConfig](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/align/modeling_align.py#L1551)

( input\_ids: typing.Optional\[torch.LongTensor\] = None pixel\_values: typing.Optional\[torch.FloatTensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None token\_type\_ids: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.Tensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None inputs\_embeds: typing.Optional\[torch.Tensor\] = None return\_loss: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None )

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids) attention\_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_): Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask) position\_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_): Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids) token\_type\_ids (`torch.LongTensor` of shape `({0})`, _optional_): Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids) head\_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_): Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
    inputs\_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, _optional_): Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix. pixel\_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`): Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [EfficientNetImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details. return\_loss (`bool`, _optional_): Whether or not to return the contrastive loss. output\_attentions (`bool`, _optional_): Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. output\_hidden\_states (`bool`, _optional_): Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. return\_dict (`bool`, _optional_): Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
    

The [AlignModel](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

#### get\_text\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/align/modeling_align.py#L1453)

( input\_ids: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None token\_type\_ids: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.Tensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None inputs\_embeds: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

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
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `({0})`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `({0}, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The text embeddings obtained by applying the projection layer to the pooled output of [AlignTextModel](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignTextModel).

The [AlignModel](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, AlignModel

>>> model = AlignModel.from_pretrained("kakaobrain/align-base")
>>> tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
>>> text_features = model.get_text_features(**inputs)
```

#### get\_image\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/align/modeling_align.py#L1506)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [EfficientNetImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The image embeddings obtained by applying the projection layer to the pooled output of [AlignVisionModel](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignVisionModel).

The [AlignModel](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, AlignModel

>>> model = AlignModel.from_pretrained("kakaobrain/align-base")
>>> processor = AutoProcessor.from_pretrained("kakaobrain/align-base")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> image_features = model.get_image_features(**inputs)
```

## AlignTextModel

### class transformers.AlignTextModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/align/modeling_align.py#L1209)

( config: AlignTextConfig add\_pooling\_layer: bool = True )

Parameters

-   **config** ([AlignConfig](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The text model from ALIGN without any head or projection on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/align/modeling_align.py#L1230)

( input\_ids: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None token\_type\_ids: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.Tensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None inputs\_embeds: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or `tuple(torch.FloatTensor)`

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
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `({0})`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `({0}, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.align.configuration_align.AlignTextConfig'>`) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    

The [AlignTextModel](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, AlignTextModel

>>> model = AlignTextModel.from_pretrained("kakaobrain/align-base")
>>> tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  
```

## AlignVisionModel

### class transformers.AlignVisionModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/align/modeling_align.py#L1335)

( config: AlignVisionConfig )

Parameters

-   **config** ([AlignConfig](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The vision model from ALIGN without any head or projection on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/align/modeling_align.py#L1359)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [EfficientNetImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.align.configuration_align.AlignVisionConfig'>`) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state after a pooling operation on the spatial dimensions.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    

The [AlignVisionModel](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, AlignVisionModel

>>> model = AlignVisionModel.from_pretrained("kakaobrain/align-base")
>>> processor = AutoProcessor.from_pretrained("kakaobrain/align-base")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  
```