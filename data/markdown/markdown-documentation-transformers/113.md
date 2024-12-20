# BLIP-2

## Overview

The BLIP-2 model was proposed in [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) by Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi. BLIP-2 leverages frozen pre-trained image encoders and large language models (LLMs) by training a lightweight, 12-layer Transformer encoder in between them, achieving state-of-the-art performance on various vision-language tasks. Most notably, BLIP-2 improves upon [Flamingo](https://arxiv.org/abs/2204.14198), an 80 billion parameter model, by 8.7% on zero-shot VQAv2 with 54x fewer trainable parameters.

The abstract from the paper is the following:

_The cost of vision-and-language pre-training has become increasingly prohibitive due to end-to-end training of large-scale models. This paper proposes BLIP-2, a generic and efficient pre-training strategy that bootstraps vision-language pre-training from off-the-shelf frozen pre-trained image encoders and frozen large language models. BLIP-2 bridges the modality gap with a lightweight Querying Transformer, which is pre-trained in two stages. The first stage bootstraps vision-language representation learning from a frozen image encoder. The second stage bootstraps vision-to-language generative learning from a frozen language model. BLIP-2 achieves state-of-the-art performance on various vision-language tasks, despite having significantly fewer trainable parameters than existing methods. For example, our model outperforms Flamingo80B by 8.7% on zero-shot VQAv2 with 54x fewer trainable parameters. We also demonstrate the model’s emerging capabilities of zero-shot image-to-text generation that can follow natural language instructions._

Tips:

-   BLIP-2 can be used for conditional text generation given an image and an optional text prompt. At inference time, it’s recommended to use the `generate` method.
-   One can use [Blip2Processor](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2Processor) to prepare images for the model, and decode the predicted tokens ID’s back to text.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/blip2_architecture.jpg) BLIP-2 architecture. Taken from the [original paper.](https://arxiv.org/abs/2301.12597)

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/salesforce/LAVIS/tree/5ee63d688ba4cebff63acee04adaef2dee9af207).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with BLIP-2.

-   Demo notebooks for BLIP-2 for image captioning, visual question answering (VQA) and chat-like conversations can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/BLIP-2).

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## Blip2Config

### class transformers.Blip2Config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip_2/configuration_blip_2.py#L250)

( vision\_config = Noneqformer\_config = Nonetext\_config = Nonenum\_query\_tokens = 32\*\*kwargs )

Parameters

-   **vision\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [Blip2VisionConfig](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2VisionConfig).
-   **qformer\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [Blip2QFormerConfig](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2QFormerConfig).
-   **text\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize any [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig).
-   **num\_query\_tokens** (`int`, _optional_, defaults to 32) — The number of query tokens passed through the Transformer.
-   **kwargs** (_optional_) — Dictionary of keyword arguments.

[Blip2Config](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2Config) is the configuration class to store the configuration of a [Blip2ForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2ForConditionalGeneration). It is used to instantiate a BLIP-2 model according to the specified arguments, defining the vision model, Q-Former model and language model configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the BLIP-2 [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import (
...     Blip2VisionConfig,
...     Blip2QFormerConfig,
...     OPTConfig,
...     Blip2Config,
...     Blip2ForConditionalGeneration,
... )

>>> 
>>> configuration = Blip2Config()

>>> 
>>> model = Blip2ForConditionalGeneration(configuration)

>>> 
>>> configuration = model.config

>>> 

>>> 
>>> vision_config = Blip2VisionConfig()
>>> qformer_config = Blip2QFormerConfig()
>>> text_config = OPTConfig()

>>> config = Blip2Config.from_text_vision_configs(vision_config, qformer_config, text_config)
```

#### from\_vision\_qformer\_text\_configs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip_2/configuration_blip_2.py#L334)

( vision\_config: Blip2VisionConfigqformer\_config: Blip2QFormerConfigtext\_config: PretrainedConfig\*\*kwargs ) → [Blip2Config](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2Config)

An instance of a configuration object

Instantiate a [Blip2Config](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2Config) (or a derived class) from a BLIP-2 vision model, Q-Former and language model configurations.

## Blip2VisionConfig

### class transformers.Blip2VisionConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip_2/configuration_blip_2.py#L33)

( hidden\_size = 1408intermediate\_size = 6144num\_hidden\_layers = 39num\_attention\_heads = 16image\_size = 224patch\_size = 14hidden\_act = 'gelu'layer\_norm\_eps = 1e-06attention\_dropout = 0.0initializer\_range = 1e-10qkv\_bias = True\*\*kwargs )

This is the configuration class to store the configuration of a [Blip2VisionModel](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2VisionModel). It is used to instantiate a BLIP-2 vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration defaults will yield a similar configuration to that of the BLIP-2 [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import Blip2VisionConfig, Blip2VisionModel

>>> 
>>> configuration = Blip2VisionConfig()

>>> 
>>> model = Blip2VisionModel(configuration)

>>> 
>>> configuration = model.config
```

## Blip2QFormerConfig

### class transformers.Blip2QFormerConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip_2/configuration_blip_2.py#L132)

( vocab\_size = 30522hidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout\_prob = 0.1attention\_probs\_dropout\_prob = 0.1max\_position\_embeddings = 512initializer\_range = 0.02layer\_norm\_eps = 1e-12pad\_token\_id = 0position\_embedding\_type = 'absolute'cross\_attention\_frequency = 2encoder\_hidden\_size = 1408\*\*kwargs )

This is the configuration class to store the configuration of a [Blip2QFormerModel](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2QFormerModel). It is used to instantiate a BLIP-2 Querying Transformer (Q-Former) model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the BLIP-2 [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) architecture. Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Note that [Blip2QFormerModel](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2QFormerModel) is very similar to [BertLMHeadModel](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertLMHeadModel) with interleaved cross-attention.

Examples:

```
>>> from transformers import Blip2QFormerConfig, Blip2QFormerModel

>>> 
>>> configuration = Blip2QFormerConfig()

>>> 
>>> model = Blip2QFormerModel(configuration)
>>> 
>>> configuration = model.config
```

## Blip2Processor

### class transformers.Blip2Processor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip_2/processing_blip_2.py#L27)

( image\_processortokenizer )

Parameters

-   **image\_processor** (`BlipImageProcessor`) — An instance of [BlipImageProcessor](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipImageProcessor). The image processor is a required input.
-   **tokenizer** (`AutoTokenizer`) — An instance of \[‘PreTrainedTokenizer\`\]. The tokenizer is a required input.

Constructs a BLIP-2 processor which wraps a BLIP image processor and an OPT/T5 tokenizer into a single processor.

[BlipProcessor](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipProcessor) offers all the functionalities of [BlipImageProcessor](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipImageProcessor) and [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See the docstring of `__call__()` and [decode()](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipProcessor.decode) for more information.

This method forwards all its arguments to PreTrainedTokenizer’s [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.batch_decode). Please refer to the docstring of this method for more information.

This method forwards all its arguments to PreTrainedTokenizer’s [decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.decode). Please refer to the docstring of this method for more information.

## Blip2VisionModel

### class transformers.Blip2VisionModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip_2/modeling_blip_2.py#L511)

( config: Blip2VisionConfig )

The [Blip2VisionModel](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2VisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## Blip2QFormerModel

### class transformers.Blip2QFormerModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip_2/modeling_blip_2.py#L1006)

( config: Blip2QFormerConfig )

Querying Transformer (Q-Former), used in BLIP-2.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip_2/modeling_blip_2.py#L1081)

( query\_embeds: FloatTensorattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None )

encoder\_hidden\_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, `optional`): Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model is configured as a decoder. encoder\_attention\_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`): Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

-   1 for tokens that are **not masked**,
-   0 for tokens that are **masked**. past\_key\_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of: shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`. use\_cache (`bool`, `optional`): If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

## Blip2Model

### class transformers.Blip2Model

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip_2/modeling_blip_2.py#L1202)

( config: Blip2Config )

Parameters

-   **config** ([Blip2Config](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

BLIP-2 Model for generating text and image features. The model consists of a vision encoder, Querying Transformer (Q-Former) and a language model.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip_2/modeling_blip_2.py#L1425)

( pixel\_values: FloatTensorinput\_ids: FloatTensorattention\_mask: typing.Optional\[torch.LongTensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.blip_2.modeling_blip_2.Blip2ForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)`

The [Blip2Model](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import Blip2Processor, Blip2Model
>>> import torch

>>> device = "cuda" if torch.cuda.is_available() else "cpu"

>>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
>>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
>>> model.to(device)
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> prompt = "Question: how many cats are there? Answer:"
>>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

>>> outputs = model(**inputs)
```

#### get\_text\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip_2/modeling_blip_2.py#L1252)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.Tensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → text\_outputs (`CausalLMOutputWithPast`, or `tuple(torch.FloatTensor)` if `return_dict=False`)

The [Blip2Model](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import torch
>>> from transformers import AutoTokenizer, Blip2Model

>>> device = "cuda" if torch.cuda.is_available() else "cpu"

>>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

>>> model.to(device)
>>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt").to(device)
>>> text_features = model.get_text_features(**inputs)
```

#### get\_image\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip_2/modeling_blip_2.py#L1315)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → vision\_outputs (`BaseModelOutputWithPooling` or tuple of `torch.FloatTensor`)

The [Blip2Model](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import torch
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Blip2Model

>>> device = "cuda" if torch.cuda.is_available() else "cpu"

>>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

>>> model.to(device)
>>> processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
>>> image_outputs = model.get_image_features(**inputs)
```

#### get\_qformer\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip_2/modeling_blip_2.py#L1363)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → vision\_outputs (`BaseModelOutputWithPooling` or tuple of `torch.FloatTensor`)

The [Blip2Model](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import torch
>>> from PIL import Image
>>> import requests
>>> from transformers import Blip2Processor, Blip2Model

>>> device = "cuda" if torch.cuda.is_available() else "cpu"

>>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
>>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
>>> model.to(device)
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
>>> qformer_outputs = model.get_qformer_features(**inputs)
```

## Blip2ForConditionalGeneration

### class transformers.Blip2ForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip_2/modeling_blip_2.py#L1568)

( config: Blip2Config )

Parameters

-   **config** ([Blip2Config](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

BLIP-2 Model for generating text given an image and an optional text prompt. The model consists of a vision encoder, Querying Transformer (Q-Former) and a language model.

One can optionally pass `input_ids` to the model, which serve as a text prompt, to make the language model continue the prompt. Otherwise, the language model starts generating text from the \[BOS\] (beginning-of-sequence) token.

Note that Flan-T5 checkpoints cannot be cast to float16. They are pre-trained using bfloat16.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip_2/modeling_blip_2.py#L1638)

( pixel\_values: FloatTensorinput\_ids: FloatTensorattention\_mask: typing.Optional\[torch.LongTensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.blip_2.modeling_blip_2.Blip2ForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)`

The [Blip2ForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2ForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

Image captioning (without providing a text prompt):

```
>>> from PIL import Image
>>> import requests
>>> from transformers import Blip2Processor, Blip2ForConditionalGeneration
>>> import torch

>>> device = "cuda" if torch.cuda.is_available() else "cpu"

>>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
>>> model = Blip2ForConditionalGeneration.from_pretrained(
...     "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
... )
>>> model.to(device)
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

>>> generated_ids = model.generate(**inputs)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
>>> print(generated_text)
two cats laying on a couch
```

Visual question answering (prompt = question):

```
>>> from PIL import Image
>>> import requests
>>> from transformers import Blip2Processor, Blip2ForConditionalGeneration
>>> import torch

>>> device = "cuda" if torch.cuda.is_available() else "cpu"

>>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
>>> model = Blip2ForConditionalGeneration.from_pretrained(
...     "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
... )  

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> prompt = "Question: how many cats are there? Answer:"
>>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)

>>> generated_ids = model.generate(**inputs)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
>>> print(generated_text)
two
```

Note that int8 inference is also supported through [bitsandbytes](https://github.com/TimDettmers/bitsandbytes). This greatly reduces the amount of memory used by the model while maintaining the same performance.

```
>>> from PIL import Image
>>> import requests
>>> from transformers import Blip2Processor, Blip2ForConditionalGeneration
>>> import torch

>>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
>>> model = Blip2ForConditionalGeneration.from_pretrained(
...     "Salesforce/blip2-flan-t5-xl", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.bfloat16
... )  

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> prompt = "Question: how many cats are there? Answer:"
>>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)

>>> generated_ids = model.generate(**inputs)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
>>> print(generated_text)
two
```

#### generate

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blip_2/modeling_blip_2.py#L1823)

( pixel\_values: FloatTensorinput\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.LongTensor\] = None\*\*generate\_kwargs ) → captions (list)

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape (batch\_size, num\_channels, height, width)) — Input images to be processed.
-   **input\_ids** (`torch.LongTensor` of shape (batch\_size, sequence\_length), _optional_) — The sequence used as a prompt for the generation.
-   **attention\_mask** (`torch.LongTensor` of shape (batch\_size, sequence\_length), _optional_) — Mask to avoid performing attention on padding token indices

A list of strings of length batch\_size \* num\_captions.

Overrides `generate` function to be able to use the model as a conditional generator.