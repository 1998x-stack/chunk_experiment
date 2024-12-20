# InstructBLIP

## Overview

The InstructBLIP model was proposed in [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](https://arxiv.org/abs/2305.06500) by Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, Steven Hoi. InstructBLIP leverages the [BLIP-2](blip2) architecture for visual instruction tuning.

The abstract from the paper is the following:

_General-purpose language models that can solve various language-domain tasks have emerged driven by the pre-training and instruction-tuning pipeline. However, building general-purpose vision-language models is challenging due to the increased task discrepancy introduced by the additional visual input. Although vision-language pre-training has been widely studied, vision-language instruction tuning remains relatively less explored. In this paper, we conduct a systematic and comprehensive study on vision-language instruction tuning based on the pre-trained BLIP-2 models. We gather a wide variety of 26 publicly available datasets, transform them into instruction tuning format and categorize them into two clusters for held-in instruction tuning and held-out zero-shot evaluation. Additionally, we introduce instruction-aware visual feature extraction, a crucial method that enables the model to extract informative features tailored to the given instruction. The resulting InstructBLIP models achieve state-of-the-art zero-shot performance across all 13 held-out datasets, substantially outperforming BLIP-2 and the larger Flamingo. Our models also lead to state-of-the-art performance when finetuned on individual downstream tasks (e.g., 90.7% accuracy on ScienceQA IMG). Furthermore, we qualitatively demonstrate the advantages of InstructBLIP over concurrent multimodal models._

Tips:

-   InstructBLIP uses the same architecture as [BLIP-2](blip2) with a tiny but important difference: it also feeds the text prompt (instruction) to the Q-Former.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/instructblip_architecture.jpg) InstructBLIP architecture. Taken from the [original paper.](https://arxiv.org/abs/2305.06500)

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip).

## InstructBlipConfig

### class transformers.InstructBlipConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/instructblip/configuration_instructblip.py#L252)

( vision\_config = Noneqformer\_config = Nonetext\_config = Nonenum\_query\_tokens = 32\*\*kwargs )

Parameters

-   **vision\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [InstructBlipVisionConfig](/docs/transformers/v4.34.0/en/model_doc/instructblip#transformers.InstructBlipVisionConfig).
-   **qformer\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [InstructBlipQFormerConfig](/docs/transformers/v4.34.0/en/model_doc/instructblip#transformers.InstructBlipQFormerConfig).
-   **text\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize any [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig).
-   **num\_query\_tokens** (`int`, _optional_, defaults to 32) — The number of query tokens passed through the Transformer.
-   **kwargs** (_optional_) — Dictionary of keyword arguments.

[InstructBlipConfig](/docs/transformers/v4.34.0/en/model_doc/instructblip#transformers.InstructBlipConfig) is the configuration class to store the configuration of a [InstructBlipForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/instructblip#transformers.InstructBlipForConditionalGeneration). It is used to instantiate a InstructBLIP model according to the specified arguments, defining the vision model, Q-Former model and language model configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the InstructBLIP [Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import (
...     InstructBlipVisionConfig,
...     InstructBlipQFormerConfig,
...     OPTConfig,
...     InstructBlipConfig,
...     InstructBlipForConditionalGeneration,
... )

>>> 
>>> configuration = InstructBlipConfig()

>>> 
>>> model = InstructBlipForConditionalGeneration(configuration)

>>> 
>>> configuration = model.config

>>> 

>>> 
>>> vision_config = InstructBlipVisionConfig()
>>> qformer_config = InstructBlipQFormerConfig()
>>> text_config = OPTConfig()

>>> config = InstructBlipConfig.from_text_vision_configs(vision_config, qformer_config, text_config)
```

#### from\_vision\_qformer\_text\_configs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/instructblip/configuration_instructblip.py#L337)

( vision\_config: InstructBlipVisionConfigqformer\_config: InstructBlipQFormerConfigtext\_config: PretrainedConfig\*\*kwargs ) → [InstructBlipConfig](/docs/transformers/v4.34.0/en/model_doc/instructblip#transformers.InstructBlipConfig)

An instance of a configuration object

Instantiate a [InstructBlipConfig](/docs/transformers/v4.34.0/en/model_doc/instructblip#transformers.InstructBlipConfig) (or a derived class) from a InstructBLIP vision model, Q-Former and language model configurations.

## InstructBlipVisionConfig

### class transformers.InstructBlipVisionConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/instructblip/configuration_instructblip.py#L33)

( hidden\_size = 1408intermediate\_size = 6144num\_hidden\_layers = 39num\_attention\_heads = 16image\_size = 224patch\_size = 14hidden\_act = 'gelu'layer\_norm\_eps = 1e-06attention\_dropout = 0.0initializer\_range = 1e-10qkv\_bias = True\*\*kwargs )

This is the configuration class to store the configuration of a [InstructBlipVisionModel](/docs/transformers/v4.34.0/en/model_doc/instructblip#transformers.InstructBlipVisionModel). It is used to instantiate a InstructBLIP vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration defaults will yield a similar configuration to that of the InstructBLIP [Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import InstructBlipVisionConfig, InstructBlipVisionModel

>>> 
>>> configuration = InstructBlipVisionConfig()

>>> 
>>> model = InstructBlipVisionModel(configuration)

>>> 
>>> configuration = model.config
```

## InstructBlipQFormerConfig

### class transformers.InstructBlipQFormerConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/instructblip/configuration_instructblip.py#L134)

( vocab\_size = 30522hidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout\_prob = 0.1attention\_probs\_dropout\_prob = 0.1max\_position\_embeddings = 512initializer\_range = 0.02layer\_norm\_eps = 1e-12pad\_token\_id = 0position\_embedding\_type = 'absolute'cross\_attention\_frequency = 2encoder\_hidden\_size = 1408\*\*kwargs )

This is the configuration class to store the configuration of a [InstructBlipQFormerModel](/docs/transformers/v4.34.0/en/model_doc/instructblip#transformers.InstructBlipQFormerModel). It is used to instantiate a InstructBLIP Querying Transformer (Q-Former) model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the InstructBLIP [Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5) architecture. Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Note that [InstructBlipQFormerModel](/docs/transformers/v4.34.0/en/model_doc/instructblip#transformers.InstructBlipQFormerModel) is very similar to [BertLMHeadModel](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertLMHeadModel) with interleaved cross-attention.

Examples:

```
>>> from transformers import InstructBlipQFormerConfig, InstructBlipQFormerModel

>>> 
>>> configuration = InstructBlipQFormerConfig()

>>> 
>>> model = InstructBlipQFormerModel(configuration)
>>> 
>>> configuration = model.config
```

## InstructBlipProcessor

### class transformers.InstructBlipProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/instructblip/processing_instructblip.py#L30)

( image\_processortokenizerqformer\_tokenizer )

Parameters

-   **image\_processor** (`BlipImageProcessor`) — An instance of [BlipImageProcessor](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipImageProcessor). The image processor is a required input.
-   **tokenizer** (`AutoTokenizer`) — An instance of \[‘PreTrainedTokenizer\`\]. The tokenizer is a required input.
-   **qformer\_tokenizer** (`AutoTokenizer`) — An instance of \[‘PreTrainedTokenizer\`\]. The Q-Former tokenizer is a required input.

Constructs an InstructBLIP processor which wraps a BLIP image processor and a LLaMa/T5 tokenizer into a single processor.

[InstructBlipProcessor](/docs/transformers/v4.34.0/en/model_doc/instructblip#transformers.InstructBlipProcessor) offers all the functionalities of [BlipImageProcessor](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipImageProcessor) and [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See the docstring of `__call__()` and [decode()](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipProcessor.decode) for more information.

This method forwards all its arguments to PreTrainedTokenizer’s [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.batch_decode). Please refer to the docstring of this method for more information.

This method forwards all its arguments to PreTrainedTokenizer’s [decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.decode). Please refer to the docstring of this method for more information.

## InstructBlipVisionModel

### class transformers.InstructBlipVisionModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/instructblip/modeling_instructblip.py#L500)

( config: InstructBlipVisionConfig )

The [InstructBlipVisionModel](/docs/transformers/v4.34.0/en/model_doc/instructblip#transformers.InstructBlipVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## InstructBlipQFormerModel

### class transformers.InstructBlipQFormerModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/instructblip/modeling_instructblip.py#L1052)

( config: InstructBlipQFormerConfig )

Querying Transformer (Q-Former), used in InstructBLIP. Slightly modified from BLIP-2 as it also takes the instruction as input.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/instructblip/modeling_instructblip.py#L1125)

( input\_ids: LongTensorattention\_mask: typing.Optional\[torch.FloatTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonequery\_embeds: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None )

encoder\_hidden\_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_): Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model is configured as a decoder. encoder\_attention\_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_): Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

-   1 for tokens that are **not masked**,
-   0 for tokens that are **masked**. past\_key\_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of: shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`. use\_cache (`bool`, _optional_): If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

## InstructBlipForConditionalGeneration

### class transformers.InstructBlipForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/instructblip/modeling_instructblip.py#L1258)

( config: InstructBlipConfig )

Parameters

-   **config** ([InstructBlipConfig](/docs/transformers/v4.34.0/en/model_doc/instructblip#transformers.InstructBlipConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

InstructBLIP Model for generating text given an image and an optional text prompt. The model consists of a vision encoder, Querying Transformer (Q-Former) and a language model.

One can optionally pass `input_ids` to the model, which serve as a text prompt, to make the language model continue the prompt. Otherwise, the language model starts generating text from the \[BOS\] (beginning-of-sequence) token.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/instructblip/modeling_instructblip.py#L1331)

( pixel\_values: FloatTensorqformer\_input\_ids: FloatTensorqformer\_attention\_mask: typing.Optional\[torch.LongTensor\] = Noneinput\_ids: typing.Optional\[torch.FloatTensor\] = Noneattention\_mask: typing.Optional\[torch.LongTensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.instructblip.modeling_instructblip.InstructBlipForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)`

The [InstructBlipForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/instructblip#transformers.InstructBlipForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
>>> import torch
>>> from PIL import Image
>>> import requests

>>> model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
>>> processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model.to(device)
>>> url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
>>> prompt = "What is unusual about this image?"
>>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

>>> outputs = model.generate(
...     **inputs,
...     do_sample=False,
...     num_beams=5,
...     max_length=256,
...     min_length=1,
...     top_p=0.9,
...     repetition_penalty=1.5,
...     length_penalty=1.0,
...     temperature=1,
... )
>>> generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
>>> print(generated_text)
The unusual aspect of this image is that a man is ironing clothes on the back of a yellow SUV, which is parked in the middle of a busy city street. This is an unconventional approach to ironing clothes, as it requires the man to balance himself and his ironing equipment on top of the vehicle while navigating through traffic. Additionally, the presence of taxis and other vehicles in the scene further emphasizes the unusual nature of this situation.
```

#### generate

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/instructblip/modeling_instructblip.py#L1486)

( pixel\_values: FloatTensorqformer\_input\_ids: typing.Optional\[torch.LongTensor\] = Noneqformer\_attention\_mask: typing.Optional\[torch.LongTensor\] = Noneinput\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.LongTensor\] = None\*\*generate\_kwargs ) → captions (list)

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape (batch\_size, num\_channels, height, width)) — Input images to be processed.
-   **qformer\_input\_ids** (`torch.LongTensor` of shape (batch\_size, sequence\_length), _optional_) — The sequence used as a prompt to be fed to the Q-Former module.
-   **qformer\_attention\_mask** (`torch.LongTensor` of shape (batch\_size, sequence\_length), _optional_) — Mask to avoid performing attention on padding token indices.
-   **input\_ids** (`torch.LongTensor` of shape (batch\_size, sequence\_length), _optional_) — The sequence used as a prompt for the generation.
-   **attention\_mask** (`torch.LongTensor` of shape (batch\_size, sequence\_length), _optional_) — Mask to avoid performing attention on padding token indices.

A list of strings of length batch\_size \* num\_captions.

Overrides `generate` function to be able to use the model as a conditional generator.