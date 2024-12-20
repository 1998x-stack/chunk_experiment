# Sample usage

## UMT5

[![Models](https://img.shields.io/badge/All_model_pages-mt5-blueviolet)](https://huggingface.co/models?filter=umt5) [![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/docs-demos/mt5-small-finetuned-arxiv-cs-finetuned-arxiv-cs-full)

## Overview

The UMT5 model was proposed in [UniMax: Fairer and More Effective Language Sampling for Large-Scale Multilingual Pretraining](https://openreview.net/forum?id=kXwdL1cWOAi) by Hyung Won Chung, Xavier Garcia, Adam Roberts, Yi Tay, Orhan Firat, Sharan Narang, Noah Constant.

The abstract from the paper is the following:

_Pretrained multilingual large language models have typically used heuristic temperature-based sampling to balance between different languages. However previous work has not systematically evaluated the efficacy of different pretraining language distributions across model scales. In this paper, we propose a new sampling method, UniMax, that delivers more uniform coverage of head languages while mitigating overfitting on tail languages by explicitly capping the number of repeats over each language’s corpus. We perform an extensive series of ablations testing a range of sampling strategies on a suite of multilingual benchmarks, while varying model scale. We find that UniMax outperforms standard temperature-based sampling, and the benefits persist as scale increases. As part of our contribution, we release: (i) an improved and refreshed mC4 multilingual corpus consisting of 29 trillion characters across 107 languages, and (ii) a suite of pretrained umT5 model checkpoints trained with UniMax sampling._

Tips:

-   UMT5 was only pre-trained on [mC4](https://huggingface.co/datasets/mc4) excluding any supervised training. Therefore, this model has to be fine-tuned before it is usable on a downstream task, unlike the original T5 model.
-   Since umT5 was pre-trained in an unsupervise manner, there’s no real advantage to using a task prefix during single-task fine-tuning. If you are doing multi-task fine-tuning, you should use a prefix.

Google has released the following variants:

-   [google/umt5-small](https://huggingface.co/google/umt5-small)
-   [google/umt5-base](https://huggingface.co/google/umt5-base)
-   [google/umt5-xl](https://huggingface.co/google/umt5-xl)
-   [google/umt5-xxl](https://huggingface.co/google/umt5-xxl).

This model was contributed by [agemagician](https://huggingface.co/agemagician) and [stefan-it](https://huggingface.co/stefan-it). The original code can be found [here](https://github.com/google-research/t5x).

One can refer to [T5’s documentation page](t5) for more tips, code examples and notebooks.

## Differences with mT5?

\`UmT5\` is based on mT5, with a non-shared relative positional bias that is computed for each layer. This means that the model set \`has\_relative\_bias\` for each layer. The conversion script is also different because the model was saved in t5x's latest checkpointing format.

```
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> model = AutoModelForSeq2SeqLM.from_pretrained("google/umt5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")

>>> inputs = tokenizer(
...     "A <extra_id_0> walks into a bar and orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>.",
...     return_tensors="pt",
... )
>>> outputs = model.generate(**inputs)
>>> print(tokenizer.batch_decode(outputs))
['<pad><extra_id_0>nyone who<extra_id_1> drink<extra_id_2> a<extra_id_3> alcohol<extra_id_4> A<extra_id_5> A. This<extra_id_6> I<extra_id_7><extra_id_52><extra_id_53></s>']
```

## UMT5Config

### class transformers.UMT5Config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/umt5/configuration_umt5.py#L31)

( vocab\_size = 250112d\_model = 512d\_kv = 64d\_ff = 1024num\_layers = 8num\_decoder\_layers = Nonenum\_heads = 6relative\_attention\_num\_buckets = 32relative\_attention\_max\_distance = 128dropout\_rate = 0.1layer\_norm\_epsilon = 1e-06initializer\_factor = 1.0feed\_forward\_proj = 'gated-gelu'is\_encoder\_decoder = Trueuse\_cache = Truetokenizer\_class = 'T5Tokenizer'tie\_word\_embeddings = Truepad\_token\_id = 0eos\_token\_id = 1decoder\_start\_token\_id = 0classifier\_dropout = 0.0\*\*kwargs )

This is the configuration class to store the configuration of a [UMT5Model](/docs/transformers/v4.34.0/en/model_doc/umt5#transformers.UMT5Model). It is used to instantiate a UMT5 model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the UMT5 [google/umt5-small](https://huggingface.co/google/umt5-small) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## UMT5Model

### class transformers.UMT5Model

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/umt5/modeling_umt5.py#L936)

( config )

Parameters

-   **config** ([UMT5Config](/docs/transformers/v4.34.0/en/model_doc/umt5#transformers.UMT5Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare UMT5 Model transformer outputting raw hidden-states without any specific head on top.

The UMT5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It’s an encoder decoder transformer pre-trained in a text-to-text denoising generative setting.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

Examples:

```
>>> from transformers import UMT5Model, AutoTokenizer

>>> model = UMT5Model.from_pretrained("google/umt5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")
>>> noisy_text = "UN Offizier sagt, dass weiter <extra_id_0> werden muss in Syrien."
>>> label = "<extra_id_0> verhandelt"
>>> inputs = tokenizer(inputs, return_tensors="pt")
>>> labels = tokenizer(label=label, return_tensors="pt")

>>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])
>>> hidden_states = outputs.last_hidden_state
```

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/umt5/modeling_umt5.py#L1003)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.BoolTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.FloatTensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

The [UMT5Model](/docs/transformers/v4.34.0/en/model_doc/umt5#transformers.UMT5Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, UMT5Model

>>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")
>>> model = UMT5Model.from_pretrained("google/umt5-small")

>>> input_ids = tokenizer(
...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
... ).input_ids  
>>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  

>>> 
>>> 
>>> decoder_input_ids = model._shift_right(decoder_input_ids)

>>> 
>>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
>>> last_hidden_states = outputs.last_hidden_state
```

## UMT5ForConditionalGeneration

### class transformers.UMT5ForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/umt5/modeling_umt5.py#L1102)

( config )

Parameters

-   **config** ([UMT5Config](/docs/transformers/v4.34.0/en/model_doc/umt5#transformers.UMT5Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

UMT5 Model with a `language modeling` head on top.

The UMT5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It’s an encoder decoder transformer pre-trained in a text-to-text denoising generative setting.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

Examples:

```
>>> from transformers import UMT5ForConditionalGeneration, AutoTokenizer

>>> model = UMT5ForConditionalGeneration.from_pretrained("google/umt5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")
>>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
>>> summary = "Weiter Verhandlung in Syrien."
>>> inputs = tokenizer(article, text_target=summary, return_tensors="pt")

>>> outputs = model(**inputs)
>>> loss = outputs.loss
```

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/umt5/modeling_umt5.py#L1171)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.BoolTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.FloatTensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The [UMT5ForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/umt5#transformers.UMT5ForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, UMT5ForConditionalGeneration

>>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")
>>> model = UMT5ForConditionalGeneration.from_pretrained("google/umt5-small")

>>> 
>>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
>>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
>>> outputs = model(input_ids=input_ids, labels=labels)
>>> loss = outputs.loss
>>> logits = outputs.logits

>>> 
>>> input_ids = tokenizer("Studies have shown that <extra_id_0> good for you", return_tensors="pt").input_ids
>>> outputs = model.generate(input_ids)
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## UMT5EncoderModel

### class transformers.UMT5EncoderModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/umt5/modeling_umt5.py#L1344)

( config )

Parameters

-   **config** ([UMT5Config](/docs/transformers/v4.34.0/en/model_doc/umt5#transformers.UMT5Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare UMT5 Model transformer outputting encoder’s raw hidden-states without any specific head on top.

The UMT5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It’s an encoder decoder transformer pre-trained in a text-to-text denoising generative setting.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

Examples:

```
>>> from transformers import UMT5EncoderModel, AutoTokenizer

>>> model = UMT5EncoderModel.from_pretrained("google/umt5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")
>>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
>>> input_ids = tokenizer(article, return_tensors="pt").input_ids
>>> outputs = model(input_ids)
>>> hidden_state = outputs.last_hidden_state
```

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/umt5/modeling_umt5.py#L1397)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

The [UMT5EncoderModel](/docs/transformers/v4.34.0/en/model_doc/umt5#transformers.UMT5EncoderModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, UMT5EncoderModel

>>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")
>>> model = UMT5EncoderModel.from_pretrained("google/umt5-small")
>>> input_ids = tokenizer(
...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
... ).input_ids  
>>> outputs = model(input_ids=input_ids)
>>> last_hidden_states = outputs.last_hidden_state
```

## UMT5ForSequenceClassification

### class transformers.UMT5ForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/umt5/modeling_umt5.py#L1448)

( config: UMT5Config )

Parameters

-   **config** ([UMT5Config](/docs/transformers/v4.34.0/en/model_doc/umt5#transformers.UMT5Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

UMT5 model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

The UMT5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It’s an encoder decoder transformer pre-trained in a text-to-text denoising generative setting.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/umt5/modeling_umt5.py#L1463)

( input\_ids: LongTensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [UMT5ForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/umt5#transformers.UMT5ForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## UMT5ForQuestionAnswering

### class transformers.UMT5ForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/umt5/modeling_umt5.py#L1582)

( config )

Parameters

-   **config** ([UMT5Config](/docs/transformers/v4.34.0/en/model_doc/umt5#transformers.UMT5Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

UMT5 Model with a span classification head on top for extractive question-answering tasks like SQuAD (linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

The UMT5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It’s an encoder decoder transformer pre-trained in a text-to-text denoising generative setting.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/umt5/modeling_umt5.py#L1627)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.BoolTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.FloatTensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = Nonestart\_positions: typing.Optional\[torch.LongTensor\] = Noneend\_positions: typing.Optional\[torch.LongTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqQuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

The [UMT5ForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/umt5#transformers.UMT5ForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.