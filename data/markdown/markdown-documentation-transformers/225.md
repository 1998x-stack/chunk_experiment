# MPT

## Overview

The MPT model was proposed by the [MosaicML](https://www.mosaicml.com/) team and released with multiple sizes and finetuned variants. The MPT models is a series of open source and commercially usable LLMs pre-trained on 1T tokens.

MPT models are GPT-style decoder-only transformers with several improvements: performance-optimized layer implementations, architecture changes that provide greater training stability, and the elimination of context length limits by replacing positional embeddings with ALiBi.

-   MPT base: MPT base pre-trained models on next token prediction
-   MPT instruct: MPT base models fine-tuned on instruction based tasks
-   MPT storywriter: MPT base models fine-tuned for 2500 steps on 65k-token excerpts of fiction books contained in the books3 corpus, this enables the model to handle very long sequences

The original code is available at the [`llm-foundry`](https://github.com/mosaicml/llm-foundry/tree/main) repository.

Read more about it [in the release blogpost](https://www.mosaicml.com/blog/mpt-7b)

Tips:

-   Learn more about some techniques behind training of the model [in this section of llm-foundry repository](https://github.com/mosaicml/llm-foundry/blob/main/TUTORIAL.md#faqs)
    
-   If you want to use the advanced version of the model (triton kernels, direct flash attention integration), you can still use the original model implementation by adding `trust_remote_code=True` when calling `from_pretrained`.
    
-   [Fine-tuning Notebook](https://colab.research.google.com/drive/1HCpQkLL7UXW8xJUJJ29X7QAeNJKO0frZ?usp=sharing) on how to fine-tune MPT-7B on a free Google Colab instance to turn the model into a Chatbot.
    

## MptConfig

### class transformers.MptConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mpt/configuration_mpt.py#L121)

( d\_model: int = 2048n\_heads: int = 16n\_layers: int = 24expansion\_ratio: int = 4max\_seq\_len: int = 2048vocab\_size: int = 50368resid\_pdrop: float = 0.0layer\_norm\_epsilon: float = 1e-05emb\_pdrop: float = 0.0learned\_pos\_emb: bool = Trueattn\_config: MptAttentionConfig = Noneinit\_device: str = 'cpu'logit\_scale: typing.Union\[float, str, NoneType\] = Noneno\_bias: bool = Trueverbose: int = 0embedding\_fraction: float = 1.0norm\_type: str = 'low\_precision\_layernorm'use\_cache: bool = Falseinitializer\_range = 0.02\*\*kwargs )

This is the configuration class to store the configuration of a [MptModel](/docs/transformers/v4.34.0/en/model_doc/mpt#transformers.MptModel). It is used to instantiate a Mpt model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to the Mpt-7b architecture [mosaicml/mpt-7b](https://huggingface.co/mosaicml/mpt-7b).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import MptConfig, MptModel

>>> 
>>> configuration = MptConfig()

>>> 
>>> model = MptModel(configuration)

>>> 
>>> configuration = model.config
```

## MptModel

### class transformers.MptModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mpt/modeling_mpt.py#L390)

( config: MptConfig )

Parameters

-   **config** ([MptConfig](/docs/transformers/v4.34.0/en/model_doc/mpt#transformers.MptConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Mpt Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mpt/modeling_mpt.py#L450)

( input\_ids: typing.Optional\[torch.LongTensor\] = Nonepast\_key\_values: typing.Union\[typing.Tuple\[typing.Tuple\[torch.Tensor, torch.Tensor\], ...\], NoneType\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions) or `tuple(torch.FloatTensor)`

The [MptModel](/docs/transformers/v4.34.0/en/model_doc/mpt#transformers.MptModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, MptModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-7b")
>>> model = MptModel.from_pretrained("mosaicml/mpt-7b")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## MptForCausalLM

### class transformers.MptForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mpt/modeling_mpt.py#L582)

( config: MptConfig )

Parameters

-   **config** ([MptConfig](/docs/transformers/v4.34.0/en/model_doc/mpt#transformers.MptConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The MPT Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mpt/modeling_mpt.py#L627)

( input\_ids: typing.Optional\[torch.LongTensor\] = Nonepast\_key\_values: typing.Union\[typing.Tuple\[typing.Tuple\[torch.Tensor, torch.Tensor\], ...\], NoneType\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

The [MptForCausalLM](/docs/transformers/v4.34.0/en/model_doc/mpt#transformers.MptForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoTokenizer, MptForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-7b")
>>> model = MptForCausalLM.from_pretrained("mosaicml/mpt-7b")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs, labels=inputs["input_ids"])
>>> loss = outputs.loss
>>> logits = outputs.logits
```

## MptForSequenceClassification

### class transformers.MptForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mpt/modeling_mpt.py#L732)

( config: MptConfig )

Parameters

-   **config** ([MptConfig](/docs/transformers/v4.34.0/en/model_doc/mpt#transformers.MptConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The MPT Model transformer with a sequence classification head on top (linear layer).

[MptForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/mpt#transformers.MptForSequenceClassification) uses the last token in order to do the classification, as other causal models (e.g. GPT-1) do.

Since it does classification on the last token, it requires to know the position of the last token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in each row of the batch).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mpt/modeling_mpt.py#L742)

( input\_ids: typing.Optional\[torch.LongTensor\] = Nonepast\_key\_values: typing.Union\[typing.Tuple\[typing.Tuple\[torch.Tensor, torch.Tensor\], ...\], NoneType\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.modeling_outputs.SequenceClassifierOutputWithPast` or `tuple(torch.FloatTensor)`

The [MptForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/mpt#transformers.MptForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example of single-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, MptForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-7b")
>>> model = MptForSequenceClassification.from_pretrained("mosaicml/mpt-7b")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = MptForSequenceClassification.from_pretrained("mosaicml/mpt-7b", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
```

Example of multi-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, MptForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-7b")
>>> model = MptForSequenceClassification.from_pretrained("mosaicml/mpt-7b", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = MptForSequenceClassification.from_pretrained(
...     "mosaicml/mpt-7b", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## MptForTokenClassification

### class transformers.MptForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mpt/modeling_mpt.py#L845)

( config: MptConfig )

Parameters

-   **config** ([MptConfig](/docs/transformers/v4.34.0/en/model_doc/mpt#transformers.MptConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

MPT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mpt/modeling_mpt.py#L863)

( input\_ids: typing.Optional\[torch.LongTensor\] = Nonepast\_key\_values: typing.Union\[typing.Tuple\[typing.Tuple\[torch.Tensor, torch.Tensor\], ...\], NoneType\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None\*\*deprecated\_arguments ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

The [MptForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/mpt#transformers.MptForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, MptForTokenClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-7b")
>>> model = MptForTokenClassification.from_pretrained("mosaicml/mpt-7b")

>>> inputs = tokenizer(
...     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
... )

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_token_class_ids = logits.argmax(-1)

>>> 
>>> 
>>> 
>>> predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]

>>> labels = predicted_token_class_ids
>>> loss = model(**inputs, labels=labels).loss
```

## MptForQuestionAnswering

### class transformers.MptForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mpt/modeling_mpt.py#L934)

( config )

Parameters

-   **config** ([MptConfig](/docs/transformers/v4.34.0/en/model_doc/mpt#transformers.MptConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The MPT Model transformer with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mpt/modeling_mpt.py#L943)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonestart\_positions: typing.Optional\[torch.LongTensor\] = Noneend\_positions: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None )

The [MptForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/mpt#transformers.MptForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.