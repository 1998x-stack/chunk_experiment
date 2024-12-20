# BLOOM

## Overview

The BLOOM model has been proposed with its various versions through the [BigScience Workshop](https://bigscience.huggingface.co/). BigScience is inspired by other open science initiatives where researchers have pooled their time and resources to collectively achieve a higher impact. The architecture of BLOOM is essentially similar to GPT3 (auto-regressive model for next token prediction), but has been trained on 46 different languages and 13 programming languages. Several smaller versions of the models have been trained on the same dataset. BLOOM is available in the following versions:

-   [bloom-560m](https://huggingface.co/bigscience/bloom-560m)
-   [bloom-1b1](https://huggingface.co/bigscience/bloom-1b1)
-   [bloom-1b7](https://huggingface.co/bigscience/bloom-1b7)
-   [bloom-3b](https://huggingface.co/bigscience/bloom-3b)
-   [bloom-7b1](https://huggingface.co/bigscience/bloom-7b1)
-   [bloom](https://huggingface.co/bigscience/bloom) (176B parameters)

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with BLOOM. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

-   [BloomForCausalLM](/docs/transformers/v4.34.0/en/model_doc/bloom#transformers.BloomForCausalLM) is supported by this [causal language modeling example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).

See also:

-   [Causal language modeling task guide](../tasks/language_modeling)
-   [Text classification task guide](../tasks/sequence_classification)
-   [Token classification task guide](../tasks/token_classification)
-   [Question answering task guide](../tasks/question_answering)

⚡️ Inference

-   A blog on [Optimization story: Bloom inference](https://huggingface.co/blog/bloom-inference-optimization).
-   A blog on [Incredibly Fast BLOOM Inference with DeepSpeed and Accelerate](https://huggingface.co/blog/bloom-inference-pytorch-scripts).

⚙️ Training

-   A blog on [The Technology Behind BLOOM Training](https://huggingface.co/blog/bloom-megatron-deepspeed).

## BloomConfig

### class transformers.BloomConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bloom/configuration_bloom.py#L42)

( vocab\_size = 250880hidden\_size = 64n\_layer = 2n\_head = 8layer\_norm\_epsilon = 1e-05initializer\_range = 0.02use\_cache = Truebos\_token\_id = 1eos\_token\_id = 2apply\_residual\_connection\_post\_layernorm = Falsehidden\_dropout = 0.0attention\_dropout = 0.0pretraining\_tp = 1slow\_but\_exact = False\*\*kwargs )

This is the configuration class to store the configuration of a [BloomModel](/docs/transformers/v4.34.0/en/model_doc/bloom#transformers.BloomModel). It is used to instantiate a Bloom model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to the Bloom architecture [bigscience/bloom](https://huggingface.co/bigscience/bloom).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import BloomConfig, BloomModel

>>> 
>>> configuration = BloomConfig()

>>> 
>>> model = BloomModel(configuration)

>>> 
>>> configuration = model.config
```

## BloomModel

### class transformers.BloomModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bloom/modeling_bloom.py#L617)

( config: BloomConfig )

Parameters

-   **config** ([BloomConfig](/docs/transformers/v4.34.0/en/model_doc/bloom#transformers.BloomConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Bloom Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bloom/modeling_bloom.py#L670)

( input\_ids: typing.Optional\[torch.LongTensor\] = Nonepast\_key\_values: typing.Union\[typing.Tuple\[typing.Tuple\[torch.Tensor, torch.Tensor\], ...\], NoneType\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.LongTensor\] = Noneinputs\_embeds: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None\*\*deprecated\_arguments ) → [transformers.modeling\_outputs.BaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions) or `tuple(torch.FloatTensor)`

The [BloomModel](/docs/transformers/v4.34.0/en/model_doc/bloom#transformers.BloomModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, BloomModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
>>> model = BloomModel.from_pretrained("bigscience/bloom-560m")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## BloomTokenizerFast

### class transformers.BloomTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bloom/tokenization_bloom_fast.py#L43)

( vocab\_file = Nonemerges\_file = Nonetokenizer\_file = Noneunk\_token = '<unk>'bos\_token = '<s>'eos\_token = '</s>'pad\_token = '<pad>'add\_prefix\_space = Falseclean\_up\_tokenization\_spaces = False\*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — Path to the vocabulary file.
-   **merges\_file** (`str`) — Path to the merges file.
-   **errors** (`str`, _optional_, defaults to `"replace"`) — Paradigm to follow when decoding bytes to UTF-8. See [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
-   **unk\_token** (`str`, _optional_, defaults to `<|endoftext|>`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **bos\_token** (`str`, _optional_, defaults to `<|endoftext|>`) — The beginning of sequence token.
-   **eos\_token** (`str`, _optional_, defaults to `<|endoftext|>`) — The end of sequence token.
-   **add\_prefix\_space** (`bool`, _optional_, defaults to `False`) — Whether or not to add an initial space to the input. This allows to treat the leading word just as any other word. (Bloom tokenizer detect beginning of words by the preceding space).
-   **trim\_offsets** (`bool`, _optional_, defaults to `True`) — Whether or not the post-processing step should trim offsets to avoid including whitespaces.

Construct a “fast” Bloom tokenizer (backed by HuggingFace’s _tokenizers_ library). Based on byte-level Byte-Pair-Encoding.

This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will

be encoded differently whether it is at the beginning of the sentence (without space) or not:

```
>>> from transformers import BloomTokenizerFast

>>> tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")
>>> tokenizer("Hello world")["input_ids"]
[59414, 8876]

>>> tokenizer(" Hello world")["input_ids"]
[86153, 8876]
```

You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer, but since the model was not pretrained this way, it might yield a decrease in performance.

When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

## BloomForCausalLM

### class transformers.BloomForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bloom/modeling_bloom.py#L822)

( config: BloomConfig )

Parameters

-   **config** ([BloomConfig](/docs/transformers/v4.34.0/en/model_doc/bloom#transformers.BloomConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Bloom Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bloom/modeling_bloom.py#L870)

( input\_ids: typing.Optional\[torch.LongTensor\] = Nonepast\_key\_values: typing.Union\[typing.Tuple\[typing.Tuple\[torch.Tensor, torch.Tensor\], ...\], NoneType\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None\*\*deprecated\_arguments ) → [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

The [BloomForCausalLM](/docs/transformers/v4.34.0/en/model_doc/bloom#transformers.BloomForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoTokenizer, BloomForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
>>> model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs, labels=inputs["input_ids"])
>>> loss = outputs.loss
>>> logits = outputs.logits
```

## BloomForSequenceClassification

### class transformers.BloomForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bloom/modeling_bloom.py#L990)

( config: BloomConfig )

Parameters

-   **config** ([BloomConfig](/docs/transformers/v4.34.0/en/model_doc/bloom#transformers.BloomConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Bloom Model transformer with a sequence classification head on top (linear layer).

[BloomForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/bloom#transformers.BloomForSequenceClassification) uses the last token in order to do the classification, as other causal models (e.g. GPT-1) do.

Since it does classification on the last token, it requires to know the position of the last token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in each row of the batch).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bloom/modeling_bloom.py#L1000)

( input\_ids: typing.Optional\[torch.LongTensor\] = Nonepast\_key\_values: typing.Union\[typing.Tuple\[typing.Tuple\[torch.Tensor, torch.Tensor\], ...\], NoneType\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None\*\*deprecated\_arguments ) → `transformers.modeling_outputs.SequenceClassifierOutputWithPast` or `tuple(torch.FloatTensor)`

The [BloomForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/bloom#transformers.BloomForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example of single-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, BloomForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
>>> model = BloomForSequenceClassification.from_pretrained("bigscience/bloom-560m")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = BloomForSequenceClassification.from_pretrained("bigscience/bloom-560m", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
```

Example of multi-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, BloomForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
>>> model = BloomForSequenceClassification.from_pretrained("bigscience/bloom-560m", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = BloomForSequenceClassification.from_pretrained(
...     "bigscience/bloom-560m", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## BloomForTokenClassification

### class transformers.BloomForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bloom/modeling_bloom.py#L1116)

( config: BloomConfig )

Parameters

-   **config** ([BloomConfig](/docs/transformers/v4.34.0/en/model_doc/bloom#transformers.BloomConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Bloom Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bloom/modeling_bloom.py#L1134)

( input\_ids: typing.Optional\[torch.LongTensor\] = Nonepast\_key\_values: typing.Union\[typing.Tuple\[typing.Tuple\[torch.Tensor, torch.Tensor\], ...\], NoneType\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None\*\*deprecated\_arguments ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

The [BloomForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/bloom#transformers.BloomForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, BloomForTokenClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
>>> model = BloomForTokenClassification.from_pretrained("bigscience/bloom-560m")

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

## BloomForQuestionAnswering

### class transformers.BloomForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bloom/modeling_bloom.py#L1217)

( config )

Parameters

-   **config** ([BloomConfig](/docs/transformers/v4.34.0/en/model_doc/bloom#transformers.BloomConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The BLOOM Model transformer with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bloom/modeling_bloom.py#L1226)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonestart\_positions: typing.Optional\[torch.LongTensor\] = Noneend\_positions: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None )

The [BloomForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/bloom#transformers.BloomForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## FlaxBloomModel

### class transformers.FlaxBloomModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bloom/modeling_flax_bloom.py#L647)

( config: BloomConfiginput\_shape: typing.Tuple = (1, 1)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([BloomConfig](/docs/transformers/v4.34.0/en/model_doc/bloom#transformers.BloomConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

The bare Bloom Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bloom/modeling_flax_bloom.py#L461)

( input\_idsattention\_mask = Nonepast\_key\_values: dict = Noneparams: dict = Nonedropout\_rng: PRNGKey = Nonetrain: bool = Falseoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutput) or `tuple(torch.FloatTensor)`

The `FlaxBloomPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxBloomModel

>>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
>>> model = FlaxBloomModel.from_pretrained("bigscience/bloom")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## FlaxBloomForCausalLM

### class transformers.FlaxBloomForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bloom/modeling_flax_bloom.py#L708)

( config: BloomConfiginput\_shape: typing.Tuple = (1, 1)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([BloomConfig](/docs/transformers/v4.34.0/en/model_doc/bloom#transformers.BloomConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

The Bloom Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bloom/modeling_flax_bloom.py#L461)

( input\_idsattention\_mask = Nonepast\_key\_values: dict = Noneparams: dict = Nonedropout\_rng: PRNGKey = Nonetrain: bool = Falseoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_flax\_outputs.FlaxMaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxMaskedLMOutput) or `tuple(torch.FloatTensor)`

The `FlaxBloomPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxBloomForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
>>> model = FlaxBloomForCausalLM.from_pretrained("bigscience/bloom")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
>>> outputs = model(**inputs)

>>> 
>>> next_token_logits = outputs.logits[:, -1]
```