# ProphetNet

[![Models](https://img.shields.io/badge/All_model_pages-prophetnet-blueviolet)](https://huggingface.co/models?filter=prophetnet) [![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/docs-demos/prophetnet-large-uncased)

**DISCLAIMER:** If you see something strange, file a [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title) and assign @patrickvonplaten

## Overview

The ProphetNet model was proposed in [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training,](https://arxiv.org/abs/2001.04063) by Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang, Ming Zhou on 13 Jan, 2020.

ProphetNet is an encoder-decoder model and can predict n-future tokens for “ngram” language modeling instead of just the next token.

The abstract from the paper is the following:

_In this paper, we present a new sequence-to-sequence pretraining model called ProphetNet, which introduces a novel self-supervised objective named future n-gram prediction and the proposed n-stream self-attention mechanism. Instead of the optimization of one-step ahead prediction in traditional sequence-to-sequence model, the ProphetNet is optimized by n-step ahead prediction which predicts the next n tokens simultaneously based on previous context tokens at each time step. The future n-gram prediction explicitly encourages the model to plan for the future tokens and prevent overfitting on strong local correlations. We pre-train ProphetNet using a base scale dataset (16GB) and a large scale dataset (160GB) respectively. Then we conduct experiments on CNN/DailyMail, Gigaword, and SQuAD 1.1 benchmarks for abstractive summarization and question generation tasks. Experimental results show that ProphetNet achieves new state-of-the-art results on all these datasets compared to the models using the same scale pretraining corpus._

Tips:

-   ProphetNet is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left.
-   The model architecture is based on the original Transformer, but replaces the “standard” self-attention mechanism in the decoder by a a main self-attention mechanism and a self and n-stream (predict) self-attention mechanism.

The Authors’ code can be found [here](https://github.com/microsoft/ProphetNet).

## Documentation resources

-   [Causal language modeling task guide](../tasks/language_modeling)
-   [Translation task guide](../tasks/translation)
-   [Summarization task guide](../tasks/summarization)

## ProphetNetConfig

### class transformers.ProphetNetConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/prophetnet/configuration_prophetnet.py#L32)

( activation\_dropout: typing.Optional\[float\] = 0.1activation\_function: typing.Union\[str, typing.Callable, NoneType\] = 'gelu'vocab\_size: typing.Optional\[int\] = 30522hidden\_size: typing.Optional\[int\] = 1024encoder\_ffn\_dim: typing.Optional\[int\] = 4096num\_encoder\_layers: typing.Optional\[int\] = 12num\_encoder\_attention\_heads: typing.Optional\[int\] = 16decoder\_ffn\_dim: typing.Optional\[int\] = 4096num\_decoder\_layers: typing.Optional\[int\] = 12num\_decoder\_attention\_heads: typing.Optional\[int\] = 16attention\_dropout: typing.Optional\[float\] = 0.1dropout: typing.Optional\[float\] = 0.1max\_position\_embeddings: typing.Optional\[int\] = 512init\_std: typing.Optional\[float\] = 0.02is\_encoder\_decoder: typing.Optional\[bool\] = Trueadd\_cross\_attention: typing.Optional\[bool\] = Truedecoder\_start\_token\_id: typing.Optional\[int\] = 0ngram: typing.Optional\[int\] = 2num\_buckets: typing.Optional\[int\] = 32relative\_max\_distance: typing.Optional\[int\] = 128disable\_ngram\_loss: typing.Optional\[bool\] = Falseeps: typing.Optional\[float\] = 0.0use\_cache: typing.Optional\[bool\] = Truepad\_token\_id: typing.Optional\[int\] = 0bos\_token\_id: typing.Optional\[int\] = 1eos\_token\_id: typing.Optional\[int\] = 2\*\*kwargs )

This is the configuration class to store the configuration of a [ProphetNetModel](/docs/transformers/v4.34.0/en/model_doc/prophetnet#transformers.ProphetNetModel). It is used to instantiate a ProphetNet model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the ProphetNet [microsoft/prophetnet-large-uncased](https://huggingface.co/microsoft/prophetnet-large-uncased) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## ProphetNetTokenizer

### class transformers.ProphetNetTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/prophetnet/tokenization_prophetnet.py#L287)

( vocab\_file: strdo\_lower\_case: typing.Optional\[bool\] = Truedo\_basic\_tokenize: typing.Optional\[bool\] = Truenever\_split: typing.Optional\[typing.Iterable\] = Noneunk\_token: typing.Optional\[str\] = '\[UNK\]'sep\_token: typing.Optional\[str\] = '\[SEP\]'x\_sep\_token: typing.Optional\[str\] = '\[X\_SEP\]'pad\_token: typing.Optional\[str\] = '\[PAD\]'mask\_token: typing.Optional\[str\] = '\[MASK\]'tokenize\_chinese\_chars: typing.Optional\[bool\] = Truestrip\_accents: typing.Optional\[bool\] = None\*\*kwargs )

Construct a ProphetNetTokenizer. Based on WordPiece.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/prophetnet/tokenization_prophetnet.py#L499)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A BERT sequence has the following format:

-   single sequence: `[CLS] X [SEP]`
-   pair of sequences: `[CLS] A [SEP] B [SEP]`

#### convert\_tokens\_to\_string

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/prophetnet/tokenization_prophetnet.py#L416)

( tokens: str )

Converts a sequence of tokens (string) in a single string.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/prophetnet/tokenization_prophetnet.py#L451)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).

Create a mask from the two sequences passed to be used in a sequence-pair classification task. A ProphetNet

sequence pair mask has the following format:

```
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence    | second sequence |
```

If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/prophetnet/tokenization_prophetnet.py#L421)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = Nonealready\_has\_special\_tokens: typing.Optional\[bool\] = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

## ProphetNet specific outputs

### class transformers.models.prophetnet.modeling\_prophetnet.ProphetNetSeq2SeqLMOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/prophetnet/modeling_prophetnet.py#L253)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Nonelogits\_ngram: typing.Optional\[torch.FloatTensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_ngram\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_ngram\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for sequence-to-sequence language models outputs.

### class transformers.models.prophetnet.modeling\_prophetnet.ProphetNetSeq2SeqModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/prophetnet/modeling_prophetnet.py#L338)

( last\_hidden\_state: FloatTensorlast\_hidden\_state\_ngram: typing.Optional\[torch.FloatTensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_ngram\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_ngram\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for model encoder’s outputs that also contains : pre-computed hidden states that can speed up sequential decoding.

### class transformers.models.prophetnet.modeling\_prophetnet.ProphetNetDecoderModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/prophetnet/modeling_prophetnet.py#L424)

( last\_hidden\_state: FloatTensorlast\_hidden\_state\_ngram: typing.Optional\[torch.FloatTensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonehidden\_states\_ngram: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonengram\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for model’s outputs that may also contain a past key/values (to speed up sequential decoding).

### class transformers.models.prophetnet.modeling\_prophetnet.ProphetNetDecoderLMOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/prophetnet/modeling_prophetnet.py#L484)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Nonelogits\_ngram: typing.Optional\[torch.FloatTensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonehidden\_states\_ngram: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonengram\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for model’s outputs that may also contain a past key/values (to speed up sequential decoding).

## ProphetNetModel

### class transformers.ProphetNetModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/prophetnet/modeling_prophetnet.py#L1746)

( config: ProphetNetConfig )

Parameters

-   **config** ([ProphetNetConfig](/docs/transformers/v4.34.0/en/model_doc/prophetnet#transformers.ProphetNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare ProphetNet Model outputting raw hidden-states without any specific head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/prophetnet/modeling_prophetnet.py#L1780)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.Tensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.BoolTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.models.prophetnet.modeling\_prophetnet.ProphetNetSeq2SeqModelOutput](/docs/transformers/v4.34.0/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput) or `tuple(torch.FloatTensor)`

The [ProphetNetModel](/docs/transformers/v4.34.0/en/model_doc/prophetnet#transformers.ProphetNetModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, ProphetNetModel

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
>>> model = ProphetNetModel.from_pretrained("microsoft/prophetnet-large-uncased")

>>> input_ids = tokenizer(
...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
... ).input_ids  
>>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  
>>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

>>> last_hidden_states = outputs.last_hidden_state  
>>> last_hidden_states_ngram = outputs.last_hidden_state_ngram  
```

## ProphetNetEncoder

### class transformers.ProphetNetEncoder

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/prophetnet/modeling_prophetnet.py#L1232)

( config: ProphetNetConfigword\_embeddings: Embedding = None )

Parameters

-   **config** ([ProphetNetConfig](/docs/transformers/v4.34.0/en/model_doc/prophetnet#transformers.ProphetNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The standalone encoder part of the ProphetNetModel. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and behavior.

word\_embeddings (`torch.nn.Embeddings` of shape `(config.vocab_size, config.hidden_size)`, _optional_): The word embedding parameters. This can be used to initialize [ProphetNetEncoder](/docs/transformers/v4.34.0/en/model_doc/prophetnet#transformers.ProphetNetEncoder) with pre-defined word embeddings instead of randomly initialized word embeddings.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/prophetnet/modeling_prophetnet.py#L1262)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

The [ProphetNetEncoder](/docs/transformers/v4.34.0/en/model_doc/prophetnet#transformers.ProphetNetEncoder) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, ProphetNetEncoder
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
>>> model = ProphetNetEncoder.from_pretrained("patrickvonplaten/prophetnet-large-uncased-standalone")
>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## ProphetNetDecoder

### class transformers.ProphetNetDecoder

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/prophetnet/modeling_prophetnet.py#L1372)

( config: ProphetNetConfigword\_embeddings: typing.Optional\[torch.nn.modules.sparse.Embedding\] = None )

Parameters

-   **config** ([ProphetNetConfig](/docs/transformers/v4.34.0/en/model_doc/prophetnet#transformers.ProphetNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The standalone decoder part of the ProphetNetModel. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and behavior.

word\_embeddings (`torch.nn.Embeddings` of shape `(config.vocab_size, config.hidden_size)`, _optional_): The word embedding parameters. This can be used to initialize [ProphetNetEncoder](/docs/transformers/v4.34.0/en/model_doc/prophetnet#transformers.ProphetNetEncoder) with pre-defined word embeddings instead of randomly initialized word embeddings.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/prophetnet/modeling_prophetnet.py#L1409)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.Tensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.models.prophetnet.modeling\_prophetnet.ProphetNetDecoderModelOutput](/docs/transformers/v4.34.0/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput) or `tuple(torch.FloatTensor)`

The [ProphetNetDecoder](/docs/transformers/v4.34.0/en/model_doc/prophetnet#transformers.ProphetNetDecoder) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, ProphetNetDecoder
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
>>> model = ProphetNetDecoder.from_pretrained("microsoft/prophetnet-large-uncased", add_cross_attention=False)
>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## ProphetNetForConditionalGeneration

### class transformers.ProphetNetForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/prophetnet/modeling_prophetnet.py#L1875)

( config: ProphetNetConfig )

Parameters

-   **config** ([ProphetNetConfig](/docs/transformers/v4.34.0/en/model_doc/prophetnet#transformers.ProphetNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The ProphetNet Model with a language modeling head. Can be used for sequence generation tasks. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/prophetnet/modeling_prophetnet.py#L1898)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.Tensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.BoolTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.models.prophetnet.modeling\_prophetnet.ProphetNetSeq2SeqLMOutput](/docs/transformers/v4.34.0/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The [ProphetNetForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/prophetnet#transformers.ProphetNetForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, ProphetNetForConditionalGeneration

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
>>> model = ProphetNetForConditionalGeneration.from_pretrained("microsoft/prophetnet-large-uncased")

>>> input_ids = tokenizer(
...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
... ).input_ids  
>>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  
>>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

>>> logits_next_token = outputs.logits  
>>> logits_ngram_next_tokens = outputs.logits_ngram  
```

## ProphetNetForCausalLM

### class transformers.ProphetNetForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/prophetnet/modeling_prophetnet.py#L2088)

( config: ProphetNetConfig )

Parameters

-   **config** ([ProphetNetConfig](/docs/transformers/v4.34.0/en/model_doc/prophetnet#transformers.ProphetNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The standalone decoder part of the ProphetNetModel with a lm head on top. The model can be used for causal language modeling. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/prophetnet/modeling_prophetnet.py#L2125)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.Tensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.models.prophetnet.modeling\_prophetnet.ProphetNetDecoderLMOutput](/docs/transformers/v4.34.0/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput) or `tuple(torch.FloatTensor)`

The [ProphetNetForCausalLM](/docs/transformers/v4.34.0/en/model_doc/prophetnet#transformers.ProphetNetForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, ProphetNetForCausalLM
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
>>> model = ProphetNetForCausalLM.from_pretrained("microsoft/prophetnet-large-uncased")
>>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> logits = outputs.logits

>>> 
>>> from transformers import BertTokenizer, EncoderDecoderModel, AutoTokenizer
>>> import torch

>>> tokenizer_enc = BertTokenizer.from_pretrained("bert-large-uncased")
>>> tokenizer_dec = AutoTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
>>> model = EncoderDecoderModel.from_encoder_decoder_pretrained(
...     "bert-large-uncased", "microsoft/prophetnet-large-uncased"
... )

>>> ARTICLE = (
...     "the us state department said wednesday it had received no "
...     "formal word from bolivia that it was expelling the us ambassador there "
...     "but said the charges made against him are `` baseless ."
... )
>>> input_ids = tokenizer_enc(ARTICLE, return_tensors="pt").input_ids
>>> labels = tokenizer_dec(
...     "us rejects charges against its ambassador in bolivia", return_tensors="pt"
... ).input_ids
>>> outputs = model(input_ids=input_ids, decoder_input_ids=labels[:, :-1], labels=labels[:, 1:])

>>> loss = outputs.loss
```