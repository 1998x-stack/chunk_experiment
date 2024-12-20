# XLM-ProphetNet

[![Models](https://img.shields.io/badge/All_model_pages-xprophetnet-blueviolet)](https://huggingface.co/models?filter=xprophetnet) [![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/docs-demos/xprophetnet-large-wiki100-cased-xglue-ntg)

**DISCLAIMER:** If you see something strange, file a [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title) and assign @patrickvonplaten

## Overview

The XLM-ProphetNet model was proposed in [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training,](https://arxiv.org/abs/2001.04063) by Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang, Ming Zhou on 13 Jan, 2020.

XLM-ProphetNet is an encoder-decoder model and can predict n-future tokens for “ngram” language modeling instead of just the next token. Its architecture is identical to ProhpetNet, but the model was trained on the multi-lingual “wiki100” Wikipedia dump.

The abstract from the paper is the following:

_In this paper, we present a new sequence-to-sequence pretraining model called ProphetNet, which introduces a novel self-supervised objective named future n-gram prediction and the proposed n-stream self-attention mechanism. Instead of the optimization of one-step ahead prediction in traditional sequence-to-sequence model, the ProphetNet is optimized by n-step ahead prediction which predicts the next n tokens simultaneously based on previous context tokens at each time step. The future n-gram prediction explicitly encourages the model to plan for the future tokens and prevent overfitting on strong local correlations. We pre-train ProphetNet using a base scale dataset (16GB) and a large scale dataset (160GB) respectively. Then we conduct experiments on CNN/DailyMail, Gigaword, and SQuAD 1.1 benchmarks for abstractive summarization and question generation tasks. Experimental results show that ProphetNet achieves new state-of-the-art results on all these datasets compared to the models using the same scale pretraining corpus._

The Authors’ code can be found [here](https://github.com/microsoft/ProphetNet).

Tips:

-   XLM-ProphetNet’s model architecture and pretraining objective is same as ProphetNet, but XLM-ProphetNet was pre-trained on the cross-lingual dataset XGLUE.

## Documentation resources

-   [Causal language modeling task guide](../tasks/language_modeling)
-   [Translation task guide](../tasks/translation)
-   [Summarization task guide](../tasks/summarization)

## XLMProphetNetConfig

### class transformers.XLMProphetNetConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlm_prophetnet/configuration_xlm_prophetnet.py#L33)

( activation\_dropout: typing.Optional\[float\] = 0.1activation\_function: typing.Union\[str, typing.Callable, NoneType\] = 'gelu'vocab\_size: typing.Optional\[int\] = 30522hidden\_size: typing.Optional\[int\] = 1024encoder\_ffn\_dim: typing.Optional\[int\] = 4096num\_encoder\_layers: typing.Optional\[int\] = 12num\_encoder\_attention\_heads: typing.Optional\[int\] = 16decoder\_ffn\_dim: typing.Optional\[int\] = 4096num\_decoder\_layers: typing.Optional\[int\] = 12num\_decoder\_attention\_heads: typing.Optional\[int\] = 16attention\_dropout: typing.Optional\[float\] = 0.1dropout: typing.Optional\[float\] = 0.1max\_position\_embeddings: typing.Optional\[int\] = 512init\_std: typing.Optional\[float\] = 0.02is\_encoder\_decoder: typing.Optional\[bool\] = Trueadd\_cross\_attention: typing.Optional\[bool\] = Truedecoder\_start\_token\_id: typing.Optional\[int\] = 0ngram: typing.Optional\[int\] = 2num\_buckets: typing.Optional\[int\] = 32relative\_max\_distance: typing.Optional\[int\] = 128disable\_ngram\_loss: typing.Optional\[bool\] = Falseeps: typing.Optional\[float\] = 0.0use\_cache: typing.Optional\[bool\] = Truepad\_token\_id: typing.Optional\[int\] = 0bos\_token\_id: typing.Optional\[int\] = 1eos\_token\_id: typing.Optional\[int\] = 2\*\*kwargs )

This is the configuration class to store the configuration of a [XLMProphetNetModel](/docs/transformers/v4.34.0/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetModel). It is used to instantiate a XLMProphetNet model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the XLMProphetNet [microsoft/xprophetnet-large-wiki100-cased](https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## XLMProphetNetTokenizer

### class transformers.XLMProphetNetTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlm_prophetnet/tokenization_xlm_prophetnet.py#L59)

( vocab\_filebos\_token = '\[SEP\]'eos\_token = '\[SEP\]'sep\_token = '\[SEP\]'unk\_token = '\[UNK\]'pad\_token = '\[PAD\]'cls\_token = '\[CLS\]'mask\_token = '\[MASK\]'sp\_model\_kwargs: typing.Union\[typing.Dict\[str, typing.Any\], NoneType\] = None\*\*kwargs )

Adapted from [RobertaTokenizer](/docs/transformers/v4.34.0/en/model_doc/roberta#transformers.RobertaTokenizer) and [XLNetTokenizer](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetTokenizer). Based on [SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlm_prophetnet/tokenization_xlm_prophetnet.py#L320)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

list of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A XLMProphetNet sequence has the following format:

-   single sequence: `X [SEP]`
-   pair of sequences: `A [SEP] B [SEP]`

Converts a sequence of tokens (strings for sub-words) in a single string.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlm_prophetnet/tokenization_xlm_prophetnet.py#L247)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. XLMProphetNet does not make use of token type ids, therefore a list of zeros is returned.

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlm_prophetnet/tokenization_xlm_prophetnet.py#L219)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = Nonealready\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

## XLMProphetNetModel

### class transformers.XLMProphetNetModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlm_prophetnet/modeling_xlm_prophetnet.py#L1770)

( config: XLMProphetNetConfig )

Parameters

-   **config** ([XLMProphetNetConfig](/docs/transformers/v4.34.0/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare XLMProphetNet Model outputting raw hidden-states without any specific head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlm_prophetnet/modeling_xlm_prophetnet.py#L1804)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.Tensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.BoolTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetSeq2SeqModelOutput` or `tuple(torch.FloatTensor)`

The [XLMProphetNetModel](/docs/transformers/v4.34.0/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, XLMProphetNetModel

>>> tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone")
>>> model = XLMProphetNetModel.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone")

>>> input_ids = tokenizer(
...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
... ).input_ids  
>>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  
>>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

>>> last_hidden_states = outputs.last_hidden_state  
>>> last_hidden_states_ngram = outputs.last_hidden_state_ngram  
```

## XLMProphetNetEncoder

### class transformers.XLMProphetNetEncoder

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlm_prophetnet/modeling_xlm_prophetnet.py#L1252)

( config: XLMProphetNetConfigword\_embeddings: Embedding = None )

Parameters

-   **config** ([XLMProphetNetConfig](/docs/transformers/v4.34.0/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The standalone encoder part of the XLMProphetNetModel. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and behavior.

word\_embeddings (`torch.nn.Embeddings` of shape `(config.vocab_size, config.hidden_size)`, _optional_): The word embedding parameters. This can be used to initialize [XLMProphetNetEncoder](/docs/transformers/v4.34.0/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetEncoder) with pre-defined word embeddings instead of randomly initialized word embeddings.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlm_prophetnet/modeling_xlm_prophetnet.py#L1282)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

The [XLMProphetNetEncoder](/docs/transformers/v4.34.0/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetEncoder) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, XLMProphetNetEncoder
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone")
>>> model = XLMProphetNetEncoder.from_pretrained("patrickvonplaten/prophetnet-large-uncased-standalone")
>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## XLMProphetNetDecoder

### class transformers.XLMProphetNetDecoder

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlm_prophetnet/modeling_xlm_prophetnet.py#L1393)

( config: XLMProphetNetConfigword\_embeddings: typing.Optional\[torch.nn.modules.sparse.Embedding\] = None )

Parameters

-   **config** ([XLMProphetNetConfig](/docs/transformers/v4.34.0/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The standalone decoder part of the XLMProphetNetModel. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and behavior.

word\_embeddings (`torch.nn.Embeddings` of shape `(config.vocab_size, config.hidden_size)`, _optional_): The word embedding parameters. This can be used to initialize [XLMProphetNetEncoder](/docs/transformers/v4.34.0/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetEncoder) with pre-defined word embeddings instead of randomly initialized word embeddings.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlm_prophetnet/modeling_xlm_prophetnet.py#L1430)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.Tensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetDecoderModelOutput` or `tuple(torch.FloatTensor)`

The [XLMProphetNetDecoder](/docs/transformers/v4.34.0/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetDecoder) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, XLMProphetNetDecoder
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone")
>>> model = XLMProphetNetDecoder.from_pretrained(
...     "patrickvonplaten/xprophetnet-large-uncased-standalone", add_cross_attention=False
... )
>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## XLMProphetNetForConditionalGeneration

### class transformers.XLMProphetNetForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlm_prophetnet/modeling_xlm_prophetnet.py#L1900)

( config: XLMProphetNetConfig )

Parameters

-   **config** ([XLMProphetNetConfig](/docs/transformers/v4.34.0/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The XLMProphetNet Model with a language modeling head. Can be used for sequence generation tasks. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlm_prophetnet/modeling_xlm_prophetnet.py#L1923)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.Tensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.BoolTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetSeq2SeqLMOutput` or `tuple(torch.FloatTensor)`

The [XLMProphetNetForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, XLMProphetNetForConditionalGeneration

>>> tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone")
>>> model = XLMProphetNetForConditionalGeneration.from_pretrained(
...     "patrickvonplaten/xprophetnet-large-uncased-standalone"
... )

>>> input_ids = tokenizer(
...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
... ).input_ids  
>>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  
>>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

>>> logits_next_token = outputs.logits  
>>> logits_ngram_next_tokens = outputs.logits_ngram  
```

## XLMProphetNetForCausalLM

### class transformers.XLMProphetNetForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlm_prophetnet/modeling_xlm_prophetnet.py#L2116)

( config: XLMProphetNetConfig )

Parameters

-   **config** ([XLMProphetNetConfig](/docs/transformers/v4.34.0/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The standalone decoder part of the XLMProphetNetModel with a lm head on top. The model can be used for causal language modeling. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlm_prophetnet/modeling_xlm_prophetnet.py#L2153)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.Tensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetDecoderLMOutput` or `tuple(torch.FloatTensor)`

The [XLMProphetNetForCausalLM](/docs/transformers/v4.34.0/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, XLMProphetNetForCausalLM
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone")
>>> model = XLMProphetNetForCausalLM.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone")
>>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> logits = outputs.logits

>>> 
>>> from transformers import BertTokenizer, EncoderDecoderModel, AutoTokenizer
>>> import torch

>>> tokenizer_enc = BertTokenizer.from_pretrained("bert-large-uncased")
>>> tokenizer_dec = AutoTokenizer.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone")
>>> model = EncoderDecoderModel.from_encoder_decoder_pretrained(
...     "bert-large-uncased", "patrickvonplaten/xprophetnet-large-uncased-standalone"
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