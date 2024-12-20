# XLNet

[![Models](https://img.shields.io/badge/All_model_pages-xlnet-blueviolet)](https://huggingface.co/models?filter=xlnet) [![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/docs-demos/xlnet-base-cased)

## Overview

The XLNet model was proposed in [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) by Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le. XLnet is an extension of the Transformer-XL model pre-trained using an autoregressive method to learn bidirectional contexts by maximizing the expected likelihood over all permutations of the input sequence factorization order.

The abstract from the paper is the following:

_With the capability of modeling bidirectional contexts, denoising autoencoding based pretraining like BERT achieves better performance than pretraining approaches based on autoregressive language modeling. However, relying on corrupting the input with masks, BERT neglects dependency between the masked positions and suffers from a pretrain-finetune discrepancy. In light of these pros and cons, we propose XLNet, a generalized autoregressive pretraining method that (1) enables learning bidirectional contexts by maximizing the expected likelihood over all permutations of the factorization order and (2) overcomes the limitations of BERT thanks to its autoregressive formulation. Furthermore, XLNet integrates ideas from Transformer-XL, the state-of-the-art autoregressive model, into pretraining. Empirically, under comparable experiment settings, XLNet outperforms BERT on 20 tasks, often by a large margin, including question answering, natural language inference, sentiment analysis, and document ranking._

Tips:

-   The specific attention pattern can be controlled at training and test time using the `perm_mask` input.
-   Due to the difficulty of training a fully auto-regressive model over various factorization order, XLNet is pretrained using only a sub-set of the output tokens as target which are selected with the `target_mapping` input.
-   To use XLNet for sequential decoding (i.e. not in fully bi-directional setting), use the `perm_mask` and `target_mapping` inputs to control the attention span and outputs (see examples in _examples/pytorch/text-generation/run\_generation.py_)
-   XLNet is one of the few models that has no sequence length limit.
-   XLNet is not a traditional autoregressive model but uses a training strategy that builds on that. It permutes the tokens in the sentence, then allows the model to use the last n tokens to predict the token n+1. Since this is all done with a mask, the sentence is actually fed in the model in the right order, but instead of masking the first n tokens for n+1, XLNet uses a mask that hides the previous tokens in some given permutation of 1,…,sequence length.
-   XLNet also uses the same recurrence mechanism as Transformer-XL to build long-term dependencies.

This model was contributed by [thomwolf](https://huggingface.co/thomwolf). The original code can be found [here](https://github.com/zihangdai/xlnet/).

## Documentation resources

-   [Text classification task guide](../tasks/sequence_classification)
-   [Token classification task guide](../tasks/token_classification)
-   [Question answering task guide](../tasks/question_answering)
-   [Causal language modeling task guide](../tasks/language_modeling)
-   [Multiple choice task guide](../tasks/multiple_choice)

## XLNetConfig

### class transformers.XLNetConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/configuration_xlnet.py#L32)

( vocab\_size = 32000d\_model = 1024n\_layer = 24n\_head = 16d\_inner = 4096ff\_activation = 'gelu'untie\_r = Trueattn\_type = 'bi'initializer\_range = 0.02layer\_norm\_eps = 1e-12dropout = 0.1mem\_len = 512reuse\_len = Noneuse\_mems\_eval = Trueuse\_mems\_train = Falsebi\_data = Falseclamp\_len = -1same\_length = Falsesummary\_type = 'last'summary\_use\_proj = Truesummary\_activation = 'tanh'summary\_last\_dropout = 0.1start\_n\_top = 5end\_n\_top = 5pad\_token\_id = 5bos\_token\_id = 1eos\_token\_id = 2\*\*kwargs )

This is the configuration class to store the configuration of a [XLNetModel](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetModel) or a [TFXLNetModel](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.TFXLNetModel). It is used to instantiate a XLNet model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the [xlnet-large-cased](https://huggingface.co/xlnet-large-cased) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import XLNetConfig, XLNetModel

>>> 
>>> configuration = XLNetConfig()

>>> 
>>> model = XLNetModel(configuration)

>>> 
>>> configuration = model.config
```

## XLNetTokenizer

### class transformers.XLNetTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/tokenization_xlnet.py#L53)

( vocab\_filedo\_lower\_case = Falseremove\_space = Truekeep\_accents = Falsebos\_token = '<s>'eos\_token = '</s>'unk\_token = '<unk>'sep\_token = '<sep>'pad\_token = '<pad>'cls\_token = '<cls>'mask\_token = '<mask>'additional\_special\_tokens = \['<eop>', '<eod>'\]sp\_model\_kwargs: typing.Union\[typing.Dict\[str, typing.Any\], NoneType\] = None\*\*kwargs )

Construct an XLNet tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/tokenization_xlnet.py#L298)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. An XLNet sequence has the following format:

-   single sequence: `X <sep> <cls>`
-   pair of sequences: `A <sep> B <sep> <cls>`

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/tokenization_xlnet.py#L323)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = Nonealready\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/tokenization_xlnet.py#L351)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).

Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLNet

sequence pair mask has the following format:

```
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence    | second sequence |
```

If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/tokenization_xlnet.py#L381)

( save\_directory: strfilename\_prefix: typing.Optional\[str\] = None )

## XLNetTokenizerFast

### class transformers.XLNetTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/tokenization_xlnet_fast.py#L63)

( vocab\_file = Nonetokenizer\_file = Nonedo\_lower\_case = Falseremove\_space = Truekeep\_accents = Falsebos\_token = '<s>'eos\_token = '</s>'unk\_token = '<unk>'sep\_token = '<sep>'pad\_token = '<pad>'cls\_token = '<cls>'mask\_token = '<mask>'additional\_special\_tokens = \['<eop>', '<eod>'\]\*\*kwargs )

Construct a “fast” XLNet tokenizer (backed by HuggingFace’s _tokenizers_ library). Based on [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/tokenization_xlnet_fast.py#L177)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. An XLNet sequence has the following format:

-   single sequence: `X <sep> <cls>`
-   pair of sequences: `A <sep> B <sep> <cls>`

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/tokenization_xlnet_fast.py#L202)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).

Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLNet

sequence pair mask has the following format:

```
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence    | second sequence |
```

If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

## XLNet specific outputs

### class transformers.models.xlnet.modeling\_xlnet.XLNetModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_xlnet.py#L579)

( last\_hidden\_state: FloatTensormems: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Output type of [XLNetModel](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetModel).

### class transformers.models.xlnet.modeling\_xlnet.XLNetLMHeadModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_xlnet.py#L613)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Nonemems: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Output type of [XLNetLMHeadModel](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetLMHeadModel).

### class transformers.models.xlnet.modeling\_xlnet.XLNetForSequenceClassificationOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_xlnet.py#L650)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Nonemems: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Output type of [XLNetForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetForSequenceClassification).

### class transformers.models.xlnet.modeling\_xlnet.XLNetForMultipleChoiceOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_xlnet.py#L718)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Nonemems: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Output type of [XLNetForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetForMultipleChoice).

### class transformers.models.xlnet.modeling\_xlnet.XLNetForTokenClassificationOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_xlnet.py#L684)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Nonemems: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Output type of `XLNetForTokenClassificationOutput`.

### class transformers.models.xlnet.modeling\_xlnet.XLNetForQuestionAnsweringSimpleOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_xlnet.py#L754)

( loss: typing.Optional\[torch.FloatTensor\] = Nonestart\_logits: FloatTensor = Noneend\_logits: FloatTensor = Nonemems: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Output type of [XLNetForQuestionAnsweringSimple](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetForQuestionAnsweringSimple).

### class transformers.models.xlnet.modeling\_xlnet.XLNetForQuestionAnsweringOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_xlnet.py#L791)

( loss: typing.Optional\[torch.FloatTensor\] = Nonestart\_top\_log\_probs: typing.Optional\[torch.FloatTensor\] = Nonestart\_top\_index: typing.Optional\[torch.LongTensor\] = Noneend\_top\_log\_probs: typing.Optional\[torch.FloatTensor\] = Noneend\_top\_index: typing.Optional\[torch.LongTensor\] = Nonecls\_logits: typing.Optional\[torch.FloatTensor\] = Nonemems: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Output type of [XLNetForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetForQuestionAnswering).

### class transformers.models.xlnet.modeling\_tf\_xlnet.TFXLNetModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_tf_xlnet.py#L802)

( last\_hidden\_state: tf.Tensor = Nonemems: List\[tf.Tensor\] | None = Nonehidden\_states: Tuple\[tf.Tensor\] | None = Noneattentions: Tuple\[tf.Tensor\] | None = None )

Output type of [TFXLNetModel](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.TFXLNetModel).

### class transformers.models.xlnet.modeling\_tf\_xlnet.TFXLNetLMHeadModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_tf_xlnet.py#L836)

( loss: tf.Tensor | None = Nonelogits: tf.Tensor = Nonemems: List\[tf.Tensor\] | None = Nonehidden\_states: Tuple\[tf.Tensor\] | None = Noneattentions: Tuple\[tf.Tensor\] | None = None )

Output type of [TFXLNetLMHeadModel](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.TFXLNetLMHeadModel).

### class transformers.models.xlnet.modeling\_tf\_xlnet.TFXLNetForSequenceClassificationOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_tf_xlnet.py#L873)

( loss: tf.Tensor | None = Nonelogits: tf.Tensor = Nonemems: List\[tf.Tensor\] | None = Nonehidden\_states: Tuple\[tf.Tensor\] | None = Noneattentions: Tuple\[tf.Tensor\] | None = None )

Output type of [TFXLNetForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.TFXLNetForSequenceClassification).

### class transformers.models.xlnet.modeling\_tf\_xlnet.TFXLNetForMultipleChoiceOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_tf_xlnet.py#L941)

( loss: tf.Tensor | None = Nonelogits: tf.Tensor = Nonemems: List\[tf.Tensor\] | None = Nonehidden\_states: Tuple\[tf.Tensor\] | None = Noneattentions: Tuple\[tf.Tensor\] | None = None )

Output type of [TFXLNetForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.TFXLNetForMultipleChoice).

### class transformers.models.xlnet.modeling\_tf\_xlnet.TFXLNetForTokenClassificationOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_tf_xlnet.py#L907)

( loss: tf.Tensor | None = Nonelogits: tf.Tensor = Nonemems: List\[tf.Tensor\] | None = Nonehidden\_states: Tuple\[tf.Tensor\] | None = Noneattentions: Tuple\[tf.Tensor\] | None = None )

Output type of `TFXLNetForTokenClassificationOutput`.

### class transformers.models.xlnet.modeling\_tf\_xlnet.TFXLNetForQuestionAnsweringSimpleOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_tf_xlnet.py#L977)

( loss: tf.Tensor | None = Nonestart\_logits: tf.Tensor = Noneend\_logits: tf.Tensor = Nonemems: List\[tf.Tensor\] | None = Nonehidden\_states: Tuple\[tf.Tensor\] | None = Noneattentions: Tuple\[tf.Tensor\] | None = None )

Output type of [TFXLNetForQuestionAnsweringSimple](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.TFXLNetForQuestionAnsweringSimple).

## XLNetModel

### class transformers.XLNetModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_xlnet.py#L931)

( config )

Parameters

-   **config** ([XLNetConfig](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare XLNet Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_xlnet.py#L1059)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonemems: typing.Optional\[torch.Tensor\] = Noneperm\_mask: typing.Optional\[torch.Tensor\] = Nonetarget\_mapping: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneinput\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Noneuse\_mems: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None\*\*kwargs ) → [transformers.models.xlnet.modeling\_xlnet.XLNetModelOutput](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetModelOutput) or `tuple(torch.FloatTensor)`

The [XLNetModel](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, XLNetModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
>>> model = XLNetModel.from_pretrained("xlnet-base-cased")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## XLNetLMHeadModel

### class transformers.XLNetLMHeadModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_xlnet.py#L1294)

( config )

Parameters

-   **config** ([XLNetConfig](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

XLNet Model with a language modeling head on top (linear layer with weights tied to the input embeddings).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_xlnet.py#L1356)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonemems: typing.Optional\[torch.Tensor\] = Noneperm\_mask: typing.Optional\[torch.Tensor\] = Nonetarget\_mapping: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneinput\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneuse\_mems: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None\*\*kwargs ) → [transformers.models.xlnet.modeling\_xlnet.XLNetLMHeadModelOutput](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput) or `tuple(torch.FloatTensor)`

The [XLNetLMHeadModel](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetLMHeadModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, XLNetLMHeadModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("xlnet-large-cased")
>>> model = XLNetLMHeadModel.from_pretrained("xlnet-large-cased")

>>> 
>>> input_ids = torch.tensor(
...     tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=False)
... ).unsqueeze(
...     0
... )  
>>> perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
>>> perm_mask[:, :, -1] = 1.0  
>>> target_mapping = torch.zeros(
...     (1, 1, input_ids.shape[1]), dtype=torch.float
... )  
>>> target_mapping[
...     0, 0, -1
... ] = 1.0  

>>> outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
>>> next_token_logits = outputs[
...     0
... ]  

>>> 
>>> input_ids = torch.tensor(
...     tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=False)
... ).unsqueeze(
...     0
... )  
>>> labels = torch.tensor(tokenizer.encode("cute", add_special_tokens=False)).unsqueeze(0)
>>> assert labels.shape[0] == 1, "only one word will be predicted"
>>> perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
>>> perm_mask[
...     :, :, -1
... ] = 1.0  
>>> target_mapping = torch.zeros(
...     (1, 1, input_ids.shape[1]), dtype=torch.float
... )  
>>> target_mapping[
...     0, 0, -1
... ] = 1.0  

>>> outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping, labels=labels)
>>> loss = outputs.loss
>>> next_token_logits = (
...     outputs.logits
... )  
```

## XLNetForSequenceClassification

### class transformers.XLNetForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_xlnet.py#L1500)

( config )

Parameters

-   **config** ([XLNetConfig](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

XLNet Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_xlnet.py#L1513)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonemems: typing.Optional\[torch.Tensor\] = Noneperm\_mask: typing.Optional\[torch.Tensor\] = Nonetarget\_mapping: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneinput\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneuse\_mems: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None\*\*kwargs ) → [transformers.models.xlnet.modeling\_xlnet.XLNetForSequenceClassificationOutput](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput) or `tuple(torch.FloatTensor)`

The [XLNetForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example of single-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, XLNetForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
>>> model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
```

Example of multi-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, XLNetForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
>>> model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = XLNetForSequenceClassification.from_pretrained(
...     "xlnet-base-cased", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## XLNetForMultipleChoice

### class transformers.XLNetForMultipleChoice

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_xlnet.py#L1696)

( config )

Parameters

-   **config** ([XLNetConfig](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

XLNet Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a softmax) e.g. for RACE/SWAG tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_xlnet.py#L1707)

( input\_ids: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneinput\_mask: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonemems: typing.Optional\[torch.Tensor\] = Noneperm\_mask: typing.Optional\[torch.Tensor\] = Nonetarget\_mapping: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneuse\_mems: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None\*\*kwargs ) → [transformers.models.xlnet.modeling\_xlnet.XLNetForMultipleChoiceOutput](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput) or `tuple(torch.FloatTensor)`

The [XLNetForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetForMultipleChoice) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, XLNetForMultipleChoice
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
>>> model = XLNetForMultipleChoice.from_pretrained("xlnet-base-cased")

>>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
>>> choice0 = "It is eaten with a fork and a knife."
>>> choice1 = "It is eaten while held in the hand."
>>> labels = torch.tensor(0).unsqueeze(0)  

>>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="pt", padding=True)
>>> outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  

>>> 
>>> loss = outputs.loss
>>> logits = outputs.logits
```

## XLNetForTokenClassification

### class transformers.XLNetForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_xlnet.py#L1609)

( config )

Parameters

-   **config** ([XLNetConfig](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

XLNet Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_xlnet.py#L1620)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonemems: typing.Optional\[torch.Tensor\] = Noneperm\_mask: typing.Optional\[torch.Tensor\] = Nonetarget\_mapping: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneinput\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneuse\_mems: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None\*\*kwargs ) → [transformers.models.xlnet.modeling\_xlnet.XLNetForTokenClassificationOutput](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput) or `tuple(torch.FloatTensor)`

The [XLNetForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, XLNetForTokenClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
>>> model = XLNetForTokenClassification.from_pretrained("xlnet-base-cased")

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

## XLNetForQuestionAnsweringSimple

### class transformers.XLNetForQuestionAnsweringSimple

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_xlnet.py#L1799)

( config )

Parameters

-   **config** ([XLNetConfig](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

XLNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_xlnet.py#L1810)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonemems: typing.Optional\[torch.Tensor\] = Noneperm\_mask: typing.Optional\[torch.Tensor\] = Nonetarget\_mapping: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneinput\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonestart\_positions: typing.Optional\[torch.Tensor\] = Noneend\_positions: typing.Optional\[torch.Tensor\] = Noneuse\_mems: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None\*\*kwargs ) → [transformers.models.xlnet.modeling\_xlnet.XLNetForQuestionAnsweringSimpleOutput](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput) or `tuple(torch.FloatTensor)`

The [XLNetForQuestionAnsweringSimple](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetForQuestionAnsweringSimple) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, XLNetForQuestionAnsweringSimple
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
>>> model = XLNetForQuestionAnsweringSimple.from_pretrained("xlnet-base-cased")

>>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

>>> inputs = tokenizer(question, text, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> answer_start_index = outputs.start_logits.argmax()
>>> answer_end_index = outputs.end_logits.argmax()

>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]

>>> 
>>> target_start_index = torch.tensor([14])
>>> target_end_index = torch.tensor([15])

>>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
>>> loss = outputs.loss
```

## XLNetForQuestionAnswering

### class transformers.XLNetForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_xlnet.py#L1909)

( config )

Parameters

-   **config** ([XLNetConfig](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

XLNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_xlnet.py#L1923)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonemems: typing.Optional\[torch.Tensor\] = Noneperm\_mask: typing.Optional\[torch.Tensor\] = Nonetarget\_mapping: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneinput\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonestart\_positions: typing.Optional\[torch.Tensor\] = Noneend\_positions: typing.Optional\[torch.Tensor\] = Noneis\_impossible: typing.Optional\[torch.Tensor\] = Nonecls\_index: typing.Optional\[torch.Tensor\] = Nonep\_mask: typing.Optional\[torch.Tensor\] = Noneuse\_mems: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None\*\*kwargs ) → [transformers.models.xlnet.modeling\_xlnet.XLNetForQuestionAnsweringOutput](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput) or `tuple(torch.FloatTensor)`

The [XLNetForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, XLNetForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
>>> model = XLNetForQuestionAnswering.from_pretrained("xlnet-base-cased")

>>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(
...     0
... )  
>>> start_positions = torch.tensor([1])
>>> end_positions = torch.tensor([3])
>>> outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)

>>> loss = outputs.loss
```

## TFXLNetModel

### class transformers.TFXLNetModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_tf_xlnet.py#L1132)

( \*args\*\*kwargs )

Parameters

-   **config** ([XLNetConfig](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare XLNet Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TensorFlow models and layers in `transformers` accept two formats as input:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional argument.

The reason the second format is supported is that Keras methods prefer this format when passing inputs to models and layers. Because of this support, when using methods like `model.fit()` things should “just work” for you - just pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first positional argument:

-   a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
-   a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
-   a dictionary with one or several input Tensors associated to the input names given in the docstring: `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

Note that when creating models and layers with [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don’t need to worry about any of this, as you can just pass inputs like you would to any other Python function!

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_tf_xlnet.py#L1137)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonemems: np.ndarray | tf.Tensor | None = Noneperm\_mask: np.ndarray | tf.Tensor | None = Nonetarget\_mapping: np.ndarray | tf.Tensor | None = Nonetoken\_type\_ids: np.ndarray | tf.Tensor | None = Noneinput\_mask: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Noneuse\_mems: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False ) → [transformers.models.xlnet.modeling\_tf\_xlnet.TFXLNetModelOutput](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.models.xlnet.modeling_tf_xlnet.TFXLNetModelOutput) or `tuple(tf.Tensor)`

The [TFXLNetModel](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.TFXLNetModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFXLNetModel
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
>>> model = TFXLNetModel.from_pretrained("xlnet-base-cased")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
>>> outputs = model(inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## TFXLNetLMHeadModel

### class transformers.TFXLNetLMHeadModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_tf_xlnet.py#L1187)

( \*args\*\*kwargs )

Parameters

-   **config** ([XLNetConfig](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

XLNet Model with a language modeling head on top (linear layer with weights tied to the input embeddings).

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TensorFlow models and layers in `transformers` accept two formats as input:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional argument.

The reason the second format is supported is that Keras methods prefer this format when passing inputs to models and layers. Because of this support, when using methods like `model.fit()` things should “just work” for you - just pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first positional argument:

-   a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
-   a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
-   a dictionary with one or several input Tensors associated to the input names given in the docstring: `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

Note that when creating models and layers with [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don’t need to worry about any of this, as you can just pass inputs like you would to any other Python function!

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_tf_xlnet.py#L1241)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonemems: np.ndarray | tf.Tensor | None = Noneperm\_mask: np.ndarray | tf.Tensor | None = Nonetarget\_mapping: np.ndarray | tf.Tensor | None = Nonetoken\_type\_ids: np.ndarray | tf.Tensor | None = Noneinput\_mask: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Noneuse\_mems: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonelabels: np.ndarray | tf.Tensor | None = Nonetraining: bool = False ) → [transformers.models.xlnet.modeling\_tf\_xlnet.TFXLNetLMHeadModelOutput](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.models.xlnet.modeling_tf_xlnet.TFXLNetLMHeadModelOutput) or `tuple(tf.Tensor)`

The [TFXLNetLMHeadModel](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.TFXLNetLMHeadModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import tensorflow as tf
>>> import numpy as np
>>> from transformers import AutoTokenizer, TFXLNetLMHeadModel

>>> tokenizer = AutoTokenizer.from_pretrained("xlnet-large-cased")
>>> model = TFXLNetLMHeadModel.from_pretrained("xlnet-large-cased")

>>> 
>>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=True))[
...     None, :
... ]  

>>> perm_mask = np.zeros((1, input_ids.shape[1], input_ids.shape[1]))
>>> perm_mask[:, :, -1] = 1.0  

>>> target_mapping = np.zeros(
...     (1, 1, input_ids.shape[1])
... )  
>>> target_mapping[
...     0, 0, -1
... ] = 1.0  

>>> outputs = model(
...     input_ids,
...     perm_mask=tf.constant(perm_mask, dtype=tf.float32),
...     target_mapping=tf.constant(target_mapping, dtype=tf.float32),
... )

>>> next_token_logits = outputs[
...     0
... ]  
```

## TFXLNetForSequenceClassification

### class transformers.TFXLNetForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_tf_xlnet.py#L1347)

( \*args\*\*kwargs )

Parameters

-   **config** ([XLNetConfig](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

XLNet Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TensorFlow models and layers in `transformers` accept two formats as input:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional argument.

The reason the second format is supported is that Keras methods prefer this format when passing inputs to models and layers. Because of this support, when using methods like `model.fit()` things should “just work” for you - just pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first positional argument:

-   a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
-   a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
-   a dictionary with one or several input Tensors associated to the input names given in the docstring: `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

Note that when creating models and layers with [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don’t need to worry about any of this, as you can just pass inputs like you would to any other Python function!

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_tf_xlnet.py#L1360)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonemems: np.ndarray | tf.Tensor | None = Noneperm\_mask: np.ndarray | tf.Tensor | None = Nonetarget\_mapping: np.ndarray | tf.Tensor | None = Nonetoken\_type\_ids: np.ndarray | tf.Tensor | None = Noneinput\_mask: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Noneuse\_mems: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonelabels: np.ndarray | tf.Tensor | None = Nonetraining: bool = False ) → [transformers.models.xlnet.modeling\_tf\_xlnet.TFXLNetForSequenceClassificationOutput](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.models.xlnet.modeling_tf_xlnet.TFXLNetForSequenceClassificationOutput) or `tuple(tf.Tensor)`

The [TFXLNetForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.TFXLNetForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFXLNetForSequenceClassification
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
>>> model = TFXLNetForSequenceClassification.from_pretrained("xlnet-base-cased")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")

>>> logits = model(**inputs).logits

>>> predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
```

```
>>> 
>>> num_labels = len(model.config.id2label)
>>> model = TFXLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=num_labels)

>>> labels = tf.constant(1)
>>> loss = model(**inputs, labels=labels).loss
```

## TFLNetForMultipleChoice

### class transformers.TFXLNetForMultipleChoice

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_tf_xlnet.py#L1434)

( \*args\*\*kwargs )

Parameters

-   **config** ([XLNetConfig](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

XLNET Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a softmax) e.g. for RocStories/SWAG tasks.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TensorFlow models and layers in `transformers` accept two formats as input:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional argument.

The reason the second format is supported is that Keras methods prefer this format when passing inputs to models and layers. Because of this support, when using methods like `model.fit()` things should “just work” for you - just pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first positional argument:

-   a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
-   a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
-   a dictionary with one or several input Tensors associated to the input names given in the docstring: `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

Note that when creating models and layers with [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don’t need to worry about any of this, as you can just pass inputs like you would to any other Python function!

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_tf_xlnet.py#L1446)

( input\_ids: TFModelInputType | None = Nonetoken\_type\_ids: np.ndarray | tf.Tensor | None = Noneinput\_mask: np.ndarray | tf.Tensor | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonemems: np.ndarray | tf.Tensor | None = Noneperm\_mask: np.ndarray | tf.Tensor | None = Nonetarget\_mapping: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Noneuse\_mems: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonelabels: np.ndarray | tf.Tensor | None = Nonetraining: bool = False ) → [transformers.models.xlnet.modeling\_tf\_xlnet.TFXLNetForMultipleChoiceOutput](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.models.xlnet.modeling_tf_xlnet.TFXLNetForMultipleChoiceOutput) or `tuple(tf.Tensor)`

The [TFXLNetForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.TFXLNetForMultipleChoice) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFXLNetForMultipleChoice
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
>>> model = TFXLNetForMultipleChoice.from_pretrained("xlnet-base-cased")

>>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
>>> choice0 = "It is eaten with a fork and a knife."
>>> choice1 = "It is eaten while held in the hand."

>>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="tf", padding=True)
>>> inputs = {k: tf.expand_dims(v, 0) for k, v in encoding.items()}
>>> outputs = model(inputs)  

>>> 
>>> logits = outputs.logits
```

## TFXLNetForTokenClassification

### class transformers.TFXLNetForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_tf_xlnet.py#L1535)

( \*args\*\*kwargs )

Parameters

-   **config** ([XLNetConfig](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

XLNet Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TensorFlow models and layers in `transformers` accept two formats as input:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional argument.

The reason the second format is supported is that Keras methods prefer this format when passing inputs to models and layers. Because of this support, when using methods like `model.fit()` things should “just work” for you - just pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first positional argument:

-   a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
-   a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
-   a dictionary with one or several input Tensors associated to the input names given in the docstring: `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

Note that when creating models and layers with [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don’t need to worry about any of this, as you can just pass inputs like you would to any other Python function!

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_tf_xlnet.py#L1545)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonemems: np.ndarray | tf.Tensor | None = Noneperm\_mask: np.ndarray | tf.Tensor | None = Nonetarget\_mapping: np.ndarray | tf.Tensor | None = Nonetoken\_type\_ids: np.ndarray | tf.Tensor | None = Noneinput\_mask: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Noneuse\_mems: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonelabels: np.ndarray | tf.Tensor | None = Nonetraining: bool = False ) → [transformers.models.xlnet.modeling\_tf\_xlnet.TFXLNetForTokenClassificationOutput](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.models.xlnet.modeling_tf_xlnet.TFXLNetForTokenClassificationOutput) or `tuple(tf.Tensor)`

The [TFXLNetForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.TFXLNetForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFXLNetForTokenClassification
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
>>> model = TFXLNetForTokenClassification.from_pretrained("xlnet-base-cased")

>>> inputs = tokenizer(
...     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="tf"
... )

>>> logits = model(**inputs).logits
>>> predicted_token_class_ids = tf.math.argmax(logits, axis=-1)

>>> 
>>> 
>>> 
>>> predicted_tokens_classes = [model.config.id2label[t] for t in predicted_token_class_ids[0].numpy().tolist()]
```

```
>>> labels = predicted_token_class_ids
>>> loss = tf.math.reduce_mean(model(**inputs, labels=labels).loss)
```

## TFXLNetForQuestionAnsweringSimple

### class transformers.TFXLNetForQuestionAnsweringSimple

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_tf_xlnet.py#L1615)

( \*args\*\*kwargs )

Parameters

-   **config** ([XLNetConfig](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

XLNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TensorFlow models and layers in `transformers` accept two formats as input:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional argument.

The reason the second format is supported is that Keras methods prefer this format when passing inputs to models and layers. Because of this support, when using methods like `model.fit()` things should “just work” for you - just pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first positional argument:

-   a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
-   a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
-   a dictionary with one or several input Tensors associated to the input names given in the docstring: `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

Note that when creating models and layers with [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don’t need to worry about any of this, as you can just pass inputs like you would to any other Python function!

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/xlnet/modeling_tf_xlnet.py#L1623)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonemems: np.ndarray | tf.Tensor | None = Noneperm\_mask: np.ndarray | tf.Tensor | None = Nonetarget\_mapping: np.ndarray | tf.Tensor | None = Nonetoken\_type\_ids: np.ndarray | tf.Tensor | None = Noneinput\_mask: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Noneuse\_mems: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonestart\_positions: np.ndarray | tf.Tensor | None = Noneend\_positions: np.ndarray | tf.Tensor | None = Nonetraining: bool = False ) → [transformers.models.xlnet.modeling\_tf\_xlnet.TFXLNetForQuestionAnsweringSimpleOutput](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.models.xlnet.modeling_tf_xlnet.TFXLNetForQuestionAnsweringSimpleOutput) or `tuple(tf.Tensor)`

The [TFXLNetForQuestionAnsweringSimple](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.TFXLNetForQuestionAnsweringSimple) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFXLNetForQuestionAnsweringSimple
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
>>> model = TFXLNetForQuestionAnsweringSimple.from_pretrained("xlnet-base-cased")

>>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

>>> inputs = tokenizer(question, text, return_tensors="tf")
>>> outputs = model(**inputs)

>>> answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
>>> answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])

>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
```

```
>>> 
>>> target_start_index = tf.constant([14])
>>> target_end_index = tf.constant([15])

>>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
>>> loss = tf.math.reduce_mean(outputs.loss)
```