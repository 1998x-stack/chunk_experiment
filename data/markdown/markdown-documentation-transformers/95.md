# ALBERT

[![Models](https://img.shields.io/badge/All_model_pages-albert-blueviolet)](https://huggingface.co/models?filter=albert) [![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/docs-demos/albert-base-v2)

## Overview

The ALBERT model was proposed in [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut. It presents two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT:

-   Splitting the embedding matrix into two smaller matrices.
-   Using repeating layers split among groups.

The abstract from the paper is the following:

_Increasing model size when pretraining natural language representations often results in improved performance on downstream tasks. However, at some point further model increases become harder due to GPU/TPU memory limitations, longer training times, and unexpected model degradation. To address these problems, we present two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT. Comprehensive empirical evidence shows that our proposed methods lead to models that scale much better compared to the original BERT. We also use a self-supervised loss that focuses on modeling inter-sentence coherence, and show it consistently helps downstream tasks with multi-sentence inputs. As a result, our best model establishes new state-of-the-art results on the GLUE, RACE, and SQuAD benchmarks while having fewer parameters compared to BERT-large._

Tips:

-   ALBERT is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left.
-   ALBERT uses repeating layers which results in a small memory footprint, however the computational cost remains similar to a BERT-like architecture with the same number of hidden layers as it has to iterate through the same number of (repeating) layers.
-   Embedding size E is different from hidden size H justified because the embeddings are context independent (one embedding vector represents one token), whereas hidden states are context dependent (one hidden state represents a sequence of tokens) so it’s more logical to have H >> E. Also, the embedding matrix is large since it’s V x E (V being the vocab size). If E < H, it has less parameters.
-   Layers are split in groups that share parameters (to save memory). Next sentence prediction is replaced by a sentence ordering prediction: in the inputs, we have two sentences A and B (that are consecutive) and we either feed A followed by B or B followed by A. The model must predict if they have been swapped or not.

This model was contributed by [lysandre](https://huggingface.co/lysandre). This model jax version was contributed by [kamalkraj](https://huggingface.co/kamalkraj). The original code can be found [here](https://github.com/google-research/ALBERT).

## Documentation resources

-   [Text classification task guide](../tasks/sequence_classification)
-   [Token classification task guide](../tasks/token_classification)
-   [Question answering task guide](../tasks/question_answering)
-   [Masked language modeling task guide](../tasks/masked_language_modeling)
-   [Multiple choice task guide](../tasks/multiple_choice)

## AlbertConfig

### class transformers.AlbertConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/configuration_albert.py#L36)

( vocab\_size = 30000embedding\_size = 128hidden\_size = 4096num\_hidden\_layers = 12num\_hidden\_groups = 1num\_attention\_heads = 64intermediate\_size = 16384inner\_group\_num = 1hidden\_act = 'gelu\_new'hidden\_dropout\_prob = 0attention\_probs\_dropout\_prob = 0max\_position\_embeddings = 512type\_vocab\_size = 2initializer\_range = 0.02layer\_norm\_eps = 1e-12classifier\_dropout\_prob = 0.1position\_embedding\_type = 'absolute'pad\_token\_id = 0bos\_token\_id = 2eos\_token\_id = 3\*\*kwargs )

This is the configuration class to store the configuration of a [AlbertModel](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertModel) or a [TFAlbertModel](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.TFAlbertModel). It is used to instantiate an ALBERT model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the ALBERT [albert-xxlarge-v2](https://huggingface.co/albert-xxlarge-v2) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import AlbertConfig, AlbertModel

>>> 
>>> albert_xxlarge_configuration = AlbertConfig()

>>> 
>>> albert_base_configuration = AlbertConfig(
...     hidden_size=768,
...     num_attention_heads=12,
...     intermediate_size=3072,
... )

>>> 
>>> model = AlbertModel(albert_xxlarge_configuration)

>>> 
>>> configuration = model.config
```

## AlbertTokenizer

### class transformers.AlbertTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/tokenization_albert.py#L59)

( vocab\_filedo\_lower\_case = Trueremove\_space = Truekeep\_accents = Falsebos\_token = '\[CLS\]'eos\_token = '\[SEP\]'unk\_token = '<unk>'sep\_token = '\[SEP\]'pad\_token = '<pad>'cls\_token = '\[CLS\]'mask\_token = '\[MASK\]'sp\_model\_kwargs: typing.Union\[typing.Dict\[str, typing.Any\], NoneType\] = None\*\*kwargs )

Construct an ALBERT tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/tokenization_albert.py#L273)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. An ALBERT sequence has the following format:

-   single sequence: `[CLS] X [SEP]`
-   pair of sequences: `[CLS] A [SEP] B [SEP]`

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/tokenization_albert.py#L298)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = Nonealready\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/tokenization_albert.py#L326)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).

Create a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT

sequence pair mask has the following format:

```
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence    | second sequence |
```

If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/tokenization_albert.py#L356)

( save\_directory: strfilename\_prefix: typing.Optional\[str\] = None )

## AlbertTokenizerFast

### class transformers.AlbertTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/tokenization_albert_fast.py#L72)

( vocab\_file = Nonetokenizer\_file = Nonedo\_lower\_case = Trueremove\_space = Truekeep\_accents = Falsebos\_token = '\[CLS\]'eos\_token = '\[SEP\]'unk\_token = '<unk>'sep\_token = '\[SEP\]'pad\_token = '<pad>'cls\_token = '\[CLS\]'mask\_token = '\[MASK\]'\*\*kwargs )

Construct a “fast” ALBERT tokenizer (backed by HuggingFace’s _tokenizers_ library). Based on [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models). This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/tokenization_albert_fast.py#L173)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

list of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. An ALBERT sequence has the following format:

-   single sequence: `[CLS] X [SEP]`
-   pair of sequences: `[CLS] A [SEP] B [SEP]`

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/tokenization_albert_fast.py#L198)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of ids.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).

Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT

sequence pair mask has the following format:

```
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence    | second sequence |
```

if token\_ids\_1 is None, only returns the first portion of the mask (0s).

## Albert specific outputs

### class transformers.models.albert.modeling\_albert.AlbertForPreTrainingOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_albert.py#L531)

( loss: typing.Optional\[torch.FloatTensor\] = Noneprediction\_logits: FloatTensor = Nonesop\_logits: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Output type of [AlbertForPreTraining](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertForPreTraining).

### class transformers.models.albert.modeling\_tf\_albert.TFAlbertForPreTrainingOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_tf_albert.py#L655)

( loss: tf.Tensor = Noneprediction\_logits: tf.Tensor = Nonesop\_logits: tf.Tensor = Nonehidden\_states: Tuple\[tf.Tensor\] | None = Noneattentions: Tuple\[tf.Tensor\] | None = None )

Output type of [TFAlbertForPreTraining](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.TFAlbertForPreTraining).

## AlbertModel

### class transformers.AlbertModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_albert.py#L634)

( config: AlbertConfigadd\_pooling\_layer: bool = True )

Parameters

-   **config** ([AlbertConfig](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare ALBERT Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_albert.py#L677)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

The [AlbertModel](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, AlbertModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
>>> model = AlbertModel.from_pretrained("albert-base-v2")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## AlbertForPreTraining

### class transformers.AlbertForPreTraining

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_albert.py#L763)

( config: AlbertConfig )

Parameters

-   **config** ([AlbertConfig](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Albert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `sentence order prediction (classification)` head.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_albert.py#L785)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Nonesentence\_order\_label: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.models.albert.modeling\_albert.AlbertForPreTrainingOutput](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.models.albert.modeling_albert.AlbertForPreTrainingOutput) or `tuple(torch.FloatTensor)`

The [AlbertForPreTraining](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, AlbertForPreTraining
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
>>> model = AlbertForPreTraining.from_pretrained("albert-base-v2")

>>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)
>>> 
>>> outputs = model(input_ids)

>>> prediction_logits = outputs.prediction_logits
>>> sop_logits = outputs.sop_logits
```

## AlbertForMaskedLM

### class transformers.AlbertForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_albert.py#L911)

( config )

Parameters

-   **config** ([AlbertConfig](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Albert Model with a `language modeling` head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_albert.py#L932)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

The [AlbertForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoTokenizer, AlbertForMaskedLM

>>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
>>> model = AlbertForMaskedLM.from_pretrained("albert-base-v2")

>>> 
>>> inputs = tokenizer("The capital of [MASK] is Paris.", return_tensors="pt")
>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
>>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
>>> tokenizer.decode(predicted_token_id)
'france'
```

```
>>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
>>> labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
>>> outputs = model(**inputs, labels=labels)
>>> round(outputs.loss.item(), 2)
0.81
```

## AlbertForSequenceClassification

### class transformers.AlbertForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_albert.py#L1025)

( config: AlbertConfig )

Parameters

-   **config** ([AlbertConfig](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Albert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_albert.py#L1038)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [AlbertForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example of single-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, AlbertForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("textattack/albert-base-v2-imdb")
>>> model = AlbertForSequenceClassification.from_pretrained("textattack/albert-base-v2-imdb")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
'LABEL_1'

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = AlbertForSequenceClassification.from_pretrained("textattack/albert-base-v2-imdb", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
0.12
```

Example of multi-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, AlbertForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("textattack/albert-base-v2-imdb")
>>> model = AlbertForSequenceClassification.from_pretrained("textattack/albert-base-v2-imdb", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = AlbertForSequenceClassification.from_pretrained(
...     "textattack/albert-base-v2-imdb", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## AlbertForMultipleChoice

### class transformers.AlbertForMultipleChoice

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_albert.py#L1312)

( config: AlbertConfig )

Parameters

-   **config** ([AlbertConfig](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Albert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a softmax) e.g. for RocStories/SWAG tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_albert.py#L1323)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.MultipleChoiceModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or `tuple(torch.FloatTensor)`

The [AlbertForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertForMultipleChoice) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, AlbertForMultipleChoice
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
>>> model = AlbertForMultipleChoice.from_pretrained("albert-base-v2")

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

## AlbertForTokenClassification

### class transformers.AlbertForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_albert.py#L1126)

( config: AlbertConfig )

Parameters

-   **config** ([AlbertConfig](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Albert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_albert.py#L1143)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

The [AlbertForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, AlbertForTokenClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
>>> model = AlbertForTokenClassification.from_pretrained("albert-base-v2")

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

## AlbertForQuestionAnswering

### class transformers.AlbertForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_albert.py#L1209)

( config: AlbertConfig )

Parameters

-   **config** ([AlbertConfig](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Albert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_albert.py#L1220)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonestart\_positions: typing.Optional\[torch.LongTensor\] = Noneend\_positions: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

The [AlbertForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, AlbertForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("twmkn9/albert-base-v2-squad2")
>>> model = AlbertForQuestionAnswering.from_pretrained("twmkn9/albert-base-v2-squad2")

>>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

>>> inputs = tokenizer(question, text, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> answer_start_index = outputs.start_logits.argmax()
>>> answer_end_index = outputs.end_logits.argmax()

>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
'a nice puppet'

>>> 
>>> target_start_index = torch.tensor([12])
>>> target_end_index = torch.tensor([13])

>>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
>>> loss = outputs.loss
>>> round(loss.item(), 2)
7.36
```

## TFAlbertModel

### class transformers.TFAlbertModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_tf_albert.py#L787)

( \*args\*\*kwargs )

Parameters

-   **config** ([AlbertConfig](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Albert Model transformer outputting raw hidden-states without any specific head on top.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_tf_albert.py#L793)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonetoken\_type\_ids: np.ndarray | tf.Tensor | None = Noneposition\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFBaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling) or `tuple(tf.Tensor)`

The [TFAlbertModel](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.TFAlbertModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFAlbertModel
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
>>> model = TFAlbertModel.from_pretrained("albert-base-v2")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
>>> outputs = model(inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## TFAlbertForPreTraining

### class transformers.TFAlbertForPreTraining

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_tf_albert.py#L836)

( \*args\*\*kwargs )

Parameters

-   **config** ([AlbertConfig](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Albert Model with two heads on top for pretraining: a `masked language modeling` head and a `sentence order prediction` (classification) head.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_tf_albert.py#L852)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonetoken\_type\_ids: np.ndarray | tf.Tensor | None = Noneposition\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonelabels: np.ndarray | tf.Tensor | None = Nonesentence\_order\_label: np.ndarray | tf.Tensor | None = Nonetraining: Optional\[bool\] = False ) → [transformers.models.albert.modeling\_tf\_albert.TFAlbertForPreTrainingOutput](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.models.albert.modeling_tf_albert.TFAlbertForPreTrainingOutput) or `tuple(tf.Tensor)`

The [TFAlbertForPreTraining](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.TFAlbertForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import tensorflow as tf
>>> from transformers import AutoTokenizer, TFAlbertForPreTraining

>>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
>>> model = TFAlbertForPreTraining.from_pretrained("albert-base-v2")

>>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]
>>> 
>>> outputs = model(input_ids)

>>> prediction_logits = outputs.prediction_logits
>>> sop_logits = outputs.sop_logits
```

## TFAlbertForMaskedLM

### class transformers.TFAlbertForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_tf_albert.py#L944)

( \*args\*\*kwargs )

Parameters

-   **config** ([AlbertConfig](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Albert Model with a `language modeling` head on top.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_tf_albert.py#L957)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonetoken\_type\_ids: np.ndarray | tf.Tensor | None = Noneposition\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonelabels: np.ndarray | tf.Tensor | None = Nonetraining: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFMaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFMaskedLMOutput) or `tuple(tf.Tensor)`

The [TFAlbertForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.TFAlbertForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import tensorflow as tf
>>> from transformers import AutoTokenizer, TFAlbertForMaskedLM

>>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
>>> model = TFAlbertForMaskedLM.from_pretrained("albert-base-v2")

>>> 
>>> inputs = tokenizer(f"The capital of [MASK] is Paris.", return_tensors="tf")
>>> logits = model(**inputs).logits

>>> 
>>> mask_token_index = tf.where(inputs.input_ids == tokenizer.mask_token_id)[0][1]
>>> predicted_token_id = tf.math.argmax(logits[0, mask_token_index], axis=-1)
>>> tokenizer.decode(predicted_token_id)
'france'
```

```
>>> labels = tokenizer("The capital of France is Paris.", return_tensors="tf")["input_ids"]
>>> labels = tf.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
>>> outputs = model(**inputs, labels=labels)
>>> round(float(outputs.loss), 2)
0.81
```

## TFAlbertForSequenceClassification

### class transformers.TFAlbertForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_tf_albert.py#L1046)

( \*args\*\*kwargs )

Parameters

-   **config** ([AlbertConfig](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Albert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_tf_albert.py#L1062)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonetoken\_type\_ids: np.ndarray | tf.Tensor | None = Noneposition\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonelabels: np.ndarray | tf.Tensor | None = Nonetraining: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSequenceClassifierOutput) or `tuple(tf.Tensor)`

The [TFAlbertForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.TFAlbertForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFAlbertForSequenceClassification
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("vumichien/albert-base-v2-imdb")
>>> model = TFAlbertForSequenceClassification.from_pretrained("vumichien/albert-base-v2-imdb")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")

>>> logits = model(**inputs).logits

>>> predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
>>> model.config.id2label[predicted_class_id]
'LABEL_1'
```

```
>>> 
>>> num_labels = len(model.config.id2label)
>>> model = TFAlbertForSequenceClassification.from_pretrained("vumichien/albert-base-v2-imdb", num_labels=num_labels)

>>> labels = tf.constant(1)
>>> loss = model(**inputs, labels=labels).loss
>>> round(float(loss), 2)
0.12
```

## TFAlbertForMultipleChoice

### class transformers.TFAlbertForMultipleChoice

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_tf_albert.py#L1306)

( \*args\*\*kwargs )

Parameters

-   **config** ([AlbertConfig](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Albert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a softmax) e.g. for RocStories/SWAG tasks.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_tf_albert.py#L1320)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonetoken\_type\_ids: np.ndarray | tf.Tensor | None = Noneposition\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonelabels: np.ndarray | tf.Tensor | None = Nonetraining: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFMultipleChoiceModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFMultipleChoiceModelOutput) or `tuple(tf.Tensor)`

The [TFAlbertForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.TFAlbertForMultipleChoice) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFAlbertForMultipleChoice
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
>>> model = TFAlbertForMultipleChoice.from_pretrained("albert-base-v2")

>>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
>>> choice0 = "It is eaten with a fork and a knife."
>>> choice1 = "It is eaten while held in the hand."

>>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="tf", padding=True)
>>> inputs = {k: tf.expand_dims(v, 0) for k, v in encoding.items()}
>>> outputs = model(inputs)  

>>> 
>>> logits = outputs.logits
```

## TFAlbertForTokenClassification

### class transformers.TFAlbertForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_tf_albert.py#L1128)

( \*args\*\*kwargs )

Parameters

-   **config** ([AlbertConfig](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Albert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_tf_albert.py#L1149)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonetoken\_type\_ids: np.ndarray | tf.Tensor | None = Noneposition\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonelabels: np.ndarray | tf.Tensor | None = Nonetraining: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFTokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFTokenClassifierOutput) or `tuple(tf.Tensor)`

The [TFAlbertForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.TFAlbertForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFAlbertForTokenClassification
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
>>> model = TFAlbertForTokenClassification.from_pretrained("albert-base-v2")

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

## TFAlbertForQuestionAnswering

### class transformers.TFAlbertForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_tf_albert.py#L1211)

( \*args\*\*kwargs )

Parameters

-   **config** ([AlbertConfig](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Albert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_tf_albert.py#L1225)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonetoken\_type\_ids: np.ndarray | tf.Tensor | None = Noneposition\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonestart\_positions: np.ndarray | tf.Tensor | None = Noneend\_positions: np.ndarray | tf.Tensor | None = Nonetraining: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFQuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFQuestionAnsweringModelOutput) or `tuple(tf.Tensor)`

The [TFAlbertForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.TFAlbertForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFAlbertForQuestionAnswering
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("vumichien/albert-base-v2-squad2")
>>> model = TFAlbertForQuestionAnswering.from_pretrained("vumichien/albert-base-v2-squad2")

>>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

>>> inputs = tokenizer(question, text, return_tensors="tf")
>>> outputs = model(**inputs)

>>> answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
>>> answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])

>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens)
'a nice puppet'
```

```
>>> 
>>> target_start_index = tf.constant([12])
>>> target_end_index = tf.constant([13])

>>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
>>> loss = tf.math.reduce_mean(outputs.loss)
>>> round(float(loss), 2)
7.36
```

## FlaxAlbertModel

### class transformers.FlaxAlbertModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_flax_albert.py#L676)

( config: AlbertConfiginput\_shape: typing.Tuple = (1, 1)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([AlbertConfig](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

The bare Albert Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_flax_albert.py#L553)

( input\_idsattention\_mask = Nonetoken\_type\_ids = Noneposition\_ids = Noneparams: dict = Nonedropout\_rng: PRNGKey = Nonetrain: bool = Falseoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

The `FlaxAlbertPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxAlbertModel

>>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
>>> model = FlaxAlbertModel.from_pretrained("albert-base-v2")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## FlaxAlbertForPreTraining

### class transformers.FlaxAlbertForPreTraining

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_flax_albert.py#L744)

( config: AlbertConfiginput\_shape: typing.Tuple = (1, 1)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([AlbertConfig](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

Albert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `sentence order prediction (classification)` head.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_flax_albert.py#L553)

( input\_idsattention\_mask = Nonetoken\_type\_ids = Noneposition\_ids = Noneparams: dict = Nonedropout\_rng: PRNGKey = Nonetrain: bool = Falseoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.albert.modeling_flax_albert.FlaxAlbertForPreTrainingOutput` or `tuple(torch.FloatTensor)`

The `FlaxAlbertPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxAlbertForPreTraining

>>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
>>> model = FlaxAlbertForPreTraining.from_pretrained("albert-base-v2")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
>>> outputs = model(**inputs)

>>> prediction_logits = outputs.prediction_logits
>>> seq_relationship_logits = outputs.sop_logits
```

## FlaxAlbertForMaskedLM

### class transformers.FlaxAlbertForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_flax_albert.py#L827)

( config: AlbertConfiginput\_shape: typing.Tuple = (1, 1)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([AlbertConfig](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

Albert Model with a `language modeling` head on top.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_flax_albert.py#L553)

( input\_idsattention\_mask = Nonetoken\_type\_ids = Noneposition\_ids = Noneparams: dict = Nonedropout\_rng: PRNGKey = Nonetrain: bool = Falseoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_flax\_outputs.FlaxMaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxMaskedLMOutput) or `tuple(torch.FloatTensor)`

The `FlaxAlbertPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxAlbertForMaskedLM

>>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
>>> model = FlaxAlbertForMaskedLM.from_pretrained("albert-base-v2")

>>> inputs = tokenizer("The capital of France is [MASK].", return_tensors="jax")

>>> outputs = model(**inputs)
>>> logits = outputs.logits
```

## FlaxAlbertForSequenceClassification

### class transformers.FlaxAlbertForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_flax_albert.py#L895)

( config: AlbertConfiginput\_shape: typing.Tuple = (1, 1)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([AlbertConfig](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

Albert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_flax_albert.py#L553)

( input\_idsattention\_mask = Nonetoken\_type\_ids = Noneposition\_ids = Noneparams: dict = Nonedropout\_rng: PRNGKey = Nonetrain: bool = Falseoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_flax\_outputs.FlaxSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The `FlaxAlbertPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxAlbertForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
>>> model = FlaxAlbertForSequenceClassification.from_pretrained("albert-base-v2")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

>>> outputs = model(**inputs)
>>> logits = outputs.logits
```

## FlaxAlbertForMultipleChoice

### class transformers.FlaxAlbertForMultipleChoice

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_flax_albert.py#L968)

( config: AlbertConfiginput\_shape: typing.Tuple = (1, 1)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([AlbertConfig](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

Albert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a softmax) e.g. for RocStories/SWAG tasks.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_flax_albert.py#L553)

( input\_idsattention\_mask = Nonetoken\_type\_ids = Noneposition\_ids = Noneparams: dict = Nonedropout\_rng: PRNGKey = Nonetrain: bool = Falseoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_flax\_outputs.FlaxMultipleChoiceModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxMultipleChoiceModelOutput) or `tuple(torch.FloatTensor)`

The `FlaxAlbertPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxAlbertForMultipleChoice

>>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
>>> model = FlaxAlbertForMultipleChoice.from_pretrained("albert-base-v2")

>>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
>>> choice0 = "It is eaten with a fork and a knife."
>>> choice1 = "It is eaten while held in the hand."

>>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="jax", padding=True)
>>> outputs = model(**{k: v[None, :] for k, v in encoding.items()})

>>> logits = outputs.logits
```

## FlaxAlbertForTokenClassification

### class transformers.FlaxAlbertForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_flax_albert.py#L1041)

( config: AlbertConfiginput\_shape: typing.Tuple = (1, 1)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([AlbertConfig](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

Albert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_flax_albert.py#L553)

( input\_idsattention\_mask = Nonetoken\_type\_ids = Noneposition\_ids = Noneparams: dict = Nonedropout\_rng: PRNGKey = Nonetrain: bool = Falseoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_flax\_outputs.FlaxTokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxTokenClassifierOutput) or `tuple(torch.FloatTensor)`

The `FlaxAlbertPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxAlbertForTokenClassification

>>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
>>> model = FlaxAlbertForTokenClassification.from_pretrained("albert-base-v2")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

>>> outputs = model(**inputs)
>>> logits = outputs.logits
```

## FlaxAlbertForQuestionAnswering

### class transformers.FlaxAlbertForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_flax_albert.py#L1109)

( config: AlbertConfiginput\_shape: typing.Tuple = (1, 1)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([AlbertConfig](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.AlbertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

Albert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_flax_albert.py#L553)

( input\_idsattention\_mask = Nonetoken\_type\_ids = Noneposition\_ids = Noneparams: dict = Nonedropout\_rng: PRNGKey = Nonetrain: bool = Falseoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_flax\_outputs.FlaxQuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxQuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

The `FlaxAlbertPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxAlbertForQuestionAnswering

>>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
>>> model = FlaxAlbertForQuestionAnswering.from_pretrained("albert-base-v2")

>>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
>>> inputs = tokenizer(question, text, return_tensors="jax")

>>> outputs = model(**inputs)
>>> start_scores = outputs.start_logits
>>> end_scores = outputs.end_logits
```