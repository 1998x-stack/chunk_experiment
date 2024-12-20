# RetriBERT

This model is in maintenance mode only, so we won’t accept any new PRs changing its code.

If you run into any issues running this model, please reinstall the last version that supported this model: v4.30.0. You can do so by running the following command: `pip install -U transformers==4.30.0`.

## Overview

The RetriBERT model was proposed in the blog post [Explain Anything Like I’m Five: A Model for Open Domain Long Form Question Answering](https://yjernite.github.io/lfqa.html). RetriBERT is a small model that uses either a single or pair of BERT encoders with lower-dimension projection for dense semantic indexing of text.

This model was contributed by [yjernite](https://huggingface.co/yjernite). Code to train and use the model can be found [here](https://github.com/huggingface/transformers/tree/main/examples/research-projects/distillation).

## RetriBertConfig

### class transformers.RetriBertConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deprecated/retribert/configuration_retribert.py#L31)

( vocab\_size = 30522hidden\_size = 768num\_hidden\_layers = 8num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout\_prob = 0.1attention\_probs\_dropout\_prob = 0.1max\_position\_embeddings = 512type\_vocab\_size = 2initializer\_range = 0.02layer\_norm\_eps = 1e-12share\_encoders = Trueprojection\_dim = 128pad\_token\_id = 0\*\*kwargs )

This is the configuration class to store the configuration of a [RetriBertModel](/docs/transformers/v4.34.0/en/model_doc/retribert#transformers.RetriBertModel). It is used to instantiate a RetriBertModel model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the RetriBERT [yjernite/retribert-base-uncased](https://huggingface.co/yjernite/retribert-base-uncased) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## RetriBertTokenizer

### class transformers.RetriBertTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deprecated/retribert/tokenization_retribert.py#L70)

( vocab\_filedo\_lower\_case = Truedo\_basic\_tokenize = Truenever\_split = Noneunk\_token = '\[UNK\]'sep\_token = '\[SEP\]'pad\_token = '\[PAD\]'cls\_token = '\[CLS\]'mask\_token = '\[MASK\]'tokenize\_chinese\_chars = Truestrip\_accents = None\*\*kwargs )

Constructs a RetriBERT tokenizer.

[RetriBertTokenizer](/docs/transformers/v4.34.0/en/model_doc/retribert#transformers.RetriBertTokenizer) is identical to [BertTokenizer](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizer) and runs end-to-end tokenization: punctuation splitting and wordpiece.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to: this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deprecated/retribert/tokenization_retribert.py#L214)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A BERT sequence has the following format:

-   single sequence: `[CLS] X [SEP]`
-   pair of sequences: `[CLS] A [SEP] B [SEP]`

Converts a sequence of tokens (string) in a single string.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deprecated/retribert/tokenization_retribert.py#L269)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).

Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence

pair mask has the following format:

```
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence    | second sequence |
```

If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deprecated/retribert/tokenization_retribert.py#L240)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = Nonealready\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

## RetriBertTokenizerFast

### class transformers.RetriBertTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deprecated/retribert/tokenization_retribert_fast.py#L54)

( vocab\_file = Nonetokenizer\_file = Nonedo\_lower\_case = Trueunk\_token = '\[UNK\]'sep\_token = '\[SEP\]'pad\_token = '\[PAD\]'cls\_token = '\[CLS\]'mask\_token = '\[MASK\]'tokenize\_chinese\_chars = Truestrip\_accents = None\*\*kwargs )

Construct a “fast” RetriBERT tokenizer (backed by HuggingFace’s _tokenizers_ library).

[RetriBertTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/retribert#transformers.RetriBertTokenizerFast) is identical to [BertTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizerFast) and runs end-to-end tokenization: punctuation splitting and wordpiece.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deprecated/retribert/tokenization_retribert_fast.py#L148)

( token\_ids\_0token\_ids\_1 = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A BERT sequence has the following format:

-   single sequence: `[CLS] X [SEP]`
-   pair of sequences: `[CLS] A [SEP] B [SEP]`

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deprecated/retribert/tokenization_retribert_fast.py#L173)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).

Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence

pair mask has the following format:

```
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence    | second sequence |
```

If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

## RetriBertModel

### class transformers.RetriBertModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deprecated/retribert/modeling_retribert.py#L88)

( config: RetriBertConfig )

Parameters

-   **config** ([RetriBertConfig](/docs/transformers/v4.34.0/en/model_doc/retribert#transformers.RetriBertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Bert Based model to embed queries or document for document retrieval.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deprecated/retribert/modeling_retribert.py#L176)

( input\_ids\_query: LongTensorattention\_mask\_query: typing.Optional\[torch.FloatTensor\]input\_ids\_doc: LongTensorattention\_mask\_doc: typing.Optional\[torch.FloatTensor\]checkpoint\_batch\_size: int = -1 ) → \`torch.FloatTensor“