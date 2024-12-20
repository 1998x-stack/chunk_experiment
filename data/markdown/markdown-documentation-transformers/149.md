# DPR

[![Models](https://img.shields.io/badge/All_model_pages-dpr-blueviolet)](https://huggingface.co/models?filter=dpr) [![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/docs-demos/dpr-question_encoder-bert-base-multilingual)

## Overview

Dense Passage Retrieval (DPR) is a set of tools and models for state-of-the-art open-domain Q&A research. It was introduced in [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) by Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih.

The abstract from the paper is the following:

_Open-domain question answering relies on efficient passage retrieval to select candidate contexts, where traditional sparse vector space models, such as TF-IDF or BM25, are the de facto method. In this work, we show that retrieval can be practically implemented using dense representations alone, where embeddings are learned from a small number of questions and passages by a simple dual-encoder framework. When evaluated on a wide range of open-domain QA datasets, our dense retriever outperforms a strong Lucene-BM25 system largely by 9%-19% absolute in terms of top-20 passage retrieval accuracy, and helps our end-to-end QA system establish new state-of-the-art on multiple open-domain QA benchmarks._

This model was contributed by [lhoestq](https://huggingface.co/lhoestq). The original code can be found [here](https://github.com/facebookresearch/DPR).

Tips:

-   DPR consists in three models:
    
    -   Question encoder: encode questions as vectors
    -   Context encoder: encode contexts as vectors
    -   Reader: extract the answer of the questions inside retrieved contexts, along with a relevance score (high if the inferred span actually answers the question).

## DPRConfig

### class transformers.DPRConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpr/configuration_dpr.py#L45)

( vocab\_size = 30522hidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout\_prob = 0.1attention\_probs\_dropout\_prob = 0.1max\_position\_embeddings = 512type\_vocab\_size = 2initializer\_range = 0.02layer\_norm\_eps = 1e-12pad\_token\_id = 0position\_embedding\_type = 'absolute'projection\_dim: int = 0\*\*kwargs )

[DPRConfig](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRConfig) is the configuration class to store the configuration of a _DPRModel_.

This is the configuration class to store the configuration of a [DPRContextEncoder](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRContextEncoder), [DPRQuestionEncoder](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRQuestionEncoder), or a [DPRReader](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRReader). It is used to instantiate the components of the DPR model according to the specified arguments, defining the model component architectures. Instantiating a configuration with the defaults will yield a similar configuration to that of the DPRContextEncoder [facebook/dpr-ctx\_encoder-single-nq-base](https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base) architecture.

This class is a subclass of [BertConfig](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertConfig). Please check the superclass for the documentation of all kwargs.

Example:

```
>>> from transformers import DPRConfig, DPRContextEncoder

>>> 
>>> configuration = DPRConfig()

>>> 
>>> model = DPRContextEncoder(configuration)

>>> 
>>> configuration = model.config
```

## DPRContextEncoderTokenizer

### class transformers.DPRContextEncoderTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpr/tokenization_dpr.py#L113)

( vocab\_filedo\_lower\_case = Truedo\_basic\_tokenize = Truenever\_split = Noneunk\_token = '\[UNK\]'sep\_token = '\[SEP\]'pad\_token = '\[PAD\]'cls\_token = '\[CLS\]'mask\_token = '\[MASK\]'tokenize\_chinese\_chars = Truestrip\_accents = None\*\*kwargs )

Construct a DPRContextEncoder tokenizer.

[DPRContextEncoderTokenizer](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRContextEncoderTokenizer) is identical to [BertTokenizer](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizer) and runs end-to-end tokenization: punctuation splitting and wordpiece.

Refer to superclass [BertTokenizer](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizer) for usage examples and documentation concerning parameters.

## DPRContextEncoderTokenizerFast

### class transformers.DPRContextEncoderTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpr/tokenization_dpr_fast.py#L114)

( vocab\_file = Nonetokenizer\_file = Nonedo\_lower\_case = Trueunk\_token = '\[UNK\]'sep\_token = '\[SEP\]'pad\_token = '\[PAD\]'cls\_token = '\[CLS\]'mask\_token = '\[MASK\]'tokenize\_chinese\_chars = Truestrip\_accents = None\*\*kwargs )

Construct a “fast” DPRContextEncoder tokenizer (backed by HuggingFace’s _tokenizers_ library).

[DPRContextEncoderTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRContextEncoderTokenizerFast) is identical to [BertTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizerFast) and runs end-to-end tokenization: punctuation splitting and wordpiece.

Refer to superclass [BertTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizerFast) for usage examples and documentation concerning parameters.

## DPRQuestionEncoderTokenizer

### class transformers.DPRQuestionEncoderTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpr/tokenization_dpr.py#L129)

( vocab\_filedo\_lower\_case = Truedo\_basic\_tokenize = Truenever\_split = Noneunk\_token = '\[UNK\]'sep\_token = '\[SEP\]'pad\_token = '\[PAD\]'cls\_token = '\[CLS\]'mask\_token = '\[MASK\]'tokenize\_chinese\_chars = Truestrip\_accents = None\*\*kwargs )

Constructs a DPRQuestionEncoder tokenizer.

[DPRQuestionEncoderTokenizer](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRQuestionEncoderTokenizer) is identical to [BertTokenizer](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizer) and runs end-to-end tokenization: punctuation splitting and wordpiece.

Refer to superclass [BertTokenizer](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizer) for usage examples and documentation concerning parameters.

## DPRQuestionEncoderTokenizerFast

### class transformers.DPRQuestionEncoderTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpr/tokenization_dpr_fast.py#L131)

( vocab\_file = Nonetokenizer\_file = Nonedo\_lower\_case = Trueunk\_token = '\[UNK\]'sep\_token = '\[SEP\]'pad\_token = '\[PAD\]'cls\_token = '\[CLS\]'mask\_token = '\[MASK\]'tokenize\_chinese\_chars = Truestrip\_accents = None\*\*kwargs )

Constructs a “fast” DPRQuestionEncoder tokenizer (backed by HuggingFace’s _tokenizers_ library).

[DPRQuestionEncoderTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRQuestionEncoderTokenizerFast) is identical to [BertTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizerFast) and runs end-to-end tokenization: punctuation splitting and wordpiece.

Refer to superclass [BertTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizerFast) for usage examples and documentation concerning parameters.

## DPRReaderTokenizer

### class transformers.DPRReaderTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpr/tokenization_dpr.py#L395)

( vocab\_filedo\_lower\_case = Truedo\_basic\_tokenize = Truenever\_split = Noneunk\_token = '\[UNK\]'sep\_token = '\[SEP\]'pad\_token = '\[PAD\]'cls\_token = '\[CLS\]'mask\_token = '\[MASK\]'tokenize\_chinese\_chars = Truestrip\_accents = None\*\*kwargs ) → `Dict[str, List[List[int]]]`

Construct a DPRReader tokenizer.

[DPRReaderTokenizer](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRReaderTokenizer) is almost identical to [BertTokenizer](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizer) and runs end-to-end tokenization: punctuation splitting and wordpiece. The difference is that is has three inputs strings: question, titles and texts that are combined to be fed to the [DPRReader](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRReader) model.

Refer to superclass [BertTokenizer](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizer) for usage examples and documentation concerning parameters.

Return a dictionary with the token ids of the input strings and other information to give to `.decode_best_spans`. It converts the strings of a question and different passages (title and text) in a sequence of IDs (integers), using the tokenizer and vocabulary. The resulting `input_ids` is a matrix of size `(n_passages, sequence_length)`

with the format:

```
[CLS] <question token ids> [SEP] <titles ids> [SEP] <texts ids>
```

## DPRReaderTokenizerFast

### class transformers.DPRReaderTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpr/tokenization_dpr_fast.py#L393)

( vocab\_file = Nonetokenizer\_file = Nonedo\_lower\_case = Trueunk\_token = '\[UNK\]'sep\_token = '\[SEP\]'pad\_token = '\[PAD\]'cls\_token = '\[CLS\]'mask\_token = '\[MASK\]'tokenize\_chinese\_chars = Truestrip\_accents = None\*\*kwargs ) → `Dict[str, List[List[int]]]`

Constructs a “fast” DPRReader tokenizer (backed by HuggingFace’s _tokenizers_ library).

[DPRReaderTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRReaderTokenizerFast) is almost identical to [BertTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizerFast) and runs end-to-end tokenization: punctuation splitting and wordpiece. The difference is that is has three inputs strings: question, titles and texts that are combined to be fed to the [DPRReader](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRReader) model.

Refer to superclass [BertTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizerFast) for usage examples and documentation concerning parameters.

Return a dictionary with the token ids of the input strings and other information to give to `.decode_best_spans`. It converts the strings of a question and different passages (title and text) in a sequence of IDs (integers), using the tokenizer and vocabulary. The resulting `input_ids` is a matrix of size `(n_passages, sequence_length)` with the format:

\[CLS\] <question token ids> \[SEP\] <titles ids> \[SEP\] <texts ids>

## DPR specific outputs

### class transformers.models.dpr.modeling\_dpr.DPRContextEncoderOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpr/modeling_dpr.py#L62)

( pooler\_output: FloatTensorhidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Class for outputs of [DPRQuestionEncoder](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRQuestionEncoder).

### class transformers.models.dpr.modeling\_dpr.DPRQuestionEncoderOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpr/modeling_dpr.py#L90)

( pooler\_output: FloatTensorhidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Class for outputs of [DPRQuestionEncoder](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRQuestionEncoder).

### class transformers.DPRReaderOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpr/modeling_dpr.py#L118)

( start\_logits: FloatTensorend\_logits: FloatTensor = Nonerelevance\_logits: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Class for outputs of [DPRQuestionEncoder](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRQuestionEncoder).

## DPRContextEncoder

### class transformers.DPRContextEncoder

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpr/modeling_dpr.py#L441)

( config: DPRConfig )

Parameters

-   **config** ([DPRConfig](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare DPRContextEncoder transformer outputting pooler outputs as context representations.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpr/modeling_dpr.py#L449)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.models.dpr.modeling\_dpr.DPRContextEncoderOutput](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.models.dpr.modeling_dpr.DPRContextEncoderOutput) or `tuple(torch.FloatTensor)`

The [DPRContextEncoder](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRContextEncoder) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

>>> tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
>>> model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
>>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="pt")["input_ids"]
>>> embeddings = model(input_ids).pooler_output
```

## DPRQuestionEncoder

### class transformers.DPRQuestionEncoder

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpr/modeling_dpr.py#L522)

( config: DPRConfig )

Parameters

-   **config** ([DPRConfig](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare DPRQuestionEncoder transformer outputting pooler outputs as question representations.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpr/modeling_dpr.py#L530)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.models.dpr.modeling\_dpr.DPRQuestionEncoderOutput](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.models.dpr.modeling_dpr.DPRQuestionEncoderOutput) or `tuple(torch.FloatTensor)`

The [DPRQuestionEncoder](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRQuestionEncoder) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

>>> tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
>>> model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
>>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="pt")["input_ids"]
>>> embeddings = model(input_ids).pooler_output
```

## DPRReader

### class transformers.DPRReader

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpr/modeling_dpr.py#L604)

( config: DPRConfig )

Parameters

-   **config** ([DPRConfig](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare DPRReader transformer outputting span predictions.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpr/modeling_dpr.py#L612)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.models.dpr.modeling\_dpr.DPRReaderOutput](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRReaderOutput) or `tuple(torch.FloatTensor)`

The [DPRReader](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRReader) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import DPRReader, DPRReaderTokenizer

>>> tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")
>>> model = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base")
>>> encoded_inputs = tokenizer(
...     questions=["What is love ?"],
...     titles=["Haddaway"],
...     texts=["'What Is Love' is a song recorded by the artist Haddaway"],
...     return_tensors="pt",
... )
>>> outputs = model(**encoded_inputs)
>>> start_logits = outputs.start_logits
>>> end_logits = outputs.end_logits
>>> relevance_logits = outputs.relevance_logits
```

## TFDPRContextEncoder

### class transformers.TFDPRContextEncoder

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpr/modeling_tf_dpr.py#L526)

( \*args\*\*kwargs )

Parameters

-   **config** ([DPRConfig](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The bare DPRContextEncoder transformer outputting pooler outputs as context representations.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Tensorflow [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TensorFlow models and layers in `transformers` accept two formats as input:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional argument.

The reason the second format is supported is that Keras methods prefer this format when passing inputs to models and layers. Because of this support, when using methods like `model.fit()` things should “just work” for you - just pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first positional argument:

-   a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
-   a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
-   a dictionary with one or several input Tensors associated to the input names given in the docstring: `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

Note that when creating models and layers with [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don’t need to worry about any of this, as you can just pass inputs like you would to any other Python function!

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpr/modeling_tf_dpr.py#L538)

( input\_ids: TFModelInputType | None = Noneattention\_mask: tf.Tensor | None = Nonetoken\_type\_ids: tf.Tensor | None = Noneinputs\_embeds: tf.Tensor | None = Noneoutput\_attentions: bool | None = Noneoutput\_hidden\_states: bool | None = Nonereturn\_dict: bool | None = Nonetraining: bool = False ) → `transformers.models.dpr.modeling_tf_dpr.TFDPRContextEncoderOutput` or `tuple(tf.Tensor)`

The [TFDPRContextEncoder](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.TFDPRContextEncoder) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import TFDPRContextEncoder, DPRContextEncoderTokenizer

>>> tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
>>> model = TFDPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", from_pt=True)
>>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="tf")["input_ids"]
>>> embeddings = model(input_ids).pooler_output
```

## TFDPRQuestionEncoder

### class transformers.TFDPRQuestionEncoder

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpr/modeling_tf_dpr.py#L607)

( \*args\*\*kwargs )

Parameters

-   **config** ([DPRConfig](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The bare DPRQuestionEncoder transformer outputting pooler outputs as question representations.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Tensorflow [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TensorFlow models and layers in `transformers` accept two formats as input:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional argument.

The reason the second format is supported is that Keras methods prefer this format when passing inputs to models and layers. Because of this support, when using methods like `model.fit()` things should “just work” for you - just pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first positional argument:

-   a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
-   a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
-   a dictionary with one or several input Tensors associated to the input names given in the docstring: `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

Note that when creating models and layers with [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don’t need to worry about any of this, as you can just pass inputs like you would to any other Python function!

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpr/modeling_tf_dpr.py#L619)

( input\_ids: TFModelInputType | None = Noneattention\_mask: tf.Tensor | None = Nonetoken\_type\_ids: tf.Tensor | None = Noneinputs\_embeds: tf.Tensor | None = Noneoutput\_attentions: bool | None = Noneoutput\_hidden\_states: bool | None = Nonereturn\_dict: bool | None = Nonetraining: bool = False ) → `transformers.models.dpr.modeling_tf_dpr.TFDPRQuestionEncoderOutput` or `tuple(tf.Tensor)`

The [TFDPRQuestionEncoder](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.TFDPRQuestionEncoder) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import TFDPRQuestionEncoder, DPRQuestionEncoderTokenizer

>>> tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
>>> model = TFDPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base", from_pt=True)
>>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="tf")["input_ids"]
>>> embeddings = model(input_ids).pooler_output
```

## TFDPRReader

### class transformers.TFDPRReader

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpr/modeling_tf_dpr.py#L687)

( \*args\*\*kwargs )

Parameters

-   **config** ([DPRConfig](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The bare DPRReader transformer outputting span predictions.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Tensorflow [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TensorFlow models and layers in `transformers` accept two formats as input:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional argument.

The reason the second format is supported is that Keras methods prefer this format when passing inputs to models and layers. Because of this support, when using methods like `model.fit()` things should “just work” for you - just pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first positional argument:

-   a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
-   a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
-   a dictionary with one or several input Tensors associated to the input names given in the docstring: `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

Note that when creating models and layers with [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don’t need to worry about any of this, as you can just pass inputs like you would to any other Python function!

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpr/modeling_tf_dpr.py#L699)

( input\_ids: TFModelInputType | None = Noneattention\_mask: tf.Tensor | None = Noneinputs\_embeds: tf.Tensor | None = Noneoutput\_attentions: bool | None = Noneoutput\_hidden\_states: bool | None = Nonereturn\_dict: bool | None = Nonetraining: bool = False ) → `transformers.models.dpr.modeling_tf_dpr.TFDPRReaderOutput` or `tuple(tf.Tensor)`

The [TFDPRReader](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.TFDPRReader) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import TFDPRReader, DPRReaderTokenizer

>>> tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")
>>> model = TFDPRReader.from_pretrained("facebook/dpr-reader-single-nq-base", from_pt=True)
>>> encoded_inputs = tokenizer(
...     questions=["What is love ?"],
...     titles=["Haddaway"],
...     texts=["'What Is Love' is a song recorded by the artist Haddaway"],
...     return_tensors="tf",
... )
>>> outputs = model(encoded_inputs)
>>> start_logits = outputs.start_logits
>>> end_logits = outputs.end_logits
>>> relevance_logits = outputs.relevance_logits
```