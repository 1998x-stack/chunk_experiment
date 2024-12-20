# SqueezeBERT

## Overview

The SqueezeBERT model was proposed in [SqueezeBERT: What can computer vision teach NLP about efficient neural networks?](https://arxiv.org/abs/2006.11316) by Forrest N. Iandola, Albert E. Shaw, Ravi Krishna, Kurt W. Keutzer. It’s a bidirectional transformer similar to the BERT model. The key difference between the BERT architecture and the SqueezeBERT architecture is that SqueezeBERT uses [grouped convolutions](https://blog.yani.io/filter-group-tutorial) instead of fully-connected layers for the Q, K, V and FFN layers.

The abstract from the paper is the following:

_Humans read and write hundreds of billions of messages every day. Further, due to the availability of large datasets, large computing systems, and better neural network models, natural language processing (NLP) technology has made significant strides in understanding, proofreading, and organizing these messages. Thus, there is a significant opportunity to deploy NLP in myriad applications to help web users, social networks, and businesses. In particular, we consider smartphones and other mobile devices as crucial platforms for deploying NLP models at scale. However, today’s highly-accurate NLP neural network models such as BERT and RoBERTa are extremely computationally expensive, with BERT-base taking 1.7 seconds to classify a text snippet on a Pixel 3 smartphone. In this work, we observe that methods such as grouped convolutions have yielded significant speedups for computer vision networks, but many of these techniques have not been adopted by NLP neural network designers. We demonstrate how to replace several operations in self-attention layers with grouped convolutions, and we use this technique in a novel network architecture called SqueezeBERT, which runs 4.3x faster than BERT-base on the Pixel 3 while achieving competitive accuracy on the GLUE test set. The SqueezeBERT code will be released._

Tips:

-   SqueezeBERT is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left.
-   SqueezeBERT is similar to BERT and therefore relies on the masked language modeling (MLM) objective. It is therefore efficient at predicting masked tokens and at NLU in general, but is not optimal for text generation. Models trained with a causal language modeling (CLM) objective are better in that regard.
-   For best results when finetuning on sequence classification tasks, it is recommended to start with the _squeezebert/squeezebert-mnli-headless_ checkpoint.

This model was contributed by [forresti](https://huggingface.co/forresti).

## Documentation resources

-   [Text classification task guide](../tasks/sequence_classification)
-   [Token classification task guide](../tasks/token_classification)
-   [Question answering task guide](../tasks/question_answering)
-   [Masked language modeling task guide](../tasks/masked_language_modeling)
-   [Multiple choice task guide](../tasks/multiple_choice)

## SqueezeBertConfig

### class transformers.SqueezeBertConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/squeezebert/configuration_squeezebert.py#L37)

( vocab\_size = 30522hidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout\_prob = 0.1attention\_probs\_dropout\_prob = 0.1max\_position\_embeddings = 512type\_vocab\_size = 2initializer\_range = 0.02layer\_norm\_eps = 1e-12pad\_token\_id = 0embedding\_size = 768q\_groups = 4k\_groups = 4v\_groups = 4post\_attention\_groups = 1intermediate\_groups = 4output\_groups = 4\*\*kwargs )

This is the configuration class to store the configuration of a [SqueezeBertModel](/docs/transformers/v4.34.0/en/model_doc/squeezebert#transformers.SqueezeBertModel). It is used to instantiate a SqueezeBERT model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the SqueezeBERT [squeezebert/squeezebert-uncased](https://huggingface.co/squeezebert/squeezebert-uncased) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import SqueezeBertConfig, SqueezeBertModel

>>> 
>>> configuration = SqueezeBertConfig()

>>> 
>>> model = SqueezeBertModel(configuration)

>>> 
>>> configuration = model.config
```

Attributes: pretrained\_config\_archive\_map (Dict\[str, str\]): A dictionary containing all the available pre-trained checkpoints.

## SqueezeBertTokenizer

### class transformers.SqueezeBertTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/squeezebert/tokenization_squeezebert.py#L79)

( vocab\_filedo\_lower\_case = Truedo\_basic\_tokenize = Truenever\_split = Noneunk\_token = '\[UNK\]'sep\_token = '\[SEP\]'pad\_token = '\[PAD\]'cls\_token = '\[CLS\]'mask\_token = '\[MASK\]'tokenize\_chinese\_chars = Truestrip\_accents = None\*\*kwargs )

Construct a SqueezeBERT tokenizer. Based on WordPiece.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/squeezebert/tokenization_squeezebert.py#L212)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A SqueezeBERT sequence has the following format:

-   single sequence: `[CLS] X [SEP]`
-   pair of sequences: `[CLS] A [SEP] B [SEP]`

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/squeezebert/tokenization_squeezebert.py#L237)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = Nonealready\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/squeezebert/tokenization_squeezebert.py#L265)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).

Create a mask from the two sequences passed to be used in a sequence-pair classification task. A SqueezeBERT

sequence pair mask has the following format:

```
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence    | second sequence |
```

If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/squeezebert/tokenization_squeezebert.py#L294)

( save\_directory: strfilename\_prefix: typing.Optional\[str\] = None )

## SqueezeBertTokenizerFast

### class transformers.SqueezeBertTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/squeezebert/tokenization_squeezebert_fast.py#L69)

( vocab\_file = Nonetokenizer\_file = Nonedo\_lower\_case = Trueunk\_token = '\[UNK\]'sep\_token = '\[SEP\]'pad\_token = '\[PAD\]'cls\_token = '\[CLS\]'mask\_token = '\[MASK\]'tokenize\_chinese\_chars = Truestrip\_accents = None\*\*kwargs )

Construct a “fast” SqueezeBERT tokenizer (backed by HuggingFace’s _tokenizers_ library). Based on WordPiece.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/squeezebert/tokenization_squeezebert_fast.py#L157)

( token\_ids\_0token\_ids\_1 = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A SqueezeBERT sequence has the following format:

-   single sequence: `[CLS] X [SEP]`
-   pair of sequences: `[CLS] A [SEP] B [SEP]`

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/squeezebert/tokenization_squeezebert_fast.py#L181)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).

Create a mask from the two sequences passed to be used in a sequence-pair classification task. A SqueezeBERT

sequence pair mask has the following format:

```
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence    | second sequence |
```

If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

## SqueezeBertModel

### class transformers.SqueezeBertModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/squeezebert/modeling_squeezebert.py#L547)

( config )

Parameters

-   **config** ([SqueezeBertConfig](/docs/transformers/v4.34.0/en/model_doc/squeezebert#transformers.SqueezeBertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare SqueezeBERT Model transformer outputting raw hidden-states without any specific head on top.

The SqueezeBERT model was proposed in [SqueezeBERT: What can computer vision teach NLP about efficient neural networks?](https://arxiv.org/abs/2006.11316) by Forrest N. Iandola, Albert E. Shaw, Ravi Krishna, and Kurt W. Keutzer

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

For best results finetuning SqueezeBERT on text classification tasks, it is recommended to use the _squeezebert/squeezebert-mnli-headless_ checkpoint as a starting point.

Hierarchy:

```
Internal class hierarchy:
SqueezeBertModel
    SqueezeBertEncoder
        SqueezeBertModule
        SqueezeBertSelfAttention
            ConvActivation
            ConvDropoutLayerNorm
```

Data layouts:

```
Input data is in [batch, sequence_length, hidden_size] format.

Data inside the encoder is in [batch, hidden_size, sequence_length] format. But, if `output_hidden_states == True`, the data from inside the encoder is returned in [batch, sequence_length, hidden_size] format.

The final output of the encoder is in [batch, sequence_length, hidden_size] format.
```

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/squeezebert/modeling_squeezebert.py#L572)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

The [SqueezeBertModel](/docs/transformers/v4.34.0/en/model_doc/squeezebert#transformers.SqueezeBertModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, SqueezeBertModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased")
>>> model = SqueezeBertModel.from_pretrained("squeezebert/squeezebert-uncased")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## SqueezeBertForMaskedLM

### class transformers.SqueezeBertForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/squeezebert/modeling_squeezebert.py#L647)

( config )

Parameters

-   **config** ([SqueezeBertConfig](/docs/transformers/v4.34.0/en/model_doc/squeezebert#transformers.SqueezeBertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

SqueezeBERT Model with a `language modeling` head on top.

The SqueezeBERT model was proposed in [SqueezeBERT: What can computer vision teach NLP about efficient neural networks?](https://arxiv.org/abs/2006.11316) by Forrest N. Iandola, Albert E. Shaw, Ravi Krishna, and Kurt W. Keutzer

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

For best results finetuning SqueezeBERT on text classification tasks, it is recommended to use the _squeezebert/squeezebert-mnli-headless_ checkpoint as a starting point.

Hierarchy:

```
Internal class hierarchy:
SqueezeBertModel
    SqueezeBertEncoder
        SqueezeBertModule
        SqueezeBertSelfAttention
            ConvActivation
            ConvDropoutLayerNorm
```

Data layouts:

```
Input data is in [batch, sequence_length, hidden_size] format.

Data inside the encoder is in [batch, hidden_size, sequence_length] format. But, if `output_hidden_states == True`, the data from inside the encoder is returned in [batch, sequence_length, hidden_size] format.

The final output of the encoder is in [batch, sequence_length, hidden_size] format.
```

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/squeezebert/modeling_squeezebert.py#L665)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

The [SqueezeBertForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/squeezebert#transformers.SqueezeBertForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, SqueezeBertForMaskedLM
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased")
>>> model = SqueezeBertForMaskedLM.from_pretrained("squeezebert/squeezebert-uncased")

>>> inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

>>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)

>>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
>>> 
>>> labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

>>> outputs = model(**inputs, labels=labels)
```

## SqueezeBertForSequenceClassification

### class transformers.SqueezeBertForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/squeezebert/modeling_squeezebert.py#L731)

( config )

Parameters

-   **config** ([SqueezeBertConfig](/docs/transformers/v4.34.0/en/model_doc/squeezebert#transformers.SqueezeBertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

SqueezeBERT Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

The SqueezeBERT model was proposed in [SqueezeBERT: What can computer vision teach NLP about efficient neural networks?](https://arxiv.org/abs/2006.11316) by Forrest N. Iandola, Albert E. Shaw, Ravi Krishna, and Kurt W. Keutzer

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

For best results finetuning SqueezeBERT on text classification tasks, it is recommended to use the _squeezebert/squeezebert-mnli-headless_ checkpoint as a starting point.

Hierarchy:

```
Internal class hierarchy:
SqueezeBertModel
    SqueezeBertEncoder
        SqueezeBertModule
        SqueezeBertSelfAttention
            ConvActivation
            ConvDropoutLayerNorm
```

Data layouts:

```
Input data is in [batch, sequence_length, hidden_size] format.

Data inside the encoder is in [batch, hidden_size, sequence_length] format. But, if `output_hidden_states == True`, the data from inside the encoder is returned in [batch, sequence_length, hidden_size] format.

The final output of the encoder is in [batch, sequence_length, hidden_size] format.
```

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/squeezebert/modeling_squeezebert.py#L744)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [SqueezeBertForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/squeezebert#transformers.SqueezeBertForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example of single-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, SqueezeBertForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased")
>>> model = SqueezeBertForSequenceClassification.from_pretrained("squeezebert/squeezebert-uncased")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = SqueezeBertForSequenceClassification.from_pretrained("squeezebert/squeezebert-uncased", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
```

Example of multi-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, SqueezeBertForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased")
>>> model = SqueezeBertForSequenceClassification.from_pretrained("squeezebert/squeezebert-uncased", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = SqueezeBertForSequenceClassification.from_pretrained(
...     "squeezebert/squeezebert-uncased", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## SqueezeBertForMultipleChoice

### class transformers.SqueezeBertForMultipleChoice

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/squeezebert/modeling_squeezebert.py#L830)

( config )

Parameters

-   **config** ([SqueezeBertConfig](/docs/transformers/v4.34.0/en/model_doc/squeezebert#transformers.SqueezeBertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

SqueezeBERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a softmax) e.g. for RocStories/SWAG tasks.

The SqueezeBERT model was proposed in [SqueezeBERT: What can computer vision teach NLP about efficient neural networks?](https://arxiv.org/abs/2006.11316) by Forrest N. Iandola, Albert E. Shaw, Ravi Krishna, and Kurt W. Keutzer

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

For best results finetuning SqueezeBERT on text classification tasks, it is recommended to use the _squeezebert/squeezebert-mnli-headless_ checkpoint as a starting point.

Hierarchy:

```
Internal class hierarchy:
SqueezeBertModel
    SqueezeBertEncoder
        SqueezeBertModule
        SqueezeBertSelfAttention
            ConvActivation
            ConvDropoutLayerNorm
```

Data layouts:

```
Input data is in [batch, sequence_length, hidden_size] format.

Data inside the encoder is in [batch, hidden_size, sequence_length] format. But, if `output_hidden_states == True`, the data from inside the encoder is returned in [batch, sequence_length, hidden_size] format.

The final output of the encoder is in [batch, sequence_length, hidden_size] format.
```

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/squeezebert/modeling_squeezebert.py#L841)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.MultipleChoiceModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or `tuple(torch.FloatTensor)`

The [SqueezeBertForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/squeezebert#transformers.SqueezeBertForMultipleChoice) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, SqueezeBertForMultipleChoice
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased")
>>> model = SqueezeBertForMultipleChoice.from_pretrained("squeezebert/squeezebert-uncased")

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

## SqueezeBertForTokenClassification

### class transformers.SqueezeBertForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/squeezebert/modeling_squeezebert.py#L923)

( config )

Parameters

-   **config** ([SqueezeBertConfig](/docs/transformers/v4.34.0/en/model_doc/squeezebert#transformers.SqueezeBertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

SqueezeBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

The SqueezeBERT model was proposed in [SqueezeBERT: What can computer vision teach NLP about efficient neural networks?](https://arxiv.org/abs/2006.11316) by Forrest N. Iandola, Albert E. Shaw, Ravi Krishna, and Kurt W. Keutzer

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

For best results finetuning SqueezeBERT on text classification tasks, it is recommended to use the _squeezebert/squeezebert-mnli-headless_ checkpoint as a starting point.

Hierarchy:

```
Internal class hierarchy:
SqueezeBertModel
    SqueezeBertEncoder
        SqueezeBertModule
        SqueezeBertSelfAttention
            ConvActivation
            ConvDropoutLayerNorm
```

Data layouts:

```
Input data is in [batch, sequence_length, hidden_size] format.

Data inside the encoder is in [batch, hidden_size, sequence_length] format. But, if `output_hidden_states == True`, the data from inside the encoder is returned in [batch, sequence_length, hidden_size] format.

The final output of the encoder is in [batch, sequence_length, hidden_size] format.
```

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/squeezebert/modeling_squeezebert.py#L935)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

The [SqueezeBertForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/squeezebert#transformers.SqueezeBertForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, SqueezeBertForTokenClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased")
>>> model = SqueezeBertForTokenClassification.from_pretrained("squeezebert/squeezebert-uncased")

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

## SqueezeBertForQuestionAnswering

### class transformers.SqueezeBertForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/squeezebert/modeling_squeezebert.py#L1001)

( config )

Parameters

-   **config** ([SqueezeBertConfig](/docs/transformers/v4.34.0/en/model_doc/squeezebert#transformers.SqueezeBertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

SqueezeBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

The SqueezeBERT model was proposed in [SqueezeBERT: What can computer vision teach NLP about efficient neural networks?](https://arxiv.org/abs/2006.11316) by Forrest N. Iandola, Albert E. Shaw, Ravi Krishna, and Kurt W. Keutzer

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

For best results finetuning SqueezeBERT on text classification tasks, it is recommended to use the _squeezebert/squeezebert-mnli-headless_ checkpoint as a starting point.

Hierarchy:

```
Internal class hierarchy:
SqueezeBertModel
    SqueezeBertEncoder
        SqueezeBertModule
        SqueezeBertSelfAttention
            ConvActivation
            ConvDropoutLayerNorm
```

Data layouts:

```
Input data is in [batch, sequence_length, hidden_size] format.

Data inside the encoder is in [batch, hidden_size, sequence_length] format. But, if `output_hidden_states == True`, the data from inside the encoder is returned in [batch, sequence_length, hidden_size] format.

The final output of the encoder is in [batch, sequence_length, hidden_size] format.
```

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/squeezebert/modeling_squeezebert.py#L1012)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonestart\_positions: typing.Optional\[torch.Tensor\] = Noneend\_positions: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

The [SqueezeBertForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/squeezebert#transformers.SqueezeBertForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, SqueezeBertForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased")
>>> model = SqueezeBertForQuestionAnswering.from_pretrained("squeezebert/squeezebert-uncased")

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