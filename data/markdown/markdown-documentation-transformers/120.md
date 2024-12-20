# CANINE

## Overview

The CANINE model was proposed in [CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation](https://arxiv.org/abs/2103.06874) by Jonathan H. Clark, Dan Garrette, Iulia Turc, John Wieting. It’s among the first papers that trains a Transformer without using an explicit tokenization step (such as Byte Pair Encoding (BPE), WordPiece or SentencePiece). Instead, the model is trained directly at a Unicode character-level. Training at a character-level inevitably comes with a longer sequence length, which CANINE solves with an efficient downsampling strategy, before applying a deep Transformer encoder.

The abstract from the paper is the following:

_Pipelined NLP systems have largely been superseded by end-to-end neural modeling, yet nearly all commonly-used models still require an explicit tokenization step. While recent tokenization approaches based on data-derived subword lexicons are less brittle than manually engineered tokenizers, these techniques are not equally suited to all languages, and the use of any fixed vocabulary may limit a model’s ability to adapt. In this paper, we present CANINE, a neural encoder that operates directly on character sequences, without explicit tokenization or vocabulary, and a pre-training strategy that operates either directly on characters or optionally uses subwords as a soft inductive bias. To use its finer-grained input effectively and efficiently, CANINE combines downsampling, which reduces the input sequence length, with a deep transformer stack, which encodes context. CANINE outperforms a comparable mBERT model by 2.8 F1 on TyDi QA, a challenging multilingual benchmark, despite having 28% fewer model parameters._

Tips:

-   CANINE uses no less than 3 Transformer encoders internally: 2 “shallow” encoders (which only consist of a single layer) and 1 “deep” encoder (which is a regular BERT encoder). First, a “shallow” encoder is used to contextualize the character embeddings, using local attention. Next, after downsampling, a “deep” encoder is applied. Finally, after upsampling, a “shallow” encoder is used to create the final character embeddings. Details regarding up- and downsampling can be found in the paper.
    
-   CANINE uses a max sequence length of 2048 characters by default. One can use [CanineTokenizer](/docs/transformers/v4.34.0/en/model_doc/canine#transformers.CanineTokenizer) to prepare text for the model.
    
-   Classification can be done by placing a linear layer on top of the final hidden state of the special \[CLS\] token (which has a predefined Unicode code point). For token classification tasks however, the downsampled sequence of tokens needs to be upsampled again to match the length of the original character sequence (which is 2048). The details for this can be found in the paper.
    
-   Models:
    
-   [google/canine-c](https://huggingface.co/google/canine-c): Pre-trained with autoregressive character loss, 12-layer, 768-hidden, 12-heads, 121M parameters (size ~500 MB).
    
-   [google/canine-s](https://huggingface.co/google/canine-s): Pre-trained with subword loss, 12-layer, 768-hidden, 12-heads, 121M parameters (size ~500 MB).
    

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/google-research/language/tree/master/language/canine).

### Example

CANINE works on raw characters, so it can be used without a tokenizer:

```
>>> from transformers import CanineModel
>>> import torch

>>> model = CanineModel.from_pretrained("google/canine-c")  

>>> text = "hello world"
>>> 
>>> input_ids = torch.tensor([[ord(char) for char in text]])

>>> outputs = model(input_ids)  
>>> pooled_output = outputs.pooler_output
>>> sequence_output = outputs.last_hidden_state
```

For batched inference and training, it is however recommended to make use of the tokenizer (to pad/truncate all sequences to the same length):

```
>>> from transformers import CanineTokenizer, CanineModel

>>> model = CanineModel.from_pretrained("google/canine-c")
>>> tokenizer = CanineTokenizer.from_pretrained("google/canine-c")

>>> inputs = ["Life is like a box of chocolates.", "You never know what you gonna get."]
>>> encoding = tokenizer(inputs, padding="longest", truncation=True, return_tensors="pt")

>>> outputs = model(**encoding)  
>>> pooled_output = outputs.pooler_output
>>> sequence_output = outputs.last_hidden_state
```

## Documentation resources

-   [Text classification task guide](../tasks/sequence_classification)
-   [Token classification task guide](../tasks/token_classification)
-   [Question answering task guide](../tasks/question_answering)
-   [Multiple choice task guide](../tasks/multiple_choice)

## CANINE specific outputs

### class transformers.models.canine.modeling\_canine.CanineModelOutputWithPooling

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/canine/modeling_canine.py#L66)

( last\_hidden\_state: FloatTensor = None pooler\_output: FloatTensor = None hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Parameters

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model (i.e. the output of the final shallow Transformer encoder).
-   **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Hidden-state of the first token of the sequence (classification token) at the last layer of the deep Transformer encoder, further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the input to each encoder + one for the output of each layer of each encoder) of shape `(batch_size, sequence_length, hidden_size)` and `(batch_size, sequence_length // config.downsampling_rate, hidden_size)`. Hidden-states of the model at the output of each layer plus the initial input to each Transformer encoder. The hidden states of the shallow encoders have length `sequence_length`, but the hidden states of the deep encoder have length `sequence_length` // `config.downsampling_rate`.
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of the 3 Transformer encoders of shape `(batch_size, num_heads, sequence_length, sequence_length)` and `(batch_size, num_heads, sequence_length // config.downsampling_rate, sequence_length // config.downsampling_rate)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Output type of [CanineModel](/docs/transformers/v4.34.0/en/model_doc/canine#transformers.CanineModel). Based on [BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling), but with slightly different `hidden_states` and `attentions`, as these also include the hidden states and attentions of the shallow Transformer encoders.

## CanineConfig

### class transformers.CanineConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/canine/configuration_canine.py#L29)

( hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 16384 type\_vocab\_size = 16 initializer\_range = 0.02 layer\_norm\_eps = 1e-12 pad\_token\_id = 0 bos\_token\_id = 57344 eos\_token\_id = 57345 downsampling\_rate = 4 upsampling\_kernel\_size = 4 num\_hash\_functions = 8 num\_hash\_buckets = 16384 local\_transformer\_stride = 128 \*\*kwargs )

Parameters

-   **hidden\_size** (`int`, _optional_, defaults to 768) — Dimension of the encoder layers and the pooler layer.
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 12) — Number of hidden layers in the deep Transformer encoder.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 12) — Number of attention heads for each attention layer in the Transformer encoders.
-   **intermediate\_size** (`int`, _optional_, defaults to 3072) — Dimension of the “intermediate” (i.e., feed-forward) layer in the Transformer encoders.
-   **hidden\_act** (`str` or `function`, _optional_, defaults to `"gelu"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
-   **hidden\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout probabilitiy for all fully connected layers in the embeddings, encoders, and pooler.
-   **attention\_probs\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout ratio for the attention probabilities.
-   **max\_position\_embeddings** (`int`, _optional_, defaults to 16384) — The maximum sequence length that this model might ever be used with.
-   **type\_vocab\_size** (`int`, _optional_, defaults to 16) — The vocabulary size of the `token_type_ids` passed when calling [CanineModel](/docs/transformers/v4.34.0/en/model_doc/canine#transformers.CanineModel).
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-12) — The epsilon used by the layer normalization layers.
-   **downsampling\_rate** (`int`, _optional_, defaults to 4) — The rate at which to downsample the original character sequence length before applying the deep Transformer encoder.
-   **upsampling\_kernel\_size** (`int`, _optional_, defaults to 4) — The kernel size (i.e. the number of characters in each window) of the convolutional projection layer when projecting back from `hidden_size`\*2 to `hidden_size`.
-   **num\_hash\_functions** (`int`, _optional_, defaults to 8) — The number of hash functions to use. Each hash function has its own embedding matrix.
-   **num\_hash\_buckets** (`int`, _optional_, defaults to 16384) — The number of hash buckets to use.
-   **local\_transformer\_stride** (`int`, _optional_, defaults to 128) — The stride of the local attention of the first shallow Transformer encoder. Defaults to 128 for good TPU/XLA memory alignment.

This is the configuration class to store the configuration of a [CanineModel](/docs/transformers/v4.34.0/en/model_doc/canine#transformers.CanineModel). It is used to instantiate an CANINE model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the CANINE [google/canine-s](https://huggingface.co/google/canine-s) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import CanineConfig, CanineModel

>>> 
>>> configuration = CanineConfig()

>>> 
>>> model = CanineModel(configuration)

>>> 
>>> configuration = model.config
```

## CanineTokenizer

### class transformers.CanineTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/canine/tokenization_canine.py#L62)

( bos\_token = '\\ue000' eos\_token = '\\ue001' sep\_token = '\\ue001' cls\_token = '\\ue000' pad\_token = '\\x00' mask\_token = '\\ue003' add\_prefix\_space = False model\_max\_length = 2048 \*\*kwargs )

Parameters

-   **model\_max\_length** (`int`, _optional_, defaults to 2048) — The maximum sentence length the model accepts.

Construct a CANINE tokenizer (i.e. a character splitter). It turns text into a sequence of characters, and then converts each character into its Unicode code point.

[CanineTokenizer](/docs/transformers/v4.34.0/en/model_doc/canine#transformers.CanineTokenizer) inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer).

Refer to superclass [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) for usage examples and documentation concerning parameters.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/canine/tokenization_canine.py#L159)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A CANINE sequence has the following format:

-   single sequence: `[CLS] X [SEP]`
-   pair of sequences: `[CLS] A [SEP] B [SEP]`

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/canine/tokenization_canine.py#L186)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None already\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/canine/tokenization_canine.py#L214)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).

Create a mask from the two sequences passed to be used in a sequence-pair classification task. A CANINE

sequence pair mask has the following format:

```
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence    | second sequence |
```

If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

## CanineModel

### class transformers.CanineModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/canine/modeling_canine.py#L992)

( config add\_pooling\_layer = True )

Parameters

-   **config** ([CanineConfig](/docs/transformers/v4.34.0/en/model_doc/canine#transformers.CanineConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare CANINE Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/canine/modeling_canine.py#L1100)

( input\_ids: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.models.canine.modeling\_canine.CanineModelOutputWithPooling](/docs/transformers/v4.34.0/en/model_doc/canine#transformers.models.canine.modeling_canine.CanineModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert _input\_ids_ indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.models.canine.modeling\_canine.CanineModelOutputWithPooling](/docs/transformers/v4.34.0/en/model_doc/canine#transformers.models.canine.modeling_canine.CanineModelOutputWithPooling) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([CanineConfig](/docs/transformers/v4.34.0/en/model_doc/canine#transformers.CanineConfig)) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model (i.e. the output of the final shallow Transformer encoder).
-   **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Hidden-state of the first token of the sequence (classification token) at the last layer of the deep Transformer encoder, further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the input to each encoder + one for the output of each layer of each encoder) of shape `(batch_size, sequence_length, hidden_size)` and `(batch_size, sequence_length // config.downsampling_rate, hidden_size)`. Hidden-states of the model at the output of each layer plus the initial input to each Transformer encoder. The hidden states of the shallow encoders have length `sequence_length`, but the hidden states of the deep encoder have length `sequence_length` // `config.downsampling_rate`.
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of the 3 Transformer encoders of shape `(batch_size, num_heads, sequence_length, sequence_length)` and `(batch_size, num_heads, sequence_length // config.downsampling_rate, sequence_length // config.downsampling_rate)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The [CanineModel](/docs/transformers/v4.34.0/en/model_doc/canine#transformers.CanineModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, CanineModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("google/canine-s")
>>> model = CanineModel.from_pretrained("google/canine-s")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## CanineForSequenceClassification

### class transformers.CanineForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/canine/modeling_canine.py#L1272)

( config )

Parameters

-   **config** ([CanineConfig](/docs/transformers/v4.34.0/en/model_doc/canine#transformers.CanineConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

CANINE Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/canine/modeling_canine.py#L1284)

( input\_ids: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert _input\_ids_ indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

A [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([CanineConfig](/docs/transformers/v4.34.0/en/model_doc/canine#transformers.CanineConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [CanineForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/canine#transformers.CanineForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example of single-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, CanineForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("google/canine-s")
>>> model = CanineForSequenceClassification.from_pretrained("google/canine-s")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = CanineForSequenceClassification.from_pretrained("google/canine-s", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
```

Example of multi-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, CanineForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("google/canine-s")
>>> model = CanineForSequenceClassification.from_pretrained("google/canine-s", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = CanineForSequenceClassification.from_pretrained(
...     "google/canine-s", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## CanineForMultipleChoice

### class transformers.CanineForMultipleChoice

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/canine/modeling_canine.py#L1369)

( config )

Parameters

-   **config** ([CanineConfig](/docs/transformers/v4.34.0/en/model_doc/canine#transformers.CanineConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

CANINE Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a softmax) e.g. for RocStories/SWAG tasks.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/canine/modeling_canine.py#L1380)

( input\_ids: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.MultipleChoiceModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert _input\_ids_ indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)

A [transformers.modeling\_outputs.MultipleChoiceModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([CanineConfig](/docs/transformers/v4.34.0/en/model_doc/canine#transformers.CanineConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape _(1,)_, _optional_, returned when `labels` is provided) — Classification loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, num_choices)`) — _num\_choices_ is the second dimension of the input tensors. (see _input\_ids_ above).
    
    Classification scores (before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [CanineForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/canine#transformers.CanineForMultipleChoice) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, CanineForMultipleChoice
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("google/canine-s")
>>> model = CanineForMultipleChoice.from_pretrained("google/canine-s")

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

## CanineForTokenClassification

### class transformers.CanineForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/canine/modeling_canine.py#L1460)

( config )

Parameters

-   **config** ([CanineConfig](/docs/transformers/v4.34.0/en/model_doc/canine#transformers.CanineConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

CANINE Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/canine/modeling_canine.py#L1472)

( input\_ids: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert _input\_ids_ indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

A [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([CanineConfig](/docs/transformers/v4.34.0/en/model_doc/canine#transformers.CanineConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [CanineForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/canine#transformers.CanineForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, CanineForTokenClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("google/canine-s")
>>> model = CanineForTokenClassification.from_pretrained("google/canine-s")

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
>>> predicted_tokens_classes
```

```
>>> labels = predicted_token_class_ids
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
```

## CanineForQuestionAnswering

### class transformers.CanineForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/canine/modeling_canine.py#L1566)

( config )

Parameters

-   **config** ([CanineConfig](/docs/transformers/v4.34.0/en/model_doc/canine#transformers.CanineConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

CANINE Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/canine/modeling_canine.py#L1577)

( input\_ids: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None start\_positions: typing.Optional\[torch.LongTensor\] = None end\_positions: typing.Optional\[torch.LongTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert _input\_ids_ indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **start\_positions** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for position (index) of the start of the labelled span for computing the token classification loss. Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence are not taken into account for computing the loss.
-   **end\_positions** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for position (index) of the end of the labelled span for computing the token classification loss. Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence are not taken into account for computing the loss.

A [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([CanineConfig](/docs/transformers/v4.34.0/en/model_doc/canine#transformers.CanineConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
    
-   **start\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-start scores (before SoftMax).
    
-   **end\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-end scores (before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [CanineForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/canine#transformers.CanineForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, CanineForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("Splend1dchan/canine-c-squad")
>>> model = CanineForQuestionAnswering.from_pretrained("Splend1dchan/canine-c-squad")

>>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

>>> inputs = tokenizer(question, text, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> answer_start_index = outputs.start_logits.argmax()
>>> answer_end_index = outputs.end_logits.argmax()

>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
'nice puppet'

>>> 
>>> target_start_index = torch.tensor([14])
>>> target_end_index = torch.tensor([15])

>>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
>>> loss = outputs.loss
>>> round(loss.item(), 2)
8.81
```