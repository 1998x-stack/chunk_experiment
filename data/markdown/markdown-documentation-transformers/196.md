# Longformer

[![Models](https://img.shields.io/badge/All_model_pages-longformer-blueviolet)](https://huggingface.co/models?filter=longformer) [![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/docs-demos/longformer-base-4096-finetuned-squadv1)

## Overview

The Longformer model was presented in [Longformer: The Long-Document Transformer](https://arxiv.org/pdf/2004.05150.pdf) by Iz Beltagy, Matthew E. Peters, Arman Cohan.

The abstract from the paper is the following:

_Transformer-based models are unable to process long sequences due to their self-attention operation, which scales quadratically with the sequence length. To address this limitation, we introduce the Longformer with an attention mechanism that scales linearly with sequence length, making it easy to process documents of thousands of tokens or longer. Longformer’s attention mechanism is a drop-in replacement for the standard self-attention and combines a local windowed attention with a task motivated global attention. Following prior work on long-sequence transformers, we evaluate Longformer on character-level language modeling and achieve state-of-the-art results on text8 and enwik8. In contrast to most prior work, we also pretrain Longformer and finetune it on a variety of downstream tasks. Our pretrained Longformer consistently outperforms RoBERTa on long document tasks and sets new state-of-the-art results on WikiHop and TriviaQA._

Tips:

-   Since the Longformer is based on RoBERTa, it doesn’t have `token_type_ids`. You don’t need to indicate which token belongs to which segment. Just separate your segments with the separation token `tokenizer.sep_token` (or `</s>`).
-   A transformer model replacing the attention matrices by sparse matrices to go faster. Often, the local context (e.g., what are the two tokens left and right?) is enough to take action for a given token. Some preselected input tokens are still given global attention, but the attention matrix has way less parameters, resulting in a speed-up. See the local attention section for more information.

This model was contributed by [beltagy](https://huggingface.co/beltagy). The Authors’ code can be found [here](https://github.com/allenai/longformer).

## Longformer Self Attention

Longformer self attention employs self attention on both a “local” context and a “global” context. Most tokens only attend “locally” to each other meaning that each token attends to its 12w\\frac{1}{2} w previous tokens and 12w\\frac{1}{2} w succeeding tokens with ww being the window length as defined in `config.attention_window`. Note that `config.attention_window` can be of type `List` to define a different ww for each layer. A selected few tokens attend “globally” to all other tokens, as it is conventionally done for all tokens in `BertSelfAttention`.

Note that “locally” and “globally” attending tokens are projected by different query, key and value matrices. Also note that every “locally” attending token not only attends to tokens within its window ww, but also to all “globally” attending tokens so that global attention is _symmetric_.

The user can define which tokens attend “locally” and which tokens attend “globally” by setting the tensor `global_attention_mask` at run-time appropriately. All Longformer models employ the following logic for `global_attention_mask`:

-   0: the token attends “locally”,
-   1: the token attends “globally”.

For more information please also refer to [forward()](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerModel.forward) method.

Using Longformer self attention, the memory and time complexity of the query-key matmul operation, which usually represents the memory and time bottleneck, can be reduced from O(ns×ns)\\mathcal{O}(n\_s \\times n\_s) to O(ns×w)\\mathcal{O}(n\_s \\times w), with nsn\_s being the sequence length and ww being the average window size. It is assumed that the number of “globally” attending tokens is insignificant as compared to the number of “locally” attending tokens.

For more information, please refer to the official [paper](https://arxiv.org/pdf/2004.05150.pdf).

## Training

[LongformerForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerForMaskedLM) is trained the exact same way [RobertaForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/roberta#transformers.RobertaForMaskedLM) is trained and should be used as follows:

```
input_ids = tokenizer.encode("This is a sentence from [MASK] training data", return_tensors="pt")
mlm_labels = tokenizer.encode("This is a sentence from the training data", return_tensors="pt")

loss = model(input_ids, labels=input_ids, masked_lm_labels=mlm_labels)[0]
```

## Documentation resources

-   [Text classification task guide](../tasks/sequence_classification)
-   [Token classification task guide](../tasks/token_classification)
-   [Question answering task guide](../tasks/question_answering)
-   [Masked language modeling task guide](../tasks/masked_language_modeling)
-   [Multiple choice task guide](../tasks/multiple_choice)

## LongformerConfig

### class transformers.LongformerConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/configuration_longformer.py#L46)

( attention\_window: typing.Union\[typing.List\[int\], int\] = 512 sep\_token\_id: int = 2 pad\_token\_id: int = 1 bos\_token\_id: int = 0 eos\_token\_id: int = 2 vocab\_size: int = 30522 hidden\_size: int = 768 num\_hidden\_layers: int = 12 num\_attention\_heads: int = 12 intermediate\_size: int = 3072 hidden\_act: str = 'gelu' hidden\_dropout\_prob: float = 0.1 attention\_probs\_dropout\_prob: float = 0.1 max\_position\_embeddings: int = 512 type\_vocab\_size: int = 2 initializer\_range: float = 0.02 layer\_norm\_eps: float = 1e-12 onnx\_export: bool = False \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 30522) — Vocabulary size of the Longformer model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [LongformerModel](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerModel) or [TFLongformerModel](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.TFLongformerModel).
-   **hidden\_size** (`int`, _optional_, defaults to 768) — Dimensionality of the encoder layers and the pooler layer.
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 12) — Number of hidden layers in the Transformer encoder.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 12) — Number of attention heads for each attention layer in the Transformer encoder.
-   **intermediate\_size** (`int`, _optional_, defaults to 3072) — Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
-   **hidden\_act** (`str` or `Callable`, _optional_, defaults to `"gelu"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
-   **hidden\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
-   **attention\_probs\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout ratio for the attention probabilities.
-   **max\_position\_embeddings** (`int`, _optional_, defaults to 512) — The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
-   **type\_vocab\_size** (`int`, _optional_, defaults to 2) — The vocabulary size of the `token_type_ids` passed when calling [LongformerModel](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerModel) or [TFLongformerModel](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.TFLongformerModel).
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-12) — The epsilon used by the layer normalization layers.
-   **attention\_window** (`int` or `List[int]`, _optional_, defaults to 512) — Size of an attention window around each token. If an `int`, use the same size for all layers. To specify a different window size for each layer, use a `List[int]` where `len(attention_window) == num_hidden_layers`.

This is the configuration class to store the configuration of a [LongformerModel](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerModel) or a [TFLongformerModel](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.TFLongformerModel). It is used to instantiate a Longformer model according to the specified arguments, defining the model architecture.

This is the configuration class to store the configuration of a [LongformerModel](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerModel). It is used to instantiate an Longformer model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the LongFormer [allenai/longformer-base-4096](https://huggingface.co/allenai/longformer-base-4096) architecture with a sequence length 4,096.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import LongformerConfig, LongformerModel

>>> 
>>> configuration = LongformerConfig()

>>> 
>>> model = LongformerModel(configuration)

>>> 
>>> configuration = model.config
```

## LongformerTokenizer

### class transformers.LongformerTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/tokenization_longformer.py#L116)

( vocab\_file merges\_file errors = 'replace' bos\_token = '<s>' eos\_token = '</s>' sep\_token = '</s>' cls\_token = '<s>' unk\_token = '<unk>' pad\_token = '<pad>' mask\_token = '<mask>' add\_prefix\_space = False \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — Path to the vocabulary file.
-   **merges\_file** (`str`) — Path to the merges file.
-   **errors** (`str`, _optional_, defaults to `"replace"`) — Paradigm to follow when decoding bytes to UTF-8. See [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
-   **bos\_token** (`str`, _optional_, defaults to `"<s>"`) — The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
    
    When building a sequence using special tokens, this is not the token that is used for the beginning of sequence. The token used is the `cls_token`.
    
-   **eos\_token** (`str`, _optional_, defaults to `"</s>"`) — The end of sequence token.
    
    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.
    
-   **sep\_token** (`str`, _optional_, defaults to `"</s>"`) — The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.
-   **cls\_token** (`str`, _optional_, defaults to `"<s>"`) — The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.
-   **unk\_token** (`str`, _optional_, defaults to `"<unk>"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **pad\_token** (`str`, _optional_, defaults to `"<pad>"`) — The token used for padding, for example when batching sequences of different lengths.
-   **mask\_token** (`str`, _optional_, defaults to `"<mask>"`) — The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.
-   **add\_prefix\_space** (`bool`, _optional_, defaults to `False`) — Whether or not to add an initial space to the input. This allows to treat the leading word just as any other word. (Longformer tokenizer detect beginning of words by the preceding space).

Constructs a Longformer tokenizer, derived from the GPT-2 tokenizer, using byte-level Byte-Pair-Encoding.

This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will

be encoded differently whether it is at the beginning of the sentence (without space) or not:

```
>>> from transformers import LongformerTokenizer

>>> tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
>>> tokenizer("Hello world")["input_ids"]
[0, 31414, 232, 2]

>>> tokenizer(" Hello world")["input_ids"]
[0, 20920, 232, 2]
```

You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/tokenization_longformer.py#L363)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A Longformer sequence has the following format:

-   single sequence: `<s> X </s>`
-   pair of sequences: `<s> A </s></s> B </s>`

Converts a sequence of tokens (string) in a single string.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/tokenization_longformer.py#L415)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. Longformer does not make use of token type ids, therefore a list of zeros is returned.

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/tokenization_longformer.py#L388)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None already\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

## LongformerTokenizerFast

### class transformers.LongformerTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/tokenization_longformer_fast.py#L91)

( vocab\_file = None merges\_file = None tokenizer\_file = None errors = 'replace' bos\_token = '<s>' eos\_token = '</s>' sep\_token = '</s>' cls\_token = '<s>' unk\_token = '<unk>' pad\_token = '<pad>' mask\_token = '<mask>' add\_prefix\_space = False trim\_offsets = True \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — Path to the vocabulary file.
-   **merges\_file** (`str`) — Path to the merges file.
-   **errors** (`str`, _optional_, defaults to `"replace"`) — Paradigm to follow when decoding bytes to UTF-8. See [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
-   **bos\_token** (`str`, _optional_, defaults to `"<s>"`) — The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
    
    When building a sequence using special tokens, this is not the token that is used for the beginning of sequence. The token used is the `cls_token`.
    
-   **eos\_token** (`str`, _optional_, defaults to `"</s>"`) — The end of sequence token.
    
    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.
    
-   **sep\_token** (`str`, _optional_, defaults to `"</s>"`) — The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.
-   **cls\_token** (`str`, _optional_, defaults to `"<s>"`) — The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.
-   **unk\_token** (`str`, _optional_, defaults to `"<unk>"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **pad\_token** (`str`, _optional_, defaults to `"<pad>"`) — The token used for padding, for example when batching sequences of different lengths.
-   **mask\_token** (`str`, _optional_, defaults to `"<mask>"`) — The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.
-   **add\_prefix\_space** (`bool`, _optional_, defaults to `False`) — Whether or not to add an initial space to the input. This allows to treat the leading word just as any other word. (Longformer tokenizer detect beginning of words by the preceding space).
-   **trim\_offsets** (`bool`, _optional_, defaults to `True`) — Whether the post processing step should trim offsets to avoid including whitespaces.

Construct a “fast” Longformer tokenizer (backed by HuggingFace’s _tokenizers_ library), derived from the GPT-2 tokenizer, using byte-level Byte-Pair-Encoding.

This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will

be encoded differently whether it is at the beginning of the sentence (without space) or not:

```
>>> from transformers import LongformerTokenizerFast

>>> tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")
>>> tokenizer("Hello world")["input_ids"]
[0, 31414, 232, 2]

>>> tokenizer(" Hello world")["input_ids"]
[0, 20920, 232, 2]
```

You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/tokenization_longformer_fast.py#L308)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. Longformer does not make use of token type ids, therefore a list of zeros is returned.

## Longformer specific outputs

### class transformers.models.longformer.modeling\_longformer.LongformerBaseModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_longformer.py#L56)

( last\_hidden\_state: FloatTensor hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None global\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Parameters

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
    
-   **global\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

Base class for Longformer’s outputs, with potential hidden states, local and global attentions.

### class transformers.models.longformer.modeling\_longformer.LongformerBaseModelOutputWithPooling

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_longformer.py#L99)

( last\_hidden\_state: FloatTensor pooler\_output: FloatTensor = None hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None global\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Parameters

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
-   **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
    
-   **global\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

Base class for Longformer’s outputs that also contains a pooling of the last hidden states.

### class transformers.models.longformer.modeling\_longformer.LongformerMaskedLMOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_longformer.py#L147)

( loss: typing.Optional\[torch.FloatTensor\] = None logits: FloatTensor = None hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None global\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Parameters

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Masked language modeling (MLM) loss.
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
    
-   **global\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

Base class for masked language models outputs.

### class transformers.models.longformer.modeling\_longformer.LongformerQuestionAnsweringModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_longformer.py#L193)

( loss: typing.Optional\[torch.FloatTensor\] = None start\_logits: FloatTensor = None end\_logits: FloatTensor = None hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None global\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Parameters

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
-   **start\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-start scores (before SoftMax).
-   **end\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-end scores (before SoftMax).
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
    
-   **global\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

Base class for outputs of question answering Longformer models.

### class transformers.models.longformer.modeling\_longformer.LongformerSequenceClassifierOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_longformer.py#L242)

( loss: typing.Optional\[torch.FloatTensor\] = None logits: FloatTensor = None hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None global\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Parameters

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
    
-   **global\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

Base class for outputs of sentence classification models.

### class transformers.models.longformer.modeling\_longformer.LongformerMultipleChoiceModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_longformer.py#L288)

( loss: typing.Optional\[torch.FloatTensor\] = None logits: FloatTensor = None hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None global\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Parameters

-   **loss** (`torch.FloatTensor` of shape _(1,)_, _optional_, returned when `labels` is provided) — Classification loss.
-   **logits** (`torch.FloatTensor` of shape `(batch_size, num_choices)`) — _num\_choices_ is the second dimension of the input tensors. (see _input\_ids_ above).
    
    Classification scores (before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
    
-   **global\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

Base class for outputs of multiple choice Longformer models.

### class transformers.models.longformer.modeling\_longformer.LongformerTokenClassifierOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_longformer.py#L336)

( loss: typing.Optional\[torch.FloatTensor\] = None logits: FloatTensor = None hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None global\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Parameters

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification loss.
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
    
-   **global\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

Base class for outputs of token classification models.

### class transformers.models.longformer.modeling\_tf\_longformer.TFLongformerBaseModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_tf_longformer.py#L69)

( last\_hidden\_state: tf.Tensor = None hidden\_states: Tuple\[tf.Tensor\] | None = None attentions: Tuple\[tf.Tensor\] | None = None global\_attentions: Tuple\[tf.Tensor\] | None = None )

Parameters

-   **last\_hidden\_state** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
    
-   **global\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

Base class for Longformer’s outputs, with potential hidden states, local and global attentions.

### class transformers.models.longformer.modeling\_tf\_longformer.TFLongformerBaseModelOutputWithPooling

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_tf_longformer.py#L112)

( last\_hidden\_state: tf.Tensor = None pooler\_output: tf.Tensor = None hidden\_states: Tuple\[tf.Tensor\] | None = None attentions: Tuple\[tf.Tensor\] | None = None global\_attentions: Tuple\[tf.Tensor\] | None = None )

Parameters

-   **last\_hidden\_state** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
-   **pooler\_output** (`tf.Tensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
    
-   **global\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

Base class for Longformer’s outputs that also contains a pooling of the last hidden states.

### class transformers.models.longformer.modeling\_tf\_longformer.TFLongformerMaskedLMOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_tf_longformer.py#L160)

( loss: tf.Tensor | None = None logits: tf.Tensor = None hidden\_states: Tuple\[tf.Tensor\] | None = None attentions: Tuple\[tf.Tensor\] | None = None global\_attentions: Tuple\[tf.Tensor\] | None = None )

Parameters

-   **loss** (`tf.Tensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Masked language modeling (MLM) loss.
-   **logits** (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
    
-   **global\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

Base class for masked language models outputs.

### class transformers.models.longformer.modeling\_tf\_longformer.TFLongformerQuestionAnsweringModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_tf_longformer.py#L206)

( loss: tf.Tensor | None = None start\_logits: tf.Tensor = None end\_logits: tf.Tensor = None hidden\_states: Tuple\[tf.Tensor\] | None = None attentions: Tuple\[tf.Tensor\] | None = None global\_attentions: Tuple\[tf.Tensor\] | None = None )

Parameters

-   **loss** (`tf.Tensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
-   **start\_logits** (`tf.Tensor` of shape `(batch_size, sequence_length)`) — Span-start scores (before SoftMax).
-   **end\_logits** (`tf.Tensor` of shape `(batch_size, sequence_length)`) — Span-end scores (before SoftMax).
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
    
-   **global\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

Base class for outputs of question answering Longformer models.

### class transformers.models.longformer.modeling\_tf\_longformer.TFLongformerSequenceClassifierOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_tf_longformer.py#L255)

( loss: tf.Tensor | None = None logits: tf.Tensor = None hidden\_states: Tuple\[tf.Tensor\] | None = None attentions: Tuple\[tf.Tensor\] | None = None global\_attentions: Tuple\[tf.Tensor\] | None = None )

Parameters

-   **loss** (`tf.Tensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
-   **logits** (`tf.Tensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
    
-   **global\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

Base class for outputs of sentence classification models.

### class transformers.models.longformer.modeling\_tf\_longformer.TFLongformerMultipleChoiceModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_tf_longformer.py#L301)

( loss: tf.Tensor | None = None logits: tf.Tensor = None hidden\_states: Tuple\[tf.Tensor\] | None = None attentions: Tuple\[tf.Tensor\] | None = None global\_attentions: Tuple\[tf.Tensor\] | None = None )

Parameters

-   **loss** (`tf.Tensor` of shape _(1,)_, _optional_, returned when `labels` is provided) — Classification loss.
-   **logits** (`tf.Tensor` of shape `(batch_size, num_choices)`) — _num\_choices_ is the second dimension of the input tensors. (see _input\_ids_ above).
    
    Classification scores (before SoftMax).
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
    
-   **global\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

Base class for outputs of multiple choice models.

### class transformers.models.longformer.modeling\_tf\_longformer.TFLongformerTokenClassifierOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_tf_longformer.py#L349)

( loss: tf.Tensor | None = None logits: tf.Tensor = None hidden\_states: Tuple\[tf.Tensor\] | None = None attentions: Tuple\[tf.Tensor\] | None = None global\_attentions: Tuple\[tf.Tensor\] | None = None )

Parameters

-   **loss** (`tf.Tensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification loss.
-   **logits** (`tf.Tensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
    
-   **global\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

Base class for outputs of token classification models.

## LongformerModel

### class transformers.LongformerModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_longformer.py#L1534)

( config add\_pooling\_layer = True )

Parameters

-   **config** ([LongformerConfig](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Longformer Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

This class copied code from [RobertaModel](/docs/transformers/v4.34.0/en/model_doc/roberta#transformers.RobertaModel) and overwrote standard self-attention with longformer self-attention to provide the ability to process long sequences following the self-attention approach described in [Longformer: the Long-Document Transformer](https://arxiv.org/abs/2004.05150) by Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer self-attention combines a local (sliding window) and global attention to extend to long documents without the O(n^2) increase in memory and compute.

The self-attention module `LongformerSelfAttention` implemented here supports the combination of local and global attention but it lacks support for autoregressive attention and dilated attention. Autoregressive and dilated attention are more relevant for autoregressive language modeling than finetuning on downstream tasks. Future release will add support for autoregressive attention, but the support for dilated attention requires a custom CUDA kernel to be memory and compute efficient.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_longformer.py#L1647)

( input\_ids: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None global\_attention\_mask: typing.Optional\[torch.Tensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None token\_type\_ids: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.Tensor\] = None inputs\_embeds: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.models.longformer.modeling\_longformer.LongformerBaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerBaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **global\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to decide the attention given on each token, local attention or global attention. Tokens with global attention attends to all other tokens, and all other tokens attend to them. This is important for task-specific finetuning because it makes the model more flexible at representing the task. For example, for classification, the ~token should be given global attention. For QA, all question tokens should also have global attention. Please refer to the [Longformer paper](https://arxiv.org/abs/2004.05150) for more details. Mask values selected in `[0, 1]`:~
    
    -   0 for local attention (a sliding window attention),
    -   1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
    
-   **head\_mask** (`torch.Tensor` of shape `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **decoder\_head\_mask** (`torch.Tensor` of shape `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.models.longformer.modeling\_longformer.LongformerBaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerBaseModelOutputWithPooling) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LongformerConfig](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerConfig)) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
-   **global\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

The [LongformerModel](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import torch
>>> from transformers import LongformerModel, AutoTokenizer

>>> model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
>>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

>>> SAMPLE_TEXT = " ".join(["Hello world! "] * 1000)  
>>> input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  

>>> attention_mask = torch.ones(
...     input_ids.shape, dtype=torch.long, device=input_ids.device
... )  
>>> global_attention_mask = torch.zeros(
...     input_ids.shape, dtype=torch.long, device=input_ids.device
... )  
>>> global_attention_mask[
...     :,
...     [
...         1,
...         4,
...         21,
...     ],
... ] = 1  
>>> 
>>> 
>>> 
>>> 
>>> outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
>>> sequence_output = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output
```

## LongformerForMaskedLM

### class transformers.LongformerForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_longformer.py#L1772)

( config )

Parameters

-   **config** ([LongformerConfig](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Longformer Model with a `language modeling` head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_longformer.py#L1790)

( input\_ids: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None global\_attention\_mask: typing.Optional\[torch.Tensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None token\_type\_ids: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.Tensor\] = None inputs\_embeds: typing.Optional\[torch.Tensor\] = None labels: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.models.longformer.modeling\_longformer.LongformerMaskedLMOutput](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerMaskedLMOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **global\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to decide the attention given on each token, local attention or global attention. Tokens with global attention attends to all other tokens, and all other tokens attend to them. This is important for task-specific finetuning because it makes the model more flexible at representing the task. For example, for classification, the ~token should be given global attention. For QA, all question tokens should also have global attention. Please refer to the [Longformer paper](https://arxiv.org/abs/2004.05150) for more details. Mask values selected in `[0, 1]`:~
    
    -   0 for local attention (a sliding window attention),
    -   1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
    
-   **head\_mask** (`torch.Tensor` of shape `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **decoder\_head\_mask** (`torch.Tensor` of shape `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
-   **kwargs** (`Dict[str, any]`, optional, defaults to _{}_) — Used to hide legacy arguments that have been deprecated.

A [transformers.models.longformer.modeling\_longformer.LongformerMaskedLMOutput](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerMaskedLMOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LongformerConfig](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Masked language modeling (MLM) loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
-   **global\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

The [LongformerForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Mask filling example:

```
>>> from transformers import AutoTokenizer, LongformerForMaskedLM

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
>>> model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096")
```

Let’s try a very long input.

```
>>> TXT = (
...     "My friends are <mask> but they eat too many carbs."
...     + " That's why I decide not to eat with them." * 300
... )
>>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
>>> logits = model(input_ids).logits

>>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
>>> probs = logits[0, masked_index].softmax(dim=0)
>>> values, predictions = probs.topk(5)

>>> tokenizer.decode(predictions).split()
['healthy', 'skinny', 'thin', 'good', 'vegetarian']
```

## LongformerForSequenceClassification

### class transformers.LongformerForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_longformer.py#L1886)

( config )

Parameters

-   **config** ([LongformerConfig](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Longformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_longformer.py#L1898)

( input\_ids: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None global\_attention\_mask: typing.Optional\[torch.Tensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None token\_type\_ids: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.Tensor\] = None inputs\_embeds: typing.Optional\[torch.Tensor\] = None labels: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.models.longformer.modeling\_longformer.LongformerSequenceClassifierOutput](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerSequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **global\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to decide the attention given on each token, local attention or global attention. Tokens with global attention attends to all other tokens, and all other tokens attend to them. This is important for task-specific finetuning because it makes the model more flexible at representing the task. For example, for classification, the ~token should be given global attention. For QA, all question tokens should also have global attention. Please refer to the [Longformer paper](https://arxiv.org/abs/2004.05150) for more details. Mask values selected in `[0, 1]`:~
    
    -   0 for local attention (a sliding window attention),
    -   1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
    
-   **head\_mask** (`torch.Tensor` of shape `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **decoder\_head\_mask** (`torch.Tensor` of shape `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

A [transformers.models.longformer.modeling\_longformer.LongformerSequenceClassifierOutput](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerSequenceClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LongformerConfig](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
-   **global\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

The [LongformerForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example of single-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, LongformerForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("jpwahle/longformer-base-plagiarism-detection")
>>> model = LongformerForSequenceClassification.from_pretrained("jpwahle/longformer-base-plagiarism-detection")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
'ORIGINAL'

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = LongformerForSequenceClassification.from_pretrained("jpwahle/longformer-base-plagiarism-detection", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
5.44
```

Example of multi-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, LongformerForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("jpwahle/longformer-base-plagiarism-detection")
>>> model = LongformerForSequenceClassification.from_pretrained("jpwahle/longformer-base-plagiarism-detection", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = LongformerForSequenceClassification.from_pretrained(
...     "jpwahle/longformer-base-plagiarism-detection", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## LongformerForMultipleChoice

### class transformers.LongformerForMultipleChoice

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_longformer.py#L2238)

( config )

Parameters

-   **config** ([LongformerConfig](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Longformer Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a softmax) e.g. for RocStories/SWAG tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_longformer.py#L2249)

( input\_ids: typing.Optional\[torch.Tensor\] = None token\_type\_ids: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None global\_attention\_mask: typing.Optional\[torch.Tensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None labels: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.Tensor\] = None inputs\_embeds: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.models.longformer.modeling\_longformer.LongformerMultipleChoiceModelOutput](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerMultipleChoiceModelOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **global\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length)`, _optional_) — Mask to decide the attention given on each token, local attention or global attention. Tokens with global attention attends to all other tokens, and all other tokens attend to them. This is important for task-specific finetuning because it makes the model more flexible at representing the task. For example, for classification, the ~token should be given global attention. For QA, all question tokens should also have global attention. Please refer to the [Longformer paper](https://arxiv.org/abs/2004.05150) for more details. Mask values selected in `[0, 1]`:~
    
    -   0 for local attention (a sliding window attention),
    -   1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
    
-   **head\_mask** (`torch.Tensor` of shape `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **decoder\_head\_mask** (`torch.Tensor` of shape `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)

A [transformers.models.longformer.modeling\_longformer.LongformerMultipleChoiceModelOutput](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerMultipleChoiceModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LongformerConfig](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape _(1,)_, _optional_, returned when `labels` is provided) — Classification loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, num_choices)`) — _num\_choices_ is the second dimension of the input tensors. (see _input\_ids_ above).
    
    Classification scores (before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
-   **global\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

The [LongformerForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerForMultipleChoice) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, LongformerForMultipleChoice
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
>>> model = LongformerForMultipleChoice.from_pretrained("allenai/longformer-base-4096")

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

## LongformerForTokenClassification

### class transformers.LongformerForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_longformer.py#L2150)

( config )

Parameters

-   **config** ([LongformerConfig](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Longformer Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_longformer.py#L2162)

( input\_ids: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None global\_attention\_mask: typing.Optional\[torch.Tensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None token\_type\_ids: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.Tensor\] = None inputs\_embeds: typing.Optional\[torch.Tensor\] = None labels: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.models.longformer.modeling\_longformer.LongformerTokenClassifierOutput](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerTokenClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **global\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to decide the attention given on each token, local attention or global attention. Tokens with global attention attends to all other tokens, and all other tokens attend to them. This is important for task-specific finetuning because it makes the model more flexible at representing the task. For example, for classification, the ~token should be given global attention. For QA, all question tokens should also have global attention. Please refer to the [Longformer paper](https://arxiv.org/abs/2004.05150) for more details. Mask values selected in `[0, 1]`:~
    
    -   0 for local attention (a sliding window attention),
    -   1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
    
-   **head\_mask** (`torch.Tensor` of shape `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **decoder\_head\_mask** (`torch.Tensor` of shape `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

A [transformers.models.longformer.modeling\_longformer.LongformerTokenClassifierOutput](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerTokenClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LongformerConfig](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
-   **global\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

The [LongformerForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, LongformerForTokenClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("brad1141/Longformer-finetuned-norm")
>>> model = LongformerForTokenClassification.from_pretrained("brad1141/Longformer-finetuned-norm")

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
['Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence']

>>> labels = predicted_token_class_ids
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
0.63
```

## LongformerForQuestionAnswering

### class transformers.LongformerForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_longformer.py#L2013)

( config )

Parameters

-   **config** ([LongformerConfig](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Longformer Model with a span classification head on top for extractive question-answering tasks like SQuAD / TriviaQA (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_longformer.py#L2024)

( input\_ids: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None global\_attention\_mask: typing.Optional\[torch.Tensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None token\_type\_ids: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.Tensor\] = None inputs\_embeds: typing.Optional\[torch.Tensor\] = None start\_positions: typing.Optional\[torch.Tensor\] = None end\_positions: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.models.longformer.modeling\_longformer.LongformerQuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerQuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **global\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to decide the attention given on each token, local attention or global attention. Tokens with global attention attends to all other tokens, and all other tokens attend to them. This is important for task-specific finetuning because it makes the model more flexible at representing the task. For example, for classification, the ~token should be given global attention. For QA, all question tokens should also have global attention. Please refer to the [Longformer paper](https://arxiv.org/abs/2004.05150) for more details. Mask values selected in `[0, 1]`:~
    
    -   0 for local attention (a sliding window attention),
    -   1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
    
-   **head\_mask** (`torch.Tensor` of shape `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **decoder\_head\_mask** (`torch.Tensor` of shape `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **start\_positions** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for position (index) of the start of the labelled span for computing the token classification loss. Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence are not taken into account for computing the loss.
-   **end\_positions** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for position (index) of the end of the labelled span for computing the token classification loss. Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence are not taken into account for computing the loss.

A [transformers.models.longformer.modeling\_longformer.LongformerQuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerQuestionAnsweringModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LongformerConfig](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
    
-   **start\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-start scores (before SoftMax).
    
-   **end\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-end scores (before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
-   **global\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

The [LongformerForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, LongformerForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
>>> model = LongformerForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")

>>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
>>> encoding = tokenizer(question, text, return_tensors="pt")
>>> input_ids = encoding["input_ids"]

>>> 
>>> 
>>> attention_mask = encoding["attention_mask"]

>>> outputs = model(input_ids, attention_mask=attention_mask)
>>> start_logits = outputs.start_logits
>>> end_logits = outputs.end_logits
>>> all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

>>> answer_tokens = all_tokens[torch.argmax(start_logits) : torch.argmax(end_logits) + 1]
>>> answer = tokenizer.decode(
...     tokenizer.convert_tokens_to_ids(answer_tokens)
... )  
```

## TFLongformerModel

### class transformers.TFLongformerModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_tf_longformer.py#L1993)

( \*args \*\*kwargs )

Parameters

-   **config** ([LongformerConfig](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Longformer Model outputting raw hidden-states without any specific head on top.

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

This class copies code from [TFRobertaModel](/docs/transformers/v4.34.0/en/model_doc/roberta#transformers.TFRobertaModel) and overwrites standard self-attention with longformer self-attention to provide the ability to process long sequences following the self-attention approach described in [Longformer: the Long-Document Transformer](https://arxiv.org/abs/2004.05150) by Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer self-attention combines a local (sliding window) and global attention to extend to long documents without the O(n^2) increase in memory and compute.

The self-attention module `TFLongformerSelfAttention` implemented here supports the combination of local and global attention but it lacks support for autoregressive attention and dilated attention. Autoregressive and dilated attention are more relevant for autoregressive language modeling than finetuning on downstream tasks. Future release will add support for autoregressive attention, but the support for dilated attention requires a custom CUDA kernel to be memory and compute efficient.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_tf_longformer.py#L2015)

( input\_ids: TFModelInputType | None = None attention\_mask: np.ndarray | tf.Tensor | None = None head\_mask: np.ndarray | tf.Tensor | None = None global\_attention\_mask: np.ndarray | tf.Tensor | None = None token\_type\_ids: np.ndarray | tf.Tensor | None = None position\_ids: np.ndarray | tf.Tensor | None = None inputs\_embeds: np.ndarray | tf.Tensor | None = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None training: Optional\[bool\] = False )

Parameters

-   **input\_ids** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **head\_mask** (`np.ndarray` or `tf.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, _optional_) — Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **global\_attention\_mask** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to decide the attention given on each token, local attention or global attention. Tokens with global attention attends to all other tokens, and all other tokens attend to them. This is important for task-specific finetuning because it makes the model more flexible at representing the task. For example, for classification, the ~token should be given global attention. For QA, all question tokens should also have global attention. Please refer to the [Longformer paper](https://arxiv.org/abs/2004.05150) for more details. Mask values selected in `[0, 1]`:~
    
    -   0 for local attention (a sliding window attention),
    -   1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
    
-   **token\_type\_ids** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **inputs\_embeds** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).

The [TFLongformerModel](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.TFLongformerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## TFLongformerForMaskedLM

### class transformers.TFLongformerForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_tf_longformer.py#L2052)

( \*args \*\*kwargs )

Parameters

-   **config** ([LongformerConfig](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Longformer Model with a `language modeling` head on top.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_tf_longformer.py#L2069)

( input\_ids: TFModelInputType | None = None attention\_mask: np.ndarray | tf.Tensor | None = None head\_mask: np.ndarray | tf.Tensor | None = None global\_attention\_mask: np.ndarray | tf.Tensor | None = None token\_type\_ids: np.ndarray | tf.Tensor | None = None position\_ids: np.ndarray | tf.Tensor | None = None inputs\_embeds: np.ndarray | tf.Tensor | None = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None labels: np.ndarray | tf.Tensor | None = None training: Optional\[bool\] = False ) → [transformers.models.longformer.modeling\_tf\_longformer.TFLongformerMaskedLMOutput](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.models.longformer.modeling_tf_longformer.TFLongformerMaskedLMOutput) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **head\_mask** (`np.ndarray` or `tf.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, _optional_) — Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **global\_attention\_mask** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to decide the attention given on each token, local attention or global attention. Tokens with global attention attends to all other tokens, and all other tokens attend to them. This is important for task-specific finetuning because it makes the model more flexible at representing the task. For example, for classification, the ~token should be given global attention. For QA, all question tokens should also have global attention. Please refer to the [Longformer paper](https://arxiv.org/abs/2004.05150) for more details. Mask values selected in `[0, 1]`:~
    
    -   0 for local attention (a sliding window attention),
    -   1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
    
-   **token\_type\_ids** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **inputs\_embeds** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).
-   **labels** (`tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

A [transformers.models.longformer.modeling\_tf\_longformer.TFLongformerMaskedLMOutput](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.models.longformer.modeling_tf_longformer.TFLongformerMaskedLMOutput) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LongformerConfig](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerConfig)) and inputs.

-   **loss** (`tf.Tensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Masked language modeling (MLM) loss.
    
-   **logits** (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
-   **global\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

The [TFLongformerForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.TFLongformerForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFLongformerForMaskedLM
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
>>> model = TFLongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096")

>>> inputs = tokenizer("The capital of France is <mask>.", return_tensors="tf")
>>> logits = model(**inputs).logits

>>> 
>>> mask_token_index = tf.where((inputs.input_ids == tokenizer.mask_token_id)[0])
>>> selected_logits = tf.gather_nd(logits[0], indices=mask_token_index)

>>> predicted_token_id = tf.math.argmax(selected_logits, axis=-1)
>>> tokenizer.decode(predicted_token_id)
' Paris'
```

```
>>> labels = tokenizer("The capital of France is Paris.", return_tensors="tf")["input_ids"]
>>> 
>>> labels = tf.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

>>> outputs = model(**inputs, labels=labels)
>>> round(float(outputs.loss), 2)
0.44
```

## TFLongformerForQuestionAnswering

### class transformers.TFLongformerForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_tf_longformer.py#L2139)

( \*args \*\*kwargs )

Parameters

-   **config** ([LongformerConfig](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Longformer Model with a span classification head on top for extractive question-answering tasks like SQuAD / TriviaQA (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_tf_longformer.py#L2154)

( input\_ids: TFModelInputType | None = None attention\_mask: np.ndarray | tf.Tensor | None = None head\_mask: np.ndarray | tf.Tensor | None = None global\_attention\_mask: np.ndarray | tf.Tensor | None = None token\_type\_ids: np.ndarray | tf.Tensor | None = None position\_ids: np.ndarray | tf.Tensor | None = None inputs\_embeds: np.ndarray | tf.Tensor | None = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None start\_positions: np.ndarray | tf.Tensor | None = None end\_positions: np.ndarray | tf.Tensor | None = None training: Optional\[bool\] = False ) → [transformers.models.longformer.modeling\_tf\_longformer.TFLongformerQuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.models.longformer.modeling_tf_longformer.TFLongformerQuestionAnsweringModelOutput) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **head\_mask** (`np.ndarray` or `tf.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, _optional_) — Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **global\_attention\_mask** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to decide the attention given on each token, local attention or global attention. Tokens with global attention attends to all other tokens, and all other tokens attend to them. This is important for task-specific finetuning because it makes the model more flexible at representing the task. For example, for classification, the ~token should be given global attention. For QA, all question tokens should also have global attention. Please refer to the [Longformer paper](https://arxiv.org/abs/2004.05150) for more details. Mask values selected in `[0, 1]`:~
    
    -   0 for local attention (a sliding window attention),
    -   1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
    
-   **token\_type\_ids** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **inputs\_embeds** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).
-   **start\_positions** (`tf.Tensor` of shape `(batch_size,)`, _optional_) — Labels for position (index) of the start of the labelled span for computing the token classification loss. Positions are clamped to the length of the sequence (_sequence\_length_). Position outside of the sequence are not taken into account for computing the loss.
-   **end\_positions** (`tf.Tensor` of shape `(batch_size,)`, _optional_) — Labels for position (index) of the end of the labelled span for computing the token classification loss. Positions are clamped to the length of the sequence (_sequence\_length_). Position outside of the sequence are not taken into account for computing the loss.

A [transformers.models.longformer.modeling\_tf\_longformer.TFLongformerQuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.models.longformer.modeling_tf_longformer.TFLongformerQuestionAnsweringModelOutput) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LongformerConfig](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerConfig)) and inputs.

-   **loss** (`tf.Tensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
    
-   **start\_logits** (`tf.Tensor` of shape `(batch_size, sequence_length)`) — Span-start scores (before SoftMax).
    
-   **end\_logits** (`tf.Tensor` of shape `(batch_size, sequence_length)`) — Span-end scores (before SoftMax).
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
-   **global\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

The [TFLongformerForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.TFLongformerForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFLongformerForQuestionAnswering
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
>>> model = TFLongformerForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")

>>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

>>> inputs = tokenizer(question, text, return_tensors="tf")
>>> outputs = model(**inputs)

>>> answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
>>> answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])

>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens)
' puppet'
```

```
>>> 
>>> target_start_index = tf.constant([14])
>>> target_end_index = tf.constant([15])

>>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
>>> loss = tf.math.reduce_mean(outputs.loss)
>>> round(float(loss), 2)
0.96
```

## TFLongformerForSequenceClassification

### class transformers.TFLongformerForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_tf_longformer.py#L2294)

( \*args \*\*kwargs )

Parameters

-   **config** ([LongformerConfig](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Longformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_tf_longformer.py#L2306)

( input\_ids: TFModelInputType | None = None attention\_mask: np.ndarray | tf.Tensor | None = None head\_mask: np.ndarray | tf.Tensor | None = None token\_type\_ids: np.ndarray | tf.Tensor | None = None position\_ids: np.ndarray | tf.Tensor | None = None global\_attention\_mask: np.ndarray | tf.Tensor | None = None inputs\_embeds: np.ndarray | tf.Tensor | None = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None labels: np.ndarray | tf.Tensor | None = None training: Optional\[bool\] = False ) → [transformers.models.longformer.modeling\_tf\_longformer.TFLongformerSequenceClassifierOutput](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.models.longformer.modeling_tf_longformer.TFLongformerSequenceClassifierOutput) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **head\_mask** (`np.ndarray` or `tf.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, _optional_) — Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **global\_attention\_mask** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to decide the attention given on each token, local attention or global attention. Tokens with global attention attends to all other tokens, and all other tokens attend to them. This is important for task-specific finetuning because it makes the model more flexible at representing the task. For example, for classification, the ~token should be given global attention. For QA, all question tokens should also have global attention. Please refer to the [Longformer paper](https://arxiv.org/abs/2004.05150) for more details. Mask values selected in `[0, 1]`:~
    
    -   0 for local attention (a sliding window attention),
    -   1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
    
-   **token\_type\_ids** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **inputs\_embeds** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).

A [transformers.models.longformer.modeling\_tf\_longformer.TFLongformerSequenceClassifierOutput](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.models.longformer.modeling_tf_longformer.TFLongformerSequenceClassifierOutput) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LongformerConfig](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerConfig)) and inputs.

-   **loss** (`tf.Tensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
    
-   **logits** (`tf.Tensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
-   **global\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

The [TFLongformerForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.TFLongformerForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFLongformerForSequenceClassification
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
>>> model = TFLongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")

>>> logits = model(**inputs).logits

>>> predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
```

```
>>> 
>>> num_labels = len(model.config.id2label)
>>> model = TFLongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=num_labels)

>>> labels = tf.constant(1)
>>> loss = model(**inputs, labels=labels).loss
```

## TFLongformerForTokenClassification

### class transformers.TFLongformerForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_tf_longformer.py#L2511)

( \*args \*\*kwargs )

Parameters

-   **config** ([LongformerConfig](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Longformer Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_tf_longformer.py#L2526)

( input\_ids: TFModelInputType | None = None attention\_mask: np.ndarray | tf.Tensor | None = None head\_mask: np.ndarray | tf.Tensor | None = None token\_type\_ids: np.ndarray | tf.Tensor | None = None position\_ids: np.ndarray | tf.Tensor | None = None global\_attention\_mask: np.ndarray | tf.Tensor | None = None inputs\_embeds: np.ndarray | tf.Tensor | None = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None labels: Optional\[Union\[np.array, tf.Tensor\]\] = None training: Optional\[bool\] = False ) → [transformers.models.longformer.modeling\_tf\_longformer.TFLongformerTokenClassifierOutput](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.models.longformer.modeling_tf_longformer.TFLongformerTokenClassifierOutput) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **head\_mask** (`np.ndarray` or `tf.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, _optional_) — Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **global\_attention\_mask** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to decide the attention given on each token, local attention or global attention. Tokens with global attention attends to all other tokens, and all other tokens attend to them. This is important for task-specific finetuning because it makes the model more flexible at representing the task. For example, for classification, the ~token should be given global attention. For QA, all question tokens should also have global attention. Please refer to the [Longformer paper](https://arxiv.org/abs/2004.05150) for more details. Mask values selected in `[0, 1]`:~
    
    -   0 for local attention (a sliding window attention),
    -   1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
    
-   **token\_type\_ids** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **inputs\_embeds** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).
-   **labels** (`tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

A [transformers.models.longformer.modeling\_tf\_longformer.TFLongformerTokenClassifierOutput](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.models.longformer.modeling_tf_longformer.TFLongformerTokenClassifierOutput) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LongformerConfig](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerConfig)) and inputs.

-   **loss** (`tf.Tensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification loss.
    
-   **logits** (`tf.Tensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
-   **global\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

The [TFLongformerForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.TFLongformerForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFLongformerForTokenClassification
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
>>> model = TFLongformerForTokenClassification.from_pretrained("allenai/longformer-base-4096")

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

## TFLongformerForMultipleChoice

### class transformers.TFLongformerForMultipleChoice

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_tf_longformer.py#L2397)

( \*args \*\*kwargs )

Parameters

-   **config** ([LongformerConfig](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Longformer Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a softmax) e.g. for RocStories/SWAG tasks.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/longformer/modeling_tf_longformer.py#L2418)

( input\_ids: TFModelInputType | None = None attention\_mask: np.ndarray | tf.Tensor | None = None head\_mask: np.ndarray | tf.Tensor | None = None token\_type\_ids: np.ndarray | tf.Tensor | None = None position\_ids: np.ndarray | tf.Tensor | None = None global\_attention\_mask: np.ndarray | tf.Tensor | None = None inputs\_embeds: np.ndarray | tf.Tensor | None = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None labels: np.ndarray | tf.Tensor | None = None training: Optional\[bool\] = False ) → [transformers.models.longformer.modeling\_tf\_longformer.TFLongformerMultipleChoiceModelOutput](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.models.longformer.modeling_tf_longformer.TFLongformerMultipleChoiceModelOutput) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, num_choices, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, num_choices, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **head\_mask** (`np.ndarray` or `tf.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, _optional_) — Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **global\_attention\_mask** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, num_choices, sequence_length)`, _optional_) — Mask to decide the attention given on each token, local attention or global attention. Tokens with global attention attends to all other tokens, and all other tokens attend to them. This is important for task-specific finetuning because it makes the model more flexible at representing the task. For example, for classification, the ~token should be given global attention. For QA, all question tokens should also have global attention. Please refer to the [Longformer paper](https://arxiv.org/abs/2004.05150) for more details. Mask values selected in `[0, 1]`:~
    
    -   0 for local attention (a sliding window attention),
    -   1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
    
-   **token\_type\_ids** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, num_choices, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, num_choices, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **inputs\_embeds** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, num_choices, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).
-   **labels** (`tf.Tensor` of shape `(batch_size,)`, _optional_) — Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]` where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)

A [transformers.models.longformer.modeling\_tf\_longformer.TFLongformerMultipleChoiceModelOutput](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.models.longformer.modeling_tf_longformer.TFLongformerMultipleChoiceModelOutput) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LongformerConfig](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerConfig)) and inputs.

-   **loss** (`tf.Tensor` of shape _(1,)_, _optional_, returned when `labels` is provided) — Classification loss.
    
-   **logits** (`tf.Tensor` of shape `(batch_size, num_choices)`) — _num\_choices_ is the second dimension of the input tensors. (see _input\_ids_ above).
    
    Classification scores (before SoftMax).
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.
    
    Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining \`attention\_window
    
    -   1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions\`.
-   **global\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.
    
    Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.
    

The [TFLongformerForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.TFLongformerForMultipleChoice) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFLongformerForMultipleChoice
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
>>> model = TFLongformerForMultipleChoice.from_pretrained("allenai/longformer-base-4096")

>>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
>>> choice0 = "It is eaten with a fork and a knife."
>>> choice1 = "It is eaten while held in the hand."

>>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="tf", padding=True)
>>> inputs = {k: tf.expand_dims(v, 0) for k, v in encoding.items()}
>>> outputs = model(inputs)  

>>> 
>>> logits = outputs.logits
```