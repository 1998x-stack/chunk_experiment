# BigBird

## Overview

The BigBird model was proposed in [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) by Zaheer, Manzil and Guruganesh, Guru and Dubey, Kumar Avinava and Ainslie, Joshua and Alberti, Chris and Ontanon, Santiago and Pham, Philip and Ravula, Anirudh and Wang, Qifan and Yang, Li and others. BigBird, is a sparse-attention based transformer which extends Transformer based models, such as BERT to much longer sequences. In addition to sparse attention, BigBird also applies global attention as well as random attention to the input sequence. Theoretically, it has been shown that applying sparse, global, and random attention approximates full attention, while being computationally much more efficient for longer sequences. As a consequence of the capability to handle longer context, BigBird has shown improved performance on various long document NLP tasks, such as question answering and summarization, compared to BERT or RoBERTa.

The abstract from the paper is the following:

_Transformers-based models, such as BERT, have been one of the most successful deep learning models for NLP. Unfortunately, one of their core limitations is the quadratic dependency (mainly in terms of memory) on the sequence length due to their full attention mechanism. To remedy this, we propose, BigBird, a sparse attention mechanism that reduces this quadratic dependency to linear. We show that BigBird is a universal approximator of sequence functions and is Turing complete, thereby preserving these properties of the quadratic, full attention model. Along the way, our theoretical analysis reveals some of the benefits of having O(1) global tokens (such as CLS), that attend to the entire sequence as part of the sparse attention mechanism. The proposed sparse attention can handle sequences of length up to 8x of what was previously possible using similar hardware. As a consequence of the capability to handle longer context, BigBird drastically improves performance on various NLP tasks such as question answering and summarization. We also propose novel applications to genomics data._

Tips:

-   For an in-detail explanation on how BigBird’s attention works, see [this blog post](https://huggingface.co/blog/big-bird).
-   BigBird comes with 2 implementations: **original\_full** & **block\_sparse**. For the sequence length < 1024, using **original\_full** is advised as there is no benefit in using **block\_sparse** attention.
-   The code currently uses window size of 3 blocks and 2 global blocks.
-   Sequence length must be divisible by block size.
-   Current implementation supports only **ITC**.
-   Current implementation doesn’t support **num\_random\_blocks = 0**
-   BigBird is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left.

This model was contributed by [vasudevgupta](https://huggingface.co/vasudevgupta). The original code can be found [here](https://github.com/google-research/bigbird).

## Documentation resources

-   [Text classification task guide](../tasks/sequence_classification)
-   [Token classification task guide](../tasks/token_classification)
-   [Question answering task guide](../tasks/question_answering)
-   [Causal language modeling task guide](../tasks/language_modeling)
-   [Masked language modeling task guide](../tasks/masked_language_modeling)
-   [Multiple choice task guide](../tasks/multiple_choice)

## BigBirdConfig

### class transformers.BigBirdConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/configuration_big_bird.py#L34)

( vocab\_size = 50358 hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu\_new' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 4096 type\_vocab\_size = 2 initializer\_range = 0.02 layer\_norm\_eps = 1e-12 use\_cache = True pad\_token\_id = 0 bos\_token\_id = 1 eos\_token\_id = 2 sep\_token\_id = 66 attention\_type = 'block\_sparse' use\_bias = True rescale\_embeddings = False block\_size = 64 num\_random\_blocks = 3 classifier\_dropout = None \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 50358) — Vocabulary size of the BigBird model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [BigBirdModel](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdModel).
-   **hidden\_size** (`int`, _optional_, defaults to 768) — Dimension of the encoder layers and the pooler layer.
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 12) — Number of hidden layers in the Transformer encoder.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 12) — Number of attention heads for each attention layer in the Transformer encoder.
-   **intermediate\_size** (`int`, _optional_, defaults to 3072) — Dimension of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
-   **hidden\_act** (`str` or `function`, _optional_, defaults to `"gelu_new"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
-   **hidden\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
-   **attention\_probs\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout ratio for the attention probabilities.
-   **max\_position\_embeddings** (`int`, _optional_, defaults to 4096) — The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 1024 or 2048 or 4096).
-   **type\_vocab\_size** (`int`, _optional_, defaults to 2) — The vocabulary size of the `token_type_ids` passed when calling [BigBirdModel](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdModel).
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-12) — The epsilon used by the layer normalization layers.
-   **is\_decoder** (`bool`, _optional_, defaults to `False`) — Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
-   **use\_cache** (`bool`, _optional_, defaults to `True`) — Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True`.
-   **attention\_type** (`str`, _optional_, defaults to `"block_sparse"`) — Whether to use block sparse attention (with n complexity) as introduced in paper or original attention layer (with n^2 complexity). Possible values are `"original_full"` and `"block_sparse"`.
-   **use\_bias** (`bool`, _optional_, defaults to `True`) — Whether to use bias in query, key, value.
-   **rescale\_embeddings** (`bool`, _optional_, defaults to `False`) — Whether to rescale embeddings with (hidden\_size \*\* 0.5).
-   **block\_size** (`int`, _optional_, defaults to 64) — Size of each block. Useful only when `attention_type == "block_sparse"`.
-   **num\_random\_blocks** (`int`, _optional_, defaults to 3) — Each query is going to attend these many number of random blocks. Useful only when `attention_type == "block_sparse"`.
-   **classifier\_dropout** (`float`, _optional_) — The dropout ratio for the classification head.

This is the configuration class to store the configuration of a [BigBirdModel](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdModel). It is used to instantiate an BigBird model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the BigBird [google/bigbird-roberta-base](https://huggingface.co/google/bigbird-roberta-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import BigBirdConfig, BigBirdModel

>>> 
>>> configuration = BigBirdConfig()

>>> 
>>> model = BigBirdModel(configuration)

>>> 
>>> configuration = model.config
```

## BigBirdTokenizer

### class transformers.BigBirdTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/tokenization_big_bird.py#L52)

( vocab\_file unk\_token = '<unk>' bos\_token = '<s>' eos\_token = '</s>' pad\_token = '<pad>' sep\_token = '\[SEP\]' mask\_token = '\[MASK\]' cls\_token = '\[CLS\]' sp\_model\_kwargs: typing.Union\[typing.Dict\[str, typing.Any\], NoneType\] = None \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — [SentencePiece](https://github.com/google/sentencepiece) file (generally has a _.spm_ extension) that contains the vocabulary necessary to instantiate a tokenizer.
-   **eos\_token** (`str`, _optional_, defaults to `"</s>"`) — The end of sequence token.
-   **bos\_token** (`str`, _optional_, defaults to `"<s>"`) — The begin of sequence token.
-   **unk\_token** (`str`, _optional_, defaults to `"<unk>"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **pad\_token** (`str`, _optional_, defaults to `"<pad>"`) — The token used for padding, for example when batching sequences of different lengths.
-   **sep\_token** (`str`, _optional_, defaults to `"[SEP]"`) — The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.
-   **cls\_token** (`str`, _optional_, defaults to `"[CLS]"`) — The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.
-   **mask\_token** (`str`, _optional_, defaults to `"[MASK]"`) — The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.
-   **sp\_model\_kwargs** (`dict`, _optional_) — Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things, to set:
    
    -   `enable_sampling`: Enable subword regularization.
        
    -   `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.
        
        -   `nbest_size = {0,1}`: No sampling is performed.
        -   `nbest_size > 1`: samples from the nbest\_size results.
        -   `nbest_size < 0`: assuming that nbest\_size is infinite and samples from the all hypothesis (lattice) using forward-filtering-and-backward-sampling algorithm.
    -   `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for BPE-dropout.
        
    

Construct a BigBird tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/tokenization_big_bird.py#L268)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A Big Bird sequence has the following format:

-   single sequence: `[CLS] X [SEP]`
-   pair of sequences: `[CLS] A [SEP] B [SEP]`

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/tokenization_big_bird.py#L293)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None already\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/tokenization_big_bird.py#L320)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).

Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence pair mask has the following format: :: 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 | first sequence | second sequence | If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/tokenization_big_bird.py#L251)

( save\_directory: str filename\_prefix: typing.Optional\[str\] = None )

## BigBirdTokenizerFast

### class transformers.BigBirdTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/tokenization_big_bird_fast.py#L68)

( vocab\_file = None tokenizer\_file = None unk\_token = '<unk>' bos\_token = '<s>' eos\_token = '</s>' pad\_token = '<pad>' sep\_token = '\[SEP\]' mask\_token = '\[MASK\]' cls\_token = '\[CLS\]' \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — [SentencePiece](https://github.com/google/sentencepiece) file (generally has a _.spm_ extension) that contains the vocabulary necessary to instantiate a tokenizer.
-   **bos\_token** (`str`, _optional_, defaults to `"<s>"`) — The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
    
    When building a sequence using special tokens, this is not the token that is used for the beginning of sequence. The token used is the `cls_token`.
    
-   **eos\_token** (`str`, _optional_, defaults to `"</s>"`) — The end of sequence token. .. note:: When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.
-   **unk\_token** (`str`, _optional_, defaults to `"<unk>"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **sep\_token** (`str`, _optional_, defaults to `"[SEP]"`) — The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.
-   **pad\_token** (`str`, _optional_, defaults to `"<pad>"`) — The token used for padding, for example when batching sequences of different lengths.
-   **cls\_token** (`str`, _optional_, defaults to `"[CLS]"`) — The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.
-   **mask\_token** (`str`, _optional_, defaults to `"[MASK]"`) — The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.

Construct a “fast” BigBird tokenizer (backed by HuggingFace’s _tokenizers_ library). Based on [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models). This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/tokenization_big_bird_fast.py#L158)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

list of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. An BigBird sequence has the following format:

-   single sequence: `[CLS] X [SEP]`
-   pair of sequences: `[CLS] A [SEP] B [SEP]`

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/tokenization_big_bird_fast.py#L214)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

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

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/tokenization_big_bird_fast.py#L183)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None already\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of ids.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Set to True if the token list is already formatted with special tokens for the model

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

## BigBird specific outputs

### class transformers.models.big\_bird.modeling\_big\_bird.BigBirdForPreTrainingOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_big_bird.py#L1854)

( loss: typing.Optional\[torch.FloatTensor\] = None prediction\_logits: FloatTensor = None seq\_relationship\_logits: FloatTensor = None hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Parameters

-   **loss** (_optional_, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) — Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
-   **prediction\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
-   **seq\_relationship\_logits** (`torch.FloatTensor` of shape `(batch_size, 2)`) — Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

Output type of [BigBirdForPreTraining](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdForPreTraining).

## BigBirdModel

### class transformers.BigBirdModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_big_bird.py#L1926)

( config add\_pooling\_layer = True )

Parameters

-   **config** ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare BigBird Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of cross-attention is added between the self-attention layers, following the architecture described in [Attention is all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_big_bird.py#L1983)

( input\_ids: LongTensor = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None encoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = None encoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = None past\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or `tuple(torch.FloatTensor)`

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
-   **encoder\_hidden\_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model is configured as a decoder.
-   **encoder\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`) — Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

A [transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    

The [BigBirdModel](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, BigBirdModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
>>> model = BigBirdModel.from_pretrained("google/bigbird-roberta-base")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## BigBirdForPreTraining

### class transformers.BigBirdForPreTraining

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_big_bird.py#L2265)

( config )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_big_bird.py#L2283)

( input\_ids: LongTensor = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.FloatTensor\] = None next\_sentence\_label: typing.Optional\[torch.LongTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.models.big\_bird.modeling\_big\_bird.BigBirdForPreTrainingOutput](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.models.big_bird.modeling_big_bird.BigBirdForPreTrainingOutput) or `tuple(torch.FloatTensor)`

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
-   **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
-   **next\_sentence\_label** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the next sequence prediction (classification) loss. If specified, nsp loss will be added to masked\_lm loss. Input should be a sequence pair (see `input_ids` docstring) Indices should be in `[0, 1]`:
    
    -   0 indicates sequence B is a continuation of sequence A,
    -   1 indicates sequence B is a random sequence.
    
-   **kwargs** (`Dict[str, any]`, optional, defaults to _{}_) — Used to hide legacy arguments that have been deprecated.

A [transformers.models.big\_bird.modeling\_big\_bird.BigBirdForPreTrainingOutput](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.models.big_bird.modeling_big_bird.BigBirdForPreTrainingOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) and inputs.

-   **loss** (_optional_, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) — Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
    
-   **prediction\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **seq\_relationship\_logits** (`torch.FloatTensor` of shape `(batch_size, 2)`) — Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [BigBirdForPreTraining](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, BigBirdForPreTraining
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
>>> model = BigBirdForPreTraining.from_pretrained("google/bigbird-roberta-base")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> prediction_logits = outputs.prediction_logits
>>> seq_relationship_logits = outputs.seq_relationship_logits
```

## BigBirdForCausalLM

### class transformers.BigBirdForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_big_bird.py#L2515)

( config )

Parameters

-   **config** ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

BigBird Model with a `language modeling` head on top for CLM fine-tuning. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_big_bird.py#L2536)

( input\_ids: LongTensor = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None encoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = None encoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = None past\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = None labels: typing.Optional\[torch.LongTensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

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
-   **encoder\_hidden\_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model is configured as a decoder.
-   **encoder\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`) — Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
-   **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

A [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Cross attentions weights after the attention softmax, used to compute the weighted average in the cross-attention heads.
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `torch.FloatTensor` tuples of length `config.n_layers`, with each tuple containing the cached key, value states of the self-attention and the cross-attention layers if model is used in encoder-decoder setting. Only relevant if `config.is_decoder = True`.
    
    Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    

The [BigBirdForCausalLM](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoTokenizer, BigBirdForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
>>> model = BigBirdForCausalLM.from_pretrained("google/bigbird-roberta-base")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs, labels=inputs["input_ids"])
>>> loss = outputs.loss
>>> logits = outputs.logits
```

## BigBirdForMaskedLM

### class transformers.BigBirdForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_big_bird.py#L2371)

( config )

Parameters

-   **config** ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

BigBird Model with a `language modeling` head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_big_bird.py#L2395)

( input\_ids: LongTensor = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None encoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = None encoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

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
-   **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

A [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Masked language modeling (MLM) loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [BigBirdForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoTokenizer, BigBirdForMaskedLM
>>> from datasets import load_dataset

>>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
>>> model = BigBirdForMaskedLM.from_pretrained("google/bigbird-roberta-base")
>>> squad_ds = load_dataset("squad_v2", split="train")
>>> 
>>> LONG_ARTICLE_TARGET = squad_ds[81514]["context"]
>>> 
>>> LONG_ARTICLE_TARGET[332:398]
'the highest values are very close to the theoretical maximum value'

>>> 
>>> LONG_ARTICLE_TO_MASK = LONG_ARTICLE_TARGET.replace("maximum", "[MASK]")
>>> inputs = tokenizer(LONG_ARTICLE_TO_MASK, return_tensors="pt")
>>> 
>>> list(inputs["input_ids"].shape)
[1, 919]

>>> with torch.no_grad():
...     logits = model(**inputs).logits
>>> 
>>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
>>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
>>> tokenizer.decode(predicted_token_id)
'maximum'
```

```
>>> labels = tokenizer(LONG_ARTICLE_TARGET, return_tensors="pt")["input_ids"]
>>> labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
>>> outputs = model(**inputs, labels=labels)
>>> round(outputs.loss.item(), 2)
1.99
```

## BigBirdForSequenceClassification

### class transformers.BigBirdForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_big_bird.py#L2678)

( config )

Parameters

-   **config** ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

BigBird Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_big_bird.py#L2689)

( input\_ids: LongTensor = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

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

A [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [BigBirdForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoTokenizer, BigBirdForSequenceClassification
>>> from datasets import load_dataset

>>> tokenizer = AutoTokenizer.from_pretrained("l-yohai/bigbird-roberta-base-mnli")
>>> model = BigBirdForSequenceClassification.from_pretrained("l-yohai/bigbird-roberta-base-mnli")
>>> squad_ds = load_dataset("squad_v2", split="train")
>>> LONG_ARTICLE = squad_ds[81514]["context"]
>>> inputs = tokenizer(LONG_ARTICLE, return_tensors="pt")
>>> 
>>> list(inputs["input_ids"].shape)
[1, 919]

>>> with torch.no_grad():
...     logits = model(**inputs).logits
>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
'LABEL_0'
```

```
>>> num_labels = len(model.config.id2label)
>>> model = BigBirdForSequenceClassification.from_pretrained(
...     "l-yohai/bigbird-roberta-base-mnli", num_labels=num_labels
... )
>>> labels = torch.tensor(1)
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
1.13
```

## BigBirdForMultipleChoice

### class transformers.BigBirdForMultipleChoice

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_big_bird.py#L2806)

( config )

Parameters

-   **config** ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

BigBird Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a softmax) e.g. for RocStories/SWAG tasks.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_big_bird.py#L2817)

( input\_ids: LongTensor = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.MultipleChoiceModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or `tuple(torch.FloatTensor)`

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

A [transformers.modeling\_outputs.MultipleChoiceModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape _(1,)_, _optional_, returned when `labels` is provided) — Classification loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, num_choices)`) — _num\_choices_ is the second dimension of the input tensors. (see _input\_ids_ above).
    
    Classification scores (before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [BigBirdForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdForMultipleChoice) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, BigBirdForMultipleChoice
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
>>> model = BigBirdForMultipleChoice.from_pretrained("google/bigbird-roberta-base")

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

## BigBirdForTokenClassification

### class transformers.BigBirdForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_big_bird.py#L2899)

( config )

Parameters

-   **config** ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

BigBird Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_big_bird.py#L2914)

( input\_ids: LongTensor = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

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

A [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [BigBirdForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, BigBirdForTokenClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
>>> model = BigBirdForTokenClassification.from_pretrained("google/bigbird-roberta-base")

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

## BigBirdForQuestionAnswering

### class transformers.BigBirdForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_big_bird.py#L2998)

( config add\_pooling\_layer = False )

Parameters

-   **config** ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

BigBird Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_big_bird.py#L3012)

( input\_ids: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None question\_lengths: typing.Optional\[torch.Tensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None start\_positions: typing.Optional\[torch.LongTensor\] = None end\_positions: typing.Optional\[torch.LongTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → `transformers.models.big_bird.modeling_big_bird.BigBirdForQuestionAnsweringModelOutput` or `tuple(torch.FloatTensor)`

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

Returns

`transformers.models.big_bird.modeling_big_bird.BigBirdForQuestionAnsweringModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.big_bird.modeling_big_bird.BigBirdForQuestionAnsweringModelOutput` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
    
-   **start\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-start scores (before SoftMax).
    
-   **end\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-end scores (before SoftMax).
    
-   **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, 1)`) — pooler output from BigBigModel
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [BigBirdForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoTokenizer, BigBirdForQuestionAnswering
>>> from datasets import load_dataset

>>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
>>> model = BigBirdForQuestionAnswering.from_pretrained("google/bigbird-roberta-base")
>>> squad_ds = load_dataset("squad_v2", split="train")
>>> 
>>> LONG_ARTICLE = squad_ds[81514]["context"]
>>> QUESTION = squad_ds[81514]["question"]
>>> QUESTION
'During daytime how high can the temperatures reach?'

>>> inputs = tokenizer(QUESTION, LONG_ARTICLE, return_tensors="pt")
>>> 
>>> list(inputs["input_ids"].shape)
[1, 929]

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> answer_start_index = outputs.start_logits.argmax()
>>> answer_end_index = outputs.end_logits.argmax()
>>> predict_answer_token_ids = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> predict_answer_token = tokenizer.decode(predict_answer_token_ids)
```

```
>>> target_start_index, target_end_index = torch.tensor([130]), torch.tensor([132])
>>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
>>> loss = outputs.loss
```

## FlaxBigBirdModel

### class transformers.FlaxBigBirdModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_flax_big_bird.py#L1890)

( config: BigBirdConfig input\_shape: typing.Optional\[tuple\] = None seed: int = 0 dtype: dtype = <class 'jax.numpy.float32'> \_do\_init: bool = True gradient\_checkpointing: bool = False \*\*kwargs )

Parameters

-   **config** ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

The bare BigBird Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_flax_big_bird.py#L1717)

( input\_ids attention\_mask = None token\_type\_ids = None position\_ids = None head\_mask = None encoder\_hidden\_states = None encoder\_attention\_mask = None params: dict = None dropout\_rng: typing.Optional\[PRNGKey\] = None indices\_rng: typing.Optional\[PRNGKey\] = None train: bool = False output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None past\_key\_values: dict = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
-   **head\_mask** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, `optional) -- Mask to nullify selected heads of the attention modules. Mask values selected in` \[0, 1\]\`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_flax\_outputs.FlaxBaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPooling) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) and inputs.

-   **last\_hidden\_state** (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`jnp.ndarray` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
    
-   **hidden\_states** (`tuple(jnp.ndarray)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The `FlaxBigBirdPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxBigBirdModel

>>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
>>> model = FlaxBigBirdModel.from_pretrained("google/bigbird-roberta-base")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## FlaxBigBirdForPreTraining

### class transformers.FlaxBigBirdForPreTraining

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_flax_big_bird.py#L1967)

( config: BigBirdConfig input\_shape: typing.Optional\[tuple\] = None seed: int = 0 dtype: dtype = <class 'jax.numpy.float32'> \_do\_init: bool = True gradient\_checkpointing: bool = False \*\*kwargs )

Parameters

-   **config** ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

BigBird Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next sentence prediction (classification)` head.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_flax_big_bird.py#L1717)

( input\_ids attention\_mask = None token\_type\_ids = None position\_ids = None head\_mask = None encoder\_hidden\_states = None encoder\_attention\_mask = None params: dict = None dropout\_rng: typing.Optional\[PRNGKey\] = None indices\_rng: typing.Optional\[PRNGKey\] = None train: bool = False output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None past\_key\_values: dict = None ) → `transformers.models.big_bird.modeling_flax_big_bird.FlaxBigBirdForPreTrainingOutput` or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
-   **head\_mask** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, `optional) -- Mask to nullify selected heads of the attention modules. Mask values selected in` \[0, 1\]\`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.big_bird.modeling_flax_big_bird.FlaxBigBirdForPreTrainingOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.big_bird.modeling_flax_big_bird.FlaxBigBirdForPreTrainingOutput` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) and inputs.

-   **prediction\_logits** (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **seq\_relationship\_logits** (`jnp.ndarray` of shape `(batch_size, 2)`) — Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
    
-   **hidden\_states** (`tuple(jnp.ndarray)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The `FlaxBigBirdPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxBigBirdForPreTraining

>>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
>>> model = FlaxBigBirdForPreTraining.from_pretrained("google/bigbird-roberta-base")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
>>> outputs = model(**inputs)

>>> prediction_logits = outputs.prediction_logits
>>> seq_relationship_logits = outputs.seq_relationship_logits
```

## FlaxBigBirdForCausalLM

### class transformers.FlaxBigBirdForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_flax_big_bird.py#L2599)

( config: BigBirdConfig input\_shape: typing.Optional\[tuple\] = None seed: int = 0 dtype: dtype = <class 'jax.numpy.float32'> \_do\_init: bool = True gradient\_checkpointing: bool = False \*\*kwargs )

Parameters

-   **config** ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

BigBird Model with a language modeling head on top (a linear layer on top of the hidden-states output) e.g for autoregressive tasks.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_flax_big_bird.py#L1717)

( input\_ids attention\_mask = None token\_type\_ids = None position\_ids = None head\_mask = None encoder\_hidden\_states = None encoder\_attention\_mask = None params: dict = None dropout\_rng: typing.Optional\[PRNGKey\] = None indices\_rng: typing.Optional\[PRNGKey\] = None train: bool = False output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None past\_key\_values: dict = None ) → [transformers.modeling\_flax\_outputs.FlaxCausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxCausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
-   **head\_mask** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, `optional) -- Mask to nullify selected heads of the attention modules. Mask values selected in` \[0, 1\]\`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_flax\_outputs.FlaxCausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxCausalLMOutputWithCrossAttentions) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) and inputs.

-   **logits** (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **hidden\_states** (`tuple(jnp.ndarray)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Cross attentions weights after the attention softmax, used to compute the weighted average in the cross-attention heads.
    
-   **past\_key\_values** (`tuple(tuple(jnp.ndarray))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `jnp.ndarray` tuples of length `config.n_layers`, with each tuple containing the cached key, value states of the self-attention and the cross-attention layers if model is used in encoder-decoder setting. Only relevant if `config.is_decoder = True`.
    
    Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    

The `FlaxBigBirdPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxBigBirdForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
>>> model = FlaxBigBirdForCausalLM.from_pretrained("google/bigbird-roberta-base")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
>>> outputs = model(**inputs)

>>> 
>>> next_token_logits = outputs.logits[:, -1]
```

## FlaxBigBirdForMaskedLM

### class transformers.FlaxBigBirdForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_flax_big_bird.py#L2060)

( config: BigBirdConfig input\_shape: typing.Optional\[tuple\] = None seed: int = 0 dtype: dtype = <class 'jax.numpy.float32'> \_do\_init: bool = True gradient\_checkpointing: bool = False \*\*kwargs )

Parameters

-   **config** ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

BigBird Model with a `language modeling` head on top.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_flax_big_bird.py#L1717)

( input\_ids attention\_mask = None token\_type\_ids = None position\_ids = None head\_mask = None encoder\_hidden\_states = None encoder\_attention\_mask = None params: dict = None dropout\_rng: typing.Optional\[PRNGKey\] = None indices\_rng: typing.Optional\[PRNGKey\] = None train: bool = False output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None past\_key\_values: dict = None ) → [transformers.modeling\_flax\_outputs.FlaxMaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxMaskedLMOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
-   **head\_mask** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, `optional) -- Mask to nullify selected heads of the attention modules. Mask values selected in` \[0, 1\]\`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_flax\_outputs.FlaxMaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxMaskedLMOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) and inputs.

-   **logits** (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **hidden\_states** (`tuple(jnp.ndarray)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The `FlaxBigBirdPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxBigBirdForMaskedLM

>>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
>>> model = FlaxBigBirdForMaskedLM.from_pretrained("google/bigbird-roberta-base")

>>> inputs = tokenizer("The capital of France is [MASK].", return_tensors="jax")

>>> outputs = model(**inputs)
>>> logits = outputs.logits
```

## FlaxBigBirdForSequenceClassification

### class transformers.FlaxBigBirdForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_flax_big_bird.py#L2150)

( config: BigBirdConfig input\_shape: typing.Optional\[tuple\] = None seed: int = 0 dtype: dtype = <class 'jax.numpy.float32'> \_do\_init: bool = True gradient\_checkpointing: bool = False \*\*kwargs )

Parameters

-   **config** ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

BigBird Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_flax_big_bird.py#L1717)

( input\_ids attention\_mask = None token\_type\_ids = None position\_ids = None head\_mask = None encoder\_hidden\_states = None encoder\_attention\_mask = None params: dict = None dropout\_rng: typing.Optional\[PRNGKey\] = None indices\_rng: typing.Optional\[PRNGKey\] = None train: bool = False output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None past\_key\_values: dict = None ) → [transformers.modeling\_flax\_outputs.FlaxSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
-   **head\_mask** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, `optional) -- Mask to nullify selected heads of the attention modules. Mask values selected in` \[0, 1\]\`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_flax\_outputs.FlaxSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSequenceClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) and inputs.

-   **logits** (`jnp.ndarray` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
    
-   **hidden\_states** (`tuple(jnp.ndarray)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The `FlaxBigBirdPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxBigBirdForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
>>> model = FlaxBigBirdForSequenceClassification.from_pretrained("google/bigbird-roberta-base")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

>>> outputs = model(**inputs)
>>> logits = outputs.logits
```

## FlaxBigBirdForMultipleChoice

### class transformers.FlaxBigBirdForMultipleChoice

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_flax_big_bird.py#L2231)

( config: BigBirdConfig input\_shape: typing.Optional\[tuple\] = None seed: int = 0 dtype: dtype = <class 'jax.numpy.float32'> \_do\_init: bool = True \*\*kwargs )

Parameters

-   **config** ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

BigBird Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a softmax) e.g. for RocStories/SWAG tasks.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_flax_big_bird.py#L1717)

( input\_ids attention\_mask = None token\_type\_ids = None position\_ids = None head\_mask = None encoder\_hidden\_states = None encoder\_attention\_mask = None params: dict = None dropout\_rng: typing.Optional\[PRNGKey\] = None indices\_rng: typing.Optional\[PRNGKey\] = None train: bool = False output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None past\_key\_values: dict = None ) → [transformers.modeling\_flax\_outputs.FlaxMultipleChoiceModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxMultipleChoiceModelOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`numpy.ndarray` of shape `(batch_size, num_choices, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`numpy.ndarray` of shape `(batch_size, num_choices, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`numpy.ndarray` of shape `(batch_size, num_choices, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`numpy.ndarray` of shape `(batch_size, num_choices, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
-   **head\_mask** (`numpy.ndarray` of shape `(batch_size, num_choices, sequence_length)`, `optional) -- Mask to nullify selected heads of the attention modules. Mask values selected in` \[0, 1\]\`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_flax\_outputs.FlaxMultipleChoiceModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxMultipleChoiceModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) and inputs.

-   **logits** (`jnp.ndarray` of shape `(batch_size, num_choices)`) — _num\_choices_ is the second dimension of the input tensors. (see _input\_ids_ above).
    
    Classification scores (before SoftMax).
    
-   **hidden\_states** (`tuple(jnp.ndarray)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The `FlaxBigBirdPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxBigBirdForMultipleChoice

>>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
>>> model = FlaxBigBirdForMultipleChoice.from_pretrained("google/bigbird-roberta-base")

>>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
>>> choice0 = "It is eaten with a fork and a knife."
>>> choice1 = "It is eaten while held in the hand."

>>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="jax", padding=True)
>>> outputs = model(**{k: v[None, :] for k, v in encoding.items()})

>>> logits = outputs.logits
```

## FlaxBigBirdForTokenClassification

### class transformers.FlaxBigBirdForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_flax_big_bird.py#L2329)

( config: BigBirdConfig input\_shape: typing.Optional\[tuple\] = None seed: int = 0 dtype: dtype = <class 'jax.numpy.float32'> \_do\_init: bool = True gradient\_checkpointing: bool = False \*\*kwargs )

Parameters

-   **config** ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

BigBird Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_flax_big_bird.py#L1717)

( input\_ids attention\_mask = None token\_type\_ids = None position\_ids = None head\_mask = None encoder\_hidden\_states = None encoder\_attention\_mask = None params: dict = None dropout\_rng: typing.Optional\[PRNGKey\] = None indices\_rng: typing.Optional\[PRNGKey\] = None train: bool = False output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None past\_key\_values: dict = None ) → [transformers.modeling\_flax\_outputs.FlaxTokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxTokenClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
-   **head\_mask** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, `optional) -- Mask to nullify selected heads of the attention modules. Mask values selected in` \[0, 1\]\`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_flax\_outputs.FlaxTokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxTokenClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) and inputs.

-   **logits** (`jnp.ndarray` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
    
-   **hidden\_states** (`tuple(jnp.ndarray)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The `FlaxBigBirdPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxBigBirdForTokenClassification

>>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
>>> model = FlaxBigBirdForTokenClassification.from_pretrained("google/bigbird-roberta-base")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

>>> outputs = model(**inputs)
>>> logits = outputs.logits
```

## FlaxBigBirdForQuestionAnswering

### class transformers.FlaxBigBirdForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_flax_big_bird.py#L2432)

( config: BigBirdConfig input\_shape: typing.Optional\[tuple\] = None seed: int = 0 dtype: dtype = <class 'jax.numpy.float32'> \_do\_init: bool = True gradient\_checkpointing: bool = False \*\*kwargs )

Parameters

-   **config** ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

BigBird Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/big_bird/modeling_flax_big_bird.py#L2435)

( input\_ids attention\_mask = None token\_type\_ids = None position\_ids = None head\_mask = None question\_lengths = None params: dict = None dropout\_rng: typing.Optional\[PRNGKey\] = None indices\_rng: typing.Optional\[PRNGKey\] = None train: bool = False output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → `transformers.models.big_bird.modeling_flax_big_bird.FlaxBigBirdForQuestionAnsweringModelOutput` or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
-   **head\_mask** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, `optional) -- Mask to nullify selected heads of the attention modules. Mask values selected in` \[0, 1\]\`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.big_bird.modeling_flax_big_bird.FlaxBigBirdForQuestionAnsweringModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.big_bird.modeling_flax_big_bird.FlaxBigBirdForQuestionAnsweringModelOutput` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([BigBirdConfig](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.BigBirdConfig)) and inputs.

-   **start\_logits** (`jnp.ndarray` of shape `(batch_size, sequence_length)`) — Span-start scores (before SoftMax).
    
-   **end\_logits** (`jnp.ndarray` of shape `(batch_size, sequence_length)`) — Span-end scores (before SoftMax).
    
-   **pooled\_output** (`jnp.ndarray` of shape `(batch_size, hidden_size)`) — pooled\_output returned by FlaxBigBirdModel.
    
-   **hidden\_states** (`tuple(jnp.ndarray)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [FlaxBigBirdForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.FlaxBigBirdForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxBigBirdForQuestionAnswering

>>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
>>> model = FlaxBigBirdForQuestionAnswering.from_pretrained("google/bigbird-roberta-base")

>>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
>>> inputs = tokenizer(question, text, return_tensors="jax")

>>> outputs = model(**inputs)
>>> start_scores = outputs.start_logits
>>> end_scores = outputs.end_logits
```