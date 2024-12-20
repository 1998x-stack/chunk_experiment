# mT5

[![Models](https://img.shields.io/badge/All_model_pages-mt5-blueviolet)](https://huggingface.co/models?filter=mt5) [![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/docs-demos/mt5-small-finetuned-arxiv-cs-finetuned-arxiv-cs-full)

## Overview

The mT5 model was presented in [mT5: A massively multilingual pre-trained text-to-text transformer](https://arxiv.org/abs/2010.11934) by Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, Colin Raffel.

The abstract from the paper is the following:

_The recent “Text-to-Text Transfer Transformer” (T5) leveraged a unified text-to-text format and scale to attain state-of-the-art results on a wide variety of English-language NLP tasks. In this paper, we introduce mT5, a multilingual variant of T5 that was pre-trained on a new Common Crawl-based dataset covering 101 languages. We detail the design and modified training of mT5 and demonstrate its state-of-the-art performance on many multilingual benchmarks. We also describe a simple technique to prevent “accidental translation” in the zero-shot setting, where a generative model chooses to (partially) translate its prediction into the wrong language. All of the code and model checkpoints used in this work are publicly available._

Note: mT5 was only pre-trained on [mC4](https://huggingface.co/datasets/mc4) excluding any supervised training. Therefore, this model has to be fine-tuned before it is usable on a downstream task, unlike the original T5 model. Since mT5 was pre-trained unsupervisedly, there’s no real advantage to using a task prefix during single-task fine-tuning. If you are doing multi-task fine-tuning, you should use a prefix.

Google has released the following variants:

-   [google/mt5-small](https://huggingface.co/google/mt5-small)
    
-   [google/mt5-base](https://huggingface.co/google/mt5-base)
    
-   [google/mt5-large](https://huggingface.co/google/mt5-large)
    
-   [google/mt5-xl](https://huggingface.co/google/mt5-xl)
    
-   [google/mt5-xxl](https://huggingface.co/google/mt5-xxl).
    

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten). The original code can be found [here](https://github.com/google-research/multilingual-t5).

## Documentation resources

-   [Translation task guide](../tasks/translation)
-   [Summarization task guide](../tasks/summarization)

## MT5Config

### class transformers.MT5Config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mt5/configuration_mt5.py#L26)

( vocab\_size = 250112 d\_model = 512 d\_kv = 64 d\_ff = 1024 num\_layers = 8 num\_decoder\_layers = None num\_heads = 6 relative\_attention\_num\_buckets = 32 relative\_attention\_max\_distance = 128 dropout\_rate = 0.1 layer\_norm\_epsilon = 1e-06 initializer\_factor = 1.0 feed\_forward\_proj = 'gated-gelu' is\_encoder\_decoder = True use\_cache = True tokenizer\_class = 'T5Tokenizer' tie\_word\_embeddings = False pad\_token\_id = 0 eos\_token\_id = 1 decoder\_start\_token\_id = 0 classifier\_dropout = 0.0 \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 250112) — Vocabulary size of the T5 model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [T5Model](/docs/transformers/v4.34.0/en/model_doc/t5#transformers.T5Model) or [TFT5Model](/docs/transformers/v4.34.0/en/model_doc/t5#transformers.TFT5Model).
-   **d\_model** (`int`, _optional_, defaults to 512) — Size of the encoder layers and the pooler layer.
-   **d\_kv** (`int`, _optional_, defaults to 64) — Size of the key, query, value projections per attention head. `d_kv` has to be equal to `d_model // num_heads`.
-   **d\_ff** (`int`, _optional_, defaults to 1024) — Size of the intermediate feed forward layer in each `T5Block`.
-   **num\_layers** (`int`, _optional_, defaults to 8) — Number of hidden layers in the Transformer encoder.
-   **num\_decoder\_layers** (`int`, _optional_) — Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
-   **num\_heads** (`int`, _optional_, defaults to 6) — Number of attention heads for each attention layer in the Transformer encoder.
-   **relative\_attention\_num\_buckets** (`int`, _optional_, defaults to 32) — The number of buckets to use for each attention layer.
-   **relative\_attention\_max\_distance** (`int`, _optional_, defaults to 128) — The maximum distance of the longer sequences for the bucket separation.
-   **dropout\_rate** (`float`, _optional_, defaults to 0.1) — The ratio for all dropout layers.
-   **classifier\_dropout** (`float`, _optional_, defaults to 0.0) — The dropout ratio for classifier.
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-6) — The epsilon used by the layer normalization layers.
-   **initializer\_factor** (`float`, _optional_, defaults to 1) — A factor for initializing all weight matrices (should be kept to 1, used internally for initialization testing).
-   **feed\_forward\_proj** (`string`, _optional_, defaults to `"gated-gelu"`) — Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`.
-   **use\_cache** (`bool`, _optional_, defaults to `True`) — Whether or not the model should return the last key/values attentions (not used by all models).

This is the configuration class to store the configuration of a [MT5Model](/docs/transformers/v4.34.0/en/model_doc/mt5#transformers.MT5Model) or a [TFMT5Model](/docs/transformers/v4.34.0/en/model_doc/mt5#transformers.TFMT5Model). It is used to instantiate a mT5 model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the mT5 [google/mt5-small](https://huggingface.co/google/mt5-small) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## MT5Tokenizer

### class transformers.T5Tokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/t5/tokenization_t5.py#L63)

( vocab\_file eos\_token = '</s>' unk\_token = '<unk>' pad\_token = '<pad>' extra\_ids = 100 additional\_special\_tokens = None sp\_model\_kwargs: typing.Union\[typing.Dict\[str, typing.Any\], NoneType\] = None legacy = None \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — [SentencePiece](https://github.com/google/sentencepiece) file (generally has a _.spm_ extension) that contains the vocabulary necessary to instantiate a tokenizer.
-   **eos\_token** (`str`, _optional_, defaults to `"</s>"`) — The end of sequence token.
    
    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.
    
-   **unk\_token** (`str`, _optional_, defaults to `"<unk>"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **pad\_token** (`str`, _optional_, defaults to `"<pad>"`) — The token used for padding, for example when batching sequences of different lengths.
-   **extra\_ids** (`int`, _optional_, defaults to 100) — Add a number of extra ids added to the vocabulary for use as sentinels. These tokens are accessible as “id{%d}>” where ”{%d}” is a number between 0 and extra\_ids-1. These tokens can be retrieved by calling get\_sentinel\_tokens method and token ids can be by calling get\_sentinel\_token\_ids method additional\_special\_tokens (`List[str]`, _optional_): Additional special tokens used by the tokenizer.
-   **sp\_model\_kwargs** (`dict`, _optional_) — Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things, to set:
    
    -   `enable_sampling`: Enable subword regularization.
        
    -   `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.
        
        -   `nbest_size = {0,1}`: No sampling is performed.
        -   `nbest_size > 1`: samples from the nbest\_size results.
        -   `nbest_size < 0`: assuming that nbest\_size is infinite and samples from the all hypothesis (lattice) using forward-filtering-and-backward-sampling algorithm.
    -   `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for BPE-dropout.
        
    
-   **legacy** (`bool`, _optional_) — Whether or not the `legacy` behaviour of the tokenizer should be used. Legacy is before the merge of #24622 and #25224 which includes fixes to properly handle tokens that appear after special tokens. A simple example:
    
    -   `legacy=True`:
    

Construct a T5 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/t5/tokenization_t5.py#L331)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A sequence has the following format:

-   single sequence: `X </s>`
-   pair of sequences: `A </s> B </s>`

Converts a sequence of tokens (string) in a single string.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/t5/tokenization_t5.py#L309)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make use of token type ids, therefore a list of zeros is returned.

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/t5/tokenization_t5.py#L262)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None already\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

#### tokenize

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/t5/tokenization_t5.py#L373)

( text: TextInput add\_special\_tokens = False \*\*kwargs )

Converts a string to a list of tokens. If `self.legacy` is set to `False`, a prefix token is added unless the first token is special.

See [T5Tokenizer](/docs/transformers/v4.34.0/en/model_doc/mt5#transformers.T5Tokenizer) for all details.

## MT5TokenizerFast

### class transformers.T5TokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/t5/tokenization_t5_fast.py#L66)

( vocab\_file = None tokenizer\_file = None eos\_token = '</s>' unk\_token = '<unk>' pad\_token = '<pad>' extra\_ids = 100 additional\_special\_tokens = None \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — [SentencePiece](https://github.com/google/sentencepiece) file (generally has a _.spm_ extension) that contains the vocabulary necessary to instantiate a tokenizer.
-   **eos\_token** (`str`, _optional_, defaults to `"</s>"`) — The end of sequence token.
    
    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.
    
-   **unk\_token** (`str`, _optional_, defaults to `"<unk>"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **pad\_token** (`str`, _optional_, defaults to `"<pad>"`) — The token used for padding, for example when batching sequences of different lengths.
-   **extra\_ids** (`int`, _optional_, defaults to 100) — Add a number of extra ids added to the vocabulary for use as sentinels. These tokens are accessible as “id{%d}>” where ”{%d}” is a number between 0 and extra\_ids-1. These tokens can be retrieved by calling get\_sentinel\_tokens method and token ids can be by calling get\_sentinel\_token\_ids method
-   **additional\_special\_tokens** (`List[str]`, _optional_) — Additional special tokens used by the tokenizer.

Construct a “fast” T5 tokenizer (backed by HuggingFace’s _tokenizers_ library). Based on [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/t5/tokenization_t5_fast.py#L193)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A sequence has the following format:

-   single sequence: `X </s>`
-   pair of sequences: `A </s> B </s>`

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/t5/tokenization_t5_fast.py#L219)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make use of token type ids, therefore a list of zeros is returned.

See [T5TokenizerFast](/docs/transformers/v4.34.0/en/model_doc/mt5#transformers.T5TokenizerFast) for all details.

## MT5Model

### class transformers.MT5Model

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mt5/modeling_mt5.py#L1330)

( config: MT5Config )

Parameters

-   **config** ([MT5Config](/docs/transformers/v4.34.0/en/model_doc/mt5#transformers.MT5Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare MT5 Model transformer outputting raw hidden-states without any specific head on top.

The MT5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It’s an encoder decoder transformer pre-trained in a text-to-text denoising generative setting.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

Examples:

```
>>> from transformers import MT5Model, AutoTokenizer

>>> model = MT5Model.from_pretrained("google/mt5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
>>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
>>> summary = "Weiter Verhandlung in Syrien."
>>> inputs = tokenizer(article, return_tensors="pt")
>>> labels = tokenizer(text_target=summary, return_tensors="pt")

>>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])
>>> hidden_states = outputs.last_hidden_state
```

Moves the model to cpu from a model parallel state.

Example:

```
model = MT5ForConditionalGeneration.from_pretrained("Mt5-xl")
device_map = {
    0: [0, 1, 2],
    1: [3, 4, 5, 6, 7, 8, 9],
    2: [10, 11, 12, 13, 14, 15, 16],
    3: [17, 18, 19, 20, 21, 22, 23],
}
model.parallelize(device_map)  
model.deparallelize()  
```

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mt5/modeling_mt5.py#L1439)

( input\_ids: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None decoder\_input\_ids: typing.Optional\[torch.LongTensor\] = None decoder\_attention\_mask: typing.Optional\[torch.BoolTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None decoder\_head\_mask: typing.Optional\[torch.FloatTensor\] = None cross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = None encoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = None past\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = None inputs\_embeds: typing.Optional\[torch.Tensor\] = None decoder\_inputs\_embeds: typing.Optional\[torch.Tensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. MT5 is a model with relative position embeddings so you should be able to pad the inputs on both the right and the left.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for detail.
    
    [What are input IDs?](../glossary#input-ids)
    
    To know more on how to prepare `input_ids` for pretraining take a look a [MT5 Training](./mt5#training).
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, _optional_) — Indices of decoder input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are decoder input IDs?](../glossary#decoder-input-ids)
    
    MT5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
    
    To know more on how to prepare `decoder_input_ids` for pretraining take a look at [MT5 Training](./mt5#training).
    
-   **decoder\_attention\_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, _optional_) — Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also be used by default.
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **decoder\_head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **encoder\_outputs** (`tuple(tuple(torch.FloatTensor)`, _optional_) — Tuple consists of (`last_hidden_state`, `optional`: _hidden\_states_, `optional`: _attentions_) `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`) — Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
    
    If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **decoder\_inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be input (see `past_key_values`). This is useful if you want more control over how to convert `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
    
    If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value of `inputs_embeds`.
    
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([MT5Config](/docs/transformers/v4.34.0/en/model_doc/mt5#transformers.MT5Config)) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the decoder of the model.
    
    If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.
    
-   **decoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
    
-   **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
    
-   **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
    
-   **encoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [MT5Model](/docs/transformers/v4.34.0/en/model_doc/mt5#transformers.MT5Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, MT5Model

>>> tokenizer = AutoTokenizer.from_pretrained("mt5-small")
>>> model = MT5Model.from_pretrained("mt5-small")

>>> input_ids = tokenizer(
...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
... ).input_ids  
>>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  

>>> 
>>> 
>>> decoder_input_ids = model._shift_right(decoder_input_ids)

>>> 
>>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
>>> last_hidden_states = outputs.last_hidden_state
```

#### parallelize

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mt5/modeling_mt5.py#L1377)

( device\_map = None )

Parameters

-   **device\_map** (`Dict[int, list]`, optional, defaults to None) — A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always automatically mapped to the first device (for esoteric reasons). That means that the first device should have fewer attention modules mapped to it than other devices. For reference, the mt5 models have the following number of attention modules:
    
    -   mt5-small: 6
    -   mt5-base: 12
    -   mt5-large: 24
    -   mt5-xl: 24
    -   mt5-xxl: 24
    

This is an experimental feature and is a subject to change at a moment’s notice.

Uses a device map to distribute attention modules of the model across several devices. If no device map is given, it will evenly distribute blocks across all devices.

Example:

```
model = MT5ForConditionalGeneration.from_pretrained("mt5-xl")
device_map = {
    0: [0, 1, 2],
    1: [3, 4, 5, 6, 7, 8, 9],
    2: [10, 11, 12, 13, 14, 15, 16],
    3: [17, 18, 19, 20, 21, 22, 23],
}
model.parallelize(device_map)
```

## MT5ForConditionalGeneration

### class transformers.MT5ForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mt5/modeling_mt5.py#L1556)

( config: MT5Config )

Parameters

-   **config** ([MT5Config](/docs/transformers/v4.34.0/en/model_doc/mt5#transformers.MT5Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

MT5 Model with a `language modeling` head on top.

The MT5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It’s an encoder decoder transformer pre-trained in a text-to-text denoising generative setting.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

Examples:

```
>>> from transformers import MT5ForConditionalGeneration, AutoTokenizer

>>> model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
>>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
>>> summary = "Weiter Verhandlung in Syrien."
>>> inputs = tokenizer(article, text_target=summary, return_tensors="pt")

>>> outputs = model(**inputs)
>>> loss = outputs.loss
```

Moves the model to cpu from a model parallel state.

Example:

```
model = MT5ForConditionalGeneration.from_pretrained("Mt5-xl")
device_map = {
    0: [0, 1, 2],
    1: [3, 4, 5, 6, 7, 8, 9],
    2: [10, 11, 12, 13, 14, 15, 16],
    3: [17, 18, 19, 20, 21, 22, 23],
}
model.parallelize(device_map)  
model.deparallelize()  
```

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mt5/modeling_mt5.py#L1669)

( input\_ids: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None decoder\_input\_ids: typing.Optional\[torch.LongTensor\] = None decoder\_attention\_mask: typing.Optional\[torch.BoolTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None decoder\_head\_mask: typing.Optional\[torch.FloatTensor\] = None cross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = None encoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = None past\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None decoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. MT5 is a model with relative position embeddings so you should be able to pad the inputs on both the right and the left.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for detail.
    
    [What are input IDs?](../glossary#input-ids)
    
    To know more on how to prepare `input_ids` for pretraining take a look a [MT5 Training](./mt5#training).
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, _optional_) — Indices of decoder input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are decoder input IDs?](../glossary#decoder-input-ids)
    
    MT5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
    
    To know more on how to prepare `decoder_input_ids` for pretraining take a look at [MT5 Training](./mt5#training).
    
-   **decoder\_attention\_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, _optional_) — Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also be used by default.
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **decoder\_head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **encoder\_outputs** (`tuple(tuple(torch.FloatTensor)`, _optional_) — Tuple consists of (`last_hidden_state`, `optional`: _hidden\_states_, `optional`: _attentions_) `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`) — Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
    
    If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **decoder\_inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be input (see `past_key_values`). This is useful if you want more control over how to convert `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
    
    If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value of `inputs_embeds`.
    
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ..., config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`

A [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([MT5Config](/docs/transformers/v4.34.0/en/model_doc/mt5#transformers.MT5Config)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Language modeling loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
    
-   **decoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
    
-   **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
    
-   **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
    
-   **encoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [MT5ForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/mt5#transformers.MT5ForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, MT5ForConditionalGeneration

>>> tokenizer = AutoTokenizer.from_pretrained("mt5-small")
>>> model = MT5ForConditionalGeneration.from_pretrained("mt5-small")

>>> 
>>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
>>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
>>> outputs = model(input_ids=input_ids, labels=labels)
>>> loss = outputs.loss
>>> logits = outputs.logits

>>> 
>>> input_ids = tokenizer(
...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
... ).input_ids  
>>> outputs = model.generate(input_ids)
>>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
>>> 
```

#### parallelize

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mt5/modeling_mt5.py#L1606)

( device\_map = None )

Parameters

-   **device\_map** (`Dict[int, list]`, optional, defaults to None) — A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always automatically mapped to the first device (for esoteric reasons). That means that the first device should have fewer attention modules mapped to it than other devices. For reference, the mt5 models have the following number of attention modules:
    
    -   mt5-small: 6
    -   mt5-base: 12
    -   mt5-large: 24
    -   mt5-xl: 24
    -   mt5-xxl: 24
    

This is an experimental feature and is a subject to change at a moment’s notice.

Uses a device map to distribute attention modules of the model across several devices. If no device map is given, it will evenly distribute blocks across all devices.

Example:

```
model = MT5ForConditionalGeneration.from_pretrained("mt5-xl")
device_map = {
    0: [0, 1, 2],
    1: [3, 4, 5, 6, 7, 8, 9],
    2: [10, 11, 12, 13, 14, 15, 16],
    3: [17, 18, 19, 20, 21, 22, 23],
}
model.parallelize(device_map)
```

## MT5EncoderModel

### class transformers.MT5EncoderModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mt5/modeling_mt5.py#L1895)

( config: MT5Config )

Parameters

-   **config** ([MT5Config](/docs/transformers/v4.34.0/en/model_doc/mt5#transformers.MT5Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare MT5 Model transformer outputting encoder’s raw hidden-states without any specific head on top.

The MT5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It’s an encoder decoder transformer pre-trained in a text-to-text denoising generative setting.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

Examples:

```
>>> from transformers import MT5EncoderModel, AutoTokenizer

>>> model = MT5EncoderModel.from_pretrained("google/mt5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
>>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
>>> input_ids = tokenizer(article, return_tensors="pt").input_ids
>>> outputs = model(input_ids)
>>> hidden_state = outputs.last_hidden_state
```

Moves the model to cpu from a model parallel state.

Example:

```
model = MT5ForConditionalGeneration.from_pretrained("Mt5-xl")
device_map = {
    0: [0, 1, 2],
    1: [3, 4, 5, 6, 7, 8, 9],
    2: [10, 11, 12, 13, 14, 15, 16],
    3: [17, 18, 19, 20, 21, 22, 23],
}
model.parallelize(device_map)  
model.deparallelize()  
```

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mt5/modeling_mt5.py#L1985)

( input\_ids: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. MT5 is a model with relative position embeddings so you should be able to pad the inputs on both the right and the left.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for detail.
    
    To know more on how to prepare `input_ids` for pretraining take a look a [MT5 Training](./mt5#training).
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([MT5Config](/docs/transformers/v4.34.0/en/model_doc/mt5#transformers.MT5Config)) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [MT5EncoderModel](/docs/transformers/v4.34.0/en/model_doc/mt5#transformers.MT5EncoderModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, MT5EncoderModel

>>> tokenizer = AutoTokenizer.from_pretrained("mt5-small")
>>> model = MT5EncoderModel.from_pretrained("mt5-small")
>>> input_ids = tokenizer(
...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
... ).input_ids  
>>> outputs = model(input_ids=input_ids)
>>> last_hidden_states = outputs.last_hidden_state
```

#### parallelize

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mt5/modeling_mt5.py#L1931)

( device\_map = None )

Parameters

-   **device\_map** (`Dict[int, list]`, optional, defaults to None) — A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always automatically mapped to the first device (for esoteric reasons). That means that the first device should have fewer attention modules mapped to it than other devices. For reference, the mt5 models have the following number of attention modules:
    
    -   mt5-small: 6
    -   mt5-base: 12
    -   mt5-large: 24
    -   mt5-xl: 24
    -   mt5-xxl: 24
    

This is an experimental feature and is a subject to change at a moment’s notice.

Uses a device map to distribute attention modules of the model across several devices. If no device map is given, it will evenly distribute blocks across all devices.

Example:

```
model = MT5ForConditionalGeneration.from_pretrained("mt5-xl")
device_map = {
    0: [0, 1, 2],
    1: [3, 4, 5, 6, 7, 8, 9],
    2: [10, 11, 12, 13, 14, 15, 16],
    3: [17, 18, 19, 20, 21, 22, 23],
}
model.parallelize(device_map)
```

## MT5ForSequenceClassification

### class transformers.MT5ForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mt5/modeling_mt5.py#L2036)

( config: MT5Config )

Parameters

-   **config** ([MT5Config](/docs/transformers/v4.34.0/en/model_doc/mt5#transformers.MT5Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

MT5 model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

The MT5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It’s an encoder decoder transformer pre-trained in a text-to-text denoising generative setting.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mt5/modeling_mt5.py#L2051)

( input\_ids: LongTensor = None attention\_mask: typing.Optional\[torch.Tensor\] = None decoder\_input\_ids: typing.Optional\[torch.LongTensor\] = None decoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None decoder\_head\_mask: typing.Optional\[torch.Tensor\] = None cross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = None encoder\_outputs: typing.Optional\[typing.List\[torch.FloatTensor\]\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None decoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. MT5 is a model with relative position embeddings so you should be able to pad the inputs on both the right and the left.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for detail.
    
    [What are input IDs?](../glossary#input-ids)
    
    To know more on how to prepare `input_ids` for pretraining take a look a [MT5 Training](./mt5#training).
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, _optional_) — Indices of decoder input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are decoder input IDs?](../glossary#decoder-input-ids)
    
    MT5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
    
    To know more on how to prepare `decoder_input_ids` for pretraining take a look at [MT5 Training](./mt5#training).
    
-   **decoder\_attention\_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, _optional_) — Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also be used by default.
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **decoder\_head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **encoder\_outputs** (`tuple(tuple(torch.FloatTensor)`, _optional_) — Tuple consists of (`last_hidden_state`, `optional`: _hidden\_states_, `optional`: _attentions_) `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`) — Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
    
    If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **decoder\_inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be input (see `past_key_values`). This is useful if you want more control over how to convert `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
    
    If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value of `inputs_embeds`.
    
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

A [transformers.modeling\_outputs.Seq2SeqSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([MT5Config](/docs/transformers/v4.34.0/en/model_doc/mt5#transformers.MT5Config)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `label` is provided) — Classification (or regression if config.num\_labels==1) loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
    
-   **decoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
    
-   **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
    
-   **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
    
-   **encoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [MT5ForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/mt5#transformers.MT5ForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## MT5ForQuestionAnswering

### class transformers.MT5ForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mt5/modeling_mt5.py#L2171)

( config: MT5Config )

Parameters

-   **config** ([MT5Config](/docs/transformers/v4.34.0/en/model_doc/mt5#transformers.MT5Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

MT5 Model with a span classification head on top for extractive question-answering tasks like SQuAD (linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

The MT5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It’s an encoder decoder transformer pre-trained in a text-to-text denoising generative setting.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mt5/modeling_mt5.py#L2220)

( input\_ids: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None decoder\_input\_ids: typing.Optional\[torch.LongTensor\] = None decoder\_attention\_mask: typing.Optional\[torch.BoolTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None decoder\_head\_mask: typing.Optional\[torch.FloatTensor\] = None cross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = None encoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = None start\_positions: typing.Optional\[torch.LongTensor\] = None end\_positions: typing.Optional\[torch.LongTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None decoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqQuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. MT5 is a model with relative position embeddings so you should be able to pad the inputs on both the right and the left.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for detail.
    
    [What are input IDs?](../glossary#input-ids)
    
    To know more on how to prepare `input_ids` for pretraining take a look a [MT5 Training](./mt5#training).
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, _optional_) — Indices of decoder input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are decoder input IDs?](../glossary#decoder-input-ids)
    
    MT5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
    
    To know more on how to prepare `decoder_input_ids` for pretraining take a look at [MT5 Training](./mt5#training).
    
-   **decoder\_attention\_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, _optional_) — Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also be used by default.
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **decoder\_head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **encoder\_outputs** (`tuple(tuple(torch.FloatTensor)`, _optional_) — Tuple consists of (`last_hidden_state`, `optional`: _hidden\_states_, `optional`: _attentions_) `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`) — Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
    
    If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **decoder\_inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be input (see `past_key_values`). This is useful if you want more control over how to convert `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
    
    If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value of `inputs_embeds`.
    
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **start\_positions** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for position (index) of the start of the labelled span for computing the token classification loss. Positions are clamped to the length of the sequence (_sequence\_length_). Position outside of the sequence are not taken into account for computing the loss.
-   **end\_positions** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for position (index) of the end of the labelled span for computing the token classification loss. Positions are clamped to the length of the sequence (_sequence\_length_). Position outside of the sequence are not taken into account for computing the loss.

A [transformers.modeling\_outputs.Seq2SeqQuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([MT5Config](/docs/transformers/v4.34.0/en/model_doc/mt5#transformers.MT5Config)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
    
-   **start\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-start scores (before SoftMax).
    
-   **end\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-end scores (before SoftMax).
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
    
-   **decoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
    
-   **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
    
-   **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
    
-   **encoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [MT5ForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/mt5#transformers.MT5ForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## TFMT5Model

### class transformers.TFMT5Model

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mt5/modeling_tf_mt5.py#L27)

( \*args \*\*kwargs )

This class overrides [TFT5Model](/docs/transformers/v4.34.0/en/model_doc/t5#transformers.TFT5Model). Please check the superclass for the appropriate documentation alongside usage examples.

Examples:

```
>>> from transformers import TFMT5Model, AutoTokenizer

>>> model = TFMT5Model.from_pretrained("google/mt5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
>>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
>>> summary = "Weiter Verhandlung in Syrien."
>>> inputs = tokenizer(article, return_tensors="tf")
>>> labels = tokenizer(text_target=summary, return_tensors="tf")

>>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])
>>> hidden_states = outputs.last_hidden_state
```

## TFMT5ForConditionalGeneration

### class transformers.TFMT5ForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mt5/modeling_tf_mt5.py#L51)

( \*args \*\*kwargs )

This class overrides [TFT5ForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/t5#transformers.TFT5ForConditionalGeneration). Please check the superclass for the appropriate documentation alongside usage examples.

Examples:

```
>>> from transformers import TFMT5ForConditionalGeneration, AutoTokenizer

>>> model = TFMT5ForConditionalGeneration.from_pretrained("google/mt5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
>>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
>>> summary = "Weiter Verhandlung in Syrien."
>>> inputs = tokenizer(article, text_target=summary, return_tensors="tf")

>>> outputs = model(**inputs)
>>> loss = outputs.loss
```

## TFMT5EncoderModel

### class transformers.TFMT5EncoderModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mt5/modeling_tf_mt5.py#L75)

( \*args \*\*kwargs )

This class overrides [TFT5EncoderModel](/docs/transformers/v4.34.0/en/model_doc/t5#transformers.TFT5EncoderModel). Please check the superclass for the appropriate documentation alongside usage examples.

Examples:

```
>>> from transformers import TFMT5EncoderModel, AutoTokenizer

>>> model = TFMT5EncoderModel.from_pretrained("google/mt5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
>>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
>>> input_ids = tokenizer(article, return_tensors="tf").input_ids
>>> outputs = model(input_ids)
>>> hidden_state = outputs.last_hidden_state
```

## FlaxMT5Model

### class transformers.FlaxMT5Model

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mt5/modeling_flax_mt5.py#L42)

( config: T5Config input\_shape: typing.Tuple\[int\] = (1, 1) seed: int = 0 dtype: dtype = <class 'jax.numpy.float32'> \_do\_init: bool = True gradient\_checkpointing: bool = False \*\*kwargs )

This class overrides [FlaxT5Model](/docs/transformers/v4.34.0/en/model_doc/t5#transformers.FlaxT5Model). Please check the superclass for the appropriate documentation alongside usage examples.

Examples:

```
>>> from transformers import FlaxMT5Model, AutoTokenizer

>>> model = FlaxMT5Model.from_pretrained("google/mt5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

>>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
>>> summary = "Weiter Verhandlung in Syrien."
>>> inputs = tokenizer(article, return_tensors="np")

>>> decoder_input_ids = tokenizer(text_target=summary, return_tensors="np").input_ids

>>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=decoder_input_ids)
>>> hidden_states = outputs.last_hidden_state
```

## FlaxMT5ForConditionalGeneration

### class transformers.FlaxMT5ForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mt5/modeling_flax_mt5.py#L94)

( config: T5Config input\_shape: typing.Tuple\[int\] = (1, 1) seed: int = 0 dtype: dtype = <class 'jax.numpy.float32'> \_do\_init: bool = True gradient\_checkpointing: bool = False \*\*kwargs )

This class overrides [FlaxT5ForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/t5#transformers.FlaxT5ForConditionalGeneration). Please check the superclass for the appropriate documentation alongside usage examples.

Examples:

```
>>> from transformers import FlaxMT5ForConditionalGeneration, AutoTokenizer

>>> model = FlaxMT5ForConditionalGeneration.from_pretrained("google/mt5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

>>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
>>> summary = "Weiter Verhandlung in Syrien."
>>> inputs = tokenizer(article, return_tensors="np")

>>> decoder_input_ids = tokenizer(text_target=summary, return_tensors="np").input_ids

>>> outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
>>> logits = outputs.logits
```

## FlaxMT5EncoderModel

### class transformers.FlaxMT5EncoderModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mt5/modeling_flax_mt5.py#L68)

( config: T5Config input\_shape: typing.Tuple\[int\] = (1, 1) seed: int = 0 dtype: dtype = <class 'jax.numpy.float32'> \_do\_init: bool = True gradient\_checkpointing: bool = False \*\*kwargs )

This class overrides [FlaxT5EncoderModel](/docs/transformers/v4.34.0/en/model_doc/t5#transformers.FlaxT5EncoderModel). Please check the superclass for the appropriate documentation alongside usage examples.

Examples:

```
>>> from transformers import FlaxT5EncoderModel, AutoTokenizer

>>> model = FlaxT5EncoderModel.from_pretrained("google/mt5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

>>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
>>> summary = "Weiter Verhandlung in Syrien."
>>> inputs = tokenizer(article, return_tensors="np")

>>> decoder_input_ids = tokenizer(text_target=summary, return_tensors="np").input_ids

>>> outputs = model(input_ids=inputs["input_ids"])
>>> hidden_states = outputs.last_hidden_state
```