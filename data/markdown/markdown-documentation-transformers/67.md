# Data Collator

Data collators are objects that will form a batch by using a list of dataset elements as input. These elements are of the same type as the elements of `train_dataset` or `eval_dataset`.

To be able to build batches, data collators may apply some processing (like padding). Some of them (like [DataCollatorForLanguageModeling](/docs/transformers/v4.34.0/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling)) also apply some random data augmentation (like random masking) on the formed batch.

Examples of use can be found in the [example scripts](../examples) or [example notebooks](../notebooks).

## Default data collator

#### transformers.default\_data\_collator

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/data_collator.py#L52)

( features: typing.List\[InputDataClass\] return\_tensors = 'pt' )

Very simple data collator that simply collates batches of dict-like objects and performs special handling for potential keys named:

-   `label`: handles a single value (int or float) per object
-   `label_ids`: handles a list of values per object

Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs to the model. See glue and ner for example of how it’s useful.

## DefaultDataCollator

### class transformers.DefaultDataCollator

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/data_collator.py#L78)

( return\_tensors: str = 'pt' )

Parameters

-   **return\_tensors** (`str`) — The type of Tensor to return. Allowable values are “np”, “pt” and “tf”.

Very simple data collator that simply collates batches of dict-like objects and performs special handling for potential keys named:

-   `label`: handles a single value (int or float) per object
-   `label_ids`: handles a list of values per object

Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs to the model. See glue and ner for example of how it’s useful.

This is an object (like other data collators) rather than a pure function like default\_data\_collator. This can be helpful if you need to set a return\_tensors value at initialization.

## DataCollatorWithPadding

### class transformers.DataCollatorWithPadding

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/data_collator.py#L215)

( tokenizer: PreTrainedTokenizerBase padding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = True max\_length: typing.Optional\[int\] = None pad\_to\_multiple\_of: typing.Optional\[int\] = None return\_tensors: str = 'pt' )

Parameters

-   **tokenizer** ([PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) or [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast)) — The tokenizer used for encoding the data.
-   **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.utils.PaddingStrategy), _optional_, defaults to `True`) — Select a strategy to pad the returned sequences (according to the model’s padding side and padding index) among:
    
    -   `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single sequence is provided).
    -   `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided.
    -   `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
    
-   **max\_length** (`int`, _optional_) — Maximum length of the returned list and optionally padding length (see above).
-   **pad\_to\_multiple\_of** (`int`, _optional_) — If set will pad the sequence to a multiple of the provided value.
    
    This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
    
-   **return\_tensors** (`str`) — The type of Tensor to return. Allowable values are “np”, “pt” and “tf”.

Data collator that will dynamically pad the inputs received.

## DataCollatorForTokenClassification

### class transformers.DataCollatorForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/data_collator.py#L266)

( tokenizer: PreTrainedTokenizerBase padding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = True max\_length: typing.Optional\[int\] = None pad\_to\_multiple\_of: typing.Optional\[int\] = None label\_pad\_token\_id: int = -100 return\_tensors: str = 'pt' )

Parameters

-   **tokenizer** ([PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) or [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast)) — The tokenizer used for encoding the data.
-   **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.utils.PaddingStrategy), _optional_, defaults to `True`) — Select a strategy to pad the returned sequences (according to the model’s padding side and padding index) among:
    
    -   `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single sequence is provided).
    -   `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided.
    -   `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
    
-   **max\_length** (`int`, _optional_) — Maximum length of the returned list and optionally padding length (see above).
-   **pad\_to\_multiple\_of** (`int`, _optional_) — If set will pad the sequence to a multiple of the provided value.
    
    This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
    
-   **label\_pad\_token\_id** (`int`, _optional_, defaults to -100) — The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
-   **return\_tensors** (`str`) — The type of Tensor to return. Allowable values are “np”, “pt” and “tf”.

Data collator that will dynamically pad the inputs received, as well as the labels.

## DataCollatorForSeq2Seq

### class transformers.DataCollatorForSeq2Seq

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/data_collator.py#L517)

( tokenizer: PreTrainedTokenizerBase model: typing.Optional\[typing.Any\] = None padding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = True max\_length: typing.Optional\[int\] = None pad\_to\_multiple\_of: typing.Optional\[int\] = None label\_pad\_token\_id: int = -100 return\_tensors: str = 'pt' )

Parameters

-   **tokenizer** ([PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) or [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast)) — The tokenizer used for encoding the data.
-   **model** ([PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel)) — The model that is being trained. If set and has the _prepare\_decoder\_input\_ids\_from\_labels_, use it to prepare the _decoder\_input\_ids_
    
    This is useful when using _label\_smoothing_ to avoid calculating loss twice.
    
-   **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.utils.PaddingStrategy), _optional_, defaults to `True`) — Select a strategy to pad the returned sequences (according to the model’s padding side and padding index) among:
    
    -   `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single sequence is provided).
    -   `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided.
    -   `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
    
-   **max\_length** (`int`, _optional_) — Maximum length of the returned list and optionally padding length (see above).
-   **pad\_to\_multiple\_of** (`int`, _optional_) — If set will pad the sequence to a multiple of the provided value.
    
    This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
    
-   **label\_pad\_token\_id** (`int`, _optional_, defaults to -100) — The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
-   **return\_tensors** (`str`) — The type of Tensor to return. Allowable values are “np”, “pt” and “tf”.

Data collator that will dynamically pad the inputs received, as well as the labels.

## DataCollatorForLanguageModeling

### class transformers.DataCollatorForLanguageModeling

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/data_collator.py#L607)

( tokenizer: PreTrainedTokenizerBase mlm: bool = True mlm\_probability: float = 0.15 pad\_to\_multiple\_of: typing.Optional\[int\] = None tf\_experimental\_compile: bool = False return\_tensors: str = 'pt' )

Parameters

-   **tokenizer** ([PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) or [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast)) — The tokenizer used for encoding the data.
-   **mlm** (`bool`, _optional_, defaults to `True`) — Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked tokens and the value to predict for the masked token.
-   **mlm\_probability** (`float`, _optional_, defaults to 0.15) — The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
-   **pad\_to\_multiple\_of** (`int`, _optional_) — If set will pad the sequence to a multiple of the provided value.
-   **return\_tensors** (`str`) — The type of Tensor to return. Allowable values are “np”, “pt” and “tf”.

Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they are not all of the same length.

For best performance, this data collator should be used with a dataset having items that are dictionaries or BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) or a [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) with the argument `return_special_tokens_mask=True`.

#### numpy\_mask\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/data_collator.py#L806)

( inputs: typing.Any special\_tokens\_mask: typing.Optional\[typing.Any\] = None )

Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.

#### tf\_mask\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/data_collator.py#L659)

( inputs: typing.Any vocab\_size mask\_token\_id special\_tokens\_mask: typing.Optional\[typing.Any\] = None )

Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.

#### torch\_mask\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/data_collator.py#L751)

( inputs: typing.Any special\_tokens\_mask: typing.Optional\[typing.Any\] = None )

Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.

## DataCollatorForWholeWordMask

### class transformers.DataCollatorForWholeWordMask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/data_collator.py#L845)

( tokenizer: PreTrainedTokenizerBase mlm: bool = True mlm\_probability: float = 0.15 pad\_to\_multiple\_of: typing.Optional\[int\] = None tf\_experimental\_compile: bool = False return\_tensors: str = 'pt' )

Data collator used for language modeling that masks entire words.

-   collates batches of tensors, honoring their tokenizer’s pad\_token
-   preprocesses batches for masked language modeling

This collator relies on details of the implementation of subword tokenization by [BertTokenizer](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizer), specifically that subword tokens are prefixed with _##_. For tokenizers that do not adhere to this scheme, this collator will produce an output that is roughly equivalent to `.DataCollatorForLanguageModeling`.

#### numpy\_mask\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/data_collator.py#L1075)

( inputs: typing.Any mask\_labels: typing.Any )

Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set ‘mask\_labels’ means we use whole word mask (wwm), we directly mask idxs according to it’s ref.

#### tf\_mask\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/data_collator.py#L1033)

( inputs: typing.Any mask\_labels: typing.Any )

Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set ‘mask\_labels’ means we use whole word mask (wwm), we directly mask idxs according to it’s ref.

#### torch\_mask\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/data_collator.py#L993)

( inputs: typing.Any mask\_labels: typing.Any )

Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set ‘mask\_labels’ means we use whole word mask (wwm), we directly mask idxs according to it’s ref.

## DataCollatorForPermutationLanguageModeling

### class transformers.DataCollatorForPermutationLanguageModeling

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/data_collator.py#L1200)

( tokenizer: PreTrainedTokenizerBase plm\_probability: float = 0.16666666666666666 max\_span\_length: int = 5 return\_tensors: str = 'pt' )

Data collator used for permutation language modeling.

-   collates batches of tensors, honoring their tokenizer’s pad\_token
-   preprocesses batches for permutation language modeling with procedures specific to XLNet

#### numpy\_mask\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/data_collator.py#L1440)

( inputs: typing.Any )

The masked tokens to be predicted for a particular sequence are determined by the following algorithm:

0.  Start from the beginning of the sequence by setting `cur_len = 0` (number of tokens processed so far).
1.  Sample a `span_length` from the interval `[1, max_span_length]` (length of span of tokens to be masked)
2.  Reserve a context of length `context_length = span_length / plm_probability` to surround span to be masked
3.  Sample a starting point `start_index` from the interval `[cur_len, cur_len + context_length - span_length]` and mask tokens `start_index:start_index + span_length`
4.  Set `cur_len = cur_len + context_length`. If `cur_len < max_len` (i.e. there are tokens remaining in the sequence to be processed), repeat from Step 1.

The masked tokens to be predicted for a particular sequence are determined by the following algorithm:

0.  Start from the beginning of the sequence by setting `cur_len = 0` (number of tokens processed so far).
1.  Sample a `span_length` from the interval `[1, max_span_length]` (length of span of tokens to be masked)
2.  Reserve a context of length `context_length = span_length / plm_probability` to surround span to be masked
3.  Sample a starting point `start_index` from the interval `[cur_len, cur_len + context_length - span_length]` and mask tokens `start_index:start_index + span_length`
4.  Set `cur_len = cur_len + context_length`. If `cur_len < max_len` (i.e. there are tokens remaining in the sequence to be processed), repeat from Step 1.

#### torch\_mask\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/data_collator.py#L1234)

( inputs: typing.Any )

The masked tokens to be predicted for a particular sequence are determined by the following algorithm:

0.  Start from the beginning of the sequence by setting `cur_len = 0` (number of tokens processed so far).
1.  Sample a `span_length` from the interval `[1, max_span_length]` (length of span of tokens to be masked)
2.  Reserve a context of length `context_length = span_length / plm_probability` to surround span to be masked
3.  Sample a starting point `start_index` from the interval `[cur_len, cur_len + context_length - span_length]` and mask tokens `start_index:start_index + span_length`
4.  Set `cur_len = cur_len + context_length`. If `cur_len < max_len` (i.e. there are tokens remaining in the sequence to be processed), repeat from Step 1.