# Utilities for Tokenizers

Most of those are only useful if you are studying the code of the tokenizers in the library.

### class transformers.PreTrainedTokenizerBase

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L1581)

( \*\*kwargs )

Parameters

-   **model\_max\_length** (`int`, _optional_) — The maximum length (in number of tokens) for the inputs to the transformer model. When the tokenizer is loaded with [from\_pretrained()](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.from_pretrained), this will be set to the value stored for the associated model in `max_model_input_sizes` (see above). If no value is provided, will default to VERY\_LARGE\_INTEGER (`int(1e30)`).
-   **padding\_side** (`str`, _optional_) — The side on which the model should have padding applied. Should be selected between \[‘right’, ‘left’\]. Default value is picked from the class attribute of the same name.
-   **truncation\_side** (`str`, _optional_) — The side on which the model should have truncation applied. Should be selected between \[‘right’, ‘left’\]. Default value is picked from the class attribute of the same name.
-   **chat\_template** (`str`, _optional_) — A Jinja template string that will be used to format lists of chat messages. See [https://huggingface.co/docs/transformers/chat\_templating](https://huggingface.co/docs/transformers/chat_templating) for a full description.
-   **model\_input\_names** (`List[string]`, _optional_) — The list of inputs accepted by the forward pass of the model (like `"token_type_ids"` or `"attention_mask"`). Default value is picked from the class attribute of the same name.
-   **bos\_token** (`str` or `tokenizers.AddedToken`, _optional_) — A special token representing the beginning of a sentence. Will be associated to `self.bos_token` and `self.bos_token_id`.
-   **eos\_token** (`str` or `tokenizers.AddedToken`, _optional_) — A special token representing the end of a sentence. Will be associated to `self.eos_token` and `self.eos_token_id`.
-   **unk\_token** (`str` or `tokenizers.AddedToken`, _optional_) — A special token representing an out-of-vocabulary token. Will be associated to `self.unk_token` and `self.unk_token_id`.
-   **sep\_token** (`str` or `tokenizers.AddedToken`, _optional_) — A special token separating two different sentences in the same input (used by BERT for instance). Will be associated to `self.sep_token` and `self.sep_token_id`.
-   **pad\_token** (`str` or `tokenizers.AddedToken`, _optional_) — A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by attention mechanisms or loss computation. Will be associated to `self.pad_token` and `self.pad_token_id`.
-   **cls\_token** (`str` or `tokenizers.AddedToken`, _optional_) — A special token representing the class of the input (used by BERT for instance). Will be associated to `self.cls_token` and `self.cls_token_id`.
-   **mask\_token** (`str` or `tokenizers.AddedToken`, _optional_) — A special token representing a masked token (used by masked-language modeling pretraining objectives, like BERT). Will be associated to `self.mask_token` and `self.mask_token_id`.
-   **additional\_special\_tokens** (tuple or list of `str` or `tokenizers.AddedToken`, _optional_) — A tuple or a list of additional special tokens. Add them here to ensure they are skipped when decoding with `skip_special_tokens` is set to True. If they are not part of the vocabulary, they will be added at the end of the vocabulary.
-   **clean\_up\_tokenization\_spaces** (`bool`, _optional_, defaults to `True`) — Whether or not the model should cleanup the spaces that were added when splitting the input text during the tokenization process.
-   **split\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the special tokens should be split during the tokenization process. The default behavior is to not split special tokens. This means that if `<s>` is the `bos_token`, then `tokenizer.tokenize("<s>") = ['<s>`\]. Otherwise, if `split_special_tokens=True`, then `tokenizer.tokenize("<s>")` will be give `['<', 's', '>']`. This argument is only supported for `slow` tokenizers for the moment.

Base class for [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) and [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast).

Handles shared (mostly boiler plate) methods for those two classes.

Class attributes (overridden by derived classes)

-   **vocab\_files\_names** (`Dict[str, str]`) — A dictionary with, as keys, the `__init__` keyword name of each vocabulary file required by the model, and as associated values, the filename for saving the associated file (string).
-   **pretrained\_vocab\_files\_map** (`Dict[str, Dict[str, str]]`) — A dictionary of dictionaries, with the high-level keys being the `__init__` keyword name of each vocabulary file required by the model, the low-level being the `short-cut-names` of the pretrained models with, as associated values, the `url` to the associated pretrained vocabulary file.
-   **max\_model\_input\_sizes** (`Dict[str, Optional[int]]`) — A dictionary with, as keys, the `short-cut-names` of the pretrained models, and as associated values, the maximum length of the sequence inputs of this model, or `None` if the model has no maximum input size.
-   **pretrained\_init\_configuration** (`Dict[str, Dict[str, Any]]`) — A dictionary with, as keys, the `short-cut-names` of the pretrained models, and as associated values, a dictionary of specific arguments to pass to the `__init__` method of the tokenizer class for this pretrained model when loading the tokenizer with the [from\_pretrained()](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.from_pretrained) method.
-   **model\_input\_names** (`List[str]`) — A list of inputs expected in the forward pass of the model.
-   **padding\_side** (`str`) — The default value for the side on which the model should have padding applied. Should be `'right'` or `'left'`.
-   **truncation\_side** (`str`) — The default value for the side on which the model should have truncation applied. Should be `'right'` or `'left'`.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L2732)

( text: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\]\] = None text\_pair: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\], NoneType\] = None text\_target: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\]\] = None text\_pair\_target: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\], NoneType\] = None add\_special\_tokens: bool = True padding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = False truncation: typing.Union\[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy\] = None max\_length: typing.Optional\[int\] = None stride: int = 0 is\_split\_into\_words: bool = False pad\_to\_multiple\_of: typing.Optional\[int\] = None return\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None return\_token\_type\_ids: typing.Optional\[bool\] = None return\_attention\_mask: typing.Optional\[bool\] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True \*\*kwargs ) → [BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

-   **text** (`str`, `List[str]`, `List[List[str]]`, _optional_) — The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
-   **text\_pair** (`str`, `List[str]`, `List[List[str]]`, _optional_) — The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
-   **text\_target** (`str`, `List[str]`, `List[List[str]]`, _optional_) — The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
-   **text\_pair\_target** (`str`, `List[str]`, `List[List[str]]`, _optional_) — The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
-   **add\_special\_tokens** (`bool`, _optional_, defaults to `True`) — Whether or not to add special tokens when encoding the sequences. This will use the underlying `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are automatically added to the input ids. This is usefull if you want to add `bos` or `eos` tokens automatically.
-   **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.utils.PaddingStrategy), _optional_, defaults to `False`) — Activates and controls padding. Accepts the following values:
    
    -   `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
    -   `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided.
    -   `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).
    
-   **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), _optional_, defaults to `False`) — Activates and controls truncation. Accepts the following values:
    
    -   `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will truncate token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is provided.
    -   `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
    -   `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
    -   `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater than the model maximum admissible input size).
    
-   **max\_length** (`int`, _optional_) — Controls the maximum length to use by one of the truncation/padding parameters.
    
    If left unset or set to `None`, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.
    
-   **stride** (`int`, _optional_, defaults to 0) — If set to a number along with `max_length`, the overflowing tokens returned when `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence returned to provide some overlap between truncated and overflowing sequences. The value of this argument defines the number of overlapping tokens.
-   **is\_split\_into\_words** (`bool`, _optional_, defaults to `False`) — Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace) which it will tokenize. This is useful for NER or token classification.
-   **pad\_to\_multiple\_of** (`int`, _optional_) — If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
-   **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.TensorType), _optional_) — If set, will return tensors instead of list of python integers. Acceptable values are:
    
    -   `'tf'`: Return TensorFlow `tf.constant` objects.
    -   `'pt'`: Return PyTorch `torch.Tensor` objects.
    -   `'np'`: Return Numpy `np.ndarray` objects.
    
-   **return\_token\_type\_ids** (`bool`, _optional_) — Whether to return token type IDs. If left to the default, will return the token type IDs according to the specific tokenizer’s default, defined by the `return_outputs` attribute.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **return\_attention\_mask** (`bool`, _optional_) — Whether to return the attention mask. If left to the default, will return the attention mask according to the specific tokenizer’s default, defined by the `return_outputs` attribute.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **return\_overflowing\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead of returning overflowing tokens.
-   **return\_special\_tokens\_mask** (`bool`, _optional_, defaults to `False`) — Whether or not to return special tokens mask information.
-   **return\_offsets\_mapping** (`bool`, _optional_, defaults to `False`) — Whether or not to return `(char_start, char_end)` for each token.
    
    This is only available on fast tokenizers inheriting from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast), if using Python’s tokenizer, this method will raise `NotImplementedError`.
    
-   **return\_length** (`bool`, _optional_, defaults to `False`) — Whether or not to return the lengths of the encoded inputs.
-   **verbose** (`bool`, _optional_, defaults to `True`) — Whether or not to print more information and warnings. \*\*kwargs — passed to the `self.tokenize()` method

A [BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

-   **input\_ids** — List of token ids to be fed to a model.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **token\_type\_ids** — List of token type ids to be fed to a model (when `return_token_type_ids=True` or if _“token\_type\_ids”_ is in `self.model_input_names`).
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **attention\_mask** — List of indices specifying which tokens should be attended to by the model (when `return_attention_mask=True` or if _“attention\_mask”_ is in `self.model_input_names`).
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **overflowing\_tokens** — List of overflowing tokens sequences (when a `max_length` is specified and `return_overflowing_tokens=True`).
    
-   **num\_truncated\_tokens** — Number of tokens truncated (when a `max_length` is specified and `return_overflowing_tokens=True`).
    
-   **special\_tokens\_mask** — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
    
-   **length** — The length of the inputs (when `return_length=True`)
    

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of sequences.

#### apply\_chat\_template

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L1717)

( conversation: typing.Union\[typing.List\[typing.Dict\[str, str\]\], ForwardRef('Conversation')\] chat\_template: typing.Optional\[str\] = None tokenize: bool = True padding: bool = False truncation: bool = False max\_length: typing.Optional\[int\] = None return\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None \*\*tokenizer\_kwargs ) → `List[int]`

Parameters

-   **conversation** (Union\[List\[Dict\[str, str\]\], “Conversation”\]) — A Conversation object or list of dicts with “role” and “content” keys, representing the chat history so far.
-   **chat\_template** (str, _optional_) — A Jinja template to use for this conversion. If this is not passed, the model’s default chat template will be used instead.
-   **tokenize** (`bool`, defaults to `True`) — Whether to tokenize the output. If `False`, the output will be a string.
-   **padding** (`bool`, defaults to `False`) — Whether to pad sequences to the maximum length. Has no effect if tokenize is `False`.
-   **truncation** (`bool`, defaults to `False`) — Whether to truncate sequences at the maximum length. Has no effect if tokenize is `False`.
-   **max\_length** (`int`, _optional_) — Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is `False`. If not specified, the tokenizer’s `max_length` attribute will be used as a default.
-   **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.TensorType), _optional_) — If set, will return tensors of a particular framework. Has no effect if tokenize is `False`. Acceptable values are:
    
    -   `'tf'`: Return TensorFlow `tf.Tensor` objects.
    -   `'pt'`: Return PyTorch `torch.Tensor` objects.
    -   `'np'`: Return NumPy `np.ndarray` objects.
    -   `'jax'`: Return JAX `jnp.ndarray` objects. \*\*tokenizer\_kwargs — Additional kwargs to pass to the tokenizer.
    

A list of token ids representing the tokenized chat so far, including control tokens. This output is ready to pass to the model, either directly or via methods like `generate()`.

Converts a Conversation object or a list of dictionaries with `"role"` and `"content"` keys to a list of token ids. This method is intended for use with chat models, and will read the tokenizer’s chat\_template attribute to determine the format and control tokens to use when converting. When chat\_template is None, it will fall back to the default\_chat\_template specified at the class level.

Temporarily sets the tokenizer for encoding the targets. Useful for tokenizer associated to sequence-to-sequence models that need a slightly different processing for the labels.

#### batch\_decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3690)

( sequences: typing.Union\[typing.List\[int\], typing.List\[typing.List\[int\]\], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')\] skip\_special\_tokens: bool = False clean\_up\_tokenization\_spaces: bool = None \*\*kwargs ) → `List[str]`

Parameters

-   **sequences** (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`) — List of tokenized input ids. Can be obtained using the `__call__` method.
-   **skip\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not to remove special tokens in the decoding.
-   **clean\_up\_tokenization\_spaces** (`bool`, _optional_) — Whether or not to clean up the tokenization spaces. If `None`, will default to `self.clean_up_tokenization_spaces`.
-   **kwargs** (additional keyword arguments, _optional_) — Will be passed to the underlying model specific decode method.

The list of decoded sentences.

Convert a list of lists of token ids into a list of strings by calling decode.

#### batch\_encode\_plus

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3029)

( batch\_text\_or\_text\_pairs: typing.Union\[typing.List\[str\], typing.List\[typing.Tuple\[str, str\]\], typing.List\[typing.List\[str\]\], typing.List\[typing.Tuple\[typing.List\[str\], typing.List\[str\]\]\], typing.List\[typing.List\[int\]\], typing.List\[typing.Tuple\[typing.List\[int\], typing.List\[int\]\]\]\] add\_special\_tokens: bool = True padding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = False truncation: typing.Union\[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy\] = None max\_length: typing.Optional\[int\] = None stride: int = 0 is\_split\_into\_words: bool = False pad\_to\_multiple\_of: typing.Optional\[int\] = None return\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None return\_token\_type\_ids: typing.Optional\[bool\] = None return\_attention\_mask: typing.Optional\[bool\] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True \*\*kwargs ) → [BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

-   **batch\_text\_or\_text\_pairs** (`List[str]`, `List[Tuple[str, str]]`, `List[List[str]]`, `List[Tuple[List[str], List[str]]]`, and for not-fast tokenizers, also `List[List[int]]`, `List[Tuple[List[int], List[int]]]`) — Batch of sequences or pair of sequences to be encoded. This can be a list of string/string-sequences/int-sequences or a list of pair of string/string-sequences/int-sequence (see details in `encode_plus`).
-   **add\_special\_tokens** (`bool`, _optional_, defaults to `True`) — Whether or not to add special tokens when encoding the sequences. This will use the underlying `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are automatically added to the input ids. This is usefull if you want to add `bos` or `eos` tokens automatically.
-   **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.utils.PaddingStrategy), _optional_, defaults to `False`) — Activates and controls padding. Accepts the following values:
    
    -   `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
    -   `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided.
    -   `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).
    
-   **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), _optional_, defaults to `False`) — Activates and controls truncation. Accepts the following values:
    
    -   `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will truncate token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is provided.
    -   `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
    -   `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
    -   `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater than the model maximum admissible input size).
    
-   **max\_length** (`int`, _optional_) — Controls the maximum length to use by one of the truncation/padding parameters.
    
    If left unset or set to `None`, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.
    
-   **stride** (`int`, _optional_, defaults to 0) — If set to a number along with `max_length`, the overflowing tokens returned when `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence returned to provide some overlap between truncated and overflowing sequences. The value of this argument defines the number of overlapping tokens.
-   **is\_split\_into\_words** (`bool`, _optional_, defaults to `False`) — Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace) which it will tokenize. This is useful for NER or token classification.
-   **pad\_to\_multiple\_of** (`int`, _optional_) — If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
-   **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.TensorType), _optional_) — If set, will return tensors instead of list of python integers. Acceptable values are:
    
    -   `'tf'`: Return TensorFlow `tf.constant` objects.
    -   `'pt'`: Return PyTorch `torch.Tensor` objects.
    -   `'np'`: Return Numpy `np.ndarray` objects.
    
-   **return\_token\_type\_ids** (`bool`, _optional_) — Whether to return token type IDs. If left to the default, will return the token type IDs according to the specific tokenizer’s default, defined by the `return_outputs` attribute.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **return\_attention\_mask** (`bool`, _optional_) — Whether to return the attention mask. If left to the default, will return the attention mask according to the specific tokenizer’s default, defined by the `return_outputs` attribute.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **return\_overflowing\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead of returning overflowing tokens.
-   **return\_special\_tokens\_mask** (`bool`, _optional_, defaults to `False`) — Whether or not to return special tokens mask information.
-   **return\_offsets\_mapping** (`bool`, _optional_, defaults to `False`) — Whether or not to return `(char_start, char_end)` for each token.
    
    This is only available on fast tokenizers inheriting from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast), if using Python’s tokenizer, this method will raise `NotImplementedError`.
    
-   **return\_length** (`bool`, _optional_, defaults to `False`) — Whether or not to return the lengths of the encoded inputs.
-   **verbose** (`bool`, _optional_, defaults to `True`) — Whether or not to print more information and warnings. \*\*kwargs — passed to the `self.tokenize()` method

A [BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

-   **input\_ids** — List of token ids to be fed to a model.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **token\_type\_ids** — List of token type ids to be fed to a model (when `return_token_type_ids=True` or if _“token\_type\_ids”_ is in `self.model_input_names`).
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **attention\_mask** — List of indices specifying which tokens should be attended to by the model (when `return_attention_mask=True` or if _“attention\_mask”_ is in `self.model_input_names`).
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **overflowing\_tokens** — List of overflowing tokens sequences (when a `max_length` is specified and `return_overflowing_tokens=True`).
    
-   **num\_truncated\_tokens** — Number of tokens truncated (when a `max_length` is specified and `return_overflowing_tokens=True`).
    
-   **special\_tokens\_mask** — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
    
-   **length** — The length of the inputs (when `return_length=True`)
    

Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.

This method is deprecated, `__call__` should be used instead.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3325)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — The first tokenized sequence.
-   **token\_ids\_1** (`List[int]`, _optional_) — The second tokenized sequence.

The model input with special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens.

This implementation does not add special tokens and this method should be overridden in a subclass.

#### clean\_up\_tokenization

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3801)

( out\_string: str ) → `str`

Parameters

-   **out\_string** (`str`) — The text to clean up.

The cleaned-up string.

Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.

#### convert\_tokens\_to\_string

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3677)

( tokens: typing.List\[str\] ) → `str`

Parameters

-   **tokens** (`List[str]`) — The token to join in a string.

The joined tokens.

Converts a sequence of tokens in a single string. The most simple way to do it is `" ".join(tokens)` but we often want to remove sub-word tokenization artifacts at the same time.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3305)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — The first tokenized sequence.
-   **token\_ids\_1** (`List[int]`, _optional_) — The second tokenized sequence.

The token type ids.

Create the token type IDs corresponding to the sequences passed. [What are token type IDs?](../glossary#token-type-ids)

Should be overridden in a subclass if the model has a special way of building those.

#### decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3724)

( token\_ids: typing.Union\[int, typing.List\[int\], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')\] skip\_special\_tokens: bool = False clean\_up\_tokenization\_spaces: bool = None \*\*kwargs ) → `str`

Parameters

-   **token\_ids** (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`) — List of tokenized input ids. Can be obtained using the `__call__` method.
-   **skip\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not to remove special tokens in the decoding.
-   **clean\_up\_tokenization\_spaces** (`bool`, _optional_) — Whether or not to clean up the tokenization spaces. If `None`, will default to `self.clean_up_tokenization_spaces`.
-   **kwargs** (additional keyword arguments, _optional_) — Will be passed to the underlying model specific decode method.

The decoded sentence.

Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special tokens and clean up tokenization spaces.

Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

#### encode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L2540)

( text: typing.Union\[str, typing.List\[str\], typing.List\[int\]\] text\_pair: typing.Union\[str, typing.List\[str\], typing.List\[int\], NoneType\] = None add\_special\_tokens: bool = True padding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = False truncation: typing.Union\[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy\] = None max\_length: typing.Optional\[int\] = None stride: int = 0 return\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None \*\*kwargs ) → `List[int]`, `torch.Tensor`, `tf.Tensor` or `np.ndarray`

Parameters

-   **text** (`str`, `List[str]` or `List[int]`) — The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids` method).
-   **text\_pair** (`str`, `List[str]` or `List[int]`, _optional_) — Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids` method).
-   **add\_special\_tokens** (`bool`, _optional_, defaults to `True`) — Whether or not to add special tokens when encoding the sequences. This will use the underlying `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are automatically added to the input ids. This is usefull if you want to add `bos` or `eos` tokens automatically.
-   **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.utils.PaddingStrategy), _optional_, defaults to `False`) — Activates and controls padding. Accepts the following values:
    
    -   `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
    -   `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided.
    -   `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).
    
-   **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), _optional_, defaults to `False`) — Activates and controls truncation. Accepts the following values:
    
    -   `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will truncate token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is provided.
    -   `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
    -   `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
    -   `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater than the model maximum admissible input size).
    
-   **max\_length** (`int`, _optional_) — Controls the maximum length to use by one of the truncation/padding parameters.
    
    If left unset or set to `None`, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.
    
-   **stride** (`int`, _optional_, defaults to 0) — If set to a number along with `max_length`, the overflowing tokens returned when `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence returned to provide some overlap between truncated and overflowing sequences. The value of this argument defines the number of overlapping tokens.
-   **is\_split\_into\_words** (`bool`, _optional_, defaults to `False`) — Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace) which it will tokenize. This is useful for NER or token classification.
-   **pad\_to\_multiple\_of** (`int`, _optional_) — If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
-   **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.TensorType), _optional_) — If set, will return tensors instead of list of python integers. Acceptable values are:
    
    -   `'tf'`: Return TensorFlow `tf.constant` objects.
    -   `'pt'`: Return PyTorch `torch.Tensor` objects.
    -   `'np'`: Return Numpy `np.ndarray` objects.
    
    \*\*kwargs — Passed along to the `.tokenize()` method.
    

Returns

`List[int]`, `torch.Tensor`, `tf.Tensor` or `np.ndarray`

The tokenized ids of the text.

Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

Same as doing `self.convert_tokens_to_ids(self.tokenize(text))`.

#### encode\_plus

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L2933)

( text: typing.Union\[str, typing.List\[str\], typing.List\[int\]\] text\_pair: typing.Union\[str, typing.List\[str\], typing.List\[int\], NoneType\] = None add\_special\_tokens: bool = True padding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = False truncation: typing.Union\[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy\] = None max\_length: typing.Optional\[int\] = None stride: int = 0 is\_split\_into\_words: bool = False pad\_to\_multiple\_of: typing.Optional\[int\] = None return\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None return\_token\_type\_ids: typing.Optional\[bool\] = None return\_attention\_mask: typing.Optional\[bool\] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True \*\*kwargs ) → [BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

-   **text** (`str`, `List[str]` or `List[int]` (the latter only for not-fast tokenizers)) — The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids` method).
-   **text\_pair** (`str`, `List[str]` or `List[int]`, _optional_) — Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids` method).
-   **add\_special\_tokens** (`bool`, _optional_, defaults to `True`) — Whether or not to add special tokens when encoding the sequences. This will use the underlying `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are automatically added to the input ids. This is usefull if you want to add `bos` or `eos` tokens automatically.
-   **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.utils.PaddingStrategy), _optional_, defaults to `False`) — Activates and controls padding. Accepts the following values:
    
    -   `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
    -   `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided.
    -   `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).
    
-   **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), _optional_, defaults to `False`) — Activates and controls truncation. Accepts the following values:
    
    -   `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will truncate token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is provided.
    -   `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
    -   `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
    -   `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater than the model maximum admissible input size).
    
-   **max\_length** (`int`, _optional_) — Controls the maximum length to use by one of the truncation/padding parameters.
    
    If left unset or set to `None`, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.
    
-   **stride** (`int`, _optional_, defaults to 0) — If set to a number along with `max_length`, the overflowing tokens returned when `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence returned to provide some overlap between truncated and overflowing sequences. The value of this argument defines the number of overlapping tokens.
-   **is\_split\_into\_words** (`bool`, _optional_, defaults to `False`) — Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace) which it will tokenize. This is useful for NER or token classification.
-   **pad\_to\_multiple\_of** (`int`, _optional_) — If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
-   **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.TensorType), _optional_) — If set, will return tensors instead of list of python integers. Acceptable values are:
    
    -   `'tf'`: Return TensorFlow `tf.constant` objects.
    -   `'pt'`: Return PyTorch `torch.Tensor` objects.
    -   `'np'`: Return Numpy `np.ndarray` objects.
    
-   **return\_token\_type\_ids** (`bool`, _optional_) — Whether to return token type IDs. If left to the default, will return the token type IDs according to the specific tokenizer’s default, defined by the `return_outputs` attribute.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **return\_attention\_mask** (`bool`, _optional_) — Whether to return the attention mask. If left to the default, will return the attention mask according to the specific tokenizer’s default, defined by the `return_outputs` attribute.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **return\_overflowing\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead of returning overflowing tokens.
-   **return\_special\_tokens\_mask** (`bool`, _optional_, defaults to `False`) — Whether or not to return special tokens mask information.
-   **return\_offsets\_mapping** (`bool`, _optional_, defaults to `False`) — Whether or not to return `(char_start, char_end)` for each token.
    
    This is only available on fast tokenizers inheriting from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast), if using Python’s tokenizer, this method will raise `NotImplementedError`.
    
-   **return\_length** (`bool`, _optional_, defaults to `False`) — Whether or not to return the lengths of the encoded inputs.
-   **verbose** (`bool`, _optional_, defaults to `True`) — Whether or not to print more information and warnings. \*\*kwargs — passed to the `self.tokenize()` method

A [BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

-   **input\_ids** — List of token ids to be fed to a model.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **token\_type\_ids** — List of token type ids to be fed to a model (when `return_token_type_ids=True` or if _“token\_type\_ids”_ is in `self.model_input_names`).
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **attention\_mask** — List of indices specifying which tokens should be attended to by the model (when `return_attention_mask=True` or if _“attention\_mask”_ is in `self.model_input_names`).
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **overflowing\_tokens** — List of overflowing tokens sequences (when a `max_length` is specified and `return_overflowing_tokens=True`).
    
-   **num\_truncated\_tokens** — Number of tokens truncated (when a `max_length` is specified and `return_overflowing_tokens=True`).
    
-   **special\_tokens\_mask** — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
    
-   **length** — The length of the inputs (when `return_length=True`)
    

Tokenize and prepare for the model a sequence or a pair of sequences.

This method is deprecated, `__call__` should be used instead.

#### from\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L1820)

( pretrained\_model\_name\_or\_path: typing.Union\[str, os.PathLike\] \*init\_inputs cache\_dir: typing.Union\[str, os.PathLike, NoneType\] = None force\_download: bool = False local\_files\_only: bool = False token: typing.Union\[bool, str, NoneType\] = None revision: str = 'main' \*\*kwargs )

Parameters

-   **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) — Can be either:
    
    -   A string, the _model id_ of a predefined tokenizer hosted inside a model repo on huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
    -   A path to a _directory_ containing vocabulary files required by the tokenizer, for instance saved using the [save\_pretrained()](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained) method, e.g., `./my_model_directory/`.
    -   (**Deprecated**, not applicable to all derived classes) A path or url to a single saved vocabulary file (if and only if the tokenizer only requires a single vocabulary file like Bert or XLNet), e.g., `./my_model_directory/vocab.txt`.
    
-   **cache\_dir** (`str` or `os.PathLike`, _optional_) — Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the standard cache should not be used.
-   **force\_download** (`bool`, _optional_, defaults to `False`) — Whether or not to force the (re-)download the vocabulary files and override the cached versions if they exist.
-   **resume\_download** (`bool`, _optional_, defaults to `False`) — Whether or not to delete incompletely received files. Attempt to resume the download if such a file exists.
-   **proxies** (`Dict[str, str]`, _optional_) — A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
-   **token** (`str` or _bool_, _optional_) — The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
-   **local\_files\_only** (`bool`, _optional_, defaults to `False`) — Whether or not to only rely on local files and not to attempt to download any files.
-   **revision** (`str`, _optional_, defaults to `"main"`) — The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git.
-   **subfolder** (`str`, _optional_) — In case the relevant files are located inside a subfolder of the model repo on huggingface.co (e.g. for facebook/rag-token-base), specify it here.
-   **inputs** (additional positional arguments, _optional_) — Will be passed along to the Tokenizer `__init__` method.
-   **kwargs** (additional keyword arguments, _optional_) — Will be passed to the Tokenizer `__init__` method. Can be used to set special tokens like `bos_token`, `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`, `mask_token`, `additional_special_tokens`. See parameters in the `__init__` for more details.

Instantiate a [PreTrainedTokenizerBase](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase) (or a derived class) from a predefined tokenizer.

Passing `token=True` is required when you want to use a private model.

Examples:

```

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-german-cased")


tokenizer = BertTokenizer.from_pretrained("./test/saved_model/")


tokenizer = BertTokenizer.from_pretrained("./test/saved_model/my_vocab.txt")


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", unk_token="<unk>")


assert tokenizer.unk_token == "<unk>"
```

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3770)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None already\_has\_special\_tokens: bool = False ) → A list of integers in the range \[0, 1\]

Parameters

-   **token\_ids\_0** (`List[int]`) — List of ids of the first sequence.
-   **token\_ids\_1** (`List[int]`, _optional_) — List of ids of the second sequence.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

Returns

A list of integers in the range \[0, 1\]

1 for a special token, 0 for a sequence token.

Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

Returns the vocabulary as a dictionary of token to index.

`tokenizer.get_vocab()[token]` is equivalent to `tokenizer.convert_tokens_to_ids(token)` when `token` is in the vocab.

#### pad

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3132)

( encoded\_inputs: typing.Union\[transformers.tokenization\_utils\_base.BatchEncoding, typing.List\[transformers.tokenization\_utils\_base.BatchEncoding\], typing.Dict\[str, typing.List\[int\]\], typing.Dict\[str, typing.List\[typing.List\[int\]\]\], typing.List\[typing.Dict\[str, typing.List\[int\]\]\]\] padding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = True max\_length: typing.Optional\[int\] = None pad\_to\_multiple\_of: typing.Optional\[int\] = None return\_attention\_mask: typing.Optional\[bool\] = None return\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None verbose: bool = True )

Parameters

-   **encoded\_inputs** ([BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding), list of [BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding), `Dict[str, List[int]]`, `Dict[str, List[List[int]]` or `List[Dict[str, List[int]]]`) — Tokenized inputs. Can represent one input ([BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding) or `Dict[str, List[int]]`) or a batch of tokenized inputs (list of [BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding), _Dict\[str, List\[List\[int\]\]\]_ or _List\[Dict\[str, List\[int\]\]\]_) so you can use this method during preprocessing as well as in a PyTorch Dataloader collate function.
    
    Instead of `List[int]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors), see the note above for the return type.
    
-   **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.utils.PaddingStrategy), _optional_, defaults to `True`) — Select a strategy to pad the returned sequences (according to the model’s padding side and padding index) among:
    
    -   `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
    -   `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided.
    -   `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).
    
-   **max\_length** (`int`, _optional_) — Maximum length of the returned list and optionally padding length (see above).
-   **pad\_to\_multiple\_of** (`int`, _optional_) — If set will pad the sequence to a multiple of the provided value.
    
    This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
    
-   **return\_attention\_mask** (`bool`, _optional_) — Whether to return the attention mask. If left to the default, will return the attention mask according to the specific tokenizer’s default, defined by the `return_outputs` attribute.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.TensorType), _optional_) — If set, will return tensors instead of list of python integers. Acceptable values are:
    
    -   `'tf'`: Return TensorFlow `tf.constant` objects.
    -   `'pt'`: Return PyTorch `torch.Tensor` objects.
    -   `'np'`: Return Numpy `np.ndarray` objects.
    
-   **verbose** (`bool`, _optional_, defaults to `True`) — Whether or not to print more information and warnings.

Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length in the batch.

Padding side (left/right) padding token ids are defined at the tokenizer level (with `self.padding_side`, `self.pad_token_id` and `self.pad_token_type_id`).

Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.

If the `encoded_inputs` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of PyTorch tensors, you will lose the specific device of your tensors however.

#### prepare\_for\_model

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3345)

( ids: typing.List\[int\] pair\_ids: typing.Optional\[typing.List\[int\]\] = None add\_special\_tokens: bool = True padding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = False truncation: typing.Union\[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy\] = None max\_length: typing.Optional\[int\] = None stride: int = 0 pad\_to\_multiple\_of: typing.Optional\[int\] = None return\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None return\_token\_type\_ids: typing.Optional\[bool\] = None return\_attention\_mask: typing.Optional\[bool\] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True prepend\_batch\_axis: bool = False \*\*kwargs ) → [BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

-   **ids** (`List[int]`) — Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and `convert_tokens_to_ids` methods.
-   **pair\_ids** (`List[int]`, _optional_) — Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize` and `convert_tokens_to_ids` methods.
-   **add\_special\_tokens** (`bool`, _optional_, defaults to `True`) — Whether or not to add special tokens when encoding the sequences. This will use the underlying `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are automatically added to the input ids. This is usefull if you want to add `bos` or `eos` tokens automatically.
-   **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.utils.PaddingStrategy), _optional_, defaults to `False`) — Activates and controls padding. Accepts the following values:
    
    -   `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
    -   `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided.
    -   `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).
    
-   **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), _optional_, defaults to `False`) — Activates and controls truncation. Accepts the following values:
    
    -   `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will truncate token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is provided.
    -   `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
    -   `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
    -   `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater than the model maximum admissible input size).
    
-   **max\_length** (`int`, _optional_) — Controls the maximum length to use by one of the truncation/padding parameters.
    
    If left unset or set to `None`, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.
    
-   **stride** (`int`, _optional_, defaults to 0) — If set to a number along with `max_length`, the overflowing tokens returned when `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence returned to provide some overlap between truncated and overflowing sequences. The value of this argument defines the number of overlapping tokens.
-   **is\_split\_into\_words** (`bool`, _optional_, defaults to `False`) — Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace) which it will tokenize. This is useful for NER or token classification.
-   **pad\_to\_multiple\_of** (`int`, _optional_) — If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
-   **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.TensorType), _optional_) — If set, will return tensors instead of list of python integers. Acceptable values are:
    
    -   `'tf'`: Return TensorFlow `tf.constant` objects.
    -   `'pt'`: Return PyTorch `torch.Tensor` objects.
    -   `'np'`: Return Numpy `np.ndarray` objects.
    
-   **return\_token\_type\_ids** (`bool`, _optional_) — Whether to return token type IDs. If left to the default, will return the token type IDs according to the specific tokenizer’s default, defined by the `return_outputs` attribute.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **return\_attention\_mask** (`bool`, _optional_) — Whether to return the attention mask. If left to the default, will return the attention mask according to the specific tokenizer’s default, defined by the `return_outputs` attribute.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **return\_overflowing\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead of returning overflowing tokens.
-   **return\_special\_tokens\_mask** (`bool`, _optional_, defaults to `False`) — Whether or not to return special tokens mask information.
-   **return\_offsets\_mapping** (`bool`, _optional_, defaults to `False`) — Whether or not to return `(char_start, char_end)` for each token.
    
    This is only available on fast tokenizers inheriting from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast), if using Python’s tokenizer, this method will raise `NotImplementedError`.
    
-   **return\_length** (`bool`, _optional_, defaults to `False`) — Whether or not to return the lengths of the encoded inputs.
-   **verbose** (`bool`, _optional_, defaults to `True`) — Whether or not to print more information and warnings. \*\*kwargs — passed to the `self.tokenize()` method

A [BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

-   **input\_ids** — List of token ids to be fed to a model.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **token\_type\_ids** — List of token type ids to be fed to a model (when `return_token_type_ids=True` or if _“token\_type\_ids”_ is in `self.model_input_names`).
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **attention\_mask** — List of indices specifying which tokens should be attended to by the model (when `return_attention_mask=True` or if _“attention\_mask”_ is in `self.model_input_names`).
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **overflowing\_tokens** — List of overflowing tokens sequences (when a `max_length` is specified and `return_overflowing_tokens=True`).
    
-   **num\_truncated\_tokens** — Number of tokens truncated (when a `max_length` is specified and `return_overflowing_tokens=True`).
    
-   **special\_tokens\_mask** — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
    
-   **length** — The length of the inputs (when `return_length=True`)
    

Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It adds special tokens, truncates sequences if overflowing while taking into account the special tokens and manages a moving window (with user defined stride) for overflowing tokens. Please Note, for _pair\_ids_ different than `None` and _truncation\_strategy = longest\_first_ or `True`, it is not possible to return overflowing tokens. Such a combination of arguments will raise an error.

#### prepare\_seq2seq\_batch

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3901)

( src\_texts: typing.List\[str\] tgt\_texts: typing.Optional\[typing.List\[str\]\] = None max\_length: typing.Optional\[int\] = None max\_target\_length: typing.Optional\[int\] = None padding: str = 'longest' return\_tensors: str = None truncation: bool = True \*\*kwargs ) → [BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

-   **src\_texts** (`List[str]`) — List of documents to summarize or source language texts.
-   **tgt\_texts** (`list`, _optional_) — List of summaries or target language texts.
-   **max\_length** (`int`, _optional_) — Controls the maximum length for encoder inputs (documents to summarize or source language texts) If left unset or set to `None`, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.
-   **max\_target\_length** (`int`, _optional_) — Controls the maximum length of decoder inputs (target language texts or summaries) If left unset or set to `None`, this will use the max\_length value.
-   **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.utils.PaddingStrategy), _optional_, defaults to `False`) — Activates and controls padding. Accepts the following values:
    
    -   `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
    -   `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided.
    -   `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).
    
-   **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.TensorType), _optional_) — If set, will return tensors instead of list of python integers. Acceptable values are:
    
    -   `'tf'`: Return TensorFlow `tf.constant` objects.
    -   `'pt'`: Return PyTorch `torch.Tensor` objects.
    -   `'np'`: Return Numpy `np.ndarray` objects.
    
-   **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), _optional_, defaults to `True`) — Activates and controls truncation. Accepts the following values:
    
    -   `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will truncate token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is provided.
    -   `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
    -   `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
    -   `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater than the model maximum admissible input size). \*\*kwargs — Additional keyword arguments passed along to `self.__call__`.
    

A [BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

-   **input\_ids** — List of token ids to be fed to the encoder.
-   **attention\_mask** — List of indices specifying which tokens should be attended to by the model.
-   **labels** — List of token ids for tgt\_texts.

The full set of keys `[input_ids, attention_mask, labels]`, will only be returned if tgt\_texts is passed. Otherwise, input\_ids, attention\_mask will be the only keys.

Prepare model inputs for translation. For best performance, translate one sentence at a time.

#### push\_to\_hub

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/hub.py#L786)

( repo\_id: str use\_temp\_dir: typing.Optional\[bool\] = None commit\_message: typing.Optional\[str\] = None private: typing.Optional\[bool\] = None token: typing.Union\[bool, str, NoneType\] = None max\_shard\_size: typing.Union\[int, str, NoneType\] = '10GB' create\_pr: bool = False safe\_serialization: bool = False revision: str = None \*\*deprecated\_kwargs )

Parameters

-   **repo\_id** (`str`) — The name of the repository you want to push your tokenizer to. It should contain your organization name when pushing to a given organization.
-   **use\_temp\_dir** (`bool`, _optional_) — Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub. Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
-   **commit\_message** (`str`, _optional_) — Message to commit while pushing. Will default to `"Upload tokenizer"`.
-   **private** (`bool`, _optional_) — Whether or not the repository created should be private.
-   **token** (`bool` or `str`, _optional_) — The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url` is not specified.
-   **max\_shard\_size** (`int` or `str`, _optional_, defaults to `"10GB"`) — Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
-   **create\_pr** (`bool`, _optional_, defaults to `False`) — Whether or not to create a PR with the uploaded files or directly commit.
-   **safe\_serialization** (`bool`, _optional_, defaults to `False`) — Whether or not to convert the model weights in safetensors format for safer serialization.
-   **revision** (`str`, _optional_) — Branch to push the uploaded files to.

Upload the tokenizer files to the 🤗 Model Hub.

Examples:

```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


tokenizer.push_to_hub("my-finetuned-bert")


tokenizer.push_to_hub("huggingface/my-finetuned-bert")
```

#### register\_for\_auto\_class

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3875)

( auto\_class = 'AutoTokenizer' )

Parameters

-   **auto\_class** (`str` or `type`, _optional_, defaults to `"AutoTokenizer"`) — The auto class to register this new tokenizer with.

Register this class with a given auto class. This should only be used for custom tokenizers as the ones in the library are already mapped with `AutoTokenizer`.

This API is experimental and may have some slight breaking changes in the next releases.

#### save\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L2315)

( save\_directory: typing.Union\[str, os.PathLike\] legacy\_format: typing.Optional\[bool\] = None filename\_prefix: typing.Optional\[str\] = None push\_to\_hub: bool = False \*\*kwargs ) → A tuple of `str`

Parameters

-   **save\_directory** (`str` or `os.PathLike`) — The path to a directory where the tokenizer will be saved.
-   **legacy\_format** (`bool`, _optional_) — Only applicable for a fast tokenizer. If unset (default), will save the tokenizer in the unified JSON format as well as in legacy format if it exists, i.e. with tokenizer specific vocabulary and a separate added\_tokens files.
    
    If `False`, will only save the tokenizer in the unified JSON format. This format is incompatible with “slow” tokenizers (not powered by the _tokenizers_ library), so the tokenizer will not be able to be loaded in the corresponding “slow” tokenizer.
    
    If `True`, will save the tokenizer in legacy format. If the “slow” tokenizer doesn’t exits, a value error is raised.
    
-   **filename\_prefix** (`str`, _optional_) — A prefix to add to the names of the files saved by the tokenizer.
-   **push\_to\_hub** (`bool`, _optional_, defaults to `False`) — Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with `repo_id` (will default to the name of `save_directory` in your namespace).
-   **kwargs** (`Dict[str, Any]`, _optional_) — Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/v4.34.0/en/main_classes/processors#transformers.ProcessorMixin.push_to_hub) method.

The files saved.

Save the full tokenizer state.

This method make sure the full tokenizer can then be re-loaded using the `~tokenization_utils_base.PreTrainedTokenizer.from_pretrained` class method..

Warning,None This won’t save modifications you may have applied to the tokenizer after the instantiation (for instance, modifying `tokenizer.do_lower_case` after creation).

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L2502)

( save\_directory: str filename\_prefix: typing.Optional\[str\] = None ) → `Tuple(str)`

Parameters

-   **save\_directory** (`str`) — The directory in which to save the vocabulary.
-   **filename\_prefix** (`str`, _optional_) — An optional prefix to add to the named of the saved files.

Paths to the files saved.

Save only the vocabulary of the tokenizer (vocabulary + added tokens).

This method won’t save the configuration and special token mappings of the tokenizer. Use `_save_pretrained()` to save the whole state of the tokenizer.

#### tokenize

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L2520)

( text: str pair: typing.Optional\[str\] = None add\_special\_tokens: bool = False \*\*kwargs ) → `List[str]`

Parameters

-   **text** (`str`) — The sequence to be encoded.
-   **pair** (`str`, _optional_) — A second sequence to be encoded with the first.
-   **add\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not to add the special tokens associated with the corresponding model.
-   **kwargs** (additional keyword arguments, _optional_) — Will be passed to the underlying model specific encode method. See details in [**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__)

The list of tokens.

Converts a string in a sequence of tokens, replacing unknown tokens with the `unk_token`.

#### truncate\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3481)

( ids: typing.List\[int\] pair\_ids: typing.Optional\[typing.List\[int\]\] = None num\_tokens\_to\_remove: int = 0 truncation\_strategy: typing.Union\[str, transformers.tokenization\_utils\_base.TruncationStrategy\] = 'longest\_first' stride: int = 0 ) → `Tuple[List[int], List[int], List[int]]`

Parameters

-   **ids** (`List[int]`) — Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and `convert_tokens_to_ids` methods.
-   **pair\_ids** (`List[int]`, _optional_) — Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize` and `convert_tokens_to_ids` methods.
-   **num\_tokens\_to\_remove** (`int`, _optional_, defaults to 0) — Number of tokens to remove using the truncation strategy.
-   **truncation\_strategy** (`str` or [TruncationStrategy](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), _optional_, defaults to `False`) — The strategy to follow for truncation. Can be:
    
    -   `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will truncate token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is provided.
    -   `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
    -   `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
    -   `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater than the model maximum admissible input size).
    
-   **stride** (`int`, _optional_, defaults to 0) — If set to a positive number, the overflowing tokens returned will contain some tokens from the main sequence returned. The value of this argument defines the number of additional tokens.

Returns

`Tuple[List[int], List[int], List[int]]`

The truncated `ids`, the truncated `pair_ids` and the list of overflowing tokens. Note: The _longest\_first_ strategy returns empty list of overflowing tokens if a pair of sequences (or a batch of pairs) is provided.

Truncates a sequence pair in-place following the strategy.

### class transformers.SpecialTokensMixin

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L795)

( verbose = True \*\*kwargs )

Parameters

-   **bos\_token** (`str` or `tokenizers.AddedToken`, _optional_) — A special token representing the beginning of a sentence.
-   **eos\_token** (`str` or `tokenizers.AddedToken`, _optional_) — A special token representing the end of a sentence.
-   **unk\_token** (`str` or `tokenizers.AddedToken`, _optional_) — A special token representing an out-of-vocabulary token.
-   **sep\_token** (`str` or `tokenizers.AddedToken`, _optional_) — A special token separating two different sentences in the same input (used by BERT for instance).
-   **pad\_token** (`str` or `tokenizers.AddedToken`, _optional_) — A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by attention mechanisms or loss computation.
-   **cls\_token** (`str` or `tokenizers.AddedToken`, _optional_) — A special token representing the class of the input (used by BERT for instance).
-   **mask\_token** (`str` or `tokenizers.AddedToken`, _optional_) — A special token representing a masked token (used by masked-language modeling pretraining objectives, like BERT).
-   **additional\_special\_tokens** (tuple or list of `str` or `tokenizers.AddedToken`, _optional_) — A tuple or a list of additional tokens, which will be marked as `special`, meaning that they will be skipped when decoding if `skip_special_tokens` is set to `True`.

A mixin derived by [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) and [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) to handle specific behaviors related to special tokens. In particular, this class hold the attributes which can be used to directly access these special tokens in a model-independent manner and allow to set and update the special tokens.

#### add\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L886)

( special\_tokens\_dict: typing.Dict\[str, typing.Union\[str, tokenizers.AddedToken\]\] replace\_additional\_special\_tokens = True ) → `int`

Parameters

-   **special\_tokens\_dict** (dictionary _str_ to _str_ or `tokenizers.AddedToken`) — Keys should be in the list of predefined special attributes: \[`bos_token`, `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`, `mask_token`, `additional_special_tokens`\].
    
    Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer assign the index of the `unk_token` to them).
    
-   **replace\_additional\_special\_tokens** (`bool`, _optional_,, defaults to `True`) — If `True`, the existing list of additional special tokens will be replaced by the list provided in `special_tokens_dict`. Otherwise, `self._additional_special_tokens` is just extended. In the former case, the tokens will NOT be removed from the tokenizer’s full vocabulary - they are only being flagged as non-special tokens. Remember, this only affects which tokens are skipped during decoding, not the `added_tokens_encoder` and `added_tokens_decoder`. This means that the previous `additional_special_tokens` are still added tokens, and will not be split by the model.

Number of tokens added to the vocabulary.

Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the current vocabulary).

When adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix of the model so that its embedding matrix matches the tokenizer.

In order to do that, please use the [resize\_token\_embeddings()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings) method.

Using `add_special_tokens` will ensure your special tokens can be used in several ways:

-   Special tokens can be skipped when decoding using `skip_special_tokens = True`.
-   Special tokens are carefully handled by the tokenizer (they are never split), similar to `AddedTokens`.
-   You can easily refer to special tokens using tokenizer class attributes like `tokenizer.cls_token`. This makes it easy to develop model-agnostic training and fine-tuning scripts.

When possible, special tokens are already registered for provided pretrained models (for instance [BertTokenizer](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizer) `cls_token` is already registered to be :obj_’\[CLS\]’_ and XLM’s one is also registered to be `'</s>'`).

Examples:

```
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

special_tokens_dict = {"cls_token": "<CLS>"}

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print("We have added", num_added_toks, "tokens")

model.resize_token_embeddings(len(tokenizer))

assert tokenizer.cls_token == "<CLS>"
```

#### add\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L987)

( new\_tokens: typing.Union\[str, tokenizers.AddedToken, typing.List\[typing.Union\[str, tokenizers.AddedToken\]\]\] special\_tokens: bool = False ) → `int`

Parameters

-   **new\_tokens** (`str`, `tokenizers.AddedToken` or a list of _str_ or `tokenizers.AddedToken`) — Tokens are only added if they are not already in the vocabulary. `tokenizers.AddedToken` wraps a string token to let you personalize its behavior: whether this token should only match against a single word, whether this token should strip all potential whitespaces on the left side, whether this token should strip all potential whitespaces on the right side, etc.
-   **special\_tokens** (`bool`, _optional_, defaults to `False`) — Can be used to specify if the token is a special token. This mostly change the normalization behavior (special tokens like CLS or \[MASK\] are usually not lower-cased for instance).
    
    See details for `tokenizers.AddedToken` in HuggingFace tokenizers library.
    

Number of tokens added to the vocabulary.

Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to it with indices starting from length of the current vocabulary and and will be isolated before the tokenization algorithm is applied. Added tokens and tokens from the vocabulary of the tokenization algorithm are therefore not treated in the same way.

Note, when adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix of the model so that its embedding matrix matches the tokenizer.

In order to do that, please use the [resize\_token\_embeddings()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings) method.

Examples:

```
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
print("We have added", num_added_toks, "tokens")

model.resize_token_embeddings(len(tokenizer))
```

The `sanitize_special_tokens` is now deprecated kept for backward compatibility and will be removed in transformers v5.

### class transformers.tokenization\_utils\_base.TruncationStrategy

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L138)

( value names = None module = None qualname = None type = None start = 1 )

Possible values for the `truncation` argument in [PreTrainedTokenizerBase.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__). Useful for tab-completion in an IDE.

### class transformers.CharSpan

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L150)

( start: int end: int )

Parameters

-   **start** (`int`) — Index of the first character in the original string.
-   **end** (`int`) — Index of the character following the last character in the original string.

Character span in the original string.

### class transformers.TokenSpan

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L163)

( start: int end: int )

Parameters

-   **start** (`int`) — Index of the first token in the span.
-   **end** (`int`) — Index of the token following the last token in the span.

Token span in an encoded string (list of tokens).