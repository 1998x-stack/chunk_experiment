A tokenizer is in charge of preparing the inputs for a model. The library contains tokenizers for all the models. Most of the tokenizers are available in two flavors: a full python implementation and a “Fast” implementation based on the Rust library [🤗 Tokenizers](https://github.com/huggingface/tokenizers). The “Fast” implementations allows:

1.  a significant speed-up in particular when doing batched tokenization and
2.  additional methods to map between the original string (character and words) and the token space (e.g. getting the index of the token comprising a given character or the span of characters corresponding to a given token).

The base classes [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) and [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) implement the common methods for encoding string inputs in model inputs (see below) and instantiating/saving python and “Fast” tokenizers either from a local file or directory or from a pretrained tokenizer provided by the library (downloaded from HuggingFace’s AWS S3 repository). They both rely on [PreTrainedTokenizerBase](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase) that contains the common methods, and [SpecialTokensMixin](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.SpecialTokensMixin).

[PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) and [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) thus implement the main methods for using all the tokenizers:

-   Tokenizing (splitting strings in sub-word token strings), converting tokens strings to ids and back, and encoding/decoding (i.e., tokenizing and converting to integers).
-   Adding new tokens to the vocabulary in a way that is independent of the underlying structure (BPE, SentencePiece…).
-   Managing special tokens (like mask, beginning-of-sentence, etc.): adding them, assigning them to attributes in the tokenizer for easy access and making sure they are not split during tokenization.

[BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding) holds the output of the [PreTrainedTokenizerBase](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)’s encoding methods (`__call__`, `encode_plus` and `batch_encode_plus`) and is derived from a Python dictionary. When the tokenizer is a pure python tokenizer, this class behaves just like a standard python dictionary and holds the various model inputs computed by these methods (`input_ids`, `attention_mask`…). When the tokenizer is a “Fast” tokenizer (i.e., backed by HuggingFace [tokenizers library](https://github.com/huggingface/tokenizers)), this class provides in addition several advanced alignment methods which can be used to map between the original string (character and words) and the token space (e.g., getting the index of the token comprising a given character or the span of characters corresponding to a given token).

# PreTrainedTokenizer

### class transformers.PreTrainedTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils.py#L336)

( \*\*kwargs )

Base class for all slow tokenizers.

Inherits from [PreTrainedTokenizerBase](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase).

Handle all the shared methods for tokenization and special tokens as well as methods downloading/caching/loading pretrained tokenizers as well as adding tokens to the vocabulary.

This class also contain the added tokens in a unified way on top of all tokenizers so we don’t have to handle the specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece…).

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

( text: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\]\] = Nonetext\_pair: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\], NoneType\] = Nonetext\_target: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\]\] = Nonetext\_pair\_target: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\], NoneType\] = Noneadd\_special\_tokens: bool = Truepadding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = Falsetruncation: typing.Union\[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy\] = Nonemax\_length: typing.Optional\[int\] = Nonestride: int = 0is\_split\_into\_words: bool = Falsepad\_to\_multiple\_of: typing.Optional\[int\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonereturn\_token\_type\_ids: typing.Optional\[bool\] = Nonereturn\_attention\_mask: typing.Optional\[bool\] = Nonereturn\_overflowing\_tokens: bool = Falsereturn\_special\_tokens\_mask: bool = Falsereturn\_offsets\_mapping: bool = Falsereturn\_length: bool = Falseverbose: bool = True\*\*kwargs ) → [BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding)

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of sequences.

#### apply\_chat\_template

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L1717)

( conversation: typing.Union\[typing.List\[typing.Dict\[str, str\]\], ForwardRef('Conversation')\]chat\_template: typing.Optional\[str\] = Nonetokenize: bool = Truepadding: bool = Falsetruncation: bool = Falsemax\_length: typing.Optional\[int\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None\*\*tokenizer\_kwargs ) → `List[int]`

Converts a Conversation object or a list of dictionaries with `"role"` and `"content"` keys to a list of token ids. This method is intended for use with chat models, and will read the tokenizer’s chat\_template attribute to determine the format and control tokens to use when converting. When chat\_template is None, it will fall back to the default\_chat\_template specified at the class level.

#### batch\_decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3690)

( sequences: typing.Union\[typing.List\[int\], typing.List\[typing.List\[int\]\], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')\]skip\_special\_tokens: bool = Falseclean\_up\_tokenization\_spaces: bool = None\*\*kwargs ) → `List[str]`

Parameters

-   **sequences** (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`) — List of tokenized input ids. Can be obtained using the `__call__` method.
-   **skip\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not to remove special tokens in the decoding.
-   **clean\_up\_tokenization\_spaces** (`bool`, _optional_) — Whether or not to clean up the tokenization spaces. If `None`, will default to `self.clean_up_tokenization_spaces`.
-   **kwargs** (additional keyword arguments, _optional_) — Will be passed to the underlying model specific decode method.

The list of decoded sentences.

Convert a list of lists of token ids into a list of strings by calling decode.

#### decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3724)

( token\_ids: typing.Union\[int, typing.List\[int\], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')\]skip\_special\_tokens: bool = Falseclean\_up\_tokenization\_spaces: bool = None\*\*kwargs ) → `str`

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

( text: typing.Union\[str, typing.List\[str\], typing.List\[int\]\]text\_pair: typing.Union\[str, typing.List\[str\], typing.List\[int\], NoneType\] = Noneadd\_special\_tokens: bool = Truepadding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = Falsetruncation: typing.Union\[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy\] = Nonemax\_length: typing.Optional\[int\] = Nonestride: int = 0return\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None\*\*kwargs ) → `List[int]`, `torch.Tensor`, `tf.Tensor` or `np.ndarray`

Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

Same as doing `self.convert_tokens_to_ids(self.tokenize(text))`.

#### push\_to\_hub

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/hub.py#L786)

( repo\_id: struse\_temp\_dir: typing.Optional\[bool\] = Nonecommit\_message: typing.Optional\[str\] = Noneprivate: typing.Optional\[bool\] = Nonetoken: typing.Union\[bool, str, NoneType\] = Nonemax\_shard\_size: typing.Union\[int, str, NoneType\] = '10GB'create\_pr: bool = Falsesafe\_serialization: bool = Falserevision: str = None\*\*deprecated\_kwargs )

Upload the tokenizer files to the 🤗 Model Hub.

Examples:

```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


tokenizer.push_to_hub("my-finetuned-bert")


tokenizer.push_to_hub("huggingface/my-finetuned-bert")
```

#### convert\_ids\_to\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils.py#L942)

( ids: typing.Union\[int, typing.List\[int\]\]skip\_special\_tokens: bool = False ) → `str` or `List[str]`

Parameters

-   **ids** (`int` or `List[int]`) — The token id (or token ids) to convert to tokens.
-   **skip\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not to remove special tokens in the decoding.

The decoded token(s).

Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and added tokens.

#### convert\_tokens\_to\_ids

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils.py#L619)

( tokens: typing.Union\[str, typing.List\[str\]\] ) → `int` or `List[int]`

Parameters

-   **tokens** (`str` or `List[str]`) — One or several token(s) to convert to token id(s).

The token id or list of token ids.

Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the vocabulary.

Returns the added tokens in the vocabulary as a dictionary of token to index. Results might be different from the fast call because for now we always add the tokens even if they are already in the vocabulary. This is something we should change.

#### num\_special\_tokens\_to\_add

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils.py#L506)

( pair: bool = False ) → `int`

Parameters

-   **pair** (`bool`, _optional_, defaults to `False`) — Whether the number of added tokens should be computed in the case of a sequence pair or a single sequence.

Number of special tokens added to sequences.

Returns the number of added tokens when encoding a sequence with special tokens.

This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put this inside your training loop.

#### prepare\_for\_tokenization

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils.py#L880)

( text: stris\_split\_into\_words: bool = False\*\*kwargs ) → `Tuple[str, Dict[str, Any]]`

Parameters

-   **text** (`str`) — The text to prepare.
-   **is\_split\_into\_words** (`bool`, _optional_, defaults to `False`) — Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace) which it will tokenize. This is useful for NER or token classification.
-   **kwargs** (`Dict[str, Any]`, _optional_) — Keyword arguments to use for the tokenization.

Returns

`Tuple[str, Dict[str, Any]]`

The prepared text and the unused kwargs.

Performs any necessary transformations before tokenization.

This method should pop the arguments from kwargs and return the remaining `kwargs` as well. We test the `kwargs` at the end of the encoding process to be sure all the arguments have been used.

#### tokenize

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils.py#L529)

( text: str\*\*kwargs ) → `List[str]`

Parameters

-   **text** (`str`) — The sequence to be encoded.
-   \***\*kwargs** (additional keyword arguments) — Passed along to the model-specific `prepare_for_tokenization` preprocessing method.

The list of tokens.

Converts a string in a sequence of tokens, using the tokenizer.

Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces). Takes care of added tokens.

## PreTrainedTokenizerFast

The [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) depend on the [tokenizers](https://huggingface.co/docs/tokenizers) library. The tokenizers obtained from the 🤗 tokenizers library can be loaded very simply into 🤗 transformers. Take a look at the [Using tokenizers from 🤗 tokenizers](../fast_tokenizers) page to understand how this is done.

### class transformers.PreTrainedTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_fast.py#L78)

( \*args\*\*kwargs )

Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).

Inherits from [PreTrainedTokenizerBase](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase).

Handles all the shared methods for tokenization and special tokens, as well as methods for downloading/caching/loading pretrained tokenizers, as well as adding tokens to the vocabulary.

This class also contains the added tokens in a unified way on top of all tokenizers so we don’t have to handle the specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece…).

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

( text: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\]\] = Nonetext\_pair: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\], NoneType\] = Nonetext\_target: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\]\] = Nonetext\_pair\_target: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\], NoneType\] = Noneadd\_special\_tokens: bool = Truepadding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = Falsetruncation: typing.Union\[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy\] = Nonemax\_length: typing.Optional\[int\] = Nonestride: int = 0is\_split\_into\_words: bool = Falsepad\_to\_multiple\_of: typing.Optional\[int\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonereturn\_token\_type\_ids: typing.Optional\[bool\] = Nonereturn\_attention\_mask: typing.Optional\[bool\] = Nonereturn\_overflowing\_tokens: bool = Falsereturn\_special\_tokens\_mask: bool = Falsereturn\_offsets\_mapping: bool = Falsereturn\_length: bool = Falseverbose: bool = True\*\*kwargs ) → [BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding)

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of sequences.

#### apply\_chat\_template

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L1717)

( conversation: typing.Union\[typing.List\[typing.Dict\[str, str\]\], ForwardRef('Conversation')\]chat\_template: typing.Optional\[str\] = Nonetokenize: bool = Truepadding: bool = Falsetruncation: bool = Falsemax\_length: typing.Optional\[int\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None\*\*tokenizer\_kwargs ) → `List[int]`

Converts a Conversation object or a list of dictionaries with `"role"` and `"content"` keys to a list of token ids. This method is intended for use with chat models, and will read the tokenizer’s chat\_template attribute to determine the format and control tokens to use when converting. When chat\_template is None, it will fall back to the default\_chat\_template specified at the class level.

#### batch\_decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3690)

( sequences: typing.Union\[typing.List\[int\], typing.List\[typing.List\[int\]\], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')\]skip\_special\_tokens: bool = Falseclean\_up\_tokenization\_spaces: bool = None\*\*kwargs ) → `List[str]`

Parameters

-   **sequences** (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`) — List of tokenized input ids. Can be obtained using the `__call__` method.
-   **skip\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not to remove special tokens in the decoding.
-   **clean\_up\_tokenization\_spaces** (`bool`, _optional_) — Whether or not to clean up the tokenization spaces. If `None`, will default to `self.clean_up_tokenization_spaces`.
-   **kwargs** (additional keyword arguments, _optional_) — Will be passed to the underlying model specific decode method.

The list of decoded sentences.

Convert a list of lists of token ids into a list of strings by calling decode.

#### decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3724)

( token\_ids: typing.Union\[int, typing.List\[int\], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')\]skip\_special\_tokens: bool = Falseclean\_up\_tokenization\_spaces: bool = None\*\*kwargs ) → `str`

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

( text: typing.Union\[str, typing.List\[str\], typing.List\[int\]\]text\_pair: typing.Union\[str, typing.List\[str\], typing.List\[int\], NoneType\] = Noneadd\_special\_tokens: bool = Truepadding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = Falsetruncation: typing.Union\[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy\] = Nonemax\_length: typing.Optional\[int\] = Nonestride: int = 0return\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None\*\*kwargs ) → `List[int]`, `torch.Tensor`, `tf.Tensor` or `np.ndarray`

Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

Same as doing `self.convert_tokens_to_ids(self.tokenize(text))`.

#### push\_to\_hub

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/hub.py#L786)

( repo\_id: struse\_temp\_dir: typing.Optional\[bool\] = Nonecommit\_message: typing.Optional\[str\] = Noneprivate: typing.Optional\[bool\] = Nonetoken: typing.Union\[bool, str, NoneType\] = Nonemax\_shard\_size: typing.Union\[int, str, NoneType\] = '10GB'create\_pr: bool = Falsesafe\_serialization: bool = Falserevision: str = None\*\*deprecated\_kwargs )

Upload the tokenizer files to the 🤗 Model Hub.

Examples:

```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


tokenizer.push_to_hub("my-finetuned-bert")


tokenizer.push_to_hub("huggingface/my-finetuned-bert")
```

#### convert\_ids\_to\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_fast.py#L337)

( ids: typing.Union\[int, typing.List\[int\]\]skip\_special\_tokens: bool = False ) → `str` or `List[str]`

Parameters

-   **ids** (`int` or `List[int]`) — The token id (or token ids) to convert to tokens.
-   **skip\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not to remove special tokens in the decoding.

The decoded token(s).

Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and added tokens.

#### convert\_tokens\_to\_ids

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_fast.py#L282)

( tokens: typing.Union\[str, typing.List\[str\]\] ) → `int` or `List[int]`

Parameters

-   **tokens** (`str` or `List[str]`) — One or several token(s) to convert to token id(s).

The token id or list of token ids.

Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the vocabulary.

Returns the added tokens in the vocabulary as a dictionary of token to index.

#### num\_special\_tokens\_to\_add

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_fast.py#L316)

( pair: bool = False ) → `int`

Parameters

-   **pair** (`bool`, _optional_, defaults to `False`) — Whether the number of added tokens should be computed in the case of a sequence pair or a single sequence.

Number of special tokens added to sequences.

Returns the number of added tokens when encoding a sequence with special tokens.

This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put this inside your training loop.

#### set\_truncation\_and\_padding

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_fast.py#L366)

( padding\_strategy: PaddingStrategytruncation\_strategy: TruncationStrategymax\_length: intstride: intpad\_to\_multiple\_of: typing.Optional\[int\] )

Parameters

-   **padding\_strategy** ([PaddingStrategy](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.utils.PaddingStrategy)) — The kind of padding that will be applied to the input
-   **truncation\_strategy** ([TruncationStrategy](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy)) — The kind of truncation that will be applied to the input
-   **max\_length** (`int`) — The maximum size of a sequence.
-   **stride** (`int`) — The stride to use when handling overflow.
-   **pad\_to\_multiple\_of** (`int`, _optional_) — If set will pad the sequence to a multiple of the provided value. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).

Define the truncation and the padding strategies for fast tokenizers (provided by HuggingFace tokenizers library) and restore the tokenizer settings afterwards.

The provided tokenizer has no padding / truncation strategy before the managed section. If your tokenizer set a padding / truncation strategy before, then it will be reset to no padding / truncation when exiting the managed section.

Trains a tokenizer on a new corpus with the same defaults (in terms of special tokens or tokenization pipeline) as the current one.

## BatchEncoding

### class transformers.BatchEncoding

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L176)

( data: typing.Union\[typing.Dict\[str, typing.Any\], NoneType\] = Noneencoding: typing.Union\[tokenizers.Encoding, typing.Sequence\[tokenizers.Encoding\], NoneType\] = Nonetensor\_type: typing.Union\[NoneType, str, transformers.utils.generic.TensorType\] = Noneprepend\_batch\_axis: bool = Falsen\_sequences: typing.Optional\[int\] = None )

Parameters

-   **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the `__call__`/`encode_plus`/`batch_encode_plus` methods (‘input\_ids’, ‘attention\_mask’, etc.).
-   **encoding** (`tokenizers.Encoding` or `Sequence[tokenizers.Encoding]`, _optional_) — If the tokenizer is a fast tokenizer which outputs additional information like mapping from word/character space to token space the `tokenizers.Encoding` instance or list of instance (for batches) hold this information.
-   **tensor\_type** (`Union[None, str, TensorType]`, _optional_) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at initialization.
-   **prepend\_batch\_axis** (`bool`, _optional_, defaults to `False`) — Whether or not to add a batch axis when converting to tensors (see `tensor_type` above).
-   **n\_sequences** (`Optional[int]`, _optional_) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at initialization.

Holds the output of the [**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__), [encode\_plus()](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode_plus) and [batch\_encode\_plus()](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_encode_plus) methods (tokens, attention\_masks, etc).

This class is derived from a python dictionary and can be used as a dictionary. In addition, this class exposes utility methods to map from word/character space to token space.

#### char\_to\_token

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L555)

( batch\_or\_char\_index: intchar\_index: typing.Optional\[int\] = Nonesequence\_index: int = 0 ) → `int`

Parameters

-   **batch\_or\_char\_index** (`int`) — Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of the word in the sequence
-   **char\_index** (`int`, _optional_) — If a batch index is provided in _batch\_or\_token\_index_, this can be the index of the word in the sequence.
-   **sequence\_index** (`int`, _optional_, defaults to 0) — If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0 or 1) the provided character index belongs to.

Index of the token.

Get the index of the token in the encoded output comprising a character in the original string for a sequence of the batch.

Can be called as:

-   `self.char_to_token(char_index)` if batch size is 1
-   `self.char_to_token(batch_index, char_index)` if batch size is greater or equal to 1

This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized words.

#### char\_to\_word

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L641)

( batch\_or\_char\_index: intchar\_index: typing.Optional\[int\] = Nonesequence\_index: int = 0 ) → `int` or `List[int]`

Parameters

-   **batch\_or\_char\_index** (`int`) — Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of the character in the original string.
-   **char\_index** (`int`, _optional_) — If a batch index is provided in _batch\_or\_token\_index_, this can be the index of the character in the original string.
-   **sequence\_index** (`int`, _optional_, defaults to 0) — If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0 or 1) the provided character index belongs to.

Index or indices of the associated encoded token(s).

Get the word in the original string corresponding to a character in the original string of a sequence of the batch.

Can be called as:

-   `self.char_to_word(char_index)` if batch size is 1
-   `self.char_to_word(batch_index, char_index)` if batch size is greater than 1

This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized words.

#### convert\_to\_tensors

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L680)

( tensor\_type: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Noneprepend\_batch\_axis: bool = False )

Parameters

-   **tensor\_type** (`str` or [TensorType](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.TensorType), _optional_) — The type of tensors to use. If `str`, should be one of the values of the enum [TensorType](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.TensorType). If `None`, no modification is done.
-   **prepend\_batch\_axis** (`int`, _optional_, defaults to `False`) — Whether or not to add the batch dimension during the conversion.

Convert the inner content to tensors.

#### sequence\_ids

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L319)

( batch\_index: int = 0 ) → `List[Optional[int]]`

Parameters

-   **batch\_index** (`int`, _optional_, defaults to 0) — The index to access in the batch.

Returns

`List[Optional[int]]`

A list indicating the sequence id corresponding to each token. Special tokens added by the tokenizer are mapped to `None` and other tokens are mapped to the index of their corresponding sequence.

Return a list mapping the tokens to the id of their original sentences:

-   `None` for special tokens added around or between sequences,
-   `0` for tokens corresponding to words in the first sequence,
-   `1` for tokens corresponding to words in the second sequence when a pair of sequences was jointly encoded.

#### to

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L773)

( device: typing.Union\[str, ForwardRef('torch.device')\] ) → [BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

-   **device** (`str` or `torch.device`) — The device to put the tensors on.

The same instance after modification.

Send all values to device by calling `v.to(device)` (PyTorch only).

#### token\_to\_chars

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L516)

( batch\_or\_token\_index: inttoken\_index: typing.Optional\[int\] = None ) → [CharSpan](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.CharSpan)

Parameters

-   **batch\_or\_token\_index** (`int`) — Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of the token in the sequence.
-   **token\_index** (`int`, _optional_) — If a batch index is provided in _batch\_or\_token\_index_, this can be the index of the token or tokens in the sequence.

Span of characters in the original string, or None, if the token (e.g. ~,~ ) doesn’t correspond to any chars in the origin string.

Get the character span corresponding to an encoded token in a sequence of the batch.

Character spans are returned as a [CharSpan](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.CharSpan) with:

-   **start** — Index of the first character in the original string associated to the token.
-   **end** — Index of the character following the last character in the original string associated to the token.

Can be called as:

-   `self.token_to_chars(token_index)` if batch size is 1
-   `self.token_to_chars(batch_index, token_index)` if batch size is greater or equal to 1

#### token\_to\_sequence

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L386)

( batch\_or\_token\_index: inttoken\_index: typing.Optional\[int\] = None ) → `int`

Parameters

-   **batch\_or\_token\_index** (`int`) — Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of the token in the sequence.
-   **token\_index** (`int`, _optional_) — If a batch index is provided in _batch\_or\_token\_index_, this can be the index of the token in the sequence.

Index of the word in the input sequence.

Get the index of the sequence represented by the given token. In the general use case, this method returns `0` for a single sequence or the first sequence of a pair, and `1` for the second sequence of a pair

Can be called as:

-   `self.token_to_sequence(token_index)` if batch size is 1
-   `self.token_to_sequence(batch_index, token_index)` if batch size is greater than 1

This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e., words are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized words.

#### token\_to\_word

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L425)

( batch\_or\_token\_index: inttoken\_index: typing.Optional\[int\] = None ) → `int`

Parameters

-   **batch\_or\_token\_index** (`int`) — Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of the token in the sequence.
-   **token\_index** (`int`, _optional_) — If a batch index is provided in _batch\_or\_token\_index_, this can be the index of the token in the sequence.

Index of the word in the input sequence.

Get the index of the word corresponding (i.e. comprising) to an encoded token in a sequence of the batch.

Can be called as:

-   `self.token_to_word(token_index)` if batch size is 1
-   `self.token_to_word(batch_index, token_index)` if batch size is greater than 1

This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e., words are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized words.

#### tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L301)

( batch\_index: int = 0 ) → `List[str]`

Parameters

-   **batch\_index** (`int`, _optional_, defaults to 0) — The index to access in the batch.

The list of tokens at that index.

Return the list of tokens (sub-parts of the input strings after word/subword splitting and before conversion to integer indices) at a given batch index (only works for the output of a fast tokenizer).

#### word\_ids

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L367)

( batch\_index: int = 0 ) → `List[Optional[int]]`

Parameters

-   **batch\_index** (`int`, _optional_, defaults to 0) — The index to access in the batch.

Returns

`List[Optional[int]]`

A list indicating the word corresponding to each token. Special tokens added by the tokenizer are mapped to `None` and other tokens are mapped to the index of their corresponding word (several tokens will be mapped to the same word index if they are parts of that word).

Return a list mapping the tokens to their actual word in the initial sentence for a fast tokenizer.

#### word\_to\_chars

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L596)

( batch\_or\_word\_index: intword\_index: typing.Optional\[int\] = Nonesequence\_index: int = 0 ) → `CharSpan` or `List[CharSpan]`

Parameters

-   **batch\_or\_word\_index** (`int`) — Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of the word in the sequence
-   **word\_index** (`int`, _optional_) — If a batch index is provided in _batch\_or\_token\_index_, this can be the index of the word in the sequence.
-   **sequence\_index** (`int`, _optional_, defaults to 0) — If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0 or 1) the provided word index belongs to.

Returns

`CharSpan` or `List[CharSpan]`

Span(s) of the associated character or characters in the string. CharSpan are NamedTuple with:

-   start: index of the first character associated to the token in the original string
-   end: index of the character following the last character associated to the token in the original string

Get the character span in the original string corresponding to given word in a sequence of the batch.

Character spans are returned as a CharSpan NamedTuple with:

-   start: index of the first character in the original string
-   end: index of the character following the last character in the original string

Can be called as:

-   `self.word_to_chars(word_index)` if batch size is 1
-   `self.word_to_chars(batch_index, word_index)` if batch size is greater or equal to 1

#### word\_to\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L463)

( batch\_or\_word\_index: intword\_index: typing.Optional\[int\] = Nonesequence\_index: int = 0 ) → ([TokenSpan](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.TokenSpan), _optional_)

Parameters

-   **batch\_or\_word\_index** (`int`) — Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of the word in the sequence.
-   **word\_index** (`int`, _optional_) — If a batch index is provided in _batch\_or\_token\_index_, this can be the index of the word in the sequence.
-   **sequence\_index** (`int`, _optional_, defaults to 0) — If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0 or 1) the provided word index belongs to.

Returns

([TokenSpan](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.TokenSpan), _optional_)

Span of tokens in the encoded sequence. Returns `None` if no tokens correspond to the word. This can happen especially when the token is a special token that has been used to format the tokenization. For example when we add a class token at the very beginning of the tokenization.

Get the encoded token span corresponding to a word in a sequence of the batch.

Token spans are returned as a [TokenSpan](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.TokenSpan) with:

-   **start** — Index of the first token.
-   **end** — Index of the token following the last token.

Can be called as:

-   `self.word_to_tokens(word_index, sequence_index: int = 0)` if batch size is 1
-   `self.word_to_tokens(batch_index, word_index, sequence_index: int = 0)` if batch size is greater or equal to 1

This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized words.

#### words

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L343)

( batch\_index: int = 0 ) → `List[Optional[int]]`

Parameters

-   **batch\_index** (`int`, _optional_, defaults to 0) — The index to access in the batch.

Returns

`List[Optional[int]]`

A list indicating the word corresponding to each token. Special tokens added by the tokenizer are mapped to `None` and other tokens are mapped to the index of their corresponding word (several tokens will be mapped to the same word index if they are parts of that word).

Return a list mapping the tokens to their actual word in the initial sentence for a fast tokenizer.