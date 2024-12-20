# General Utilities

This page lists all of Transformers general utility functions that are found in the file `utils.py`.

Most of those are only useful if you are studying the general code in the library.

## Enums and namedtuples

### class transformers.utils.ExplicitEnum

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/generic.py#L426)

( value names = None module = None qualname = None type = None start = 1 )

Enum with more explicit error message for missing values.

### class transformers.utils.PaddingStrategy

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/generic.py#L438)

( value names = None module = None qualname = None type = None start = 1 )

Possible values for the `padding` argument in [PreTrainedTokenizerBase.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__). Useful for tab-completion in an IDE.

### class transformers.TensorType

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/generic.py#L449)

( value names = None module = None qualname = None type = None start = 1 )

Possible values for the `return_tensors` argument in [PreTrainedTokenizerBase.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__). Useful for tab-completion in an IDE.

## Special Decorators

#### transformers.add\_start\_docstrings

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/doc.py#L23)

( \*docstr )

#### transformers.utils.add\_start\_docstrings\_to\_model\_forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/doc.py#L31)

( \*docstr )

#### transformers.add\_end\_docstrings

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/doc.py#L53)

( \*docstr )

#### transformers.utils.add\_code\_sample\_docstrings

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/doc.py#L1064)

( \*docstr processor\_class = None checkpoint = None output\_type = None config\_class = None mask = '\[MASK\]' qa\_target\_start\_index = 14 qa\_target\_end\_index = 15 model\_cls = None modality = None expected\_output = None expected\_loss = None real\_checkpoint = None )

#### transformers.utils.replace\_return\_docstrings

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/doc.py#L1152)

( output\_type = None config\_class = None )

## Special Properties

### class transformers.utils.cached\_property

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/generic.py#L36)

( fget = None fset = None fdel = None doc = None )

Descriptor that mimics @property but caches output in member variable.

From tensorflow\_datasets

Built-in in functools from Python 3.8.

## Other Utilities

### class transformers.utils.\_LazyModule

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/import_utils.py#L1233)

( name module\_file import\_structure module\_spec = None extra\_objects = None )

Module class that surfaces all objects but only performs associated imports when the objects are requested.