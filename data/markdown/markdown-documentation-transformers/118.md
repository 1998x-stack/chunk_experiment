# ByT5

## Overview

The ByT5 model was presented in [ByT5: Towards a token-free future with pre-trained byte-to-byte models](https://arxiv.org/abs/2105.13626) by Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, Colin Raffel.

The abstract from the paper is the following:

_Most widely-used pre-trained language models operate on sequences of tokens corresponding to word or subword units. Encoding text as a sequence of tokens requires a tokenizer, which is typically created as an independent artifact from the model. Token-free models that instead operate directly on raw text (bytes or characters) have many benefits: they can process text in any language out of the box, they are more robust to noise, and they minimize technical debt by removing complex and error-prone text preprocessing pipelines. Since byte or character sequences are longer than token sequences, past work on token-free models has often introduced new model architectures designed to amortize the cost of operating directly on raw text. In this paper, we show that a standard Transformer architecture can be used with minimal modifications to process byte sequences. We carefully characterize the trade-offs in terms of parameter count, training FLOPs, and inference speed, and show that byte-level models are competitive with their token-level counterparts. We also demonstrate that byte-level models are significantly more robust to noise and perform better on tasks that are sensitive to spelling and pronunciation. As part of our contribution, we release a new set of pre-trained byte-level Transformer models based on the T5 architecture, as well as all code and data used in our experiments._

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten). The original code can be found [here](https://github.com/google-research/byt5).

ByT5’s architecture is based on the T5v1.1 model, so one can refer to [T5v1.1’s documentation page](t5v1.1). They only differ in how inputs should be prepared for the model, see the code examples below.

Since ByT5 was pre-trained unsupervisedly, there’s no real advantage to using a task prefix during single-task fine-tuning. If you are doing multi-task fine-tuning, you should use a prefix.

### Example

ByT5 works on raw UTF-8 bytes, so it can be used without a tokenizer:

```
>>> from transformers import T5ForConditionalGeneration
>>> import torch

>>> model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")

>>> num_special_tokens = 3
>>> 
>>> 

>>> input_ids = torch.tensor([list("Life is like a box of chocolates.".encode("utf-8"))]) + num_special_tokens

>>> labels = torch.tensor([list("La vie est comme une boîte de chocolat.".encode("utf-8"))]) + num_special_tokens

>>> loss = model(input_ids, labels=labels).loss
>>> loss.item()
2.66
```

For batched inference and training it is however recommended to make use of the tokenizer:

```
>>> from transformers import T5ForConditionalGeneration, AutoTokenizer

>>> model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

>>> model_inputs = tokenizer(
...     ["Life is like a box of chocolates.", "Today is Monday."], padding="longest", return_tensors="pt"
... )
>>> labels_dict = tokenizer(
...     ["La vie est comme une boîte de chocolat.", "Aujourd'hui c'est lundi."], padding="longest", return_tensors="pt"
... )
>>> labels = labels_dict.input_ids

>>> loss = model(**model_inputs, labels=labels).loss
>>> loss.item()
17.9
```

Similar to [T5](t5), ByT5 was trained on the span-mask denoising task. However, since the model works directly on characters, the pretraining task is a bit different. Let’s corrupt some characters of the input sentence `"The dog chases a ball in the park."` and ask ByT5 to predict them for us.

```
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-base")

>>> input_ids_prompt = "The dog chases a ball in the park."
>>> input_ids = tokenizer(input_ids_prompt).input_ids

>>> 
>>> 
>>> 
>>> 
>>> 
>>> 
>>> 
>>> 

>>> input_ids = torch.tensor([input_ids[:8] + [258] + input_ids[14:21] + [257] + input_ids[28:]])
>>> input_ids
tensor([[ 87, 107, 104,  35, 103, 114, 106,  35, 258,  35, 100,  35, 101, 100, 111, 111, 257,  35, 115, 100, 117, 110,  49,   1]])

>>> 
>>> output_ids = model.generate(input_ids, max_length=100)[0].tolist()
>>> output_ids
[0, 258, 108, 118,  35, 119, 107, 104,  35, 114, 113, 104,  35, 122, 107, 114,  35, 103, 114, 104, 118, 257,  35, 108, 113,  35, 119, 107, 104,  35, 103, 108, 118, 102, 114, 256, 108, 113,  35, 119, 107, 104, 35, 115, 100, 117, 110,  49,  35,  87, 107, 104,  35, 103, 114, 106, 35, 108, 118,  35, 119, 107, 104,  35, 114, 113, 104,  35, 122, 107, 114,  35, 103, 114, 104, 118,  35, 100,  35, 101, 100, 111, 111,  35, 108, 113, 255,  35, 108, 113,  35, 119, 107, 104,  35, 115, 100, 117, 110,  49]

>>> 

>>> 
>>> output_ids_list = []
>>> start_token = 0
>>> sentinel_token = 258
>>> while sentinel_token in output_ids:
...     split_idx = output_ids.index(sentinel_token)
...     output_ids_list.append(output_ids[start_token:split_idx])
...     start_token = split_idx
...     sentinel_token -= 1

>>> output_ids_list.append(output_ids[start_token:])
>>> output_string = tokenizer.batch_decode(output_ids_list)
>>> output_string
['<pad>', 'is the one who does', ' in the disco', 'in the park. The dog is the one who does a ball in', ' in the park.']
```

## ByT5Tokenizer

### class transformers.ByT5Tokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/byt5/tokenization_byt5.py#L28)

( eos\_token = '</s>' unk\_token = '<unk>' pad\_token = '<pad>' extra\_ids = 125 additional\_special\_tokens = None \*\*kwargs )

Parameters

-   **eos\_token** (`str`, _optional_, defaults to `"</s>"`) — The end of sequence token.
    
    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.
    
-   **unk\_token** (`str`, _optional_, defaults to `"<unk>"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **pad\_token** (`str`, _optional_, defaults to `"<pad>"`) — The token used for padding, for example when batching sequences of different lengths.
-   **extra\_ids** (`int`, _optional_, defaults to 100) — Add a number of extra ids added to the end of the vocabulary for use as sentinels. These tokens are accessible as “id{%d}>” where ”{%d}” is a number between 0 and extra\_ids-1. Extra tokens are indexed from the end of the vocabulary up to beginning (“” is the last token in the vocabulary like in ByT5 preprocessing see [here](https://github.com/google-research/text-to-text-transfer-transformer/blob/9fd7b14a769417be33bc6c850f9598764913c833/t5/data/preprocessors.py#L2117)).
-   **additional\_special\_tokens** (`List[str]`, _optional_) — Additional special tokens used by the tokenizer.

Construct a ByT5 tokenizer. ByT5 simply uses raw bytes utf-8 encoding.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/byt5/tokenization_byt5.py#L172)

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/byt5/tokenization_byt5.py#L150)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. ByT5 does not make use of token type ids, therefore a list of zeros is returned.

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/byt5/tokenization_byt5.py#L111)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None already\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

See [ByT5Tokenizer](/docs/transformers/v4.34.0/en/model_doc/byt5#transformers.ByT5Tokenizer) for all details.