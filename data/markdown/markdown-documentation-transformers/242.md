# PhoBERT

## Overview

The PhoBERT model was proposed in [PhoBERT: Pre-trained language models for Vietnamese](https://www.aclweb.org/anthology/2020.findings-emnlp.92.pdf) by Dat Quoc Nguyen, Anh Tuan Nguyen.

The abstract from the paper is the following:

_We present PhoBERT with two versions, PhoBERT-base and PhoBERT-large, the first public large-scale monolingual language models pre-trained for Vietnamese. Experimental results show that PhoBERT consistently outperforms the recent best pre-trained multilingual model XLM-R (Conneau et al., 2020) and improves the state-of-the-art in multiple Vietnamese-specific NLP tasks including Part-of-speech tagging, Dependency parsing, Named-entity recognition and Natural language inference._

Example of use:

```
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> phobert = AutoModel.from_pretrained("vinai/phobert-base")
>>> tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

>>> 
>>> line = "Tôi là sinh_viên trường đại_học Công_nghệ ."

>>> input_ids = torch.tensor([tokenizer.encode(line)])

>>> with torch.no_grad():
...     features = phobert(input_ids)  

>>> 
>>> 
>>> 
```

This model was contributed by [dqnguyen](https://huggingface.co/dqnguyen). The original code can be found [here](https://github.com/VinAIResearch/PhoBERT).

## PhobertTokenizer

### class transformers.PhobertTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/phobert/tokenization_phobert.py#L68)

( vocab\_filemerges\_filebos\_token = '<s>'eos\_token = '</s>'sep\_token = '</s>'cls\_token = '<s>'unk\_token = '<unk>'pad\_token = '<pad>'mask\_token = '<mask>'\*\*kwargs )

Construct a PhoBERT tokenizer. Based on Byte-Pair-Encoding.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

Loads a pre-existing dictionary from a text file and adds its symbols to this instance.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/phobert/tokenization_phobert.py#L165)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A PhoBERT sequence has the following format:

-   single sequence: `<s> X </s>`
-   pair of sequences: `<s> A </s></s> B </s>`

Converts a sequence of tokens (string) in a single string.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/phobert/tokenization_phobert.py#L219)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. PhoBERT does not make use of token type ids, therefore a list of zeros is returned.

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/phobert/tokenization_phobert.py#L191)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = Nonealready\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.