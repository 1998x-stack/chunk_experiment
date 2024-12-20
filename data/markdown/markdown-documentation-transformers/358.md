# CodeLlama

## Overview

The Code Llama model was proposed in [Code Llama: Open Foundation Models for Code](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/) by Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, Gabriel Synnaeve.

The abstract from the paper is the following:

_We release Code Llama, a family of large language models for code based on Llama 2 providing state-of-the-art performance among open models, infilling capabilities, support for large input contexts, and zero-shot instruction following ability for programming tasks. We provide multiple flavors to cover a wide range of applications: foundation models (Code Llama), Python specializations (Code Llama - Python), and instruction-following models (Code Llama - Instruct) with 7B, 13B and 34B parameters each. All models are trained on sequences of 16k tokens and show improvements on inputs with up to 100k tokens. 7B and 13B Code Llama and Code Llama - Instruct variants support infilling based on surrounding content. Code Llama reaches state-of-the-art performance among open models on several code benchmarks, with scores of up to 53% and 55% on HumanEval and MBPP, respectively. Notably, Code Llama - Python 7B outperforms Llama 2 70B on HumanEval and MBPP, and all our models outperform every other publicly available model on MultiPL-E. We release Code Llama under a permissive license that allows for both research and commercial use._

Check out all Code Llama models [here](https://huggingface.co/models?search=code_llama) and the officially released ones in the [codellama org](https://huggingface.co/codellama).

The `Llama2` family models, on which Code Llama is based, were trained using `bfloat16`, but the original inference uses `float16`. Let’s look at the different precisions:

-   `float32`: PyTorch convention on model initialization is to load models in `float32`, no matter with which `dtype` the model weights were stored. `transformers` also follows this convention for consistency with PyTorch. This will be picked by default. If you want the `AutoModel` API to cast the load the checkpoints with the storage weights type, you must specify `torch_dtype="auto"`, e.g. `model = AutoModelForCausalLM.from_pretrained("path", torch_dtype = "auto")`.
-   `bfloat16`: Code Llama was trained with this precision, so we recommend using it for further training or fine-tuning.
-   `float16`: We recommend running inference using this precision, as it’s usually faster than `bfloat16`, and evaluation metrics show no discernible degradation with respect to `bfloat16`. You can also run inference using `bfloat16`, and we recommend you check inference results with both `float16` and `bfloat16` after fine-tuning.

As mentioned above, the `dtype` of the storage weights is mostly irrelevant unless you are using `torch_dtype="auto"` when initializing a model using. The reason is that the model will first be downloaded (using the `dtype` of the checkpoints online) and then will be casted to the default `dtype` of `torch` (becomes `torch.float32`). If there is a specified `torch_dtype`, it will be used instead.

Tips:

-   These models have the same architecture as the `Llama2` models
-   The infilling task is supported out of the box. You should be using the `tokenizer.fill_token` where you want your input to be filled.
-   The model conversion script is the same as for the `Llama2` family:

Here is a sample usage

```
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

Note that executing the script requires enough CPU RAM to host the whole model in float16 precision (even if the biggest versions come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).

-   After conversion, the model and tokenizer can be loaded via:

```
>>> from transformers import LlamaForCausalLM, CodeLlamaTokenizer

>>> tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
>>> model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")
>>> PROMPT = '''def remove_non_ascii(s: str) -> str:
    """ <FILL_ME>
    return result
'''
>>> input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
>>> generated_ids = model.generate(input_ids, max_new_tokens=128)

>>> filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
>>> print(PROMPT.replace("<FILL_ME>", filling))
def remove_non_ascii(s: str) -> str:
    """ Remove non-ASCII characters from a string.

    Args:
        s: The string to remove non-ASCII characters from.

    Returns:
        The string with non-ASCII characters removed.
    """
    result = ""
    for c in s:
        if ord(c) < 128:
            result += c
    return result
```

If you only want the infilled part:

```
>>> from transformers import pipeline
>>> import torch

>>> generator = pipeline("text-generation",model="codellama/CodeLlama-7b-hf",torch_dtype=torch.float16, device_map="auto")
>>> generator('def remove_non_ascii(s: str) -> str:\n    """ <FILL_ME>\n    return result', max_new_tokens = 128, return_type = 1)
```

Under the hood, the tokenizer [automatically splits by `<FILL_ME>`](https://huggingface.co/docs/transformers/main/model_doc/code_llama#transformers.CodeLlamaTokenizer.fill_token) to create a formatted input string that follows [the original training pattern](https://github.com/facebookresearch/codellama/blob/cb51c14ec761370ba2e2bc351374a79265d0465e/llama/generation.py#L402). This is more robust than preparing the pattern yourself: it avoids pitfalls, such as token glueing, that are very hard to debug. To see how much CPU and GPU memory you need for this model or others, try [this calculator](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) which can help determine that value.

-   The LLaMA tokenizer is a BPE model based on [sentencepiece](https://github.com/google/sentencepiece). One quirk of sentencepiece is that when decoding a sequence, if the first token is the start of the word (e.g. “Banana”), the tokenizer does not prepend the prefix space to the string.

This model was contributed by [ArthurZucker](https://huggingface.co/ArthurZ). The original code of the authors can be found [here](https://github.com/facebookresearch/llama).

## CodeLlamaTokenizer

### class transformers.CodeLlamaTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/code_llama/tokenization_code_llama.py#L59)

( vocab\_fileunk\_token = '<unk>'bos\_token = '<s>'eos\_token = '</s>'prefix\_token = '▁<PRE>'middle\_token = '▁<MID>'suffix\_token = '▁<SUF>'eot\_token = '▁<EOT>'fill\_token = '<FILL\_ME>'suffix\_first = Falsesp\_model\_kwargs: typing.Union\[typing.Dict\[str, typing.Any\], NoneType\] = Noneadd\_bos\_token = Trueadd\_eos\_token = Falseclean\_up\_tokenization\_spaces = Falseadditional\_special\_tokens = Noneuse\_default\_system\_prompt = False\*\*kwargs )

Construct a CodeLlama tokenizer. Based on byte-level Byte-Pair-Encoding. The default padding token is unset as there is no padding token in the original model.

The default configuration match that of [codellama/CodeLlama-7b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf/blob/main/tokenizer_config.json) which supports prompt infilling.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/code_llama/tokenization_code_llama.py#L361)

( token\_ids\_0token\_ids\_1 = None )

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/code_llama/tokenization_code_llama.py#L373)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = Nonealready\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/code_llama/tokenization_code_llama.py#L411)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

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

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/code_llama/tokenization_code_llama.py#L333)

( save\_directoryfilename\_prefix: typing.Optional\[str\] = None ) → `Tuple(str)`

Parameters

-   **save\_directory** (`str`) — The directory in which to save the vocabulary.

Paths to the files saved.

Save the vocabulary and special tokens file to a directory.

## CodeLlamaTokenizerFast

### class transformers.CodeLlamaTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/code_llama/tokenization_code_llama_fast.py#L52)

( vocab\_file = Nonetokenizer\_file = Noneclean\_up\_tokenization\_spaces = Falseunk\_token = '<unk>'bos\_token = '<s>'eos\_token = '</s>'prefix\_token = '▁<PRE>'middle\_token = '▁<MID>'suffix\_token = '▁<SUF>'eot\_token = '▁<EOT>'fill\_token = '<FILL\_ME>'additional\_special\_tokens = Noneadd\_bos\_token = Trueadd\_eos\_token = Falseuse\_default\_system\_prompt = False\*\*kwargs )

Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding.

This uses notably ByteFallback and no normalization.

```
>>> from transformers import CodeLlamaTokenizerFast

>>> tokenizer = CodeLlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
>>> tokenizer.encode("Hello this is a test")
[1, 15043, 445, 338, 263, 1243]
```

If you want to change the `bos_token` or the `eos_token`, make sure to specify them when initializing the model, or call `tokenizer.update_post_processor()` to make sure that the post-processing is correctly done (otherwise the values of the first token and final token of an encoded sequence will not be correct). For more details, checkout \[post-processors\] ([https://huggingface.co/docs/tokenizers/api/post-processors](https://huggingface.co/docs/tokenizers/api/post-processors)) documentation.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods. The default configuration match that of [codellama/CodeLlama-7b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf/blob/main/tokenizer_config.json) which supports prompt infilling.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/code_llama/tokenization_code_llama_fast.py#L396)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

list of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. The special tokens depend on calling set\_lang.

An NLLB sequence has the following format, where `X` represents the sequence:

-   `input_ids` (for encoder) `X [eos, src_lang_code]`
-   `decoder_input_ids`: (for decoder) `X [eos, tgt_lang_code]`

BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a separator.

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3770)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = Nonealready\_has\_special\_tokens: bool = False ) → A list of integers in the range \[0, 1\]

Parameters

-   **token\_ids\_0** (`List[int]`) — List of ids of the first sequence.
-   **token\_ids\_1** (`List[int]`, _optional_) — List of ids of the second sequence.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

Returns

A list of integers in the range \[0, 1\]

1 for a special token, 0 for a sequence token.

Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3305)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — The first tokenized sequence.
-   **token\_ids\_1** (`List[int]`, _optional_) — The second tokenized sequence.

The token type ids.

Create the token type IDs corresponding to the sequences passed. [What are token type IDs?](../glossary#token-type-ids)

Should be overridden in a subclass if the model has a special way of building those.

Updates the underlying post processor with the current `bos_token` and `eos_token`.

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/code_llama/tokenization_code_llama_fast.py#L325)

( save\_directory: strfilename\_prefix: typing.Optional\[str\] = None )