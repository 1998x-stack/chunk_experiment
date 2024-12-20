# GPT-NeoX-Japanese

## Overview

We introduce GPT-NeoX-Japanese, which is an autoregressive language model for Japanese, trained on top of [https://github.com/EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox). Japanese is a unique language with its large vocabulary and a combination of hiragana, katakana, and kanji writing scripts. To address this distinct structure of the Japanese language, we use a [special sub-word tokenizer](https://github.com/tanreinama/Japanese-BPEEncoder_V2). We are very grateful to _tanreinama_ for open-sourcing this incredibly helpful tokenizer. Following the recommendations from Google’s research on [PaLM](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html), we have removed bias parameters from transformer blocks, achieving better model performance. Please refer [this article](https://medium.com/ml-abeja/training-a-better-gpt-2-93b157662ae4) in detail.

Development of the model was led by [Shinya Otani](https://github.com/SO0529), [Takayoshi Makabe](https://github.com/spider-man-tm), [Anuj Arora](https://github.com/Anuj040), and [Kyo Hattori](https://github.com/go5paopao) from [ABEJA, Inc.](https://www.abejainc.com/). For more information on this model-building activity, please refer [here (ja)](https://tech-blog.abeja.asia/entry/abeja-gpt-project-202207).

### Generation

The `generate()` method can be used to generate text using GPT NeoX Japanese model.

```
>>> from transformers import GPTNeoXJapaneseForCausalLM, GPTNeoXJapaneseTokenizer

>>> model = GPTNeoXJapaneseForCausalLM.from_pretrained("abeja/gpt-neox-japanese-2.7b")
>>> tokenizer = GPTNeoXJapaneseTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")

>>> prompt = "人とAIが協調するためには、"

>>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

>>> gen_tokens = model.generate(
...     input_ids,
...     do_sample=True,
...     temperature=0.9,
...     max_length=100,
... )
>>> gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

>>> print(gen_text)
人とAIが協調するためには、AIと人が共存し、AIを正しく理解する必要があります。
```

## Documentation resources

-   [Causal language modeling task guide](../tasks/language_modeling)

## GPTNeoXJapaneseConfig

### class transformers.GPTNeoXJapaneseConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt_neox_japanese/configuration_gpt_neox_japanese.py#L28)

( vocab\_size = 32000hidden\_size = 2560num\_hidden\_layers = 32num\_attention\_heads = 32intermediate\_multiple\_size = 4hidden\_act = 'gelu'rotary\_pct = 1.0rotary\_emb\_base = 10000max\_position\_embeddings = 2048initializer\_range = 0.02layer\_norm\_eps = 1e-05use\_cache = Truebos\_token\_id = 31996eos\_token\_id = 31999attention\_dropout = 0.1hidden\_dropout = 0.0\*\*kwargs )

This is the configuration class to store the configuration of a `GPTNeoXModelJapanese`. It is used to instantiate a GPTNeoX model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the GPTNeoXJapanese [abeja/gpt-neox-japanese-2.7b](https://huggingface.co/abeja/gpt-neox-japanese-2.7b) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information. Default configs is set as 2.7B model

```
>>> from transformers import GPTNeoXJapaneseConfig, GPTNeoXJapaneseModel

>>> 
>>> configuration = GPTNeoXJapaneseConfig()

>>> 
>>> model = GPTNeoXJapaneseModel(configuration)

>>> 
>>> configuration = model.config
```

## GPTNeoXJapaneseTokenizer

### class transformers.GPTNeoXJapaneseTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt_neox_japanese/tokenization_gpt_neox_japanese.py#L66)

( vocab\_fileemoji\_fileunk\_token = '<|endoftext|>'pad\_token = '<|endoftext|>'bos\_token = '<|startoftext|>'eos\_token = '<|endoftext|>'do\_clean\_text = False\*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — File containing the vocabulary.
-   **emoji\_file** (`str`) — File containing the emoji.
-   **unk\_token** (`str`, _optional_, defaults to `"<|endoftext|>"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **pad\_token** (`str`, _optional_, defaults to `"<|endoftext|>"`) — The token used for padding
-   **bos\_token** (`str`, _optional_, defaults to `"<|startoftext|>"`) — The beginning of sequence token.
-   **eos\_token** (`str`, _optional_, defaults to `"<|endoftext|>"`) — The end of sequence token.
-   **do\_clean\_text** (`bool`, _optional_, defaults to `False`) — Whether or not to clean text for URL, EMAIL, TEL, Japanese DATE and Japanese PRICE.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) and is based on Japanese special Sub-Word-Encoding that is used in this repository ([https://github.com/tanreinama/Japanese-BPEEncoder\_V2](https://github.com/tanreinama/Japanese-BPEEncoder_V2)). Check the repository for details. Japanese has a relatively large vocabulary and there is no separation between words. Furthermore, the language is a combination of hiragana, katakana, and kanji, and variants such as “1” and “①” are often used. In order to cope with these, this tokenizer has the following features

-   Subword-by-subword segmentation, which is intermediate between byte strings and morphological analysis.
-   BPEs are created for each Kanji, Hiragana, and Katakana character, and there are no BPEs that cross character types, such as Kanji + Hiragana or Hiragana + Katakana.
-   All-byte encoding that does not require <unk>.
-   Independent of UTF codes such as 2-byte and 3-byte characters
-   Conversion of heterographs to the same token\_id
-   Emoji and Emoticon are grouped into 12 types as special tags.

Example:

```
>>> from transformers import GPTNeoXJapaneseTokenizer

>>> tokenizer = GPTNeoXJapaneseTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")
>>> 
>>> tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"]
[30014, 26883, 26638, 27228, 25, 26650, 31732, 31679, 27809, 26638, 17749, 31592, 17749, 31593, 321, 1281]

>>> 
>>> tokenizer.decode(tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"])
'吾輩は猫である🐯。実は慶応(慶応)大学出身'
```

Converts a sequence of tokens (string) in a single string.

## GPTNeoXJapaneseModel

### class transformers.GPTNeoXJapaneseModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt_neox_japanese/modeling_gpt_neox_japanese.py#L440)

( config )

Parameters

-   **config** ([~GPTNeoXJapaneseConfig](/docs/transformers/v4.34.0/en/model_doc/gpt_neox_japanese#transformers.GPTNeoXJapaneseConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare GPTNeoXJapanese Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt_neox_japanese/modeling_gpt_neox_japanese.py#L460)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

The [GPTNeoXJapaneseModel](/docs/transformers/v4.34.0/en/model_doc/gpt_neox_japanese#transformers.GPTNeoXJapaneseModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, GPTNeoXJapaneseModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")
>>> model = GPTNeoXJapaneseModel.from_pretrained("abeja/gpt-neox-japanese-2.7b")

>>> inputs = tokenizer("日本語のGPT-neoxがHugging Faceで使えます😀", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## GPTNeoXJapaneseForCausalLM

### class transformers.GPTNeoXJapaneseForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt_neox_japanese/modeling_gpt_neox_japanese.py#L595)

( config )

Parameters

-   **config** ([~GPTNeoXJapaneseConfig](/docs/transformers/v4.34.0/en/model_doc/gpt_neox_japanese#transformers.GPTNeoXJapaneseConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

GPTNeoXJapanese Model with a `language modeling` head on top for Classifier Model fine-tuning. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt_neox_japanese/modeling_gpt_neox_japanese.py#L614)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

The [GPTNeoXJapaneseForCausalLM](/docs/transformers/v4.34.0/en/model_doc/gpt_neox_japanese#transformers.GPTNeoXJapaneseForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, GPTNeoXJapaneseForCausalLM, GPTNeoXJapaneseConfig
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")
>>> config = GPTNeoXJapaneseConfig.from_pretrained("abeja/gpt-neox-japanese-2.7b")
>>> config.is_decoder = True
>>> model = GPTNeoXJapaneseForCausalLM.from_pretrained("abeja/gpt-neox-japanese-2.7b", config=config)

>>> inputs = tokenizer("日本語のGPT-neoxがHugging Faceで使えます😀", return_tensors="pt")
>>> outputs = model(**inputs)

>>> prediction_logits = outputs.logits
```