# GPTSAN-japanese

## Overview

The GPTSAN-japanese model was released in the repository by Toshiyuki Sakamoto (tanreinama).

GPTSAN is a Japanese language model using Switch Transformer. It has the same structure as the model introduced as Prefix LM in the T5 paper, and support both Text Generation and Masked Language Modeling tasks. These basic tasks similarly can fine-tune for translation or summarization.

### Generation

The `generate()` method can be used to generate text using GPTSAN-Japanese model.

```
>>> from transformers import AutoModel, AutoTokenizer
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
>>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese").cuda()
>>> x_tok = tokenizer("は、", prefix_text="織田信長", return_tensors="pt")
>>> torch.manual_seed(0)
>>> gen_tok = model.generate(x_tok.input_ids.cuda(), token_type_ids=x_tok.token_type_ids.cuda(), max_new_tokens=20)
>>> tokenizer.decode(gen_tok[0])
'織田信長は、2004年に『戦国BASARA』のために、豊臣秀吉'
```

## GPTSAN Features

GPTSAN has some unique features. It has a model structure of Prefix-LM. It works as a shifted Masked Language Model for Prefix Input tokens. Un-prefixed inputs behave like normal generative models. The Spout vector is a GPTSAN specific input. Spout is pre-trained with random inputs, but you can specify a class of text or an arbitrary vector during fine-tuning. This allows you to indicate the tendency of the generated text. GPTSAN has a sparse Feed Forward based on Switch-Transformer. You can also add other layers and train them partially. See the original GPTSAN repository for details.

### Prefix-LM Model

GPTSAN has the structure of the model named Prefix-LM in the `T5` paper. (The original GPTSAN repository calls it `hybrid`) In GPTSAN, the `Prefix` part of Prefix-LM, that is, the input position that can be referenced by both tokens, can be specified with any length. Arbitrary lengths can also be specified differently for each batch. This length applies to the text entered in `prefix_text` for the tokenizer. The tokenizer returns the mask of the `Prefix` part of Prefix-LM as `token_type_ids`. The model treats the part where `token_type_ids` is 1 as a `Prefix` part, that is, the input can refer to both tokens before and after.

Tips:

Specifying the Prefix part is done with a mask passed to self-attention. When token\_type\_ids=None or all zero, it is equivalent to regular causal mask

for example:

> > > x\_token = tokenizer(“ｱｲｳｴ”) input\_ids: | SOT | SEG | ｱ | ｲ | ｳ | ｴ | token\_type\_ids: | 1 | 0 | 0 | 0 | 0 | 0 | prefix\_lm\_mask: SOT | 1 0 0 0 0 0 | SEG | 1 1 0 0 0 0 | ｱ | 1 1 1 0 0 0 | ｲ | 1 1 1 1 0 0 | ｳ | 1 1 1 1 1 0 | ｴ | 1 1 1 1 1 1 |

> > > x\_token = tokenizer("", prefix\_text=“ｱｲｳｴ”) input\_ids: | SOT | ｱ | ｲ | ｳ | ｴ | SEG | token\_type\_ids: | 1 | 1 | 1 | 1 | 1 | 0 | prefix\_lm\_mask: SOT | 1 1 1 1 1 0 | ｱ | 1 1 1 1 1 0 | ｲ | 1 1 1 1 1 0 | ｳ | 1 1 1 1 1 0 | ｴ | 1 1 1 1 1 0 | SEG | 1 1 1 1 1 1 |

> > > x\_token = tokenizer(“ｳｴ”, prefix\_text=“ｱｲ”) input\_ids: | SOT | ｱ | ｲ | SEG | ｳ | ｴ | token\_type\_ids: | 1 | 1 | 1 | 0 | 0 | 0 | prefix\_lm\_mask: SOT | 1 1 1 0 0 0 | ｱ | 1 1 1 0 0 0 | ｲ | 1 1 1 0 0 0 | SEG | 1 1 1 1 0 0 | ｳ | 1 1 1 1 1 0 | ｴ | 1 1 1 1 1 1 |

### Spout Vector

A Spout Vector is a special vector for controlling text generation. This vector is treated as the first embedding in self-attention to bring extraneous attention to the generated tokens. In the pre-trained model published from `Tanrei/GPTSAN-japanese`, the Spout Vector is a 128-dimensional vector that passes through 8 fully connected layers in the model and is projected into the space acting as external attention. The Spout Vector projected by the fully connected layer is split to be passed to all self-attentions.

## GPTSanJapaneseConfig

### class transformers.GPTSanJapaneseConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gptsan_japanese/configuration_gptsan_japanese.py#L29)

( vocab\_size = 36000max\_position\_embeddings = 1280d\_model = 1024d\_ff = 8192d\_ext = 4096d\_spout = 128num\_switch\_layers = 10num\_ext\_layers = 0num\_heads = 16num\_experts = 16expert\_capacity = 128dropout\_rate = 0.0layer\_norm\_epsilon = 1e-05router\_bias = Falserouter\_jitter\_noise = 0.0router\_dtype = 'float32'router\_ignore\_padding\_tokens = Falseoutput\_hidden\_states = Falseoutput\_attentions = Falseinitializer\_factor = 0.002output\_router\_logits = Falseuse\_cache = Trueseparator\_token\_id = 35998pad\_token\_id = 35995eos\_token\_id = 35999\*\*kwargs )

This is the configuration class to store the configuration of a [GPTSanJapaneseModel](/docs/transformers/v4.34.0/en/model_doc/gptsan-japanese#transformers.GPTSanJapaneseModel). It is used to instantiate a GPTSANJapanese model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the GPTSANJapanese [Tanrei/GPTSAN-japanese](https://huggingface.co/Tanrei/GPTSAN-japanese) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## GPTSanJapaneseTokenizer

### class transformers.GPTSanJapaneseTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gptsan_japanese/tokenization_gptsan_japanese.py#L74)

( vocab\_fileemoji\_fileunk\_token = '<|nottoken|>'pad\_token = '<|separator|>'bos\_token = '<|startoftext|>'eos\_token = '<|endoftext|>'sep\_token = '<|segmenter|>'do\_clean\_text = False\*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — File containing the vocabulary.
-   **emoji\_file** (`str`) — File containing the emoji.
-   **unk\_token** (`str`, _optional_, defaults to `"<|nottoken|>"`) — The token used for unknown charactor
-   **pad\_token** (`str`, _optional_, defaults to `"<|separator|>"`) — The token used for padding
-   **bos\_token** (`str`, _optional_, defaults to `"<|startoftext|>""`) — The beginning of sequence token.
-   **eos\_token** (`str`, _optional_, defaults to `"<|endoftext|>"`) — The end of sequence token.
-   **sep\_token** (`str`, _optional_, defaults to `"<|segmenter|>"`) — A special token to separate token to prefix part and general input part.
-   **do\_clean\_text** (`bool`, _optional_, defaults to `False`) — Whether or not to clean text for URL, EMAIL, TEL, Japanese DATE and Japanese PRICE.

This tokenizer is based on GPTNeoXJapaneseTokenizer and has the following modifications

-   Decoding byte0~byte255 tokens correctly
-   Added bagofword token handling
-   Return token\_type\_ids for Prefix-LM model The bagofword token represents a repetition of the previous token and is converted to 3 consecutive tokens when decoding In addition, the original Japanese special Sub-Word-Encoding has been released in this repository ([https://github.com/tanreinama/Japanese-BPEEncoder\_V2](https://github.com/tanreinama/Japanese-BPEEncoder_V2)). The token\_type\_ids is a mask indicating the prefix input position of the Prefix-LM model. To specify a prefix position, specify a prefix input for prefix\_text, or specify a sentence of the prefix part and the part after it as a text pair of batch input.

Example:

```
>>> from transformers import GPTSanJapaneseTokenizer

>>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
>>> 
>>> tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"]
[35993, 35998, 34347, 31459, 30647, 31448, 25, 30659, 35729, 35676, 32417, 30647, 17750, 35589, 17750, 35590, 321, 1281]

>>> 
>>> tokenizer.decode(tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"])
'吾輩は猫である🐯。実は慶応(慶応)大学出身'
```

Example for Prefix-LM:

```
>>> from transformers import GPTSanJapaneseTokenizer

>>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
>>> tokenizer("実は慶応(慶應)大学出身", prefix_text="吾輩は猫である🐯。")["input_ids"]
[35993, 34347, 31459, 30647, 31448, 25, 30659, 35729, 35676, 35998, 32417, 30647, 17750, 35589, 17750, 35590, 321, 1281]

>>> 
>>> tokenizer("実は慶応(慶應)大学出身", prefix_text="吾輩は猫である🐯。")["token_type_ids"]
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

Example for batch encode:

```
>>> from transformers import GPTSanJapaneseTokenizer

>>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
>>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["input_ids"]
[[35993, 8640, 25948, 35998, 30647, 35675, 35999, 35999], [35993, 10382, 9868, 35998, 30646, 9459, 30646, 35675]]

>>> 
>>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["token_type_ids"]
[[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]]

>>> 
>>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["attention_mask"]
[[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]]
```

Converts a sequence of tokens (string) in a single string.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gptsan_japanese/tokenization_gptsan_japanese.py#L302)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None )

The tokenizer returns token\_type\_ids as separators between the Prefix part and the rest. token\_type\_ids is 1 for the Prefix part and 0 for the rest of the token.

Example:

```
>>> from transformers import GPTSanJapaneseTokenizer

>>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
>>> x_token = tokenizer("ｱｲｳｴ")
>>> 
>>> 

>>> x_token = tokenizer("", prefix_text="ｱｲｳｴ")
>>> 
>>> 

>>> x_token = tokenizer("ｳｴ", prefix_text="ｱｲ")
>>> 
>>> 
```

## GPTSanJapaneseModel

### class transformers.GPTSanJapaneseModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gptsan_japanese/modeling_gptsan_japanese.py#L868)

( config: GPTSanJapaneseConfig )

Parameters

-   **config** ([GPTSanJapaneseConfig](/docs/transformers/v4.34.0/en/model_doc/gptsan-japanese#transformers.GPTSanJapaneseConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare GPTSAN-japanese Model transformer outputting raw hidden-states without any specific head on top.

The [GPTSAN-japanese](https://github.com/tanreinama/GPTSAN) model was proposed in General-purpose Swich transformer based Japanese language model

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gptsan_japanese/modeling_gptsan_japanese.py#L902)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.FloatTensor\] = Nonespout: typing.Optional\[torch.FloatTensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Falseinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Noneoutput\_router\_logits: typing.Optional\[bool\] = Nonenum\_precontext: typing.Optional\[torch.LongTensor\] = None )

The [GPTSanJapaneseModel](/docs/transformers/v4.34.0/en/model_doc/gptsan-japanese#transformers.GPTSanJapaneseModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## GPTSanJapaneseForConditionalGeneration

### class transformers.GPTSanJapaneseForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gptsan_japanese/modeling_gptsan_japanese.py#L1113)

( config: GPTSanJapaneseConfig )

Parameters

-   **config** ([GPTSanJapaneseConfig](/docs/transformers/v4.34.0/en/model_doc/gptsan-japanese#transformers.GPTSanJapaneseConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare GPTSAN-japanese Model with a language modeling head.

The [GPTSAN-japanese](https://github.com/tanreinama/GPTSAN) model was proposed in General-purpose Swich transformer based Japanese language model

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gptsan_japanese/modeling_gptsan_japanese.py#L1124)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.FloatTensor\] = Nonespout: typing.Optional\[torch.FloatTensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Falseinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Noneoutput\_router\_logits: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.LongTensor\] = None )

The [GPTSanJapaneseForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/gptsan-japanese#transformers.GPTSanJapaneseForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

Text Generation with regular LM Model

```
>>> from transformers import AutoModel, AutoTokenizer, trainer_utils

>>> device = "cuda"
>>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese").to(device)
>>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
>>> x_token = tokenizer("織田信長は、", return_tensors="pt")
>>> trainer_utils.set_seed(30)
>>> input_ids = x_token.input_ids.to(device)
>>> gen_token = model.generate(input_ids, max_new_tokens=50)
>>> tokenizer.decode(gen_token[0])
"織田信長は、政治・軍事の中枢まで掌握した政治家であり、日本史上類を見ない驚異的な軍事侵攻を続け..."
```

Text Generation with Prefix-LM Model

```
>>> from transformers import AutoModel, AutoTokenizer, trainer_utils

>>> device = "cuda"
>>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese").to(device)
>>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
>>> x_token = tokenizer("", prefix_text="織田信長は、", return_tensors="pt")
>>> trainer_utils.set_seed(30)
>>> input_ids = x_token.input_ids.to(device)
>>> token_type_ids = x_token.token_type_ids.to(device)
>>> gen_token = model.generate(input_ids, token_type_ids=token_type_ids, max_new_tokens=50)
>>> tokenizer.decode(gen_token[0])
"織田信長は、政治・外交で数々の戦果を上げるが、1568年からは、いわゆる本能寺の変で細川晴元に暗殺される..."
```

Simultaneously Text Generation And Masked Language Model

```
>>> from transformers import AutoModel, AutoTokenizer, trainer_utils

>>> device = "cuda"
>>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese").to(device)
>>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
>>> masked_sentence = "武田信玄は、<|inputmask|>時代ファンならぜひ押さえ<|inputmask|>きたい名将の一人。"
>>> x_token = tokenizer("", prefix_text=masked_sentence, return_tensors="pt")
>>> trainer_utils.set_seed(30)
>>> input_ids = x_token.input_ids.to(device)
>>> token_type_ids = x_token.token_type_ids.to(device)
>>> out_lm_token = model.generate(input_ids, token_type_ids=token_type_ids, max_new_tokens=50)
>>> out_mlm_token = model(input_ids, token_type_ids=token_type_ids).logits.argmax(axis=-1)
>>> tokenizer.decode(out_mlm_token[0])
"武田信玄は、戦国時代ファンならぜひ押さえておきたい名将の一人。"

>>> tokenizer.decode(out_lm_token[0][input_ids.shape[1] :])
"武田氏の三代に渡った武田家のひとり\n甲斐市に住む、日本史上最大の戦国大名。..."
```