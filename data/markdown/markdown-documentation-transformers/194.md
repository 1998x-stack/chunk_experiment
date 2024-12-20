# LLaMA

## Overview

The LLaMA model was proposed in [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) by Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample. It is a collection of foundation language models ranging from 7B to 65B parameters.

The abstract from the paper is the following:

_We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We train our models on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly available datasets exclusively, without resorting to proprietary and inaccessible datasets. In particular, LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA-65B is competitive with the best models, Chinchilla-70B and PaLM-540B. We release all our models to the research community._

Tips:

-   Weights for the LLaMA models can be obtained from by filling out [this form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform?usp=send_form)
-   After downloading the weights, they will need to be converted to the Hugging Face Transformers format using the [conversion script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py). The script can be called with the following (example) command:

```
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

-   After conversion, the model and tokenizer can be loaded via:

```
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("/output/path")
model = LlamaForCausalLM.from_pretrained("/output/path")
```

Note that executing the script requires enough CPU RAM to host the whole model in float16 precision (even if the biggest versions come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM). For the 65B model, it’s thus 130GB of RAM needed.

-   The LLaMA tokenizer is a BPE model based on [sentencepiece](https://github.com/google/sentencepiece). One quirk of sentencepiece is that when decoding a sequence, if the first token is the start of the word (e.g. “Banana”), the tokenizer does not prepend the prefix space to the string.

This model was contributed by [zphang](https://huggingface.co/zphang) with contributions from [BlackSamorez](https://huggingface.co/BlackSamorez). The code of the implementation in Hugging Face is based on GPT-NeoX [here](https://github.com/EleutherAI/gpt-neox). The original code of the authors can be found [here](https://github.com/facebookresearch/llama).

Based on the original LLaMA model, Meta AI has released some follow-up works:

-   **Llama2**: Llama2 is an improved version of Llama with some architectural tweaks (Grouped Query Attention), and is pre-trained on 2Trillion tokens. Refer to the documentation of Llama2 which can be found [here](llama2).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with LLaMA. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

-   A [notebook](https://colab.research.google.com/github/bigscience-workshop/petals/blob/main/examples/prompt-tuning-sst2.ipynb#scrollTo=f04ba4d2) on how to use prompt tuning to adapt the LLaMA model for text classification task. 🌎

-   [StackLLaMA: A hands-on guide to train LLaMA with RLHF](https://huggingface.co/blog/stackllama#stackllama-a-hands-on-guide-to-train-llama-with-rlhf), a blog post about how to train LLaMA to answer questions on [Stack Exchange](https://stackexchange.com/) with RLHF.

⚗️ Optimization

-   A [notebook](https://colab.research.google.com/drive/1SQUXq1AMZPSLD4mk3A3swUIc6Y2dclme?usp=sharing) on how to fine-tune LLaMA model using xturing library on GPU which has limited memory. 🌎

⚡️ Inference

-   A [notebook](https://colab.research.google.com/github/DominguesM/alpaca-lora-ptbr-7b/blob/main/notebooks/02%20-%20Evaluate.ipynb) on how to run the LLaMA Model using PeftModel from the 🤗 PEFT library. 🌎
-   A [notebook](https://colab.research.google.com/drive/1l2GiSSPbajVyp2Nk3CFT4t3uH6-5TiBe?usp=sharing) on how to load a PEFT adapter LLaMA model with LangChain. 🌎

🚀 Deploy

-   A [notebook](https://colab.research.google.com/github/lxe/simple-llama-finetuner/blob/master/Simple_LLaMA_FineTuner.ipynb#scrollTo=3PM_DilAZD8T) on how to fine-tune LLaMA model using LoRA method via the 🤗 PEFT library with intuitive UI. 🌎
-   A [notebook](https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart-foundation-models/text-generation-open-llama.ipynb) on how to deploy Open-LLaMA model for text generation on Amazon SageMaker. 🌎

## LlamaConfig

### class transformers.LlamaConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/llama/configuration_llama.py#L31)

( vocab\_size = 32000 hidden\_size = 4096 intermediate\_size = 11008 num\_hidden\_layers = 32 num\_attention\_heads = 32 num\_key\_value\_heads = None hidden\_act = 'silu' max\_position\_embeddings = 2048 initializer\_range = 0.02 rms\_norm\_eps = 1e-06 use\_cache = True pad\_token\_id = None bos\_token\_id = 1 eos\_token\_id = 2 pretraining\_tp = 1 tie\_word\_embeddings = False rope\_theta = 10000.0 rope\_scaling = None attention\_bias = False \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 32000) — Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [LlamaModel](/docs/transformers/v4.34.0/en/model_doc/llama2#transformers.LlamaModel)
-   **hidden\_size** (`int`, _optional_, defaults to 4096) — Dimension of the hidden representations.
-   **intermediate\_size** (`int`, _optional_, defaults to 11008) — Dimension of the MLP representations.
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 32) — Number of hidden layers in the Transformer encoder.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 32) — Number of attention heads for each attention layer in the Transformer encoder.
-   **num\_key\_value\_heads** (`int`, _optional_) — This is the number of key\_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details checkout [this paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to` num\_attention\_heads\`.
-   **pretraining\_tp** (`int`, _optional_, defaults to `1`) — Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is necessary to ensure exact reproducibility of the pretraining results. Please refer to [this issue](https://github.com/pytorch/pytorch/issues/76232).
-   **hidden\_act** (`str` or `function`, _optional_, defaults to `"silu"`) — The non-linear activation function (function or string) in the decoder.
-   **max\_position\_embeddings** (`int`, _optional_, defaults to 2048) — The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens, Llama 2 up to 4096, CodeLlama up to 16384.
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **rms\_norm\_eps** (`float`, _optional_, defaults to 1e-12) — The epsilon used by the rms normalization layers.
-   **use\_cache** (`bool`, _optional_, defaults to `True`) — Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True`.
-   **tie\_word\_embeddings(`bool`,** _optional_, defaults to `False`) — Whether to tie weight embeddings
-   **rope\_theta** (`float`, _optional_, defaults to 10000.0) — The base period of the RoPE embeddings.
-   **rope\_scaling** (`Dict`, _optional_) — Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling strategies: linear and dynamic. Their scaling factor must be an float greater than 1. The expected format is `{"type": strategy name, "factor": scaling factor}`. When using this flag, don’t update `max_position_embeddings` to the expected new maximum. See the following thread for more information on how these scaling strategies behave: [https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically\_scaled\_rope\_further\_increases/](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/). This is an experimental feature, subject to breaking API changes in future versions.
-   **attention\_bias** (`bool`, defaults to `False`) — Whether to use a bias in the query, key, value and output projection layers during self-attention.
    
    Example —
    

This is the configuration class to store the configuration of a [LlamaModel](/docs/transformers/v4.34.0/en/model_doc/llama2#transformers.LlamaModel). It is used to instantiate an LLaMA model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the LLaMA-7B.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

```
>>> from transformers import LlamaModel, LlamaConfig

>>> 
>>> configuration = LlamaConfig()

>>> 
>>> model = LlamaModel(configuration)

>>> 
>>> configuration = model.config
```

## LlamaTokenizer

### class transformers.LlamaTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/llama/tokenization_llama.py#L66)

( vocab\_file unk\_token = '<unk>' bos\_token = '<s>' eos\_token = '</s>' pad\_token = None sp\_model\_kwargs: typing.Union\[typing.Dict\[str, typing.Any\], NoneType\] = None add\_bos\_token = True add\_eos\_token = False clean\_up\_tokenization\_spaces = False use\_default\_system\_prompt = True spaces\_between\_special\_tokens = False legacy = None \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — Path to the vocabulary file.
-   **legacy** (`bool`, _optional_) — Whether or not the `legacy` behavior of the tokenizer should be used. Legacy is before the merge of #24622 and #25224 which includes fixes to properly handle tokens that appear after special tokens. A simple example:
    
    -   `legacy=True`:
    

Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding. The default padding token is unset as there is no padding token in the original model.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/llama/tokenization_llama.py#L296)

( token\_ids\_0 token\_ids\_1 = None )

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/llama/tokenization_llama.py#L307)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None already\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/llama/tokenization_llama.py#L344)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/llama/tokenization_llama.py#L269)

( save\_directory filename\_prefix: typing.Optional\[str\] = None ) → `Tuple(str)`

Parameters

-   **save\_directory** (`str`) — The directory in which to save the vocabulary.

Paths to the files saved.

Save the vocabulary and special tokens file to a directory.

## LlamaTokenizerFast

### class transformers.LlamaTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/llama/tokenization_llama_fast.py#L57)

( vocab\_file = None tokenizer\_file = None clean\_up\_tokenization\_spaces = False unk\_token = '<unk>' bos\_token = '<s>' eos\_token = '</s>' add\_bos\_token = True add\_eos\_token = False use\_default\_system\_prompt = True \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .model extension) that contains the vocabulary necessary to instantiate a tokenizer.
-   **tokenizer\_file** (`str`) — [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that contains everything needed to load the tokenizer.
-   **clean\_up\_tokenization\_spaces** (`str`, _optional_, defaults to `False`) — Wether to cleanup spaces after decoding, cleanup consists in removing potential artifacts like extra spaces.
-   **bos\_token** (`str`, _optional_, defaults to `"<s>"`) — The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
-   **eos\_token** (`str`, _optional_, defaults to `"</s>"`) — The end of sequence token.
-   **unk\_token** (`str`, _optional_, defaults to `"<unk>"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding.

This uses notably ByteFallback and no normalization.

```
from transformers import LlamaTokenizerFast

tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
tokenizer.encode("Hello this is a test")
>>> [1, 15043, 445, 338, 263, 1243]
```

If you want to change the `bos_token` or the `eos_token`, make sure to specify them when initializing the model, or call `tokenizer.update_post_processor()` to make sure that the post-processing is correctly done (otherwise the values of the first token and final token of an encoded sequence will not be correct). For more details, checkout \[post-processors\] ([https://huggingface.co/docs/tokenizers/api/post-processors](https://huggingface.co/docs/tokenizers/api/post-processors)) documentation.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/llama/tokenization_llama_fast.py#L255)

( token\_ids\_0 token\_ids\_1 = None )

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

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3305)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — The first tokenized sequence.
-   **token\_ids\_1** (`List[int]`, _optional_) — The second tokenized sequence.

The token type ids.

Create the token type IDs corresponding to the sequences passed. [What are token type IDs?](../glossary#token-type-ids)

Should be overridden in a subclass if the model has a special way of building those.

Updates the underlying post processor with the current `bos_token` and `eos_token`.

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/llama/tokenization_llama_fast.py#L182)

( save\_directory: str filename\_prefix: typing.Optional\[str\] = None )

## LlamaModel

### class transformers.LlamaModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/llama/modeling_llama.py#L776)

( config: LlamaConfig )

Parameters

-   **config** ([LlamaConfig](/docs/transformers/v4.34.0/en/model_doc/llama2#transformers.LlamaConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights. config — LlamaConfig

The bare LLaMA Model outputting raw hidden-states without any specific head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

Transformer decoder consisting of _config.num\_hidden\_layers_ layers. Each layer is a `LlamaDecoderLayer`

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/llama/modeling_llama.py#L827)

( input\_ids: LongTensor = None attention\_mask: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None past\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None )

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    If `past_key_values` is used, optionally only the last `input_ids` have to be input (see `past_key_values`).
    
    If you want to change padding behavior, you should read `modeling_opt._prepare_decoder_attention_mask` and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
    If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

The [LlamaModel](/docs/transformers/v4.34.0/en/model_doc/llama2#transformers.LlamaModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## LlamaForCausalLM

### class transformers.LlamaForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/llama/modeling_llama.py#L960)

( config )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/llama/modeling_llama.py#L990)

( input\_ids: LongTensor = None attention\_mask: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None past\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    If `past_key_values` is used, optionally only the last `input_ids` have to be input (see `past_key_values`).
    
    If you want to change padding behavior, you should read `modeling_opt._prepare_decoder_attention_mask` and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
    If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
    
    Args — labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_): Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    

A [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LlamaConfig](/docs/transformers/v4.34.0/en/model_doc/llama2#transformers.LlamaConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [LlamaForCausalLM](/docs/transformers/v4.34.0/en/model_doc/llama2#transformers.LlamaForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, LlamaForCausalLM

>>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
>>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

>>> prompt = "Hey, are you conscious? Can you talk to me?"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> 
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
```

## LlamaForSequenceClassification

### class transformers.LlamaForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/llama/modeling_llama.py#L1139)

( config )

Parameters

-   **config** ([LlamaConfig](/docs/transformers/v4.34.0/en/model_doc/llama2#transformers.LlamaConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The LLaMa Model transformer with a sequence classification head on top (linear layer).

[LlamaForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/llama2#transformers.LlamaForSequenceClassification) uses the last token in order to do the classification, as other causal models (e.g. GPT-2) do.

Since it does classification on the last token, it requires to know the position of the last token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in each row of the batch).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/llama/modeling_llama.py#L1155)

( input\_ids: LongTensor = None attention\_mask: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None past\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None )

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    If `past_key_values` is used, optionally only the last `input_ids` have to be input (see `past_key_values`).
    
    If you want to change padding behavior, you should read `modeling_opt._prepare_decoder_attention_mask` and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
    If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

The [LlamaForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/llama2#transformers.LlamaForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.