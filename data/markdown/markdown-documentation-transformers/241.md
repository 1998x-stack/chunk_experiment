# Persimmon

## Overview

The Persimmon model was created by [ADEPT](https://www.adept.ai/blog/persimmon-8b), and authored by Erich Elsen, Augustus Odena, Maxwell Nye, Sağnak Taşırlar, Tri Dao, Curtis Hawthorne, Deepak Moparthi, Arushi Somani.

The authors introduced Persimmon-8B, a decoder model based on the classic transformers architecture, with query and key normalization. Persimmon-8B is a fully permissively-licensed model with approximately 8 billion parameters, released under the Apache license. Some of the key attributes of Persimmon-8B are long context size (16K), performance, and capabilities for multimodal extensions.

The authors showcase their approach to model evaluation, focusing on practical text generation, mirroring how users interact with language models. The work also includes a comparative analysis, pitting Persimmon-8B against other prominent models (MPT 7B Instruct and Llama 2 Base 7B 1-Shot), across various evaluation tasks. The results demonstrate Persimmon-8B’s competitive performance, even with limited training data.

In terms of model details, the work outlines the architecture and training methodology of Persimmon-8B, providing insights into its design choices, sequence length, and dataset composition. The authors present a fast inference code that outperforms traditional implementations through operator fusion and CUDA graph utilization while maintaining code coherence. They express their anticipation of how the community will leverage this contribution to drive innovation, hinting at further upcoming releases as part of an ongoing series of developments.

The `Persimmon` models were trained using `bfloat16`, but the original inference uses `float16` The checkpoints uploaded on the hub use `torch_dtype = 'float16'` which will be used by the `AutoModel` API to cast the checkpoints from `torch.float32` to `torch.float16`.

The `dtype` of the online weights is mostly irrelevant, unless you are using `torch_dtype="auto"` when initializing a model using `model = AutoModelForCausalLM.from_pretrained("path", torch_dtype = "auto")`. The reason is that the model will first be downloaded ( using the `dtype` of the checkpoints online) then it will be cast to the default `dtype` of `torch` (becomes `torch.float32`). Users should specify the `torch_dtype` they want, and if they don’t it will be `torch.float32`.

Finetuning the model in `float16` is not recommended and known to produce `nan`, as such the model should be fine-tuned in `bfloat16`.

Tips:

-   To convert the model, you need to clone the original repository using `git clone https://github.com/persimmon-ai-labs/adept-inference`, then get the checkpoints:

```
git clone https://github.com/persimmon-ai-labs/adept-inference
wget https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_base_model_release.tar
tar -xvf 8b_base_model_release.tar
python src/transformers/models/persimmon/convert_persimmon_weights_to_hf.py  --input_dir /path/to/downloaded/persimmon/weights/ --output_dir /output/path \
    --pt_model_path /path/to/8b_chat_model_release/iter_0001251/mp_rank_00/model_optim_rng.pt
    --ada_lib_path /path/to/adept-inference
```

For the chat model:

```
wget https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_chat_model_release.tar
tar -xvf 8b_base_model_release.tar
```

Thereafter, models can be loaded via:

```
from transformers import PersimmonForCausalLM, PersimmonTokenizer

model = PersimmonForCausalLM.from_pretrained("/output/path")
tokenizer = PersimmonTokenizer.from_pretrained("/output/path")
```

This model was contributed by [ArthurZ](https://huggingface.co/ArthurZ). The original code can be found [here](https://github.com/persimmon-ai-labs/adept-inference).

-   Perismmon uses a `sentencepiece` based tokenizer, with a `Unigram` model. It supports bytefallback, which is only available in `tokenizers==0.14.0` for the fast tokenizer. The `LlamaTokenizer` is used as it is a standard wrapper around sentencepiece. The `chat` template will be updated with the templating functions in a follow up PR!
    
-   The authors suggest to use the following prompt format for the chat mode: `f"human: {prompt}\n\nadept:"`
    

## PersimmonConfig

### class transformers.PersimmonConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/persimmon/configuration_persimmon.py#L28)

( vocab\_size = 262144hidden\_size = 4096intermediate\_size = 16384num\_hidden\_layers = 36num\_attention\_heads = 64hidden\_act = 'relu2'max\_position\_embeddings = 16384initializer\_range = 0.02layer\_norm\_eps = 1e-05use\_cache = Truetie\_word\_embeddings = Falserope\_theta = 25000.0rope\_scaling = Noneqk\_layernorm = Truehidden\_dropout = 0.0attention\_dropout = 0.0partial\_rotary\_factor = 0.5pad\_token\_id = Nonebos\_token\_id = 1eos\_token\_id = 2\*\*kwargs )

This is the configuration class to store the configuration of a [PersimmonModel](/docs/transformers/v4.34.0/en/model_doc/persimmon#transformers.PersimmonModel). It is used to instantiate an Persimmon model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the [adept/persimmon-8b-base](https://huggingface.co/adept/persimmon-8b-base).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

```
>>> from transformers import PersimmonModel, PersimmonConfig

>>> 
>>> configuration = PersimmonConfig()
```

## PersimmonModel

### class transformers.PersimmonModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/persimmon/modeling_persimmon.py#L546)

( config: PersimmonConfig )

Parameters

-   **config** ([PersimmonConfig](/docs/transformers/v4.34.0/en/model_doc/persimmon#transformers.PersimmonConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights. config — PersimmonConfig

The bare Persimmon Model outputting raw hidden-states without any specific head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

Transformer decoder consisting of _config.num\_hidden\_layers_ layers. Each layer is a `PersimmonDecoderLayer`

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/persimmon/modeling_persimmon.py#L598)

( input\_ids: LongTensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None )

The [PersimmonModel](/docs/transformers/v4.34.0/en/model_doc/persimmon#transformers.PersimmonModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## PersimmonForCausalLM

### class transformers.PersimmonForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/persimmon/modeling_persimmon.py#L726)

( config )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/persimmon/modeling_persimmon.py#L763)

( input\_ids: LongTensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

The [PersimmonForCausalLM](/docs/transformers/v4.34.0/en/model_doc/persimmon#transformers.PersimmonForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, PersimmonForCausalLM

>>> model = PersimmonForCausalLM.from_pretrained("adept/persimmon-8b-base")
>>> tokenizer = AutoTokenizer.from_pretrained("adept/persimmon-8b-base")

>>> prompt = "human: Hey, what should I eat for dinner?"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> 
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
'human: Hey, what should I eat for dinner?\n\ncat: 🐱\n\nhuman: 😐\n\n'
```

## PersimmonForSequenceClassification

### class transformers.PersimmonForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/persimmon/modeling_persimmon.py#L908)

( config )

Parameters

-   **config** ([PersimmonConfig](/docs/transformers/v4.34.0/en/model_doc/persimmon#transformers.PersimmonConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Persimmon transformer with a sequence classification head on top (linear layer).

[PersimmonForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/persimmon#transformers.PersimmonForSequenceClassification) uses the last token in order to do the classification, as other causal models (e.g. GPT-2) do.

Since it does classification on the last token, it requires to know the position of the last token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in each row of the batch).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/persimmon/modeling_persimmon.py#L924)

( input\_ids: LongTensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None )

The [PersimmonForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/persimmon#transformers.PersimmonForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.