# Open-Llama

This model is in maintenance mode only, so we won’t accept any new PRs changing its code.

If you run into any issues running this model, please reinstall the last version that supported this model: v4.31.0. You can do so by running the following command: `pip install -U transformers==4.31.0`.

This model differs from the [OpenLLaMA models](https://huggingface.co/models?search=openllama) on the Hugging Face Hub, which primarily use the [LLaMA](llama) architecture.

## Overview

The Open-Llama model was proposed in [Open-Llama project](https://github.com/s-JoL/Open-Llama) by community developer s-JoL.

The model is mainly based on LLaMA with some modifications, incorporating memory-efficient attention from Xformers, stable embedding from Bloom, and shared input-output embedding from PaLM. And the model is pre-trained on both Chinese and English, which gives it better performance on Chinese language tasks.

This model was contributed by [s-JoL](https://huggingface.co/s-JoL). The original code can be found [Open-Llama](https://github.com/s-JoL/Open-Llama). Checkpoint and usage can be found at [s-JoL/Open-Llama-V1](https://huggingface.co/s-JoL/Open-Llama-V1).

## OpenLlamaConfig

### class transformers.OpenLlamaConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deprecated/open_llama/configuration_open_llama.py#L33)

( vocab\_size = 100000hidden\_size = 4096intermediate\_size = 11008num\_hidden\_layers = 32num\_attention\_heads = 32hidden\_act = 'silu'max\_position\_embeddings = 2048initializer\_range = 0.02rms\_norm\_eps = 1e-06use\_cache = Truepad\_token\_id = 0bos\_token\_id = 1eos\_token\_id = 2tie\_word\_embeddings = Falseuse\_memory\_efficient\_attention = Truehidden\_dropout\_prob = 0.1attention\_dropout\_prob = 0.1use\_stable\_embedding = Trueshared\_input\_output\_embedding = Truerope\_scaling = None\*\*kwargs )

This is the configuration class to store the configuration of a [OpenLlamaModel](/docs/transformers/v4.34.0/en/model_doc/open-llama#transformers.OpenLlamaModel). It is used to instantiate an Open-Llama model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the [s-JoL/Open-Llama-V1](https://huggingface.co/s-JoL/Open-Llama-V1).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

```
>>> from transformers import OpenLlamaModel, OpenLlamaConfig

>>> 
>>> configuration = OpenLlamaConfig()

>>> 
>>> model = OpenLlamaModel(configuration)

>>> 
>>> configuration = model.config
```

## OpenLlamaModel

### class transformers.OpenLlamaModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deprecated/open_llama/modeling_open_llama.py#L533)

( config: OpenLlamaConfig )

Parameters

-   **config** ([OpenLlamaConfig](/docs/transformers/v4.34.0/en/model_doc/open-llama#transformers.OpenLlamaConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights. config — OpenLlamaConfig

The bare Open-Llama Model outputting raw hidden-states without any specific head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

Transformer decoder consisting of _config.num\_hidden\_layers_ layers. Each layer is a `OpenLlamaDecoderLayer`

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deprecated/open_llama/modeling_open_llama.py#L588)

( input\_ids: LongTensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None )

The [OpenLlamaModel](/docs/transformers/v4.34.0/en/model_doc/open-llama#transformers.OpenLlamaModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## OpenLlamaForCausalLM

### class transformers.OpenLlamaForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deprecated/open_llama/modeling_open_llama.py#L721)

( config )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deprecated/open_llama/modeling_open_llama.py#L751)

( input\_ids: LongTensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

The [OpenLlamaForCausalLM](/docs/transformers/v4.34.0/en/model_doc/open-llama#transformers.OpenLlamaForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, OpenLlamaForCausalLM

>>> model = OpenLlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
>>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

>>> prompt = "Hey, are you conscious? Can you talk to me?"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> 
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
```

## OpenLlamaForSequenceClassification

### class transformers.OpenLlamaForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deprecated/open_llama/modeling_open_llama.py#L902)

( config )

Parameters

-   **config** ([OpenLlamaConfig](/docs/transformers/v4.34.0/en/model_doc/open-llama#transformers.OpenLlamaConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The LLaMa Model transformer with a sequence classification head on top (linear layer).

[OpenLlamaForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/open-llama#transformers.OpenLlamaForSequenceClassification) uses the last token in order to do the classification, as other causal models (e.g. GPT-2) do.

Since it does classification on the last token, it requires to know the position of the last token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in each row of the batch).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deprecated/open_llama/modeling_open_llama.py#L918)

( input\_ids: LongTensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None )

The [OpenLlamaForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/open-llama#transformers.OpenLlamaForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.