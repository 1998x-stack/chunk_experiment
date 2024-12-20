# ERNIE

## Overview

ERNIE is a series of powerful models proposed by baidu, especially in Chinese tasks, including \[ERNIE1.0\](https://arxiv.org/abs/1904.09223), \[ERNIE2.0\](https://ojs.aaai.org/index.php/AAAI/article/view/6428), \[ERNIE3.0\](https://arxiv.org/abs/2107.02137), \[ERNIE-Gram\](https://arxiv.org/abs/2010.12148), \[ERNIE-health\](https://arxiv.org/abs/2110.07244), etc.

These models are contributed by [nghuyong](https://huggingface.co/nghuyong) and the official code can be found in [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP) (in PaddlePaddle).

### How to use

Take \`ernie-1.0-base-zh\` as an example:

```
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0-base-zh")
model = AutoModel.from_pretrained("nghuyong/ernie-1.0-base-zh")
```

### Supported Models

| Model Name | Language | Description |
| --- | --- | --- |
| ernie-1.0-base-zh | Chinese | Layer:12, Heads:12, Hidden:768 |
| ernie-2.0-base-en | English | Layer:12, Heads:12, Hidden:768 |
| ernie-2.0-large-en | English | Layer:24, Heads:16, Hidden:1024 |
| ernie-3.0-base-zh | Chinese | Layer:12, Heads:12, Hidden:768 |
| ernie-3.0-medium-zh | Chinese | Layer:6, Heads:12, Hidden:768 |
| ernie-3.0-mini-zh | Chinese | Layer:6, Heads:12, Hidden:384 |
| ernie-3.0-micro-zh | Chinese | Layer:4, Heads:12, Hidden:384 |
| ernie-3.0-nano-zh | Chinese | Layer:4, Heads:12, Hidden:312 |
| ernie-health-zh | Chinese | Layer:12, Heads:12, Hidden:768 |
| ernie-gram-zh | Chinese | Layer:12, Heads:12, Hidden:768 |

You can find all the supported models from huggingface’s model hub: [huggingface.co/nghuyong](https://huggingface.co/nghuyong), and model details from paddle’s official repo: [PaddleNLP](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers/ERNIE/contents.html) and [ERNIE](https://github.com/PaddlePaddle/ERNIE/blob/repro).

## Documentation resources

-   [Text classification task guide](../tasks/sequence_classification)
-   [Token classification task guide](../tasks/token_classification)
-   [Question answering task guide](../tasks/question_answering)
-   [Causal language modeling task guide](../tasks/language_modeling)
-   [Masked language modeling task guide](../tasks/masked_language_modeling)
-   [Multiple choice task guide](../tasks/multiple_choice)

## ErnieConfig

### class transformers.ErnieConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ernie/configuration_ernie.py#L41)

( vocab\_size = 30522hidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout\_prob = 0.1attention\_probs\_dropout\_prob = 0.1max\_position\_embeddings = 512type\_vocab\_size = 2task\_type\_vocab\_size = 3use\_task\_id = Falseinitializer\_range = 0.02layer\_norm\_eps = 1e-12pad\_token\_id = 0position\_embedding\_type = 'absolute'use\_cache = Trueclassifier\_dropout = None\*\*kwargs )

This is the configuration class to store the configuration of a [ErnieModel](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.ErnieModel) or a `TFErnieModel`. It is used to instantiate a ERNIE model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the ERNIE [nghuyong/ernie-3.0-base-zh](https://huggingface.co/nghuyong/ernie-3.0-base-zh) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import ErnieConfig, ErnieModel

>>> 
>>> configuration = ErnieConfig()

>>> 
>>> model = ErnieModel(configuration)

>>> 
>>> configuration = model.config
```

## Ernie specific outputs

### class transformers.models.ernie.modeling\_ernie.ErnieForPreTrainingOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ernie/modeling_ernie.py#L690)

( loss: typing.Optional\[torch.FloatTensor\] = Noneprediction\_logits: FloatTensor = Noneseq\_relationship\_logits: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Output type of [ErnieForPreTraining](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.ErnieForPreTraining).

## ErnieModel

### class transformers.ErnieModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ernie/modeling_ernie.py#L798)

( configadd\_pooling\_layer = True )

Parameters

-   **config** ([ErnieConfig](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.ErnieConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Ernie Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of cross-attention is added between the self-attention layers, following the architecture described in [Attention is all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ernie/modeling_ernie.py#L841)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Nonetask\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.Tensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or `tuple(torch.FloatTensor)`

The [ErnieModel](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.ErnieModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, ErnieModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0-base-zh")
>>> model = ErnieModel.from_pretrained("nghuyong/ernie-1.0-base-zh")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## ErnieForPreTraining

### class transformers.ErnieForPreTraining

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ernie/modeling_ernie.py#L987)

( config )

Parameters

-   **config** ([ErnieConfig](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.ErnieConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Ernie Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next sentence prediction (classification)` head.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ernie/modeling_ernie.py#L1008)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Nonetask\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Nonenext\_sentence\_label: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.models.ernie.modeling\_ernie.ErnieForPreTrainingOutput](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.models.ernie.modeling_ernie.ErnieForPreTrainingOutput) or `tuple(torch.FloatTensor)`

The [ErnieForPreTraining](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.ErnieForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, ErnieForPreTraining
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0-base-zh")
>>> model = ErnieForPreTraining.from_pretrained("nghuyong/ernie-1.0-base-zh")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> prediction_logits = outputs.prediction_logits
>>> seq_relationship_logits = outputs.seq_relationship_logits
```

## ErnieForCausalLM

### class transformers.ErnieForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ernie/modeling_ernie.py#L1098)

( config )

Parameters

-   **config** ([ErnieConfig](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.ErnieConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Ernie Model with a `language modeling` head on top for CLM fine-tuning.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ernie/modeling_ernie.py#L1122)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Nonetask\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.Tensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.Tensor\]\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

The [ErnieForCausalLM](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.ErnieForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoTokenizer, ErnieForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0-base-zh")
>>> model = ErnieForCausalLM.from_pretrained("nghuyong/ernie-1.0-base-zh")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs, labels=inputs["input_ids"])
>>> loss = outputs.loss
>>> logits = outputs.logits
```

## ErnieForMaskedLM

### class transformers.ErnieForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ernie/modeling_ernie.py#L1246)

( config )

Parameters

-   **config** ([ErnieConfig](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.ErnieConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Ernie Model with a `language modeling` head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ernie/modeling_ernie.py#L1273)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Nonetask\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.Tensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

The [ErnieForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.ErnieForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, ErnieForMaskedLM
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0-base-zh")
>>> model = ErnieForMaskedLM.from_pretrained("nghuyong/ernie-1.0-base-zh")

>>> inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

>>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
>>> tokenizer.decode(predicted_token_id)
'paris'

>>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
>>> 
>>> labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

>>> outputs = model(**inputs, labels=labels)
>>> round(outputs.loss.item(), 2)
0.88
```

## ErnieForNextSentencePrediction

### class transformers.ErnieForNextSentencePrediction

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ernie/modeling_ernie.py#L1362)

( config )

Parameters

-   **config** ([ErnieConfig](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.ErnieConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Ernie Model with a `next sentence prediction (classification)` head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ernie/modeling_ernie.py#L1373)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Nonetask\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None\*\*kwargs ) → [transformers.modeling\_outputs.NextSentencePredictorOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.NextSentencePredictorOutput) or `tuple(torch.FloatTensor)`

The [ErnieForNextSentencePrediction](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.ErnieForNextSentencePrediction) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, ErnieForNextSentencePrediction
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0-base-zh")
>>> model = ErnieForNextSentencePrediction.from_pretrained("nghuyong/ernie-1.0-base-zh")

>>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
>>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
>>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

>>> outputs = model(**encoding, labels=torch.LongTensor([1]))
>>> logits = outputs.logits
>>> assert logits[0, 0] < logits[0, 1]  
```

## ErnieForSequenceClassification

### class transformers.ErnieForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ernie/modeling_ernie.py#L1470)

( config )

Parameters

-   **config** ([ErnieConfig](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.ErnieConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Ernie Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ernie/modeling_ernie.py#L1487)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Nonetask\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None )

The [ErnieForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.ErnieForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## ErnieForMultipleChoice

### class transformers.ErnieForMultipleChoice

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ernie/modeling_ernie.py#L1569)

( config )

Parameters

-   **config** ([ErnieConfig](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.ErnieConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Ernie Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a softmax) e.g. for RocStories/SWAG tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ernie/modeling_ernie.py#L1584)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Nonetask\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.MultipleChoiceModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or `tuple(torch.FloatTensor)`

The [ErnieForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.ErnieForMultipleChoice) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, ErnieForMultipleChoice
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0-base-zh")
>>> model = ErnieForMultipleChoice.from_pretrained("nghuyong/ernie-1.0-base-zh")

>>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
>>> choice0 = "It is eaten with a fork and a knife."
>>> choice1 = "It is eaten while held in the hand."
>>> labels = torch.tensor(0).unsqueeze(0)  

>>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="pt", padding=True)
>>> outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  

>>> 
>>> loss = outputs.loss
>>> logits = outputs.logits
```

## ErnieForTokenClassification

### class transformers.ErnieForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ernie/modeling_ernie.py#L1666)

( config )

Parameters

-   **config** ([ErnieConfig](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.ErnieConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Ernie Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ernie/modeling_ernie.py#L1682)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Nonetask\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None )

The [ErnieForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.ErnieForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## ErnieForQuestionAnswering

### class transformers.ErnieForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ernie/modeling_ernie.py#L1745)

( config )

Parameters

-   **config** ([ErnieConfig](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.ErnieConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Ernie Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ernie/modeling_ernie.py#L1757)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Nonetask\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonestart\_positions: typing.Optional\[torch.Tensor\] = Noneend\_positions: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None )

The [ErnieForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.ErnieForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.