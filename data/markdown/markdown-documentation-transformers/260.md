# RWKV

## Overview

The RWKV model was proposed in [this repo](https://github.com/BlinkDL/RWKV-LM)

It suggests a tweak in the traditional Transformer attention to make it linear. This way, the model can be used as recurrent network: passing inputs for timestamp 0 and timestamp 1 together is the same as passing inputs at timestamp 0, then inputs at timestamp 1 along with the state of timestamp 0 (see example below).

This can be more efficient than a regular Transformer and can deal with sentence of any length (even if the model uses a fixed context length for training).

This model was contributed by [sgugger](https://huggingface.co/sgugger). The original code can be found [here](https://github.com/BlinkDL/RWKV-LM).

Example of use as an RNN:

```
import torch
from transformers import AutoTokenizer, RwkvConfig, RwkvModel

model = RwkvModel.from_pretrained("sgugger/rwkv-430M-pile")
tokenizer = AutoTokenizer.from_pretrained("sgugger/rwkv-430M-pile")

inputs = tokenizer("This is an example.", return_tensors="pt")

outputs = model(inputs["input_ids"])
output_whole = outputs.last_hidden_state

outputs = model(inputs["input_ids"][:, :2])
output_one = outputs.last_hidden_state


outputs = model(inputs["input_ids"][:, 2:], state=outputs.state)
output_two = outputs.last_hidden_state

torch.allclose(torch.cat([output_one, output_two], dim=1), output_whole, atol=1e-5)
```

If you want to make sure the model stops generating when `'\n\n'` is detected, we recommend using the following stopping criteria:

```
from transformers import StoppingCriteria

class RwkvStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = [187,187], eos_token_id = 537):
        self.eos_sequence = eos_sequence
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_2_ids = input_ids[:,-2:].tolist()
        return self.eos_sequence in last_2_ids


output = model.generate(inputs["input_ids"], max_new_tokens=64, stopping_criteria = [RwkvStoppingCriteria()])
```

## RwkvConfig

### class transformers.RwkvConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rwkv/configuration_rwkv.py#L38)

( vocab\_size = 50277context\_length = 1024hidden\_size = 4096num\_hidden\_layers = 32attention\_hidden\_size = Noneintermediate\_size = Nonelayer\_norm\_epsilon = 1e-05bos\_token\_id = 0eos\_token\_id = 0rescale\_every = 6tie\_word\_embeddings = Falseuse\_cache = True\*\*kwargs )

This is the configuration class to store the configuration of a [RwkvModel](/docs/transformers/v4.34.0/en/model_doc/rwkv#transformers.RwkvModel). It is used to instantiate a RWKV model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the RWVK-4 [RWKV/rwkv-4-169m-pile](https://huggingface.co/RWKV/rwkv-4-169m-pile) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import RwkvConfig, RwkvModel

>>> 
>>> configuration = RwkvConfig()

>>> 
>>> model = RwkvModel(configuration)

>>> 
>>> configuration = model.config
```

## RwkvModel

### class transformers.RwkvModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rwkv/modeling_rwkv.py#L600)

( config )

Parameters

-   **config** ([RwkvConfig](/docs/transformers/v4.34.0/en/model_doc/rwkv#transformers.RwkvConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare RWKV Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rwkv/modeling_rwkv.py#L621)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.LongTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonestate: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.rwkv.modeling_rwkv.RwkvOutput` or `tuple(torch.FloatTensor)`

The [RwkvModel](/docs/transformers/v4.34.0/en/model_doc/rwkv#transformers.RwkvModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, RwkvModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-4-169m-pile")
>>> model = RwkvModel.from_pretrained("RWKV/rwkv-4-169m-pile")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## RwkvLMHeadModel

### class transformers.RwkvForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rwkv/modeling_rwkv.py#L776)

( config )

Parameters

-   **config** ([RwkvConfig](/docs/transformers/v4.34.0/en/model_doc/rwkv#transformers.RwkvConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The RWKV Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rwkv/modeling_rwkv.py#L807)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.LongTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonestate: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.rwkv.modeling_rwkv.RwkvCausalLMOutput` or `tuple(torch.FloatTensor)`

The [RwkvForCausalLM](/docs/transformers/v4.34.0/en/model_doc/rwkv#transformers.RwkvForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoTokenizer, RwkvForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-4-169m-pile")
>>> model = RwkvForCausalLM.from_pretrained("RWKV/rwkv-4-169m-pile")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs, labels=inputs["input_ids"])
>>> loss = outputs.loss
>>> logits = outputs.logits
```

## Rwkv attention and the recurrent formulas

In a traditional auto-regressive Transformer, attention is written as

O\=softmax(QKT/d)VO = \\hbox{softmax}(QK^{T} / \\sqrt{d}) V

with QQ, KK and VV are matrices of shape `seq_len x hidden_size` named query, key and value (they are actually bigger matrices with a batch dimension and an attention head dimension but we’re only interested in the last two, which is where the matrix product is taken, so for the sake of simplicity we only consider those two). The product QKTQK^{T} then has shape `seq_len x seq_len` and we can take the maxtrix product with VV to get the output OO of the same shape as the others.

Replacing the softmax by its value gives:

Oi\=∑j\=1ieQiKjT/dVj∑j\=1ieQiKjT/dO\_{i} = \\frac{\\sum\_{j=1}^{i} e^{Q\_{i} K\_{j}^{T} / \\sqrt{d}} V\_{j}}{\\sum\_{j=1}^{i} e^{Q\_{i} K\_{j}^{T} / \\sqrt{d}}}

Note that the entries in QKTQK^{T} corresponding to j\>ij > i are masked (the sum stops at j) because the attention is not allowed to look at future tokens (only past ones).

In comparison, the RWKV attention is given by

Oi\=σ(Ri)∑j\=1ieWi−j+KjVj∑j\=1ieWi−j+KjO\_{i} = \\sigma(R\_{i}) \\frac{\\sum\_{j=1}^{i} e^{W\_{i-j} + K\_{j}} V\_{j}}{\\sum\_{j=1}^{i} e^{W\_{i-j} + K\_{j}}}

where RR is a new matrix called receptance by the author, KK and VV are still the key and value (σ\\sigma here is the sigmoid function). WW is a new vector that represents the position of the token and is given by

W0\=u and Wk\=(k−1)w for k≥1W\_{0} = u \\hbox{ and } W\_{k} = (k-1)w \\hbox{ for } k \\geq 1

with uu and ww learnable parameters called in the code `time_first` and `time_decay` respectively. The numerator and denominator can both be expressed recursively. Naming them NiN\_{i} and DiD\_{i} we have:

Ni\=eu+KiVi+N^i where N^i\=eKi−1Vi−1+ew+Ki−2Vi−2⋯+e(i−2)w+K1V1N\_{i} = e^{u + K\_{i}} V\_{i} + \\hat{N}\_{i} \\hbox{ where } \\hat{N}\_{i} = e^{K\_{i-1}} V\_{i-1} + e^{w + K\_{i-2}} V\_{i-2} \\cdots + e^{(i-2)w + K\_{1}} V\_{1}

so N^i\\hat{N}\_{i} (called `numerator_state` in the code) satistfies

N^0\=0 and N^j+1\=eKjVj+ewN^j\\hat{N}\_{0} = 0 \\hbox{ and } \\hat{N}\_{j+1} = e^{K\_{j}} V\_{j} + e^{w} \\hat{N}\_{j}

and

Di\=eu+Ki+D^i where D^i\=eKi−1+ew+Ki−2⋯+e(i−2)w+K1D\_{i} = e^{u + K\_{i}} + \\hat{D}\_{i} \\hbox{ where } \\hat{D}\_{i} = e^{K\_{i-1}} + e^{w + K\_{i-2}} \\cdots + e^{(i-2)w + K\_{1}}

so D^i\\hat{D}\_{i} (called `denominator_state` in the code) satistfies

D^0\=0 and D^j+1\=eKj+ewD^j\\hat{D}\_{0} = 0 \\hbox{ and } \\hat{D}\_{j+1} = e^{K\_{j}} + e^{w} \\hat{D}\_{j}

The actual recurrent formula used are a tiny bit more complex, as for numerical stability we don’t want to compute exponentials of big numbers. Usually the softmax is not computed as is, but the exponential of the maximum term is divided of the numerator and denominator:

exi∑j\=1nexj\=exi−M∑j\=1nexj−M\\frac{e^{x\_{i}}}{\\sum\_{j=1}^{n} e^{x\_{j}}} = \\frac{e^{x\_{i} - M}}{\\sum\_{j=1}^{n} e^{x\_{j} - M}}

with MM the maximum of all xjx\_{j}. So here on top of saving the numerator state (N^\\hat{N}) and the denominator state (D^\\hat{D}) we also keep track of the maximum of all terms encountered in the exponentials. So we actually use

N~i\=e−MiN^i and D~i\=e−MiD^i\\tilde{N}\_{i} = e^{-M\_{i}} \\hat{N}\_{i} \\hbox{ and } \\tilde{D}\_{i} = e^{-M\_{i}} \\hat{D}\_{i}

defined by the following recurrent formulas:

N~0\=0 and N~j+1\=eKj−qVj+ew+Mj−qN~j where q\=max⁡(Kj,w+Mj)\\tilde{N}\_{0} = 0 \\hbox{ and } \\tilde{N}\_{j+1} = e^{K\_{j} - q} V\_{j} + e^{w + M\_{j} - q} \\tilde{N}\_{j} \\hbox{ where } q = \\max(K\_{j}, w + M\_{j})

and

D~0\=0 and D~j+1\=eKj−q+ew+Mj−qD~j where q\=max⁡(Kj,w+Mj)\\tilde{D}\_{0} = 0 \\hbox{ and } \\tilde{D}\_{j+1} = e^{K\_{j} - q} + e^{w + M\_{j} - q} \\tilde{D}\_{j} \\hbox{ where } q = \\max(K\_{j}, w + M\_{j})

and Mj+1\=qM\_{j+1} = q. With those, we can then compute

Ni\=eu+Ki−qVi+eMiN~i where q\=max⁡(u+Ki,Mi)N\_{i} = e^{u + K\_{i} - q} V\_{i} + e^{M\_{i}} \\tilde{N}\_{i} \\hbox{ where } q = \\max(u + K\_{i}, M\_{i})

and

Di\=eu+Ki−q+eMiD~i where q\=max⁡(u+Ki,Mi)D\_{i} = e^{u + K\_{i} - q} + e^{M\_{i}} \\tilde{D}\_{i} \\hbox{ where } q = \\max(u + K\_{i}, M\_{i})

which finally gives us

Oi\=σ(Ri)NiDiO\_{i} = \\sigma(R\_{i}) \\frac{N\_{i}}{D\_{i}}