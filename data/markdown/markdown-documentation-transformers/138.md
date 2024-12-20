# Decision Transformer

## Overview

The Decision Transformer model was proposed in [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)  
by Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch.

The abstract from the paper is the following:

_We introduce a framework that abstracts Reinforcement Learning (RL) as a sequence modeling problem. This allows us to draw upon the simplicity and scalability of the Transformer architecture, and associated advances in language modeling such as GPT-x and BERT. In particular, we present Decision Transformer, an architecture that casts the problem of RL as conditional sequence modeling. Unlike prior approaches to RL that fit value functions or compute policy gradients, Decision Transformer simply outputs the optimal actions by leveraging a causally masked Transformer. By conditioning an autoregressive model on the desired return (reward), past states, and actions, our Decision Transformer model can generate future actions that achieve the desired return. Despite its simplicity, Decision Transformer matches or exceeds the performance of state-of-the-art model-free offline RL baselines on Atari, OpenAI Gym, and Key-to-Door tasks._

Tips:

This version of the model is for tasks where the state is a vector, image-based states will come soon.

This model was contributed by [edbeeching](https://huggingface.co/edbeeching). The original code can be found [here](https://github.com/kzl/decision-transformer).

## DecisionTransformerConfig

### class transformers.DecisionTransformerConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/decision_transformer/configuration_decision_transformer.py#L31)

( state\_dim = 17act\_dim = 4hidden\_size = 128max\_ep\_len = 4096action\_tanh = Truevocab\_size = 1n\_positions = 1024n\_layer = 3n\_head = 1n\_inner = Noneactivation\_function = 'relu'resid\_pdrop = 0.1embd\_pdrop = 0.1attn\_pdrop = 0.1layer\_norm\_epsilon = 1e-05initializer\_range = 0.02scale\_attn\_weights = Trueuse\_cache = Truebos\_token\_id = 50256eos\_token\_id = 50256scale\_attn\_by\_inverse\_layer\_idx = Falsereorder\_and\_upcast\_attn = False\*\*kwargs )

This is the configuration class to store the configuration of a [DecisionTransformerModel](/docs/transformers/v4.34.0/en/model_doc/decision_transformer#transformers.DecisionTransformerModel). It is used to instantiate a Decision Transformer model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the standard DecisionTransformer architecture. Many of the config options are used to instatiate the GPT2 model that is used as part of the architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import DecisionTransformerConfig, DecisionTransformerModel

>>> 
>>> configuration = DecisionTransformerConfig()

>>> 
>>> model = DecisionTransformerModel(configuration)

>>> 
>>> configuration = model.config
```

## DecisionTransformerGPT2Model

### class transformers.DecisionTransformerGPT2Model

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/decision_transformer/modeling_decision_transformer.py#L477)

( config )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/decision_transformer/modeling_decision_transformer.py#L507)

( input\_ids: typing.Optional\[torch.LongTensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.Tensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None )

## DecisionTransformerModel

### class transformers.DecisionTransformerModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/decision_transformer/modeling_decision_transformer.py#L796)

( config )

Parameters

-   **config** ([~DecisionTransformerConfig](/docs/transformers/v4.34.0/en/model_doc/decision_transformer#transformers.DecisionTransformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Decision Transformer Model This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

The model builds upon the GPT2 architecture to perform autoregressive prediction of actions in an offline RL setting. Refer to the paper for more details: [https://arxiv.org/abs/2106.01345](https://arxiv.org/abs/2106.01345)

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/decision_transformer/modeling_decision_transformer.py#L829)

( states: typing.Optional\[torch.FloatTensor\] = Noneactions: typing.Optional\[torch.FloatTensor\] = Nonerewards: typing.Optional\[torch.FloatTensor\] = Nonereturns\_to\_go: typing.Optional\[torch.FloatTensor\] = Nonetimesteps: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.decision_transformer.modeling_decision_transformer.DecisionTransformerOutput` or `tuple(torch.FloatTensor)`

The [DecisionTransformerModel](/docs/transformers/v4.34.0/en/model_doc/decision_transformer#transformers.DecisionTransformerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import DecisionTransformerModel
>>> import torch

>>> model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-medium")
>>> 
>>> model = model.to(device)
>>> model.eval()

>>> env = gym.make("Hopper-v3")
>>> state_dim = env.observation_space.shape[0]
>>> act_dim = env.action_space.shape[0]

>>> state = env.reset()
>>> states = torch.from_numpy(state).reshape(1, 1, state_dim).to(device=device, dtype=torch.float32)
>>> actions = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)
>>> rewards = torch.zeros(1, 1, device=device, dtype=torch.float32)
>>> target_return = torch.tensor(TARGET_RETURN, dtype=torch.float32).reshape(1, 1)
>>> timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
>>> attention_mask = torch.zeros(1, 1, device=device, dtype=torch.float32)

>>> 
>>> with torch.no_grad():
...     state_preds, action_preds, return_preds = model(
...         states=states,
...         actions=actions,
...         rewards=rewards,
...         returns_to_go=target_return,
...         timesteps=timesteps,
...         attention_mask=attention_mask,
...         return_dict=False,
...     )
```