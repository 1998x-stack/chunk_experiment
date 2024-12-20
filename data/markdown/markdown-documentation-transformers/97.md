# Autoformer

## Overview

The Autoformer model was proposed in [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/abs/2106.13008) by Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long.

This model augments the Transformer as a deep decomposition architecture, which can progressively decompose the trend and seasonal components during the forecasting process.

The abstract from the paper is the following:

_Extending the forecasting time is a critical demand for real applications, such as extreme weather early warning and long-term energy consumption planning. This paper studies the long-term forecasting problem of time series. Prior Transformer-based models adopt various self-attention mechanisms to discover the long-range dependencies. However, intricate temporal patterns of the long-term future prohibit the model from finding reliable dependencies. Also, Transformers have to adopt the sparse versions of point-wise self-attentions for long series efficiency, resulting in the information utilization bottleneck. Going beyond Transformers, we design Autoformer as a novel decomposition architecture with an Auto-Correlation mechanism. We break with the pre-processing convention of series decomposition and renovate it as a basic inner block of deep models. This design empowers Autoformer with progressive decomposition capacities for complex time series. Further, inspired by the stochastic process theory, we design the Auto-Correlation mechanism based on the series periodicity, which conducts the dependencies discovery and representation aggregation at the sub-series level. Auto-Correlation outperforms self-attention in both efficiency and accuracy. In long-term forecasting, Autoformer yields state-of-the-art accuracy, with a 38% relative improvement on six benchmarks, covering five practical applications: energy, traffic, economics, weather and disease._

This model was contributed by [elisim](https://huggingface.co/elisim) and [kashif](https://huggingface.co/kashif). The original code can be found [here](https://github.com/thuml/Autoformer).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

-   Check out the Autoformer blog-post in HuggingFace blog: [Yes, Transformers are Effective for Time Series Forecasting (+ Autoformer)](https://huggingface.co/blog/autoformer)

## AutoformerConfig

### class transformers.AutoformerConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/autoformer/configuration_autoformer.py#L30)

( prediction\_length: typing.Optional\[int\] = Nonecontext\_length: typing.Optional\[int\] = Nonedistribution\_output: str = 'student\_t'loss: str = 'nll'input\_size: int = 1lags\_sequence: typing.List\[int\] = \[1, 2, 3, 4, 5, 6, 7\]scaling: bool = Truenum\_time\_features: int = 0num\_dynamic\_real\_features: int = 0num\_static\_categorical\_features: int = 0num\_static\_real\_features: int = 0cardinality: typing.Optional\[typing.List\[int\]\] = Noneembedding\_dimension: typing.Optional\[typing.List\[int\]\] = Noned\_model: int = 64encoder\_attention\_heads: int = 2decoder\_attention\_heads: int = 2encoder\_layers: int = 2decoder\_layers: int = 2encoder\_ffn\_dim: int = 32decoder\_ffn\_dim: int = 32activation\_function: str = 'gelu'dropout: float = 0.1encoder\_layerdrop: float = 0.1decoder\_layerdrop: float = 0.1attention\_dropout: float = 0.1activation\_dropout: float = 0.1num\_parallel\_samples: int = 100init\_std: float = 0.02use\_cache: bool = Trueis\_encoder\_decoder = Truelabel\_length: int = 10moving\_average: int = 25autocorrelation\_factor: int = 3\*\*kwargs )

This is the configuration class to store the configuration of an [AutoformerModel](/docs/transformers/v4.34.0/en/model_doc/autoformer#transformers.AutoformerModel). It is used to instantiate an Autoformer model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Autoformer [huggingface/autoformer-tourism-monthly](https://huggingface.co/huggingface/autoformer-tourism-monthly) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

```
>>> from transformers import AutoformerConfig, AutoformerModel

>>> 
>>> configuration = AutoformerConfig()

>>> 
>>> model = AutoformerModel(configuration)

>>> 
>>> configuration = model.config
```

## AutoformerModel

### class transformers.AutoformerModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/autoformer/modeling_autoformer.py#L1500)

( config: AutoformerConfig )

Parameters

-   **config** ([AutoformerConfig](/docs/transformers/v4.34.0/en/model_doc/autoformer#transformers.AutoformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Autoformer Model outputting raw hidden-states without any specific head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/autoformer/modeling_autoformer.py#L1674)

( past\_values: Tensorpast\_time\_features: Tensorpast\_observed\_mask: Tensorstatic\_categorical\_features: typing.Optional\[torch.Tensor\] = Nonestatic\_real\_features: typing.Optional\[torch.Tensor\] = Nonefuture\_values: typing.Optional\[torch.Tensor\] = Nonefuture\_time\_features: typing.Optional\[torch.Tensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneuse\_cache: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.autoformer.modeling_autoformer.AutoformerModelOutput` or `tuple(torch.FloatTensor)`

The [AutoformerModel](/docs/transformers/v4.34.0/en/model_doc/autoformer#transformers.AutoformerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from huggingface_hub import hf_hub_download
>>> import torch
>>> from transformers import AutoformerModel

>>> file = hf_hub_download(
...     repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
... )
>>> batch = torch.load(file)

>>> model = AutoformerModel.from_pretrained("huggingface/autoformer-tourism-monthly")

>>> 
>>> 
>>> outputs = model(
...     past_values=batch["past_values"],
...     past_time_features=batch["past_time_features"],
...     past_observed_mask=batch["past_observed_mask"],
...     static_categorical_features=batch["static_categorical_features"],
...     future_values=batch["future_values"],
...     future_time_features=batch["future_time_features"],
... )

>>> last_hidden_state = outputs.last_hidden_state
```

## AutoformerForPrediction

### class transformers.AutoformerForPrediction

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/autoformer/modeling_autoformer.py#L1836)

( config: AutoformerConfig )

Parameters

-   **config** ([AutoformerConfig](/docs/transformers/v4.34.0/en/model_doc/autoformer#transformers.AutoformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Autoformer Model with a distribution head on top for time-series forecasting. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/autoformer/modeling_autoformer.py#L1876)

( past\_values: Tensorpast\_time\_features: Tensorpast\_observed\_mask: Tensorstatic\_categorical\_features: typing.Optional\[torch.Tensor\] = Nonestatic\_real\_features: typing.Optional\[torch.Tensor\] = Nonefuture\_values: typing.Optional\[torch.Tensor\] = Nonefuture\_time\_features: typing.Optional\[torch.Tensor\] = Nonefuture\_observed\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneuse\_cache: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqTSPredictionOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSPredictionOutput) or `tuple(torch.FloatTensor)`

The [AutoformerForPrediction](/docs/transformers/v4.34.0/en/model_doc/autoformer#transformers.AutoformerForPrediction) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from huggingface_hub import hf_hub_download
>>> import torch
>>> from transformers import AutoformerForPrediction

>>> file = hf_hub_download(
...     repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
... )
>>> batch = torch.load(file)

>>> model = AutoformerForPrediction.from_pretrained("huggingface/autoformer-tourism-monthly")

>>> 
>>> 
>>> outputs = model(
...     past_values=batch["past_values"],
...     past_time_features=batch["past_time_features"],
...     past_observed_mask=batch["past_observed_mask"],
...     static_categorical_features=batch["static_categorical_features"],
...     static_real_features=batch["static_real_features"],
...     future_values=batch["future_values"],
...     future_time_features=batch["future_time_features"],
... )

>>> loss = outputs.loss
>>> loss.backward()

>>> 
>>> 
>>> 
>>> outputs = model.generate(
...     past_values=batch["past_values"],
...     past_time_features=batch["past_time_features"],
...     past_observed_mask=batch["past_observed_mask"],
...     static_categorical_features=batch["static_categorical_features"],
...     static_real_features=batch["static_real_features"],
...     future_time_features=batch["future_time_features"],
... )

>>> mean_prediction = outputs.sequences.mean(dim=1)
```