In many cases, the architecture you want to use can be guessed from the name or the path of the pretrained model you are supplying to the `from_pretrained()` method. AutoClasses are here to do this job for you so that you automatically retrieve the relevant model given the name/path to the pretrained weights/config/vocabulary.

Instantiating one of [AutoConfig](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoConfig), [AutoModel](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoModel), and [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer) will directly create a class of the relevant architecture. For instance

```
model = AutoModel.from_pretrained("bert-base-cased")
```

will create a model that is an instance of [BertModel](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertModel).

There is one class of `AutoModel` for each task, and for each backend (PyTorch, TensorFlow, or Flax).

# Extending the Auto Classes

Each of the auto classes has a method to be extended with your custom classes. For instance, if you have defined a custom class of model `NewModel`, make sure you have a `NewModelConfig` then you can add those to the auto classes like this:

```
from transformers import AutoConfig, AutoModel

AutoConfig.register("new-model", NewModelConfig)
AutoModel.register(NewModelConfig, NewModel)
```

You will then be able to use the auto classes like you would usually do!

If your `NewModelConfig` is a subclass of `~transformer.PretrainedConfig`, make sure its `model_type` attribute is set to the same key you use when registering the config (here `"new-model"`).

Likewise, if your `NewModel` is a subclass of [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel), make sure its `config_class` attribute is set to the same class you use when registering the model (here `NewModelConfig`).

## AutoConfig

This is a generic configuration class that will be instantiated as one of the configuration classes of the library when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoConfig.from_pretrained) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### register

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/configuration_auto.py#L1065)

( model\_typeconfigexist\_ok = False )

Parameters

-   **model\_type** (`str`) — The model type like “bert” or “gpt”.
-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The config to register.

Register a new configuration for this class.

## AutoTokenizer

## AutoFeatureExtractor

This is a generic feature extractor class that will be instantiated as one of the feature extractor classes of the library when created with the [AutoFeatureExtractor.from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoFeatureExtractor.from_pretrained) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

( config\_classfeature\_extractor\_classexist\_ok = False )

Parameters

-   **config\_class** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The configuration corresponding to the model to register.
-   **feature\_extractor\_class** (`FeatureExtractorMixin`) — The feature extractor to register.

Register a new feature extractor for this class.

## AutoImageProcessor

### class transformers.AutoImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/image_processing_auto.py#L243)

( )

This is a generic image processor class that will be instantiated as one of the image processor classes of the library when created with the [AutoImageProcessor.from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor.from_pretrained) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### register

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/image_processing_auto.py#L409)

( config\_classimage\_processor\_classexist\_ok = False )

Parameters

-   **config\_class** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The configuration corresponding to the model to register.
-   **image\_processor\_class** ([ImageProcessingMixin](/docs/transformers/v4.34.0/en/main_classes/image_processor#transformers.ImageProcessingMixin)) — The image processor to register.

Register a new image processor for this class.

## AutoProcessor

This is a generic processor class that will be instantiated as one of the processor classes of the library when created with the [AutoProcessor.from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoProcessor.from_pretrained) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### register

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/processing_auto.py#L321)

( config\_classprocessor\_classexist\_ok = False )

Parameters

-   **config\_class** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The configuration corresponding to the model to register.
-   **processor\_class** (`FeatureExtractorMixin`) — The processor to register.

Register a new processor for this class.

## Generic model classes

The following auto classes are available for instantiating a base model class without a specific head.

### AutoModel

### class transformers.AutoModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1224)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the base model classes of the library when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the base model classes of the library from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModel

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = AutoModel.from_config(config)
```

### TFAutoModel

### class transformers.TFAutoModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_tf_auto.py#L529)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the base model classes of the library when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the base model classes of the library from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, TFAutoModel

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = TFAutoModel.from_config(config)
```

### FlaxAutoModel

### class transformers.FlaxAutoModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_flax_auto.py#L274)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the base model classes of the library when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the base model classes of the library from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, FlaxAutoModel

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = FlaxAutoModel.from_config(config)
```

## Generic pretraining classes

The following auto classes are available for instantiating a model with a pretraining head.

### AutoModelForPreTraining

### class transformers.AutoModelForPreTraining

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1231)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a pretraining head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a pretraining head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForPreTraining

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = AutoModelForPreTraining.from_config(config)
```

### TFAutoModelForPreTraining

### class transformers.TFAutoModelForPreTraining

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_tf_auto.py#L545)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a pretraining head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a pretraining head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForPreTraining

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = TFAutoModelForPreTraining.from_config(config)
```

### FlaxAutoModelForPreTraining

### class transformers.FlaxAutoModelForPreTraining

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_flax_auto.py#L281)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a pretraining head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a pretraining head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, FlaxAutoModelForPreTraining

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = FlaxAutoModelForPreTraining.from_config(config)
```

## Natural Language Processing

The following auto classes are available for the following natural language processing tasks.

### AutoModelForCausalLM

### class transformers.AutoModelForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1246)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a causal language modeling head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a causal language modeling head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForCausalLM

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = AutoModelForCausalLM.from_config(config)
```

### TFAutoModelForCausalLM

### class transformers.TFAutoModelForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_tf_auto.py#L560)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a causal language modeling head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a causal language modeling head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForCausalLM

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = TFAutoModelForCausalLM.from_config(config)
```

Instantiate one of the model classes of the library (with a causal language modeling head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **bert** — [TFBertLMHeadModel](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.TFBertLMHeadModel) (BERT model)
-   **camembert** — [TFCamembertForCausalLM](/docs/transformers/v4.34.0/en/model_doc/camembert#transformers.TFCamembertForCausalLM) (CamemBERT model)
-   **ctrl** — [TFCTRLLMHeadModel](/docs/transformers/v4.34.0/en/model_doc/ctrl#transformers.TFCTRLLMHeadModel) (CTRL model)
-   **gpt-sw3** — [TFGPT2LMHeadModel](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.TFGPT2LMHeadModel) (GPT-Sw3 model)
-   **gpt2** — [TFGPT2LMHeadModel](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.TFGPT2LMHeadModel) (OpenAI GPT-2 model)
-   **gptj** — [TFGPTJForCausalLM](/docs/transformers/v4.34.0/en/model_doc/gptj#transformers.TFGPTJForCausalLM) (GPT-J model)
-   **openai-gpt** — [TFOpenAIGPTLMHeadModel](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.TFOpenAIGPTLMHeadModel) (OpenAI GPT model)
-   **opt** — [TFOPTForCausalLM](/docs/transformers/v4.34.0/en/model_doc/opt#transformers.TFOPTForCausalLM) (OPT model)
-   **rembert** — [TFRemBertForCausalLM](/docs/transformers/v4.34.0/en/model_doc/rembert#transformers.TFRemBertForCausalLM) (RemBERT model)
-   **roberta** — [TFRobertaForCausalLM](/docs/transformers/v4.34.0/en/model_doc/roberta#transformers.TFRobertaForCausalLM) (RoBERTa model)
-   **roberta-prelayernorm** — [TFRobertaPreLayerNormForCausalLM](/docs/transformers/v4.34.0/en/model_doc/roberta-prelayernorm#transformers.TFRobertaPreLayerNormForCausalLM) (RoBERTa-PreLayerNorm model)
-   **roformer** — [TFRoFormerForCausalLM](/docs/transformers/v4.34.0/en/model_doc/roformer#transformers.TFRoFormerForCausalLM) (RoFormer model)
-   **transfo-xl** — [TFTransfoXLLMHeadModel](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TFTransfoXLLMHeadModel) (Transformer-XL model)
-   **xglm** — [TFXGLMForCausalLM](/docs/transformers/v4.34.0/en/model_doc/xglm#transformers.TFXGLMForCausalLM) (XGLM model)
-   **xlm** — [TFXLMWithLMHeadModel](/docs/transformers/v4.34.0/en/model_doc/xlm#transformers.TFXLMWithLMHeadModel) (XLM model)
-   **xlm-roberta** — [TFXLMRobertaForCausalLM](/docs/transformers/v4.34.0/en/model_doc/xlm-roberta#transformers.TFXLMRobertaForCausalLM) (XLM-RoBERTa model)
-   **xlnet** — [TFXLNetLMHeadModel](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.TFXLNetLMHeadModel) (XLNet model)

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForCausalLM

>>> 
>>> model = TFAutoModelForCausalLM.from_pretrained("bert-base-cased")

>>> 
>>> model = TFAutoModelForCausalLM.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./pt_model/bert_pt_model_config.json")
>>> model = TFAutoModelForCausalLM.from_pretrained(
...     "./pt_model/bert_pytorch_model.bin", from_pt=True, config=config
... )
```

### FlaxAutoModelForCausalLM

### class transformers.FlaxAutoModelForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_flax_auto.py#L288)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a causal language modeling head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a causal language modeling head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, FlaxAutoModelForCausalLM

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = FlaxAutoModelForCausalLM.from_config(config)
```

Instantiate one of the model classes of the library (with a causal language modeling head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **bart** — [FlaxBartForCausalLM](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.FlaxBartForCausalLM) (BART model)
-   **bert** — [FlaxBertForCausalLM](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.FlaxBertForCausalLM) (BERT model)
-   **big\_bird** — [FlaxBigBirdForCausalLM](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.FlaxBigBirdForCausalLM) (BigBird model)
-   **bloom** — [FlaxBloomForCausalLM](/docs/transformers/v4.34.0/en/model_doc/bloom#transformers.FlaxBloomForCausalLM) (BLOOM model)
-   **electra** — [FlaxElectraForCausalLM](/docs/transformers/v4.34.0/en/model_doc/electra#transformers.FlaxElectraForCausalLM) (ELECTRA model)
-   **gpt-sw3** — [FlaxGPT2LMHeadModel](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.FlaxGPT2LMHeadModel) (GPT-Sw3 model)
-   **gpt2** — [FlaxGPT2LMHeadModel](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.FlaxGPT2LMHeadModel) (OpenAI GPT-2 model)
-   **gpt\_neo** — [FlaxGPTNeoForCausalLM](/docs/transformers/v4.34.0/en/model_doc/gpt_neo#transformers.FlaxGPTNeoForCausalLM) (GPT Neo model)
-   **gptj** — [FlaxGPTJForCausalLM](/docs/transformers/v4.34.0/en/model_doc/gptj#transformers.FlaxGPTJForCausalLM) (GPT-J model)
-   **opt** — [FlaxOPTForCausalLM](/docs/transformers/v4.34.0/en/model_doc/opt#transformers.FlaxOPTForCausalLM) (OPT model)
-   **roberta** — [FlaxRobertaForCausalLM](/docs/transformers/v4.34.0/en/model_doc/roberta#transformers.FlaxRobertaForCausalLM) (RoBERTa model)
-   **roberta-prelayernorm** — [FlaxRobertaPreLayerNormForCausalLM](/docs/transformers/v4.34.0/en/model_doc/roberta-prelayernorm#transformers.FlaxRobertaPreLayerNormForCausalLM) (RoBERTa-PreLayerNorm model)
-   **xglm** — [FlaxXGLMForCausalLM](/docs/transformers/v4.34.0/en/model_doc/xglm#transformers.FlaxXGLMForCausalLM) (XGLM model)
-   **xlm-roberta** — [FlaxXLMRobertaForCausalLM](/docs/transformers/v4.34.0/en/model_doc/xlm-roberta#transformers.FlaxXLMRobertaForCausalLM) (XLM-RoBERTa model)

Examples:

```
>>> from transformers import AutoConfig, FlaxAutoModelForCausalLM

>>> 
>>> model = FlaxAutoModelForCausalLM.from_pretrained("bert-base-cased")

>>> 
>>> model = FlaxAutoModelForCausalLM.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./pt_model/bert_pt_model_config.json")
>>> model = FlaxAutoModelForCausalLM.from_pretrained(
...     "./pt_model/bert_pytorch_model.bin", from_pt=True, config=config
... )
```

### AutoModelForMaskedLM

### class transformers.AutoModelForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1253)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a masked language modeling head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a masked language modeling head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForMaskedLM

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = AutoModelForMaskedLM.from_config(config)
```

### TFAutoModelForMaskedLM

### class transformers.TFAutoModelForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_tf_auto.py#L610)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a masked language modeling head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a masked language modeling head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForMaskedLM

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = TFAutoModelForMaskedLM.from_config(config)
```

### FlaxAutoModelForMaskedLM

### class transformers.FlaxAutoModelForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_flax_auto.py#L295)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a masked language modeling head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a masked language modeling head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, FlaxAutoModelForMaskedLM

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = FlaxAutoModelForMaskedLM.from_config(config)
```

Instantiate one of the model classes of the library (with a masked language modeling head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **albert** — [FlaxAlbertForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.FlaxAlbertForMaskedLM) (ALBERT model)
-   **bart** — [FlaxBartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.FlaxBartForConditionalGeneration) (BART model)
-   **bert** — [FlaxBertForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.FlaxBertForMaskedLM) (BERT model)
-   **big\_bird** — [FlaxBigBirdForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.FlaxBigBirdForMaskedLM) (BigBird model)
-   **distilbert** — [FlaxDistilBertForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/distilbert#transformers.FlaxDistilBertForMaskedLM) (DistilBERT model)
-   **electra** — [FlaxElectraForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/electra#transformers.FlaxElectraForMaskedLM) (ELECTRA model)
-   **mbart** — [FlaxMBartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/mbart#transformers.FlaxMBartForConditionalGeneration) (mBART model)
-   **roberta** — [FlaxRobertaForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/roberta#transformers.FlaxRobertaForMaskedLM) (RoBERTa model)
-   **roberta-prelayernorm** — [FlaxRobertaPreLayerNormForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/roberta-prelayernorm#transformers.FlaxRobertaPreLayerNormForMaskedLM) (RoBERTa-PreLayerNorm model)
-   **roformer** — [FlaxRoFormerForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/roformer#transformers.FlaxRoFormerForMaskedLM) (RoFormer model)
-   **xlm-roberta** — [FlaxXLMRobertaForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/xlm-roberta#transformers.FlaxXLMRobertaForMaskedLM) (XLM-RoBERTa model)

Examples:

```
>>> from transformers import AutoConfig, FlaxAutoModelForMaskedLM

>>> 
>>> model = FlaxAutoModelForMaskedLM.from_pretrained("bert-base-cased")

>>> 
>>> model = FlaxAutoModelForMaskedLM.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./pt_model/bert_pt_model_config.json")
>>> model = FlaxAutoModelForMaskedLM.from_pretrained(
...     "./pt_model/bert_pytorch_model.bin", from_pt=True, config=config
... )
```

### AutoModelForMaskGeneration

### class transformers.AutoModelForMaskGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1212)

( \*args\*\*kwargs )

### TFAutoModelForMaskGeneration

### class transformers.TFAutoModelForMaskGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_tf_auto.py#L521)

( \*args\*\*kwargs )

### AutoModelForSeq2SeqLM

### class transformers.AutoModelForSeq2SeqLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1260)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a sequence-to-sequence language modeling head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a sequence-to-sequence language modeling head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForSeq2SeqLM

>>> 
>>> config = AutoConfig.from_pretrained("t5-base")
>>> model = AutoModelForSeq2SeqLM.from_config(config)
```

### TFAutoModelForSeq2SeqLM

### class transformers.TFAutoModelForSeq2SeqLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_tf_auto.py#L617)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a sequence-to-sequence language modeling head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a sequence-to-sequence language modeling head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForSeq2SeqLM

>>> 
>>> config = AutoConfig.from_pretrained("t5-base")
>>> model = TFAutoModelForSeq2SeqLM.from_config(config)
```

### FlaxAutoModelForSeq2SeqLM

### class transformers.FlaxAutoModelForSeq2SeqLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_flax_auto.py#L302)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a sequence-to-sequence language modeling head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a sequence-to-sequence language modeling head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, FlaxAutoModelForSeq2SeqLM

>>> 
>>> config = AutoConfig.from_pretrained("t5-base")
>>> model = FlaxAutoModelForSeq2SeqLM.from_config(config)
```

### AutoModelForSequenceClassification

### class transformers.AutoModelForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1269)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a sequence classification head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a sequence classification head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForSequenceClassification

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = AutoModelForSequenceClassification.from_config(config)
```

### TFAutoModelForSequenceClassification

### class transformers.TFAutoModelForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_tf_auto.py#L626)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a sequence classification head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a sequence classification head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForSequenceClassification

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = TFAutoModelForSequenceClassification.from_config(config)
```

### FlaxAutoModelForSequenceClassification

### class transformers.FlaxAutoModelForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_flax_auto.py#L311)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a sequence classification head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a sequence classification head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, FlaxAutoModelForSequenceClassification

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = FlaxAutoModelForSequenceClassification.from_config(config)
```

### AutoModelForMultipleChoice

### class transformers.AutoModelForMultipleChoice

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1325)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a multiple choice head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a multiple choice head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForMultipleChoice

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = AutoModelForMultipleChoice.from_config(config)
```

### TFAutoModelForMultipleChoice

### class transformers.TFAutoModelForMultipleChoice

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_tf_auto.py#L673)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a multiple choice head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a multiple choice head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForMultipleChoice

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = TFAutoModelForMultipleChoice.from_config(config)
```

### FlaxAutoModelForMultipleChoice

### class transformers.FlaxAutoModelForMultipleChoice

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_flax_auto.py#L336)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a multiple choice head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a multiple choice head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, FlaxAutoModelForMultipleChoice

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = FlaxAutoModelForMultipleChoice.from_config(config)
```

Instantiate one of the model classes of the library (with a multiple choice head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **albert** — [FlaxAlbertForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/albert#transformers.FlaxAlbertForMultipleChoice) (ALBERT model)
-   **bert** — [FlaxBertForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.FlaxBertForMultipleChoice) (BERT model)
-   **big\_bird** — [FlaxBigBirdForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/big_bird#transformers.FlaxBigBirdForMultipleChoice) (BigBird model)
-   **distilbert** — [FlaxDistilBertForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/distilbert#transformers.FlaxDistilBertForMultipleChoice) (DistilBERT model)
-   **electra** — [FlaxElectraForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/electra#transformers.FlaxElectraForMultipleChoice) (ELECTRA model)
-   **roberta** — [FlaxRobertaForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/roberta#transformers.FlaxRobertaForMultipleChoice) (RoBERTa model)
-   **roberta-prelayernorm** — [FlaxRobertaPreLayerNormForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/roberta-prelayernorm#transformers.FlaxRobertaPreLayerNormForMultipleChoice) (RoBERTa-PreLayerNorm model)
-   **roformer** — [FlaxRoFormerForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/roformer#transformers.FlaxRoFormerForMultipleChoice) (RoFormer model)
-   **xlm-roberta** — [FlaxXLMRobertaForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/xlm-roberta#transformers.FlaxXLMRobertaForMultipleChoice) (XLM-RoBERTa model)

Examples:

```
>>> from transformers import AutoConfig, FlaxAutoModelForMultipleChoice

>>> 
>>> model = FlaxAutoModelForMultipleChoice.from_pretrained("bert-base-cased")

>>> 
>>> model = FlaxAutoModelForMultipleChoice.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./pt_model/bert_pt_model_config.json")
>>> model = FlaxAutoModelForMultipleChoice.from_pretrained(
...     "./pt_model/bert_pytorch_model.bin", from_pt=True, config=config
... )
```

### AutoModelForNextSentencePrediction

### class transformers.AutoModelForNextSentencePrediction

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1332)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a next sentence prediction head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [BertConfig](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertConfig) configuration class: [BertForNextSentencePrediction](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertForNextSentencePrediction) (BERT model)
    -   [ErnieConfig](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.ErnieConfig) configuration class: [ErnieForNextSentencePrediction](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.ErnieForNextSentencePrediction) (ERNIE model)
    -   [FNetConfig](/docs/transformers/v4.34.0/en/model_doc/fnet#transformers.FNetConfig) configuration class: [FNetForNextSentencePrediction](/docs/transformers/v4.34.0/en/model_doc/fnet#transformers.FNetForNextSentencePrediction) (FNet model)
    -   [MegatronBertConfig](/docs/transformers/v4.34.0/en/model_doc/megatron-bert#transformers.MegatronBertConfig) configuration class: [MegatronBertForNextSentencePrediction](/docs/transformers/v4.34.0/en/model_doc/megatron-bert#transformers.MegatronBertForNextSentencePrediction) (Megatron-BERT model)
    -   [MobileBertConfig](/docs/transformers/v4.34.0/en/model_doc/mobilebert#transformers.MobileBertConfig) configuration class: [MobileBertForNextSentencePrediction](/docs/transformers/v4.34.0/en/model_doc/mobilebert#transformers.MobileBertForNextSentencePrediction) (MobileBERT model)
    -   [NezhaConfig](/docs/transformers/v4.34.0/en/model_doc/nezha#transformers.NezhaConfig) configuration class: [NezhaForNextSentencePrediction](/docs/transformers/v4.34.0/en/model_doc/nezha#transformers.NezhaForNextSentencePrediction) (Nezha model)
    -   [QDQBertConfig](/docs/transformers/v4.34.0/en/model_doc/qdqbert#transformers.QDQBertConfig) configuration class: [QDQBertForNextSentencePrediction](/docs/transformers/v4.34.0/en/model_doc/qdqbert#transformers.QDQBertForNextSentencePrediction) (QDQBert model)
    

Instantiates one of the model classes of the library (with a next sentence prediction head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForNextSentencePrediction

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = AutoModelForNextSentencePrediction.from_config(config)
```

Instantiate one of the model classes of the library (with a next sentence prediction head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **bert** — [BertForNextSentencePrediction](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertForNextSentencePrediction) (BERT model)
-   **ernie** — [ErnieForNextSentencePrediction](/docs/transformers/v4.34.0/en/model_doc/ernie#transformers.ErnieForNextSentencePrediction) (ERNIE model)
-   **fnet** — [FNetForNextSentencePrediction](/docs/transformers/v4.34.0/en/model_doc/fnet#transformers.FNetForNextSentencePrediction) (FNet model)
-   **megatron-bert** — [MegatronBertForNextSentencePrediction](/docs/transformers/v4.34.0/en/model_doc/megatron-bert#transformers.MegatronBertForNextSentencePrediction) (Megatron-BERT model)
-   **mobilebert** — [MobileBertForNextSentencePrediction](/docs/transformers/v4.34.0/en/model_doc/mobilebert#transformers.MobileBertForNextSentencePrediction) (MobileBERT model)
-   **nezha** — [NezhaForNextSentencePrediction](/docs/transformers/v4.34.0/en/model_doc/nezha#transformers.NezhaForNextSentencePrediction) (Nezha model)
-   **qdqbert** — [QDQBertForNextSentencePrediction](/docs/transformers/v4.34.0/en/model_doc/qdqbert#transformers.QDQBertForNextSentencePrediction) (QDQBert model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:

```
>>> from transformers import AutoConfig, AutoModelForNextSentencePrediction

>>> 
>>> model = AutoModelForNextSentencePrediction.from_pretrained("bert-base-cased")

>>> 
>>> model = AutoModelForNextSentencePrediction.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForNextSentencePrediction.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### TFAutoModelForNextSentencePrediction

### class transformers.TFAutoModelForNextSentencePrediction

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_tf_auto.py#L680)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a next sentence prediction head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [BertConfig](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertConfig) configuration class: [TFBertForNextSentencePrediction](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.TFBertForNextSentencePrediction) (BERT model)
    -   [MobileBertConfig](/docs/transformers/v4.34.0/en/model_doc/mobilebert#transformers.MobileBertConfig) configuration class: [TFMobileBertForNextSentencePrediction](/docs/transformers/v4.34.0/en/model_doc/mobilebert#transformers.TFMobileBertForNextSentencePrediction) (MobileBERT model)
    

Instantiates one of the model classes of the library (with a next sentence prediction head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForNextSentencePrediction

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = TFAutoModelForNextSentencePrediction.from_config(config)
```

Instantiate one of the model classes of the library (with a next sentence prediction head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **bert** — [TFBertForNextSentencePrediction](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.TFBertForNextSentencePrediction) (BERT model)
-   **mobilebert** — [TFMobileBertForNextSentencePrediction](/docs/transformers/v4.34.0/en/model_doc/mobilebert#transformers.TFMobileBertForNextSentencePrediction) (MobileBERT model)

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForNextSentencePrediction

>>> 
>>> model = TFAutoModelForNextSentencePrediction.from_pretrained("bert-base-cased")

>>> 
>>> model = TFAutoModelForNextSentencePrediction.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./pt_model/bert_pt_model_config.json")
>>> model = TFAutoModelForNextSentencePrediction.from_pretrained(
...     "./pt_model/bert_pytorch_model.bin", from_pt=True, config=config
... )
```

### FlaxAutoModelForNextSentencePrediction

### class transformers.FlaxAutoModelForNextSentencePrediction

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_flax_auto.py#L343)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a next sentence prediction head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [BertConfig](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertConfig) configuration class: [FlaxBertForNextSentencePrediction](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.FlaxBertForNextSentencePrediction) (BERT model)
    

Instantiates one of the model classes of the library (with a next sentence prediction head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, FlaxAutoModelForNextSentencePrediction

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = FlaxAutoModelForNextSentencePrediction.from_config(config)
```

Instantiate one of the model classes of the library (with a next sentence prediction head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **bert** — [FlaxBertForNextSentencePrediction](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.FlaxBertForNextSentencePrediction) (BERT model)

Examples:

```
>>> from transformers import AutoConfig, FlaxAutoModelForNextSentencePrediction

>>> 
>>> model = FlaxAutoModelForNextSentencePrediction.from_pretrained("bert-base-cased")

>>> 
>>> model = FlaxAutoModelForNextSentencePrediction.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./pt_model/bert_pt_model_config.json")
>>> model = FlaxAutoModelForNextSentencePrediction.from_pretrained(
...     "./pt_model/bert_pytorch_model.bin", from_pt=True, config=config
... )
```

### AutoModelForTokenClassification

### class transformers.AutoModelForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1318)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a token classification head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a token classification head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForTokenClassification

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = AutoModelForTokenClassification.from_config(config)
```

### TFAutoModelForTokenClassification

### class transformers.TFAutoModelForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_tf_auto.py#L664)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a token classification head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a token classification head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForTokenClassification

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = TFAutoModelForTokenClassification.from_config(config)
```

### FlaxAutoModelForTokenClassification

### class transformers.FlaxAutoModelForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_flax_auto.py#L327)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a token classification head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a token classification head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, FlaxAutoModelForTokenClassification

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = FlaxAutoModelForTokenClassification.from_config(config)
```

### AutoModelForQuestionAnswering

### class transformers.AutoModelForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1278)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a question answering head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a question answering head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForQuestionAnswering

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = AutoModelForQuestionAnswering.from_config(config)
```

### TFAutoModelForQuestionAnswering

### class transformers.TFAutoModelForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_tf_auto.py#L635)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a question answering head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a question answering head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForQuestionAnswering

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = TFAutoModelForQuestionAnswering.from_config(config)
```

### FlaxAutoModelForQuestionAnswering

### class transformers.FlaxAutoModelForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_flax_auto.py#L320)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a question answering head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a question answering head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, FlaxAutoModelForQuestionAnswering

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = FlaxAutoModelForQuestionAnswering.from_config(config)
```

### AutoModelForTextEncoding

### class transformers.AutoModelForTextEncoding

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1216)

( \*args\*\*kwargs )

### TFAutoModelForTextEncoding

### class transformers.TFAutoModelForTextEncoding

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_tf_auto.py#L525)

( \*args\*\*kwargs )

## Computer vision

The following auto classes are available for the following computer vision tasks.

### AutoModelForDepthEstimation

### class transformers.AutoModelForDepthEstimation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1407)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a depth estimation head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [DPTConfig](/docs/transformers/v4.34.0/en/model_doc/dpt#transformers.DPTConfig) configuration class: [DPTForDepthEstimation](/docs/transformers/v4.34.0/en/model_doc/dpt#transformers.DPTForDepthEstimation) (DPT model)
    -   [GLPNConfig](/docs/transformers/v4.34.0/en/model_doc/glpn#transformers.GLPNConfig) configuration class: [GLPNForDepthEstimation](/docs/transformers/v4.34.0/en/model_doc/glpn#transformers.GLPNForDepthEstimation) (GLPN model)
    

Instantiates one of the model classes of the library (with a depth estimation head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForDepthEstimation

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = AutoModelForDepthEstimation.from_config(config)
```

Instantiate one of the model classes of the library (with a depth estimation head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **dpt** — [DPTForDepthEstimation](/docs/transformers/v4.34.0/en/model_doc/dpt#transformers.DPTForDepthEstimation) (DPT model)
-   **glpn** — [GLPNForDepthEstimation](/docs/transformers/v4.34.0/en/model_doc/glpn#transformers.GLPNForDepthEstimation) (GLPN model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:

```
>>> from transformers import AutoConfig, AutoModelForDepthEstimation

>>> 
>>> model = AutoModelForDepthEstimation.from_pretrained("bert-base-cased")

>>> 
>>> model = AutoModelForDepthEstimation.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForDepthEstimation.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForImageClassification

### class transformers.AutoModelForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1341)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a image classification head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a image classification head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForImageClassification

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = AutoModelForImageClassification.from_config(config)
```

### TFAutoModelForImageClassification

### class transformers.TFAutoModelForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_tf_auto.py#L576)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a image classification head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a image classification head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForImageClassification

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = TFAutoModelForImageClassification.from_config(config)
```

### FlaxAutoModelForImageClassification

### class transformers.FlaxAutoModelForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_flax_auto.py#L352)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a image classification head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiate one of the model classes of the library (with a image classification head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **beit** — [FlaxBeitForImageClassification](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.FlaxBeitForImageClassification) (BEiT model)
-   **regnet** — [FlaxRegNetForImageClassification](/docs/transformers/v4.34.0/en/model_doc/regnet#transformers.FlaxRegNetForImageClassification) (RegNet model)
-   **resnet** — [FlaxResNetForImageClassification](/docs/transformers/v4.34.0/en/model_doc/resnet#transformers.FlaxResNetForImageClassification) (ResNet model)
-   **vit** — [FlaxViTForImageClassification](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.FlaxViTForImageClassification) (ViT model)

Examples:

```
>>> from transformers import AutoConfig, FlaxAutoModelForImageClassification

>>> 
>>> model = FlaxAutoModelForImageClassification.from_pretrained("bert-base-cased")

>>> 
>>> model = FlaxAutoModelForImageClassification.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./pt_model/bert_pt_model_config.json")
>>> model = FlaxAutoModelForImageClassification.from_pretrained(
...     "./pt_model/bert_pytorch_model.bin", from_pt=True, config=config
... )
```

### AutoModelForVideoClassification

### class transformers.AutoModelForVideoClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1414)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a video classification head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [TimesformerConfig](/docs/transformers/v4.34.0/en/model_doc/timesformer#transformers.TimesformerConfig) configuration class: [TimesformerForVideoClassification](/docs/transformers/v4.34.0/en/model_doc/timesformer#transformers.TimesformerForVideoClassification) (TimeSformer model)
    -   [VideoMAEConfig](/docs/transformers/v4.34.0/en/model_doc/videomae#transformers.VideoMAEConfig) configuration class: [VideoMAEForVideoClassification](/docs/transformers/v4.34.0/en/model_doc/videomae#transformers.VideoMAEForVideoClassification) (VideoMAE model)
    -   [VivitConfig](/docs/transformers/v4.34.0/en/model_doc/vivit#transformers.VivitConfig) configuration class: [VivitForVideoClassification](/docs/transformers/v4.34.0/en/model_doc/vivit#transformers.VivitForVideoClassification) (ViViT model)
    

Instantiates one of the model classes of the library (with a video classification head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForVideoClassification

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = AutoModelForVideoClassification.from_config(config)
```

Instantiate one of the model classes of the library (with a video classification head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **timesformer** — [TimesformerForVideoClassification](/docs/transformers/v4.34.0/en/model_doc/timesformer#transformers.TimesformerForVideoClassification) (TimeSformer model)
-   **videomae** — [VideoMAEForVideoClassification](/docs/transformers/v4.34.0/en/model_doc/videomae#transformers.VideoMAEForVideoClassification) (VideoMAE model)
-   **vivit** — [VivitForVideoClassification](/docs/transformers/v4.34.0/en/model_doc/vivit#transformers.VivitForVideoClassification) (ViViT model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:

```
>>> from transformers import AutoConfig, AutoModelForVideoClassification

>>> 
>>> model = AutoModelForVideoClassification.from_pretrained("bert-base-cased")

>>> 
>>> model = AutoModelForVideoClassification.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForVideoClassification.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForMaskedImageModeling

### class transformers.AutoModelForMaskedImageModeling

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1479)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a masked image modeling head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiate one of the model classes of the library (with a masked image modeling head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **deit** — [DeiTForMaskedImageModeling](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTForMaskedImageModeling) (DeiT model)
-   **focalnet** — [FocalNetForMaskedImageModeling](/docs/transformers/v4.34.0/en/model_doc/focalnet#transformers.FocalNetForMaskedImageModeling) (FocalNet model)
-   **swin** — [SwinForMaskedImageModeling](/docs/transformers/v4.34.0/en/model_doc/swin#transformers.SwinForMaskedImageModeling) (Swin Transformer model)
-   **swinv2** — [Swinv2ForMaskedImageModeling](/docs/transformers/v4.34.0/en/model_doc/swinv2#transformers.Swinv2ForMaskedImageModeling) (Swin Transformer V2 model)
-   **vit** — [ViTForMaskedImageModeling](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTForMaskedImageModeling) (ViT model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:

```
>>> from transformers import AutoConfig, AutoModelForMaskedImageModeling

>>> 
>>> model = AutoModelForMaskedImageModeling.from_pretrained("bert-base-cased")

>>> 
>>> model = AutoModelForMaskedImageModeling.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForMaskedImageModeling.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### TFAutoModelForMaskedImageModeling

### class transformers.TFAutoModelForMaskedImageModeling

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_tf_auto.py#L567)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a masked image modeling head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [DeiTConfig](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTConfig) configuration class: [TFDeiTForMaskedImageModeling](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.TFDeiTForMaskedImageModeling) (DeiT model)
    -   [SwinConfig](/docs/transformers/v4.34.0/en/model_doc/swin#transformers.SwinConfig) configuration class: [TFSwinForMaskedImageModeling](/docs/transformers/v4.34.0/en/model_doc/swin#transformers.TFSwinForMaskedImageModeling) (Swin Transformer model)
    

Instantiates one of the model classes of the library (with a masked image modeling head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForMaskedImageModeling

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = TFAutoModelForMaskedImageModeling.from_config(config)
```

Instantiate one of the model classes of the library (with a masked image modeling head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **deit** — [TFDeiTForMaskedImageModeling](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.TFDeiTForMaskedImageModeling) (DeiT model)
-   **swin** — [TFSwinForMaskedImageModeling](/docs/transformers/v4.34.0/en/model_doc/swin#transformers.TFSwinForMaskedImageModeling) (Swin Transformer model)

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForMaskedImageModeling

>>> 
>>> model = TFAutoModelForMaskedImageModeling.from_pretrained("bert-base-cased")

>>> 
>>> model = TFAutoModelForMaskedImageModeling.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./pt_model/bert_pt_model_config.json")
>>> model = TFAutoModelForMaskedImageModeling.from_pretrained(
...     "./pt_model/bert_pytorch_model.bin", from_pt=True, config=config
... )
```

### AutoModelForObjectDetection

### class transformers.AutoModelForObjectDetection

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1391)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a object detection head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [ConditionalDetrConfig](/docs/transformers/v4.34.0/en/model_doc/conditional_detr#transformers.ConditionalDetrConfig) configuration class: [ConditionalDetrForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/conditional_detr#transformers.ConditionalDetrForObjectDetection) (Conditional DETR model)
    -   [DeformableDetrConfig](/docs/transformers/v4.34.0/en/model_doc/deformable_detr#transformers.DeformableDetrConfig) configuration class: [DeformableDetrForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/deformable_detr#transformers.DeformableDetrForObjectDetection) (Deformable DETR model)
    -   [DetaConfig](/docs/transformers/v4.34.0/en/model_doc/deta#transformers.DetaConfig) configuration class: [DetaForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/deta#transformers.DetaForObjectDetection) (DETA model)
    -   [DetrConfig](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrConfig) configuration class: [DetrForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForObjectDetection) (DETR model)
    -   [TableTransformerConfig](/docs/transformers/v4.34.0/en/model_doc/table-transformer#transformers.TableTransformerConfig) configuration class: [TableTransformerForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/table-transformer#transformers.TableTransformerForObjectDetection) (Table Transformer model)
    -   [YolosConfig](/docs/transformers/v4.34.0/en/model_doc/yolos#transformers.YolosConfig) configuration class: [YolosForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/yolos#transformers.YolosForObjectDetection) (YOLOS model)
    

Instantiates one of the model classes of the library (with a object detection head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForObjectDetection

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = AutoModelForObjectDetection.from_config(config)
```

Instantiate one of the model classes of the library (with a object detection head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **conditional\_detr** — [ConditionalDetrForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/conditional_detr#transformers.ConditionalDetrForObjectDetection) (Conditional DETR model)
-   **deformable\_detr** — [DeformableDetrForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/deformable_detr#transformers.DeformableDetrForObjectDetection) (Deformable DETR model)
-   **deta** — [DetaForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/deta#transformers.DetaForObjectDetection) (DETA model)
-   **detr** — [DetrForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForObjectDetection) (DETR model)
-   **table-transformer** — [TableTransformerForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/table-transformer#transformers.TableTransformerForObjectDetection) (Table Transformer model)
-   **yolos** — [YolosForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/yolos#transformers.YolosForObjectDetection) (YOLOS model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:

```
>>> from transformers import AutoConfig, AutoModelForObjectDetection

>>> 
>>> model = AutoModelForObjectDetection.from_pretrained("bert-base-cased")

>>> 
>>> model = AutoModelForObjectDetection.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForObjectDetection.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForImageSegmentation

### class transformers.AutoModelForImageSegmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1357)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a image segmentation head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [DetrConfig](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrConfig) configuration class: [DetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForSegmentation) (DETR model)
    

Instantiates one of the model classes of the library (with a image segmentation head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForImageSegmentation

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = AutoModelForImageSegmentation.from_config(config)
```

Instantiate one of the model classes of the library (with a image segmentation head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **detr** — [DetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForSegmentation) (DETR model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:

```
>>> from transformers import AutoConfig, AutoModelForImageSegmentation

>>> 
>>> model = AutoModelForImageSegmentation.from_pretrained("bert-base-cased")

>>> 
>>> model = AutoModelForImageSegmentation.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForImageSegmentation.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForImageToImage

### class transformers.AutoModelForImageToImage

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1220)

( \*args\*\*kwargs )

### AutoModelForSemanticSegmentation

### class transformers.AutoModelForSemanticSegmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1364)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a semantic segmentation head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a semantic segmentation head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForSemanticSegmentation

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = AutoModelForSemanticSegmentation.from_config(config)
```

Instantiate one of the model classes of the library (with a semantic segmentation head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **beit** — [BeitForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitForSemanticSegmentation) (BEiT model)
-   **data2vec-vision** — [Data2VecVisionForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/data2vec#transformers.Data2VecVisionForSemanticSegmentation) (Data2VecVision model)
-   **dpt** — [DPTForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/dpt#transformers.DPTForSemanticSegmentation) (DPT model)
-   **mobilenet\_v2** — [MobileNetV2ForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/mobilenet_v2#transformers.MobileNetV2ForSemanticSegmentation) (MobileNetV2 model)
-   **mobilevit** — [MobileViTForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTForSemanticSegmentation) (MobileViT model)
-   **mobilevitv2** — [MobileViTV2ForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/mobilevitv2#transformers.MobileViTV2ForSemanticSegmentation) (MobileViTV2 model)
-   **segformer** — [SegformerForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.SegformerForSemanticSegmentation) (SegFormer model)
-   **upernet** — [UperNetForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/upernet#transformers.UperNetForSemanticSegmentation) (UPerNet model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:

```
>>> from transformers import AutoConfig, AutoModelForSemanticSegmentation

>>> 
>>> model = AutoModelForSemanticSegmentation.from_pretrained("bert-base-cased")

>>> 
>>> model = AutoModelForSemanticSegmentation.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForSemanticSegmentation.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### TFAutoModelForSemanticSegmentation

### class transformers.TFAutoModelForSemanticSegmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_tf_auto.py#L594)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a semantic segmentation head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiate one of the model classes of the library (with a semantic segmentation head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **data2vec-vision** — [TFData2VecVisionForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/data2vec#transformers.TFData2VecVisionForSemanticSegmentation) (Data2VecVision model)
-   **mobilevit** — [TFMobileViTForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.TFMobileViTForSemanticSegmentation) (MobileViT model)
-   **segformer** — [TFSegformerForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.TFSegformerForSemanticSegmentation) (SegFormer model)

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForSemanticSegmentation

>>> 
>>> model = TFAutoModelForSemanticSegmentation.from_pretrained("bert-base-cased")

>>> 
>>> model = TFAutoModelForSemanticSegmentation.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./pt_model/bert_pt_model_config.json")
>>> model = TFAutoModelForSemanticSegmentation.from_pretrained(
...     "./pt_model/bert_pytorch_model.bin", from_pt=True, config=config
... )
```

### AutoModelForInstanceSegmentation

### class transformers.AutoModelForInstanceSegmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1382)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a instance segmentation head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [MaskFormerConfig](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerConfig) configuration class: [MaskFormerForInstanceSegmentation](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation) (MaskFormer model)
    

Instantiates one of the model classes of the library (with a instance segmentation head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForInstanceSegmentation

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = AutoModelForInstanceSegmentation.from_config(config)
```

Instantiate one of the model classes of the library (with a instance segmentation head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **maskformer** — [MaskFormerForInstanceSegmentation](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation) (MaskFormer model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:

```
>>> from transformers import AutoConfig, AutoModelForInstanceSegmentation

>>> 
>>> model = AutoModelForInstanceSegmentation.from_pretrained("bert-base-cased")

>>> 
>>> model = AutoModelForInstanceSegmentation.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForInstanceSegmentation.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForUniversalSegmentation

### class transformers.AutoModelForUniversalSegmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1373)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a universal image segmentation head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiate one of the model classes of the library (with a universal image segmentation head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **detr** — [DetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForSegmentation) (DETR model)
-   **mask2former** — [Mask2FormerForUniversalSegmentation](/docs/transformers/v4.34.0/en/model_doc/mask2former#transformers.Mask2FormerForUniversalSegmentation) (Mask2Former model)
-   **maskformer** — [MaskFormerForInstanceSegmentation](/docs/transformers/v4.34.0/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation) (MaskFormer model)
-   **oneformer** — [OneFormerForUniversalSegmentation](/docs/transformers/v4.34.0/en/model_doc/oneformer#transformers.OneFormerForUniversalSegmentation) (OneFormer model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:

```
>>> from transformers import AutoConfig, AutoModelForUniversalSegmentation

>>> 
>>> model = AutoModelForUniversalSegmentation.from_pretrained("bert-base-cased")

>>> 
>>> model = AutoModelForUniversalSegmentation.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForUniversalSegmentation.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForZeroShotImageClassification

### class transformers.AutoModelForZeroShotImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1348)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a zero-shot image classification head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [AlignConfig](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignConfig) configuration class: [AlignModel](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignModel) (ALIGN model)
    -   [AltCLIPConfig](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPConfig) configuration class: [AltCLIPModel](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPModel) (AltCLIP model)
    -   [BlipConfig](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipConfig) configuration class: [BlipModel](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipModel) (BLIP model)
    -   [CLIPConfig](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPConfig) configuration class: [CLIPModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPModel) (CLIP model)
    -   [CLIPSegConfig](/docs/transformers/v4.34.0/en/model_doc/clipseg#transformers.CLIPSegConfig) configuration class: [CLIPSegModel](/docs/transformers/v4.34.0/en/model_doc/clipseg#transformers.CLIPSegModel) (CLIPSeg model)
    -   [ChineseCLIPConfig](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPConfig) configuration class: [ChineseCLIPModel](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPModel) (Chinese-CLIP model)
    

Instantiates one of the model classes of the library (with a zero-shot image classification head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForZeroShotImageClassification

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = AutoModelForZeroShotImageClassification.from_config(config)
```

Instantiate one of the model classes of the library (with a zero-shot image classification head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **align** — [AlignModel](/docs/transformers/v4.34.0/en/model_doc/align#transformers.AlignModel) (ALIGN model)
-   **altclip** — [AltCLIPModel](/docs/transformers/v4.34.0/en/model_doc/altclip#transformers.AltCLIPModel) (AltCLIP model)
-   **blip** — [BlipModel](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipModel) (BLIP model)
-   **chinese\_clip** — [ChineseCLIPModel](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPModel) (Chinese-CLIP model)
-   **clip** — [CLIPModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPModel) (CLIP model)
-   **clipseg** — [CLIPSegModel](/docs/transformers/v4.34.0/en/model_doc/clipseg#transformers.CLIPSegModel) (CLIPSeg model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:

```
>>> from transformers import AutoConfig, AutoModelForZeroShotImageClassification

>>> 
>>> model = AutoModelForZeroShotImageClassification.from_pretrained("bert-base-cased")

>>> 
>>> model = AutoModelForZeroShotImageClassification.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForZeroShotImageClassification.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### TFAutoModelForZeroShotImageClassification

### class transformers.TFAutoModelForZeroShotImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_tf_auto.py#L585)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a zero-shot image classification head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [BlipConfig](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipConfig) configuration class: [TFBlipModel](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.TFBlipModel) (BLIP model)
    -   [CLIPConfig](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPConfig) configuration class: [TFCLIPModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.TFCLIPModel) (CLIP model)
    

Instantiates one of the model classes of the library (with a zero-shot image classification head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForZeroShotImageClassification

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = TFAutoModelForZeroShotImageClassification.from_config(config)
```

Instantiate one of the model classes of the library (with a zero-shot image classification head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **blip** — [TFBlipModel](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.TFBlipModel) (BLIP model)
-   **clip** — [TFCLIPModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.TFCLIPModel) (CLIP model)

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForZeroShotImageClassification

>>> 
>>> model = TFAutoModelForZeroShotImageClassification.from_pretrained("bert-base-cased")

>>> 
>>> model = TFAutoModelForZeroShotImageClassification.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./pt_model/bert_pt_model_config.json")
>>> model = TFAutoModelForZeroShotImageClassification.from_pretrained(
...     "./pt_model/bert_pytorch_model.bin", from_pt=True, config=config
... )
```

### AutoModelForZeroShotObjectDetection

### class transformers.AutoModelForZeroShotObjectDetection

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1398)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a zero-shot object detection head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [OwlViTConfig](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTConfig) configuration class: [OwlViTForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTForObjectDetection) (OWL-ViT model)
    

Instantiates one of the model classes of the library (with a zero-shot object detection head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForZeroShotObjectDetection

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = AutoModelForZeroShotObjectDetection.from_config(config)
```

Instantiate one of the model classes of the library (with a zero-shot object detection head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **owlvit** — [OwlViTForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/owlvit#transformers.OwlViTForObjectDetection) (OWL-ViT model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:

```
>>> from transformers import AutoConfig, AutoModelForZeroShotObjectDetection

>>> 
>>> model = AutoModelForZeroShotObjectDetection.from_pretrained("bert-base-cased")

>>> 
>>> model = AutoModelForZeroShotObjectDetection.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForZeroShotObjectDetection.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

## Audio

The following auto classes are available for the following audio tasks.

### AutoModelForAudioClassification

### class transformers.AutoModelForAudioClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1428)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a audio classification head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a audio classification head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForAudioClassification

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = AutoModelForAudioClassification.from_config(config)
```

Instantiate one of the model classes of the library (with a audio classification head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **audio-spectrogram-transformer** — [ASTForAudioClassification](/docs/transformers/v4.34.0/en/model_doc/audio-spectrogram-transformer#transformers.ASTForAudioClassification) (Audio Spectrogram Transformer model)
-   **data2vec-audio** — [Data2VecAudioForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/data2vec#transformers.Data2VecAudioForSequenceClassification) (Data2VecAudio model)
-   **hubert** — [HubertForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/hubert#transformers.HubertForSequenceClassification) (Hubert model)
-   **sew** — [SEWForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/sew#transformers.SEWForSequenceClassification) (SEW model)
-   **sew-d** — [SEWDForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/sew-d#transformers.SEWDForSequenceClassification) (SEW-D model)
-   **unispeech** — [UniSpeechForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/unispeech#transformers.UniSpeechForSequenceClassification) (UniSpeech model)
-   **unispeech-sat** — [UniSpeechSatForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/unispeech-sat#transformers.UniSpeechSatForSequenceClassification) (UniSpeechSat model)
-   **wav2vec2** — [Wav2Vec2ForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2ForSequenceClassification) (Wav2Vec2 model)
-   **wav2vec2-conformer** — [Wav2Vec2ConformerForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerForSequenceClassification) (Wav2Vec2-Conformer model)
-   **wavlm** — [WavLMForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/wavlm#transformers.WavLMForSequenceClassification) (WavLM model)
-   **whisper** — [WhisperForAudioClassification](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperForAudioClassification) (Whisper model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:

```
>>> from transformers import AutoConfig, AutoModelForAudioClassification

>>> 
>>> model = AutoModelForAudioClassification.from_pretrained("bert-base-cased")

>>> 
>>> model = AutoModelForAudioClassification.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForAudioClassification.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForAudioFrameClassification

### class transformers.TFAutoModelForAudioClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_tf_auto.py#L536)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a audio classification head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [Wav2Vec2Config](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Config) configuration class: [TFWav2Vec2ForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.TFWav2Vec2ForSequenceClassification) (Wav2Vec2 model)
    

Instantiates one of the model classes of the library (with a audio classification head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForAudioClassification

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = TFAutoModelForAudioClassification.from_config(config)
```

Instantiate one of the model classes of the library (with a audio classification head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **wav2vec2** — [TFWav2Vec2ForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.TFWav2Vec2ForSequenceClassification) (Wav2Vec2 model)

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForAudioClassification

>>> 
>>> model = TFAutoModelForAudioClassification.from_pretrained("bert-base-cased")

>>> 
>>> model = TFAutoModelForAudioClassification.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./pt_model/bert_pt_model_config.json")
>>> model = TFAutoModelForAudioClassification.from_pretrained(
...     "./pt_model/bert_pytorch_model.bin", from_pt=True, config=config
... )
```

### TFAutoModelForAudioFrameClassification

### class transformers.AutoModelForAudioFrameClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1451)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a audio frame (token) classification head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [Data2VecAudioConfig](/docs/transformers/v4.34.0/en/model_doc/data2vec#transformers.Data2VecAudioConfig) configuration class: [Data2VecAudioForAudioFrameClassification](/docs/transformers/v4.34.0/en/model_doc/data2vec#transformers.Data2VecAudioForAudioFrameClassification) (Data2VecAudio model)
    -   [UniSpeechSatConfig](/docs/transformers/v4.34.0/en/model_doc/unispeech-sat#transformers.UniSpeechSatConfig) configuration class: [UniSpeechSatForAudioFrameClassification](/docs/transformers/v4.34.0/en/model_doc/unispeech-sat#transformers.UniSpeechSatForAudioFrameClassification) (UniSpeechSat model)
    -   [Wav2Vec2Config](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Config) configuration class: [Wav2Vec2ForAudioFrameClassification](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2ForAudioFrameClassification) (Wav2Vec2 model)
    -   [Wav2Vec2ConformerConfig](/docs/transformers/v4.34.0/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerConfig) configuration class: [Wav2Vec2ConformerForAudioFrameClassification](/docs/transformers/v4.34.0/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerForAudioFrameClassification) (Wav2Vec2-Conformer model)
    -   [WavLMConfig](/docs/transformers/v4.34.0/en/model_doc/wavlm#transformers.WavLMConfig) configuration class: [WavLMForAudioFrameClassification](/docs/transformers/v4.34.0/en/model_doc/wavlm#transformers.WavLMForAudioFrameClassification) (WavLM model)
    

Instantiates one of the model classes of the library (with a audio frame (token) classification head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForAudioFrameClassification

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = AutoModelForAudioFrameClassification.from_config(config)
```

Instantiate one of the model classes of the library (with a audio frame (token) classification head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **data2vec-audio** — [Data2VecAudioForAudioFrameClassification](/docs/transformers/v4.34.0/en/model_doc/data2vec#transformers.Data2VecAudioForAudioFrameClassification) (Data2VecAudio model)
-   **unispeech-sat** — [UniSpeechSatForAudioFrameClassification](/docs/transformers/v4.34.0/en/model_doc/unispeech-sat#transformers.UniSpeechSatForAudioFrameClassification) (UniSpeechSat model)
-   **wav2vec2** — [Wav2Vec2ForAudioFrameClassification](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2ForAudioFrameClassification) (Wav2Vec2 model)
-   **wav2vec2-conformer** — [Wav2Vec2ConformerForAudioFrameClassification](/docs/transformers/v4.34.0/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerForAudioFrameClassification) (Wav2Vec2-Conformer model)
-   **wavlm** — [WavLMForAudioFrameClassification](/docs/transformers/v4.34.0/en/model_doc/wavlm#transformers.WavLMForAudioFrameClassification) (WavLM model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:

```
>>> from transformers import AutoConfig, AutoModelForAudioFrameClassification

>>> 
>>> model = AutoModelForAudioFrameClassification.from_pretrained("bert-base-cased")

>>> 
>>> model = AutoModelForAudioFrameClassification.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForAudioFrameClassification.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForCTC

### class transformers.AutoModelForCTC

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1435)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a connectionist temporal classification head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiates one of the model classes of the library (with a connectionist temporal classification head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForCTC

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = AutoModelForCTC.from_config(config)
```

Instantiate one of the model classes of the library (with a connectionist temporal classification head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **data2vec-audio** — [Data2VecAudioForCTC](/docs/transformers/v4.34.0/en/model_doc/data2vec#transformers.Data2VecAudioForCTC) (Data2VecAudio model)
-   **hubert** — [HubertForCTC](/docs/transformers/v4.34.0/en/model_doc/hubert#transformers.HubertForCTC) (Hubert model)
-   **mctct** — [MCTCTForCTC](/docs/transformers/v4.34.0/en/model_doc/mctct#transformers.MCTCTForCTC) (M-CTC-T model)
-   **sew** — [SEWForCTC](/docs/transformers/v4.34.0/en/model_doc/sew#transformers.SEWForCTC) (SEW model)
-   **sew-d** — [SEWDForCTC](/docs/transformers/v4.34.0/en/model_doc/sew-d#transformers.SEWDForCTC) (SEW-D model)
-   **unispeech** — [UniSpeechForCTC](/docs/transformers/v4.34.0/en/model_doc/unispeech#transformers.UniSpeechForCTC) (UniSpeech model)
-   **unispeech-sat** — [UniSpeechSatForCTC](/docs/transformers/v4.34.0/en/model_doc/unispeech-sat#transformers.UniSpeechSatForCTC) (UniSpeechSat model)
-   **wav2vec2** — [Wav2Vec2ForCTC](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC) (Wav2Vec2 model)
-   **wav2vec2-conformer** — [Wav2Vec2ConformerForCTC](/docs/transformers/v4.34.0/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerForCTC) (Wav2Vec2-Conformer model)
-   **wavlm** — [WavLMForCTC](/docs/transformers/v4.34.0/en/model_doc/wavlm#transformers.WavLMForCTC) (WavLM model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:

```
>>> from transformers import AutoConfig, AutoModelForCTC

>>> 
>>> model = AutoModelForCTC.from_pretrained("bert-base-cased")

>>> 
>>> model = AutoModelForCTC.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForCTC.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForSpeechSeq2Seq

### class transformers.AutoModelForSpeechSeq2Seq

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1442)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a sequence-to-sequence speech-to-text modeling head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [Pop2PianoConfig](/docs/transformers/v4.34.0/en/model_doc/pop2piano#transformers.Pop2PianoConfig) configuration class: [Pop2PianoForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/pop2piano#transformers.Pop2PianoForConditionalGeneration) (Pop2Piano model)
    -   [Speech2TextConfig](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextConfig) configuration class: [Speech2TextForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextForConditionalGeneration) (Speech2Text model)
    -   [SpeechEncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderConfig) configuration class: [SpeechEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel) (Speech Encoder decoder model)
    -   [SpeechT5Config](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Config) configuration class: [SpeechT5ForSpeechToText](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5ForSpeechToText) (SpeechT5 model)
    -   [WhisperConfig](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperConfig) configuration class: [WhisperForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperForConditionalGeneration) (Whisper model)
    

Instantiates one of the model classes of the library (with a sequence-to-sequence speech-to-text modeling head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForSpeechSeq2Seq

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = AutoModelForSpeechSeq2Seq.from_config(config)
```

Instantiate one of the model classes of the library (with a sequence-to-sequence speech-to-text modeling head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **pop2piano** — [Pop2PianoForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/pop2piano#transformers.Pop2PianoForConditionalGeneration) (Pop2Piano model)
-   **speech-encoder-decoder** — [SpeechEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel) (Speech Encoder decoder model)
-   **speech\_to\_text** — [Speech2TextForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextForConditionalGeneration) (Speech2Text model)
-   **speecht5** — [SpeechT5ForSpeechToText](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5ForSpeechToText) (SpeechT5 model)
-   **whisper** — [WhisperForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperForConditionalGeneration) (Whisper model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:

```
>>> from transformers import AutoConfig, AutoModelForSpeechSeq2Seq

>>> 
>>> model = AutoModelForSpeechSeq2Seq.from_pretrained("bert-base-cased")

>>> 
>>> model = AutoModelForSpeechSeq2Seq.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForSpeechSeq2Seq.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### TFAutoModelForSpeechSeq2Seq

### class transformers.TFAutoModelForSpeechSeq2Seq

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_tf_auto.py#L689)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a sequence-to-sequence speech-to-text modeling head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [Speech2TextConfig](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextConfig) configuration class: [TFSpeech2TextForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.TFSpeech2TextForConditionalGeneration) (Speech2Text model)
    -   [WhisperConfig](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperConfig) configuration class: [TFWhisperForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.TFWhisperForConditionalGeneration) (Whisper model)
    

Instantiates one of the model classes of the library (with a sequence-to-sequence speech-to-text modeling head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForSpeechSeq2Seq

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = TFAutoModelForSpeechSeq2Seq.from_config(config)
```

Instantiate one of the model classes of the library (with a sequence-to-sequence speech-to-text modeling head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **speech\_to\_text** — [TFSpeech2TextForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.TFSpeech2TextForConditionalGeneration) (Speech2Text model)
-   **whisper** — [TFWhisperForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.TFWhisperForConditionalGeneration) (Whisper model)

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForSpeechSeq2Seq

>>> 
>>> model = TFAutoModelForSpeechSeq2Seq.from_pretrained("bert-base-cased")

>>> 
>>> model = TFAutoModelForSpeechSeq2Seq.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./pt_model/bert_pt_model_config.json")
>>> model = TFAutoModelForSpeechSeq2Seq.from_pretrained(
...     "./pt_model/bert_pytorch_model.bin", from_pt=True, config=config
... )
```

### FlaxAutoModelForSpeechSeq2Seq

### class transformers.FlaxAutoModelForSpeechSeq2Seq

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_flax_auto.py#L368)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a sequence-to-sequence speech-to-text modeling head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [SpeechEncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderConfig) configuration class: [FlaxSpeechEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.FlaxSpeechEncoderDecoderModel) (Speech Encoder decoder model)
    -   [WhisperConfig](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperConfig) configuration class: [FlaxWhisperForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.FlaxWhisperForConditionalGeneration) (Whisper model)
    

Instantiates one of the model classes of the library (with a sequence-to-sequence speech-to-text modeling head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, FlaxAutoModelForSpeechSeq2Seq

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = FlaxAutoModelForSpeechSeq2Seq.from_config(config)
```

Instantiate one of the model classes of the library (with a sequence-to-sequence speech-to-text modeling head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **speech-encoder-decoder** — [FlaxSpeechEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.FlaxSpeechEncoderDecoderModel) (Speech Encoder decoder model)
-   **whisper** — [FlaxWhisperForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.FlaxWhisperForConditionalGeneration) (Whisper model)

Examples:

```
>>> from transformers import AutoConfig, FlaxAutoModelForSpeechSeq2Seq

>>> 
>>> model = FlaxAutoModelForSpeechSeq2Seq.from_pretrained("bert-base-cased")

>>> 
>>> model = FlaxAutoModelForSpeechSeq2Seq.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./pt_model/bert_pt_model_config.json")
>>> model = FlaxAutoModelForSpeechSeq2Seq.from_pretrained(
...     "./pt_model/bert_pytorch_model.bin", from_pt=True, config=config
... )
```

### AutoModelForAudioXVector

### class transformers.AutoModelForAudioXVector

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1460)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a audio retrieval via x-vector head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

Instantiate one of the model classes of the library (with a audio retrieval via x-vector head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **data2vec-audio** — [Data2VecAudioForXVector](/docs/transformers/v4.34.0/en/model_doc/data2vec#transformers.Data2VecAudioForXVector) (Data2VecAudio model)
-   **unispeech-sat** — [UniSpeechSatForXVector](/docs/transformers/v4.34.0/en/model_doc/unispeech-sat#transformers.UniSpeechSatForXVector) (UniSpeechSat model)
-   **wav2vec2** — [Wav2Vec2ForXVector](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2ForXVector) (Wav2Vec2 model)
-   **wav2vec2-conformer** — [Wav2Vec2ConformerForXVector](/docs/transformers/v4.34.0/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerForXVector) (Wav2Vec2-Conformer model)
-   **wavlm** — [WavLMForXVector](/docs/transformers/v4.34.0/en/model_doc/wavlm#transformers.WavLMForXVector) (WavLM model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:

```
>>> from transformers import AutoConfig, AutoModelForAudioXVector

>>> 
>>> model = AutoModelForAudioXVector.from_pretrained("bert-base-cased")

>>> 
>>> model = AutoModelForAudioXVector.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForAudioXVector.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForTextToSpectrogram

### class transformers.AutoModelForTextToSpectrogram

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1464)

( \*args\*\*kwargs )

### AutoModelForTextToWaveform

### class transformers.AutoModelForTextToWaveform

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1468)

( \*args\*\*kwargs )

## Multimodal

The following auto classes are available for the following multimodal tasks.

### AutoModelForTableQuestionAnswering

### class transformers.AutoModelForTableQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1285)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a table question answering head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [TapasConfig](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasConfig) configuration class: [TapasForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasForQuestionAnswering) (TAPAS model)
    

Instantiates one of the model classes of the library (with a table question answering head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForTableQuestionAnswering

>>> 
>>> config = AutoConfig.from_pretrained("google/tapas-base-finetuned-wtq")
>>> model = AutoModelForTableQuestionAnswering.from_config(config)
```

Instantiate one of the model classes of the library (with a table question answering head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **tapas** — [TapasForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasForQuestionAnswering) (TAPAS model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:

```
>>> from transformers import AutoConfig, AutoModelForTableQuestionAnswering

>>> 
>>> model = AutoModelForTableQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")

>>> 
>>> model = AutoModelForTableQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./tf_model/tapas_tf_model_config.json")
>>> model = AutoModelForTableQuestionAnswering.from_pretrained(
...     "./tf_model/tapas_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### TFAutoModelForTableQuestionAnswering

### class transformers.TFAutoModelForTableQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_tf_auto.py#L653)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a table question answering head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [TapasConfig](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasConfig) configuration class: [TFTapasForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TFTapasForQuestionAnswering) (TAPAS model)
    

Instantiates one of the model classes of the library (with a table question answering head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForTableQuestionAnswering

>>> 
>>> config = AutoConfig.from_pretrained("google/tapas-base-finetuned-wtq")
>>> model = TFAutoModelForTableQuestionAnswering.from_config(config)
```

Instantiate one of the model classes of the library (with a table question answering head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **tapas** — [TFTapasForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TFTapasForQuestionAnswering) (TAPAS model)

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForTableQuestionAnswering

>>> 
>>> model = TFAutoModelForTableQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")

>>> 
>>> model = TFAutoModelForTableQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./pt_model/tapas_pt_model_config.json")
>>> model = TFAutoModelForTableQuestionAnswering.from_pretrained(
...     "./pt_model/tapas_pytorch_model.bin", from_pt=True, config=config
... )
```

### AutoModelForDocumentQuestionAnswering

### class transformers.AutoModelForDocumentQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1307)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a document question answering head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [LayoutLMConfig](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMConfig) configuration class: [LayoutLMForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMForQuestionAnswering) (LayoutLM model)
    -   [LayoutLMv2Config](/docs/transformers/v4.34.0/en/model_doc/layoutlmv2#transformers.LayoutLMv2Config) configuration class: [LayoutLMv2ForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForQuestionAnswering) (LayoutLMv2 model)
    -   [LayoutLMv3Config](/docs/transformers/v4.34.0/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config) configuration class: [LayoutLMv3ForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForQuestionAnswering) (LayoutLMv3 model)
    

Instantiates one of the model classes of the library (with a document question answering head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForDocumentQuestionAnswering

>>> 
>>> config = AutoConfig.from_pretrained("impira/layoutlm-document-qa", revision="52e01b3")
>>> model = AutoModelForDocumentQuestionAnswering.from_config(config)
```

Instantiate one of the model classes of the library (with a document question answering head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **layoutlm** — [LayoutLMForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMForQuestionAnswering) (LayoutLM model)
-   **layoutlmv2** — [LayoutLMv2ForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForQuestionAnswering) (LayoutLMv2 model)
-   **layoutlmv3** — [LayoutLMv3ForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForQuestionAnswering) (LayoutLMv3 model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:

```
>>> from transformers import AutoConfig, AutoModelForDocumentQuestionAnswering

>>> 
>>> model = AutoModelForDocumentQuestionAnswering.from_pretrained("impira/layoutlm-document-qa", revision="52e01b3")

>>> 
>>> model = AutoModelForDocumentQuestionAnswering.from_pretrained("impira/layoutlm-document-qa", revision="52e01b3", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./tf_model/layoutlm_tf_model_config.json")
>>> model = AutoModelForDocumentQuestionAnswering.from_pretrained(
...     "./tf_model/layoutlm_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### TFAutoModelForDocumentQuestionAnswering

### class transformers.TFAutoModelForDocumentQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_tf_auto.py#L642)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a document question answering head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [LayoutLMConfig](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMConfig) configuration class: [TFLayoutLMForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.TFLayoutLMForQuestionAnswering) (LayoutLM model)
    -   [LayoutLMv3Config](/docs/transformers/v4.34.0/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config) configuration class: [TFLayoutLMv3ForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/layoutlmv3#transformers.TFLayoutLMv3ForQuestionAnswering) (LayoutLMv3 model)
    

Instantiates one of the model classes of the library (with a document question answering head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForDocumentQuestionAnswering

>>> 
>>> config = AutoConfig.from_pretrained("impira/layoutlm-document-qa", revision="52e01b3")
>>> model = TFAutoModelForDocumentQuestionAnswering.from_config(config)
```

Instantiate one of the model classes of the library (with a document question answering head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **layoutlm** — [TFLayoutLMForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.TFLayoutLMForQuestionAnswering) (LayoutLM model)
-   **layoutlmv3** — [TFLayoutLMv3ForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/layoutlmv3#transformers.TFLayoutLMv3ForQuestionAnswering) (LayoutLMv3 model)

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForDocumentQuestionAnswering

>>> 
>>> model = TFAutoModelForDocumentQuestionAnswering.from_pretrained("impira/layoutlm-document-qa", revision="52e01b3")

>>> 
>>> model = TFAutoModelForDocumentQuestionAnswering.from_pretrained("impira/layoutlm-document-qa", revision="52e01b3", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./pt_model/layoutlm_pt_model_config.json")
>>> model = TFAutoModelForDocumentQuestionAnswering.from_pretrained(
...     "./pt_model/layoutlm_pytorch_model.bin", from_pt=True, config=config
... )
```

### AutoModelForVisualQuestionAnswering

### class transformers.AutoModelForVisualQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1296)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a visual question answering head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [Blip2Config](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2Config) configuration class: [Blip2ForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2ForConditionalGeneration) (BLIP-2 model)
    -   [ViltConfig](/docs/transformers/v4.34.0/en/model_doc/vilt#transformers.ViltConfig) configuration class: [ViltForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/vilt#transformers.ViltForQuestionAnswering) (ViLT model)
    

Instantiates one of the model classes of the library (with a visual question answering head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForVisualQuestionAnswering

>>> 
>>> config = AutoConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
>>> model = AutoModelForVisualQuestionAnswering.from_config(config)
```

Instantiate one of the model classes of the library (with a visual question answering head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **blip-2** — [Blip2ForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2ForConditionalGeneration) (BLIP-2 model)
-   **vilt** — [ViltForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/vilt#transformers.ViltForQuestionAnswering) (ViLT model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:

```
>>> from transformers import AutoConfig, AutoModelForVisualQuestionAnswering

>>> 
>>> model = AutoModelForVisualQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

>>> 
>>> model = AutoModelForVisualQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./tf_model/vilt_tf_model_config.json")
>>> model = AutoModelForVisualQuestionAnswering.from_pretrained(
...     "./tf_model/vilt_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForVision2Seq

### class transformers.AutoModelForVision2Seq

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_auto.py#L1421)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a vision-to-text modeling head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [Blip2Config](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2Config) configuration class: [Blip2ForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2ForConditionalGeneration) (BLIP-2 model)
    -   [BlipConfig](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipConfig) configuration class: [BlipForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipForConditionalGeneration) (BLIP model)
    -   [GitConfig](/docs/transformers/v4.34.0/en/model_doc/git#transformers.GitConfig) configuration class: [GitForCausalLM](/docs/transformers/v4.34.0/en/model_doc/git#transformers.GitForCausalLM) (GIT model)
    -   [InstructBlipConfig](/docs/transformers/v4.34.0/en/model_doc/instructblip#transformers.InstructBlipConfig) configuration class: [InstructBlipForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/instructblip#transformers.InstructBlipForConditionalGeneration) (InstructBLIP model)
    -   [Pix2StructConfig](/docs/transformers/v4.34.0/en/model_doc/pix2struct#transformers.Pix2StructConfig) configuration class: [Pix2StructForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/pix2struct#transformers.Pix2StructForConditionalGeneration) (Pix2Struct model)
    -   [VisionEncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderConfig) configuration class: [VisionEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel) (Vision Encoder decoder model)
    

Instantiates one of the model classes of the library (with a vision-to-text modeling head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, AutoModelForVision2Seq

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = AutoModelForVision2Seq.from_config(config)
```

Instantiate one of the model classes of the library (with a vision-to-text modeling head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **blip** — [BlipForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipForConditionalGeneration) (BLIP model)
-   **blip-2** — [Blip2ForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/blip-2#transformers.Blip2ForConditionalGeneration) (BLIP-2 model)
-   **git** — [GitForCausalLM](/docs/transformers/v4.34.0/en/model_doc/git#transformers.GitForCausalLM) (GIT model)
-   **instructblip** — [InstructBlipForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/instructblip#transformers.InstructBlipForConditionalGeneration) (InstructBLIP model)
-   **pix2struct** — [Pix2StructForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/pix2struct#transformers.Pix2StructForConditionalGeneration) (Pix2Struct model)
-   **vision-encoder-decoder** — [VisionEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel) (Vision Encoder decoder model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:

```
>>> from transformers import AutoConfig, AutoModelForVision2Seq

>>> 
>>> model = AutoModelForVision2Seq.from_pretrained("bert-base-cased")

>>> 
>>> model = AutoModelForVision2Seq.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForVision2Seq.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### TFAutoModelForVision2Seq

### class transformers.TFAutoModelForVision2Seq

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_tf_auto.py#L603)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a vision-to-text modeling head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [BlipConfig](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.BlipConfig) configuration class: [TFBlipForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.TFBlipForConditionalGeneration) (BLIP model)
    -   [VisionEncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderConfig) configuration class: [TFVisionEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.TFVisionEncoderDecoderModel) (Vision Encoder decoder model)
    

Instantiates one of the model classes of the library (with a vision-to-text modeling head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForVision2Seq

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = TFAutoModelForVision2Seq.from_config(config)
```

Instantiate one of the model classes of the library (with a vision-to-text modeling head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **blip** — [TFBlipForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/blip#transformers.TFBlipForConditionalGeneration) (BLIP model)
-   **vision-encoder-decoder** — [TFVisionEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.TFVisionEncoderDecoderModel) (Vision Encoder decoder model)

Examples:

```
>>> from transformers import AutoConfig, TFAutoModelForVision2Seq

>>> 
>>> model = TFAutoModelForVision2Seq.from_pretrained("bert-base-cased")

>>> 
>>> model = TFAutoModelForVision2Seq.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./pt_model/bert_pt_model_config.json")
>>> model = TFAutoModelForVision2Seq.from_pretrained(
...     "./pt_model/bert_pytorch_model.bin", from_pt=True, config=config
... )
```

### FlaxAutoModelForVision2Seq

### class transformers.FlaxAutoModelForVision2Seq

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/modeling_flax_auto.py#L361)

( \*args\*\*kwargs )

This is a generic model class that will be instantiated as one of the model classes of the library (with a vision-to-text modeling head) when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_config) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/auto/auto_factory.py#L417)

( \*\*kwargs )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The model class to instantiate is selected based on the configuration class:
    
    -   [VisionEncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderConfig) configuration class: [FlaxVisionEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.FlaxVisionEncoderDecoderModel) (Vision Encoder decoder model)
    

Instantiates one of the model classes of the library (with a vision-to-text modeling head) from a configuration.

Note: Loading a model from its configuration file does **not** load the model weights. It only affects the model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) to load the model weights.

Examples:

```
>>> from transformers import AutoConfig, FlaxAutoModelForVision2Seq

>>> 
>>> config = AutoConfig.from_pretrained("bert-base-cased")
>>> model = FlaxAutoModelForVision2Seq.from_config(config)
```

Instantiate one of the model classes of the library (with a vision-to-text modeling head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

-   **vision-encoder-decoder** — [FlaxVisionEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.FlaxVisionEncoderDecoderModel) (Vision Encoder decoder model)

Examples:

```
>>> from transformers import AutoConfig, FlaxAutoModelForVision2Seq

>>> 
>>> model = FlaxAutoModelForVision2Seq.from_pretrained("bert-base-cased")

>>> 
>>> model = FlaxAutoModelForVision2Seq.from_pretrained("bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> 
>>> config = AutoConfig.from_pretrained("./pt_model/bert_pt_model_config.json")
>>> model = FlaxAutoModelForVision2Seq.from_pretrained(
...     "./pt_model/bert_pytorch_model.bin", from_pt=True, config=config
... )
```