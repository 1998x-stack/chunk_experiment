# Exporting 🤗 Transformers models to ONNX

🤗 Transformers provides a `transformers.onnx` package that enables you to convert model checkpoints to an ONNX graph by leveraging configuration objects.

See the [guide](../serialization) on exporting 🤗 Transformers models for more details.

## ONNX Configurations

We provide three abstract classes that you should inherit from, depending on the type of model architecture you wish to export:

-   Encoder-based models inherit from [OnnxConfig](/docs/transformers/v4.34.0/en/main_classes/onnx#transformers.onnx.OnnxConfig)
-   Decoder-based models inherit from [OnnxConfigWithPast](/docs/transformers/v4.34.0/en/main_classes/onnx#transformers.onnx.OnnxConfigWithPast)
-   Encoder-decoder models inherit from [OnnxSeq2SeqConfigWithPast](/docs/transformers/v4.34.0/en/main_classes/onnx#transformers.onnx.OnnxSeq2SeqConfigWithPast)

### OnnxConfig

### class transformers.onnx.OnnxConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/onnx/config.py#L68)

( config: PretrainedConfigtask: str = 'default'patching\_specs: typing.List\[transformers.onnx.config.PatchingSpec\] = None )

Base class for ONNX exportable model describing metadata on how to export the model through the ONNX format.

#### flatten\_output\_collection\_property

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/onnx/config.py#L424)

( name: strfield: typing.Iterable\[typing.Any\] ) → (Dict\[str, Any\])

Outputs with flattened structure and key mapping this new structure.

Flatten any potential nested structure expanding the name of the field with the index of the element within the structure.

#### from\_model\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/onnx/config.py#L127)

( config: PretrainedConfigtask: str = 'default' )

Instantiate a OnnxConfig for a specific model

#### generate\_dummy\_inputs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/onnx/config.py#L280)

( preprocessor: typing.Union\[ForwardRef('PreTrainedTokenizerBase'), ForwardRef('FeatureExtractionMixin'), ForwardRef('ImageProcessingMixin')\]batch\_size: int = -1seq\_length: int = -1num\_choices: int = -1is\_pair: bool = Falseframework: typing.Optional\[transformers.utils.generic.TensorType\] = Nonenum\_channels: int = 3image\_width: int = 40image\_height: int = 40sampling\_rate: int = 22050time\_duration: float = 5.0frequency: int = 220tokenizer: PreTrainedTokenizerBase = None )

Generate inputs to provide to the ONNX exporter for the specific framework

#### generate\_dummy\_inputs\_onnxruntime

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/onnx/config.py#L400)

( reference\_model\_inputs: typing.Mapping\[str, typing.Any\] ) → `Mapping[str, Tensor]`

Parameters

-   **reference\_model\_inputs** (\[`Mapping[str, Tensor]`) — Reference inputs for the model.

Returns

`Mapping[str, Tensor]`

The mapping holding the kwargs to provide to the model’s forward function

Generate inputs for ONNX Runtime using the reference model inputs. Override this to run inference with seq2seq models which have the encoder and decoder exported as separate ONNX files.

#### use\_external\_data\_format

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/onnx/config.py#L241)

( num\_parameters: int )

Flag indicating if the model requires using external data format

### OnnxConfigWithPast

### class transformers.onnx.OnnxConfigWithPast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/onnx/config.py#L443)

( config: PretrainedConfigtask: str = 'default'patching\_specs: typing.List\[transformers.onnx.config.PatchingSpec\] = Noneuse\_past: bool = False )

#### fill\_with\_past\_key\_values\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/onnx/config.py#L550)

( inputs\_or\_outputs: typing.Mapping\[str, typing.Mapping\[int, str\]\]direction: strinverted\_values\_shape: bool = False )

Fill the input\_or\_outputs mapping with past\_key\_values dynamic axes considering.

#### with\_past

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/onnx/config.py#L454)

( config: PretrainedConfigtask: str = 'default' )

Instantiate a OnnxConfig with `use_past` attribute set to True

### OnnxSeq2SeqConfigWithPast

### class transformers.onnx.OnnxSeq2SeqConfigWithPast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/onnx/config.py#L590)

( config: PretrainedConfigtask: str = 'default'patching\_specs: typing.List\[transformers.onnx.config.PatchingSpec\] = Noneuse\_past: bool = False )

## ONNX Features

Each ONNX configuration is associated with a set of _features_ that enable you to export models for different types of topologies or tasks.

### FeaturesManager

### class transformers.onnx.FeaturesManager

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/onnx/features.py#L85)

( )

#### check\_supported\_model\_or\_raise

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/onnx/features.py#L711)

( model: typing.Union\[ForwardRef('PreTrainedModel'), ForwardRef('TFPreTrainedModel')\]feature: str = 'default' )

Check whether or not the model has the requested features.

#### determine\_framework

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/onnx/features.py#L628)

( model: strframework: str = None )

Parameters

-   **model** (`str`) — The name of the model to export.
-   **framework** (`str`, _optional_, defaults to `None`) — The framework to use for the export. See above for priority if none provided.

Determines the framework to use for the export.

The priority is in the following order:

1.  User input via `framework`.
2.  If local checkpoint is provided, use the same framework as the checkpoint.
3.  Available framework in environment, with priority given to PyTorch

#### get\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/onnx/features.py#L736)

( model\_type: strfeature: str ) → `OnnxConfig`

Parameters

-   **model\_type** (`str`) — The model type to retrieve the config for.
-   **feature** (`str`) — The feature to retrieve the config for.

config for the combination

Gets the OnnxConfig for a model\_type and feature combination.

#### get\_model\_class\_for\_feature

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/onnx/features.py#L601)

( feature: strframework: str = 'pt' )

Parameters

-   **feature** (`str`) — The feature required.
-   **framework** (`str`, _optional_, defaults to `"pt"`) — The framework to use for the export.

Attempts to retrieve an AutoModel class from a feature name.

#### get\_model\_from\_feature

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/onnx/features.py#L678)

( feature: strmodel: strframework: str = Nonecache\_dir: str = None )

Parameters

-   **feature** (`str`) — The feature required.
-   **model** (`str`) — The name of the model to export.
-   **framework** (`str`, _optional_, defaults to `None`) — The framework to use for the export. See `FeaturesManager.determine_framework` for the priority should none be provided.

Attempts to retrieve a model from a model’s name and the feature to be enabled.

#### get\_supported\_features\_for\_model\_type

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/onnx/features.py#L556)

( model\_type: strmodel\_name: typing.Optional\[str\] = None )

Parameters

-   **model\_type** (`str`) — The model type to retrieve the supported features for.
-   **model\_name** (`str`, _optional_) — The name attribute of the model object, only used for the exception message.

Tries to retrieve the feature -> OnnxConfig constructor map from the model type.