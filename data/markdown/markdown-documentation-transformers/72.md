# Models

The base classes [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel), [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel), and [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel) implement the common methods for loading/saving a model either from a local file or directory, or from a pretrained model configuration provided by the library (downloaded from HuggingFace’s AWS S3 repository).

[PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel) and [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel) also implement a few methods which are common among all the models to:

-   resize the input token embeddings when new tokens are added to the vocabulary
-   prune the attention heads of the model.

The other methods that are common to each model are defined in [ModuleUtilsMixin](/docs/transformers/v4.34.0/en/main_classes/model#transformers.modeling_utils.ModuleUtilsMixin) (for the PyTorch models) and `~modeling_tf_utils.TFModuleUtilsMixin` (for the TensorFlow models) or for text generation, [GenerationMixin](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin) (for the PyTorch models), [TFGenerationMixin](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.TFGenerationMixin) (for the TensorFlow models) and [FlaxGenerationMixin](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.FlaxGenerationMixin) (for the Flax/JAX models).

## PreTrainedModel

### class transformers.PreTrainedModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L1069)

( config: PretrainedConfig\*inputs\*\*kwargs )

Base class for all models.

[PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel) takes care of storing the configuration of the models and handles methods for loading, downloading and saving models as well as a few methods common to all models to:

-   resize the input embeddings,
-   prune heads in the self-attention heads.

Class attributes (overridden by derived classes):

-   **config\_class** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — A subclass of [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) to use as configuration class for this model architecture.
    
-   **load\_tf\_weights** (`Callable`) — A python _method_ for loading a TensorFlow checkpoint in a PyTorch model, taking as arguments:
    
    -   **model** ([PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel)) — An instance of the model on which to load the TensorFlow checkpoint.
    -   **config** (`PreTrainedConfig`) — An instance of the configuration associated to the model.
    -   **path** (`str`) — A path to the TensorFlow checkpoint.
-   **base\_model\_prefix** (`str`) — A string indicating the attribute associated to the base model in derived classes of the same architecture adding modules on top of the base model.
    
-   **is\_parallelizable** (`bool`) — A flag indicating whether this model supports model parallelization.
    
-   **main\_input\_name** (`str`) — The name of the principal input to the model (often `input_ids` for NLP models, `pixel_values` for vision models and `input_values` for speech models).
    

#### push\_to\_hub

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/hub.py#L786)

( repo\_id: struse\_temp\_dir: typing.Optional\[bool\] = Nonecommit\_message: typing.Optional\[str\] = Noneprivate: typing.Optional\[bool\] = Nonetoken: typing.Union\[bool, str, NoneType\] = Nonemax\_shard\_size: typing.Union\[int, str, NoneType\] = '10GB'create\_pr: bool = Falsesafe\_serialization: bool = Falserevision: str = None\*\*deprecated\_kwargs )

Upload the model file to the 🤗 Model Hub.

Examples:

```
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-cased")


model.push_to_hub("my-finetuned-bert")


model.push_to_hub("huggingface/my-finetuned-bert")
```

#### can\_generate

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L1232)

( ) → `bool`

Whether this model can generate sequences with `.generate()`.

Returns whether this model can generate sequences with `.generate()`.

Removes the `_require_grads_hook`.

Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping the model weights fixed.

#### from\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L2201)

( pretrained\_model\_name\_or\_path: typing.Union\[str, os.PathLike, NoneType\]\*model\_argsconfig: typing.Union\[transformers.configuration\_utils.PretrainedConfig, str, os.PathLike, NoneType\] = Nonecache\_dir: typing.Union\[str, os.PathLike, NoneType\] = Noneignore\_mismatched\_sizes: bool = Falseforce\_download: bool = Falselocal\_files\_only: bool = Falsetoken: typing.Union\[bool, str, NoneType\] = Nonerevision: str = 'main'use\_safetensors: bool = None\*\*kwargs )

Instantiate a pretrained pytorch model from a pre-trained model configuration.

The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train the model, you should first set it back in training mode with `model.train()`.

The warning _Weights from XXX not initialized from pretrained model_ means that the weights of XXX do not come pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning task.

The warning _Weights from XXX not used in YYY_ means that the layer XXX is not used by YYY, therefore those weights are discarded.

Activate the special [“offline-mode”](https://huggingface.co/transformers/installation.html#offline-mode) to use this method in a firewalled environment.

Examples:

```
>>> from transformers import BertConfig, BertModel

>>> 
>>> model = BertModel.from_pretrained("bert-base-uncased")
>>> 
>>> model = BertModel.from_pretrained("./test/saved_model/")
>>> 
>>> model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
>>> assert model.config.output_attentions == True
>>> 
>>> config = BertConfig.from_json_file("./tf_model/my_tf_model_config.json")
>>> model = BertModel.from_pretrained("./tf_model/my_tf_checkpoint.ckpt.index", from_tf=True, config=config)
>>> 
>>> model = BertModel.from_pretrained("bert-base-uncased", from_flax=True)
```

-   `low_cpu_mem_usage` algorithm:

This is an experimental function that loads the model using ~1x model size CPU memory

Here is how it works:

1.  save which state\_dict keys we have
2.  drop state\_dict before the model is created, since the latter takes 1x model size CPU memory
3.  after the model has been instantiated switch to the meta device all params/buffers that are going to be replaced from the loaded state\_dict
4.  load state\_dict 2nd time
5.  replace the params/buffers from the state\_dict

Currently, it can’t handle deepspeed ZeRO stage 3 and ignores loading errors

#### get\_input\_embeddings

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L1341)

( ) → `nn.Module`

A torch module mapping vocabulary to hidden states.

Returns the model’s input embeddings.

#### get\_memory\_footprint

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L2141)

( return\_buffers = True )

Parameters

-   **return\_buffers** (`bool`, _optional_, defaults to `True`) — Whether to return the size of the buffer tensors in the computation of the memory footprint. Buffers are tensors that do not require gradients and not registered as parameters. E.g. mean and std in batch norm layers. Please see: [https://discuss.pytorch.org/t/what-pytorch-means-by-buffers/120266/2](https://discuss.pytorch.org/t/what-pytorch-means-by-buffers/120266/2)

Get the memory footprint of a model. This will return the memory footprint of the current model in bytes. Useful to benchmark the memory footprint of the current model and design some tests. Solution inspired from the PyTorch discussions: [https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2](https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2)

#### get\_output\_embeddings

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L1367)

( ) → `nn.Module`

A torch module mapping hidden states to vocabulary.

Returns the model’s output embeddings.

Deactivates gradient checkpointing for the current model.

Note that in other frameworks this feature can be referred to as “activation checkpointing” or “checkpoint activations”.

Activates gradient checkpointing for the current model.

Note that in other frameworks this feature can be referred to as “activation checkpointing” or “checkpoint activations”.

If needed prunes and maybe initializes weights. If using a custom `PreTrainedModel`, you need to implement any initialization logic in `_init_weights`.

A method executed at the end of each Transformer model initialization, to execute code that needs the model’s modules properly initialized (such as weight initialization).

#### prune\_heads

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L1802)

( heads\_to\_prune: typing.Dict\[int, typing.List\[int\]\] )

Parameters

-   **heads\_to\_prune** (`Dict[int, List[int]]`) — Dictionary with keys being selected layer indices (`int`) and associated values being the list of heads to prune in said layer (list of `int`). For instance {1: \[0, 2\], 2: \[2, 3\]} will prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.

Prunes heads of the base model.

#### register\_for\_auto\_class

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L3852)

( auto\_class = 'AutoModel' )

Parameters

-   **auto\_class** (`str` or `type`, _optional_, defaults to `"AutoModel"`) — The auto class to register this new model with.

Register this class with a given auto class. This should only be used for custom models as the ones in the library are already mapped with an auto class.

This API is experimental and may have some slight breaking changes in the next releases.

#### resize\_token\_embeddings

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L1507)

( new\_num\_tokens: typing.Optional\[int\] = Nonepad\_to\_multiple\_of: typing.Optional\[int\] = None ) → `torch.nn.Embedding`

Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

Reverts the transformation from [to\_bettertransformer()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.to_bettertransformer) so that the original modeling is used, for example in order to save the model.

#### save\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L1860)

( save\_directory: typing.Union\[str, os.PathLike\]is\_main\_process: bool = Truestate\_dict: typing.Optional\[dict\] = Nonesave\_function: typing.Callable = <function save at 0x7f0b4e4faf70>push\_to\_hub: bool = Falsemax\_shard\_size: typing.Union\[int, str\] = '10GB'safe\_serialization: bool = Falsevariant: typing.Optional\[str\] = Nonetoken: typing.Union\[bool, str, NoneType\] = Nonesave\_peft\_format: bool = True\*\*kwargs )

Save a model and its configuration file to a directory, so that it can be re-loaded using the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) class method.

#### set\_input\_embeddings

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L1354)

( value: Module )

Parameters

-   **value** (`nn.Module`) — A module mapping vocabulary to hidden states.

Set model’s input embeddings.

Tie the weights between the input embeddings and the output embeddings.

If the `torchscript` flag is set in the configuration, can’t handle parameter sharing so we are cloning the weights instead.

#### warn\_if\_padding\_and\_no\_attention\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L3928)

( input\_idsattention\_mask )

Shows a one-time warning if the input\_ids appear to contain padding and no attention mask was given.

### Large model loading

In Transformers 4.20.0, the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method has been reworked to accommodate large models using [Accelerate](https://huggingface.co/docs/accelerate/big_modeling). This requires Accelerate >= 0.9.0 and PyTorch >= 1.9.0. Instead of creating the full model, then loading the pretrained weights inside it (which takes twice the size of the model in RAM, one for the randomly initialized model, one for the weights), there is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded.

This option can be activated with `low_cpu_mem_usage=True`. The model is first created on the Meta device (with empty weights) and the state dict is then loaded inside it (shard by shard in the case of a sharded checkpoint). This way the maximum RAM used is the full size of the model only.

```
from transformers import AutoModelForSeq2SeqLM

t0pp = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", low_cpu_mem_usage=True)
```

Moreover, you can directly place the model on different devices if it doesn’t fully fit in RAM (only works for inference for now). With `device_map="auto"`, Accelerate will determine where to put each layer to maximize the use of your fastest devices (GPUs) and offload the rest on the CPU, or even the hard drive if you don’t have enough GPU RAM (or CPU RAM). Even if the model is split across several devices, it will run as you would normally expect.

When passing a `device_map`, `low_cpu_mem_usage` is automatically set to `True`, so you don’t need to specify it:

```
from transformers import AutoModelForSeq2SeqLM

t0pp = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", device_map="auto")
```

You can inspect how the model was split across devices by looking at its `hf_device_map` attribute:

```
{'shared': 0,
 'decoder.embed_tokens': 0,
 'encoder': 0,
 'decoder.block.0': 0,
 'decoder.block.1': 1,
 'decoder.block.2': 1,
 'decoder.block.3': 1,
 'decoder.block.4': 1,
 'decoder.block.5': 1,
 'decoder.block.6': 1,
 'decoder.block.7': 1,
 'decoder.block.8': 1,
 'decoder.block.9': 1,
 'decoder.block.10': 1,
 'decoder.block.11': 1,
 'decoder.block.12': 1,
 'decoder.block.13': 1,
 'decoder.block.14': 1,
 'decoder.block.15': 1,
 'decoder.block.16': 1,
 'decoder.block.17': 1,
 'decoder.block.18': 1,
 'decoder.block.19': 1,
 'decoder.block.20': 1,
 'decoder.block.21': 1,
 'decoder.block.22': 'cpu',
 'decoder.block.23': 'cpu',
 'decoder.final_layer_norm': 'cpu',
 'decoder.dropout': 'cpu',
 'lm_head': 'cpu'}
```

You can also write your own device map following the same format (a dictionary layer name to device). It should map all parameters of the model to a given device, but you don’t have to detail where all the submodules of one layer go if that layer is entirely on the same device. For instance, the following device map would work properly for T0pp (as long as you have the GPU memory):

```
device_map = {"shared": 0, "encoder": 0, "decoder": 1, "lm_head": 1}
```

Another way to minimize the memory impact of your model is to instantiate it at a lower precision dtype (like `torch.float16`) or use direct quantization techniques as described below.

### Model Instantiation dtype

Under Pytorch a model normally gets instantiated with `torch.float32` format. This can be an issue if one tries to load a model whose weights are in fp16, since it’d require twice as much memory. To overcome this limitation, you can either explicitly pass the desired `dtype` using `torch_dtype` argument:

```
model = T5ForConditionalGeneration.from_pretrained("t5", torch_dtype=torch.float16)
```

or, if you want the model to always load in the most optimal memory pattern, you can use the special value `"auto"`, and then `dtype` will be automatically derived from the model’s weights:

```
model = T5ForConditionalGeneration.from_pretrained("t5", torch_dtype="auto")
```

Models instantiated from scratch can also be told which `dtype` to use with:

```
config = T5Config.from_pretrained("t5")
model = AutoModel.from_config(config)
```

Due to Pytorch design, this functionality is only available for floating dtypes.

## ModuleUtilsMixin

### class transformers.modeling\_utils.ModuleUtilsMixin

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L765)

( )

A few utilities for `torch.nn.Modules`, to be used as a mixin.

Add a memory hook before and after each sub-module forward pass to record increase in memory consumption.

Increase in memory consumption is stored in a `mem_rss_diff` attribute for each module and can be reset to zero with `model.reset_memory_hooks_state()`.

#### estimate\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L1021)

( input\_dict: typing.Dict\[str, typing.Union\[torch.Tensor, typing.Any\]\] ) → `int`

Parameters

-   **inputs** (`dict`) — The model inputs.

The total number of tokens.

Helper function to estimate the total number of tokens from the model inputs.

#### floating\_point\_ops

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L1042)

( input\_dict: typing.Dict\[str, typing.Union\[torch.Tensor, typing.Any\]\]exclude\_embeddings: bool = True ) → `int`

Parameters

-   **batch\_size** (`int`) — The batch size for the forward pass.
-   **sequence\_length** (`int`) — The number of tokens in each line of the batch.
-   **exclude\_embeddings** (`bool`, _optional_, defaults to `True`) — Whether or not to count embedding and softmax operations.

The number of floating-point operations.

Get number of (optionally, non-embeddings) floating-point operations for the forward and backward passes of a batch with this transformer model. Default approximation neglects the quadratic dependency on the number of tokens (valid if `12 * d_model << sequence_length`) as laid out in [this paper](https://arxiv.org/pdf/2001.08361.pdf) section 2.1. Should be overridden for transformers with parameter re-use e.g. Albert or Universal Transformers, or if doing long-range modeling with very high sequence lengths.

#### get\_extended\_attention\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L884)

( attention\_mask: Tensorinput\_shape: typing.Tuple\[int\]device: device = Nonedtype: torch.float32 = None )

Parameters

-   **attention\_mask** (`torch.Tensor`) — Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
-   **input\_shape** (`Tuple[int]`) — The shape of the input to the model.

Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

#### get\_head\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L936)

( head\_mask: typing.Optional\[torch.Tensor\]num\_hidden\_layers: intis\_attention\_chunked: bool = False )

Parameters

-   **head\_mask** (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, _optional_) — The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
-   **num\_hidden\_layers** (`int`) — The number of hidden layers in the model.
-   **is\_attention\_chunked** (`bool`, _optional_, defaults to `False`) — Whether or not the attentions scores are computed by chunks or not.

Prepare the head mask if needed.

#### invert\_attention\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L832)

( encoder\_attention\_mask: Tensor ) → `torch.Tensor`

Parameters

-   **encoder\_attention\_mask** (`torch.Tensor`) — An attention mask.

The inverted attention mask.

Invert an attention mask (e.g., switches 0. and 1.).

#### num\_parameters

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L974)

( only\_trainable: bool = Falseexclude\_embeddings: bool = False ) → `int`

Parameters

-   **only\_trainable** (`bool`, _optional_, defaults to `False`) — Whether or not to return only the number of trainable parameters
-   **exclude\_embeddings** (`bool`, _optional_, defaults to `False`) — Whether or not to return only the number of non-embeddings parameters

The number of parameters.

Get number of (optionally, trainable or non-embeddings) parameters in the module.

## TFPreTrainedModel

### class transformers.TFPreTrainedModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L1057)

( \*args\*\*kwargs )

Base class for all TF models.

[TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel) takes care of storing the configuration of the models and handles methods for loading, downloading and saving models as well as a few methods common to all models to:

-   resize the input embeddings,
-   prune heads in the self-attention heads.

Class attributes (overridden by derived classes):

-   **config\_class** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — A subclass of [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) to use as configuration class for this model architecture.
-   **base\_model\_prefix** (`str`) — A string indicating the attribute associated to the base model in derived classes of the same architecture adding modules on top of the base model.
-   **main\_input\_name** (`str`) — The name of the principal input to the model (often `input_ids` for NLP models, `pixel_values` for vision models and `input_values` for speech models).

#### push\_to\_hub

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L3050)

( repo\_id: struse\_temp\_dir: Optional\[bool\] = Nonecommit\_message: Optional\[str\] = Noneprivate: Optional\[bool\] = Nonemax\_shard\_size: Optional\[Union\[int, str\]\] = '10GB'token: Optional\[Union\[bool, str\]\] = Noneuse\_auth\_token: Optional\[Union\[bool, str\]\] = Nonecreate\_pr: bool = False\*\*base\_model\_card\_args )

Upload the model files to the 🤗 Model Hub while synchronizing a local clone of the repo in `repo_path_or_name`.

Examples:

```
from transformers import TFAutoModel

model = TFAutoModel.from_pretrained("bert-base-cased")


model.push_to_hub("my-finetuned-bert")


model.push_to_hub("huggingface/my-finetuned-bert")
```

#### can\_generate

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L1302)

( ) → `bool`

Whether this model can generate sequences with `.generate()`.

Returns whether this model can generate sequences with `.generate()`.

#### compile

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L1497)

( optimizer = 'rmsprop'loss = 'auto\_with\_warning'metrics = Noneloss\_weights = Noneweighted\_metrics = Nonerun\_eagerly = Nonesteps\_per\_execution = None\*\*kwargs )

This is a thin wrapper that sets the model’s loss output head as the loss if the user does not specify a loss function themselves.

#### create\_model\_card

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L1792)

( output\_dirmodel\_name: strlanguage: Optional\[str\] = Nonelicense: Optional\[str\] = Nonetags: Optional\[str\] = Nonefinetuned\_from: Optional\[str\] = Nonetasks: Optional\[str\] = Nonedataset\_tags: Optional\[Union\[str, List\[str\]\]\] = Nonedataset: Optional\[Union\[str, List\[str\]\]\] = Nonedataset\_args: Optional\[Union\[str, List\[str\]\]\] = None )

Creates a draft of a model card using the information available to the `Trainer`.

#### eager\_serving

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L1214)

( inputs )

Parameters

-   **inputs** (`Dict[str, tf.Tensor]`) — The input of the saved model as a dictionary of tensors.

Method used for serving the model. This method is deprecated, and will be removed.

#### from\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L2499)

( pretrained\_model\_name\_or\_path: Optional\[Union\[str, os.PathLike\]\]\*model\_argsconfig: Optional\[Union\[PretrainedConfig, str, os.PathLike\]\] = Nonecache\_dir: Optional\[Union\[str, os.PathLike\]\] = Noneignore\_mismatched\_sizes: bool = Falseforce\_download: bool = Falselocal\_files\_only: bool = Falsetoken: Optional\[Union\[str, bool\]\] = Nonerevision: str = 'main'\*\*kwargs )

Instantiate a pretrained TF 2.0 model from a pre-trained model configuration.

The warning _Weights from XXX not initialized from pretrained model_ means that the weights of XXX do not come pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning task.

The warning _Weights from XXX not used in YYY_ means that the layer XXX is not used by YYY, therefore those weights are discarded.

Examples:

```
>>> from transformers import BertConfig, TFBertModel

>>> 
>>> model = TFBertModel.from_pretrained("bert-base-uncased")
>>> 
>>> model = TFBertModel.from_pretrained("./test/saved_model/")
>>> 
>>> model = TFBertModel.from_pretrained("bert-base-uncased", output_attentions=True)
>>> assert model.config.output_attentions == True
>>> 
>>> config = BertConfig.from_json_file("./pt_model/my_pt_model_config.json")
>>> model = TFBertModel.from_pretrained("./pt_model/my_pytorch_model.bin", from_pt=True, config=config)
```

#### get\_bias

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L1932)

( ) → `tf.Variable`

The weights representing the bias, None if not an LM model.

Dict of bias attached to an LM head. The key represents the name of the bias attribute.

#### get\_head\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L1169)

( head\_mask: tf.Tensor | Nonenum\_hidden\_layers: int )

Parameters

-   **head\_mask** (`tf.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, _optional_) — The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
-   **num\_hidden\_layers** (`int`) — The number of hidden layers in the model.

Prepare the head mask if needed.

#### get\_input\_embeddings

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L1316)

( ) → `tf.Variable`

The embeddings layer mapping vocabulary to hidden states.

Returns the model’s input embeddings layer.

#### get\_lm\_head

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L1965)

( ) → `tf.keras.layers.Layer`

Returns

`tf.keras.layers.Layer`

The LM head layer if the model has one, None if not.

The LM Head layer. This method must be overwritten by all the models that have a lm head.

#### get\_output\_embeddings

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L1872)

( ) → `tf.Variable`

The new weights mapping vocabulary to hidden states.

Returns the model’s output embeddings

#### get\_output\_layer\_with\_bias

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L1909)

( ) → `tf.keras.layers.Layer`

Returns

`tf.keras.layers.Layer`

The layer that handles the bias, None if not an LM model.

Get the layer that handles a bias attribute in case the model has an LM head with weights tied to the embeddings

#### get\_prefix\_bias\_name

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L1922)

( ) → `str`

The \_prefix name of the bias.

Get the concatenated \_prefix name of the bias from the model name to the parent layer

#### load\_repo\_checkpoint

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L1343)

( repo\_path\_or\_name ) → `dict`

Parameters

-   **repo\_path\_or\_name** (`str`) — Can either be a repository name for your {object} in the Hub or a path to a local folder (in which case the repository will have the name of that local folder).

A dictionary of extra metadata from the checkpoint, most commonly an “epoch” count.

Loads a saved checkpoint (model weights and optimizer state) from a repo. Returns the current epoch count when the checkpoint was made.

#### prepare\_tf\_dataset

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L1392)

( dataset: 'datasets.Dataset'batch\_size: int = 8shuffle: bool = Truetokenizer: Optional\['PreTrainedTokenizerBase'\] = Nonecollate\_fn: Optional\[Callable\] = Nonecollate\_fn\_args: Optional\[Dict\[str, Any\]\] = Nonedrop\_remainder: Optional\[bool\] = Noneprefetch: bool = True ) → `Dataset`

Wraps a HuggingFace [Dataset](https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/main_classes#datasets.Dataset) as a `tf.data.Dataset` with collation and batching. This method is designed to create a “ready-to-use” dataset that can be passed directly to Keras methods like `fit()` without further modification. The method will drop columns from the dataset if they don’t match input names for the model. If you want to specify the column names to return rather than using the names that match this model, we recommend using `Dataset.to_tf_dataset()` instead.

#### prune\_heads

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L2312)

( heads\_to\_prune )

Parameters

-   **heads\_to\_prune** (`Dict[int, List[int]]`) — Dictionary with keys being selected layer indices (`int`) and associated values being the list of heads to prune in said layer (list of `int`). For instance {1: \[0, 2\], 2: \[2, 3\]} will prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.

Prunes heads of the base model.

#### register\_for\_auto\_class

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L3158)

( auto\_class = 'TFAutoModel' )

Parameters

-   **auto\_class** (`str` or `type`, _optional_, defaults to `"TFAutoModel"`) — The auto class to register this new model with.

Register this class with a given auto class. This should only be used for custom models as the ones in the library are already mapped with an auto class.

This API is experimental and may have some slight breaking changes in the next releases.

#### resize\_token\_embeddings

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L1974)

( new\_num\_tokens: Optional\[int\] = None ) → `tf.Variable` or `tf.keras.layers.Embedding`

Parameters

-   **new\_num\_tokens** (`int`, _optional_) — The number of new tokens in the embedding matrix. Increasing the size will add newly initialized vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just returns a pointer to the input tokens without doing anything.

Returns

`tf.Variable` or `tf.keras.layers.Embedding`

Pointer to the input tokens of the model.

Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

#### save\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L2324)

( save\_directorysaved\_model = Falseversion = 1push\_to\_hub = Falsesignatures = Nonemax\_shard\_size: Union\[int, str\] = '10GB'create\_pr: bool = Falsesafe\_serialization: bool = Falsetoken: Optional\[Union\[str, bool\]\] = None\*\*kwargs )

Save a model and its configuration file to a directory, so that it can be re-loaded using the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) class method.

#### serving



( inputs )

Parameters

-   **Method** used for serving the model. Does not have a specific signature, but will be specialized as concrete —
-   **functions** when saving with `save_pretrained`. — inputs (`Dict[str, tf.Tensor]`): The input of the saved model as a dictionary of tensors.

Prepare the output of the saved model. Can be overridden if specific serving modifications are required.

#### set\_bias

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L1949)

( value )

Parameters

-   **value** (`Dict[tf.Variable]`) — All the new bias attached to an LM head.

Set all the bias in the LM head.

#### set\_input\_embeddings

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L1852)

( value )

Parameters

-   **value** (`tf.Variable`) — The new weights mapping hidden states to vocabulary.

Set model’s input embeddings

#### set\_output\_embeddings

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L1892)

( value )

Parameters

-   **value** (`tf.Variable`) — The new weights mapping hidden states to vocabulary.

Set model’s output embeddings

A modification of Keras’s default `train_step` that correctly handles matching outputs to labels for our models and supports directly training on the loss output head. In addition, it ensures input keys are copied to the labels where appropriate. It will also copy label keys into the input dict when using the dummy loss, to ensure that they are available to the model during the forward pass.

A modification of Keras’s default `train_step` that correctly handles matching outputs to labels for our models and supports directly training on the loss output head. In addition, it ensures input keys are copied to the labels where appropriate. It will also copy label keys into the input dict when using the dummy loss, to ensure that they are available to the model during the forward pass.

## TFModelUtilsMixin

### class transformers.modeling\_tf\_utils.TFModelUtilsMixin

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L105)

( )

A few utilities for `tf.keras.Model`, to be used as a mixin.

#### num\_parameters

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L110)

( only\_trainable: bool = False ) → `int`

Parameters

-   **only\_trainable** (`bool`, _optional_, defaults to `False`) — Whether or not to return only the number of trainable parameters

The number of parameters.

Get the number of (optionally, trainable) parameters in the model.

## FlaxPreTrainedModel

### class transformers.FlaxPreTrainedModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_flax_utils.py#L158)

( config: PretrainedConfigmodule: Moduleinput\_shape: typing.Tuple = (1, 1)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True )

Base class for all models.

[FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel) takes care of storing the configuration of the models and handles methods for loading, downloading and saving models.

Class attributes (overridden by derived classes):

-   **config\_class** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — A subclass of [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) to use as configuration class for this model architecture.
-   **base\_model\_prefix** (`str`) — A string indicating the attribute associated to the base model in derived classes of the same architecture adding modules on top of the base model.
-   **main\_input\_name** (`str`) — The name of the principal input to the model (often `input_ids` for NLP models, `pixel_values` for vision models and `input_values` for speech models).

#### push\_to\_hub

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/hub.py#L786)

( repo\_id: struse\_temp\_dir: typing.Optional\[bool\] = Nonecommit\_message: typing.Optional\[str\] = Noneprivate: typing.Optional\[bool\] = Nonetoken: typing.Union\[bool, str, NoneType\] = Nonemax\_shard\_size: typing.Union\[int, str, NoneType\] = '10GB'create\_pr: bool = Falsesafe\_serialization: bool = Falserevision: str = None\*\*deprecated\_kwargs )

Upload the model checkpoint to the 🤗 Model Hub.

Examples:

```
from transformers import FlaxAutoModel

model = FlaxAutoModel.from_pretrained("bert-base-cased")


model.push_to_hub("my-finetuned-bert")


model.push_to_hub("huggingface/my-finetuned-bert")
```

Returns whether this model can generate sequences with `.generate()`. Returns: `bool`: Whether this model can generate sequences with `.generate()`.

#### from\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_flax_utils.py#L484)

( pretrained\_model\_name\_or\_path: typing.Union\[str, os.PathLike\]dtype: dtype = <class 'jax.numpy.float32'>\*model\_argsconfig: typing.Union\[transformers.configuration\_utils.PretrainedConfig, str, os.PathLike, NoneType\] = Nonecache\_dir: typing.Union\[str, os.PathLike, NoneType\] = Noneignore\_mismatched\_sizes: bool = Falseforce\_download: bool = Falselocal\_files\_only: bool = Falsetoken: typing.Union\[bool, str, NoneType\] = Nonerevision: str = 'main'\*\*kwargs )

Instantiate a pretrained flax model from a pre-trained model configuration.

The warning _Weights from XXX not initialized from pretrained model_ means that the weights of XXX do not come pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning task.

The warning _Weights from XXX not used in YYY_ means that the layer XXX is not used by YYY, therefore those weights are discarded.

Examples:

```
>>> from transformers import BertConfig, FlaxBertModel

>>> 
>>> model = FlaxBertModel.from_pretrained("bert-base-cased")
>>> 
>>> model = FlaxBertModel.from_pretrained("./test/saved_model/")
>>> 
>>> config = BertConfig.from_json_file("./pt_model/config.json")
>>> model = FlaxBertModel.from_pretrained("./pt_model/pytorch_model.bin", from_pt=True, config=config)
```

#### load\_flax\_sharded\_weights

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_flax_utils.py#L425)

( shard\_files ) → `Dict`

Parameters

-   **shard\_files** (`List[str]` — The list of shard files to load.

A nested dictionary of the model parameters, in the expected format for flax models : `{'model': {'params': {'...'}}}`.

This is the same as `flax.serialization.from_bytes` (https:lax.readthedocs.io/en/latest/\_modules/flax/serialization.html#from\_bytes) but for a sharded checkpoint.

This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being loaded in the model.

#### register\_for\_auto\_class

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_flax_utils.py#L1152)

( auto\_class = 'FlaxAutoModel' )

Parameters

-   **auto\_class** (`str` or `type`, _optional_, defaults to `"FlaxAutoModel"`) — The auto class to register this new model with.

Register this class with a given auto class. This should only be used for custom models as the ones in the library are already mapped with an auto class.

This API is experimental and may have some slight breaking changes in the next releases.

#### save\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_flax_utils.py#L1025)

( save\_directory: typing.Union\[str, os.PathLike\]params = Nonepush\_to\_hub = Falsemax\_shard\_size = '10GB'token: typing.Union\[bool, str, NoneType\] = None\*\*kwargs )

Save a model and its configuration file to a directory, so that it can be re-loaded using the `[from_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained)` class method

#### to\_bf16

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_flax_utils.py#L320)

( params: typing.Union\[typing.Dict, flax.core.frozen\_dict.FrozenDict\]mask: typing.Any = None )

Parameters

-   **params** (`Union[Dict, FrozenDict]`) — A `PyTree` of model parameters.
-   **mask** (`Union[Dict, FrozenDict]`) — A `PyTree` with same structure as the `params` tree. The leaves should be booleans, `True` for params you want to cast, and should be `False` for those you want to skip.

Cast the floating-point `params` to `jax.numpy.bfloat16`. This returns a new `params` tree and does not cast the `params` in place.

This method can be used on TPU to explicitly convert the model parameters to bfloat16 precision to do full half-precision training or to save weights in bfloat16 for inference in order to save memory and improve speed.

Examples:

```
>>> from transformers import FlaxBertModel

>>> 
>>> model = FlaxBertModel.from_pretrained("bert-base-cased")
>>> 
>>> model.params = model.to_bf16(model.params)
>>> 
>>> 
>>> from flax import traverse_util

>>> model = FlaxBertModel.from_pretrained("bert-base-cased")
>>> flat_params = traverse_util.flatten_dict(model.params)
>>> mask = {
...     path: (path[-2] != ("LayerNorm", "bias") and path[-2:] != ("LayerNorm", "scale"))
...     for path in flat_params
... }
>>> mask = traverse_util.unflatten_dict(mask)
>>> model.params = model.to_bf16(model.params, mask)
```

#### to\_fp16

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_flax_utils.py#L386)

( params: typing.Union\[typing.Dict, flax.core.frozen\_dict.FrozenDict\]mask: typing.Any = None )

Parameters

-   **params** (`Union[Dict, FrozenDict]`) — A `PyTree` of model parameters.
-   **mask** (`Union[Dict, FrozenDict]`) — A `PyTree` with same structure as the `params` tree. The leaves should be booleans, `True` for params you want to cast, and should be `False` for those you want to skip

Cast the floating-point `parmas` to `jax.numpy.float16`. This returns a new `params` tree and does not cast the `params` in place.

This method can be used on GPU to explicitly convert the model parameters to float16 precision to do full half-precision training or to save weights in float16 for inference in order to save memory and improve speed.

Examples:

```
>>> from transformers import FlaxBertModel

>>> 
>>> model = FlaxBertModel.from_pretrained("bert-base-cased")
>>> 
>>> model.params = model.to_fp16(model.params)
>>> 
>>> 
>>> from flax import traverse_util

>>> model = FlaxBertModel.from_pretrained("bert-base-cased")
>>> flat_params = traverse_util.flatten_dict(model.params)
>>> mask = {
...     path: (path[-2] != ("LayerNorm", "bias") and path[-2:] != ("LayerNorm", "scale"))
...     for path in flat_params
... }
>>> mask = traverse_util.unflatten_dict(mask)
>>> model.params = model.to_fp16(model.params, mask)
```

#### to\_fp32

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_flax_utils.py#L359)

( params: typing.Union\[typing.Dict, flax.core.frozen\_dict.FrozenDict\]mask: typing.Any = None )

Parameters

-   **params** (`Union[Dict, FrozenDict]`) — A `PyTree` of model parameters.
-   **mask** (`Union[Dict, FrozenDict]`) — A `PyTree` with same structure as the `params` tree. The leaves should be booleans, `True` for params you want to cast, and should be `False` for those you want to skip

Cast the floating-point `parmas` to `jax.numpy.float32`. This method can be used to explicitly convert the model parameters to fp32 precision. This returns a new `params` tree and does not cast the `params` in place.

Examples:

```
>>> from transformers import FlaxBertModel

>>> 
>>> model = FlaxBertModel.from_pretrained("bert-base-cased")
>>> 
>>> 
>>> model.params = model.to_f16(model.params)
>>> 
>>> model.params = model.to_fp32(model.params)
```

## Pushing to the Hub

### class transformers.utils.PushToHubMixin

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/hub.py#L672)

( )

A Mixin containing the functionality to push a model or tokenizer to the hub.

#### push\_to\_hub

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/hub.py#L786)

( repo\_id: struse\_temp\_dir: typing.Optional\[bool\] = Nonecommit\_message: typing.Optional\[str\] = Noneprivate: typing.Optional\[bool\] = Nonetoken: typing.Union\[bool, str, NoneType\] = Nonemax\_shard\_size: typing.Union\[int, str, NoneType\] = '10GB'create\_pr: bool = Falsesafe\_serialization: bool = Falserevision: str = None\*\*deprecated\_kwargs )

Upload the {object\_files} to the 🤗 Model Hub.

Examples:

```
from transformers import {object_class}

{object} = {object_class}.from_pretrained("bert-base-cased")


{object}.push_to_hub("my-finetuned-bert")


{object}.push_to_hub("huggingface/my-finetuned-bert")
```

## Sharded checkpoints

#### transformers.modeling\_utils.load\_sharded\_checkpoint

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L373)

( modelfolderstrict = Trueprefer\_safe = True ) → `NamedTuple`

Parameters

-   **model** (`torch.nn.Module`) — The model in which to load the checkpoint.
-   **folder** (`str` or `os.PathLike`) — A path to a folder containing the sharded checkpoint.
-   **strict** (`bool`, \*optional`, defaults to` True\`) — Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.
-   **prefer\_safe** (`bool`, _optional_, defaults to `False`) — If both safetensors and PyTorch save files are present in checkpoint and `prefer_safe` is True, the safetensors files will be loaded. Otherwise, PyTorch files are always loaded when possible.

A named tuple with `missing_keys` and `unexpected_keys` fields

-   `missing_keys` is a list of str containing the missing keys
-   `unexpected_keys` is a list of str containing the unexpected keys

This is the same as [`torch.nn.Module.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict) but for a sharded checkpoint.

This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being loaded in the model.