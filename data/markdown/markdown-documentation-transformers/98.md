# Bark

## Overview

Bark is a transformer-based text-to-speech model proposed by Suno AI in [suno-ai/bark](https://github.com/suno-ai/bark).

Bark is made of 4 main models:

-   [BarkSemanticModel](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkSemanticModel) (also referred to as the ‘text’ model): a causal auto-regressive transformer model that takes as input tokenized text, and predicts semantic text tokens that capture the meaning of the text.
-   [BarkCoarseModel](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkCoarseModel) (also referred to as the ‘coarse acoustics’ model): a causal autoregressive transformer, that takes as input the results of the [BarkSemanticModel](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkSemanticModel) model. It aims at predicting the first two audio codebooks necessary for EnCodec.
-   [BarkFineModel](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkFineModel) (the ‘fine acoustics’ model), this time a non-causal autoencoder transformer, which iteratively predicts the last codebooks based on the sum of the previous codebooks embeddings.
-   having predicted all the codebook channels from the [EncodecModel](/docs/transformers/v4.34.0/en/model_doc/encodec#transformers.EncodecModel), Bark uses it to decode the output audio array.

It should be noted that each of the first three modules can support conditional speaker embeddings to condition the output sound according to specific predefined voice.

### Optimizing Bark

Bark can be optimized with just a few extra lines of code, which **significantly reduces its memory footprint** and **accelerates inference**.

#### Using half-precision

You can speed up inference and reduce memory footprint by 50% simply by loading the model in half-precision.

```
from transformers import BarkModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)
```

#### Using 🤗 Better Transformer

Better Transformer is an 🤗 Optimum feature that performs kernel fusion under the hood. You can gain 20% to 30% in speed with zero performance degradation. It only requires one line of code to export the model to 🤗 Better Transformer:

```
model =  model.to_bettertransformer()
```

Note that 🤗 Optimum must be installed before using this feature. [Here’s how to install it.](https://huggingface.co/docs/optimum/installation)

#### Using CPU offload

As mentioned above, Bark is made up of 4 sub-models, which are called up sequentially during audio generation. In other words, while one sub-model is in use, the other sub-models are idle.

If you’re using a CUDA device, a simple solution to benefit from an 80% reduction in memory footprint is to offload the GPU’s submodels when they’re idle. This operation is called CPU offloading. You can use it with one line of code.

```
model.enable_cpu_offload()
```

Note that 🤗 Accelerate must be installed before using this feature. [Here’s how to install it.](https://huggingface.co/docs/accelerate/basic_tutorials/install)

#### Combining optimizaton techniques

You can combine optimization techniques, and use CPU offload, half-precision and 🤗 Better Transformer all at once.

```
from transformers import BarkModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)


model = BetterTransformer.transform(model, keep_original_model=False)


model.enable_cpu_offload()
```

Find out more on inference optimization techniques [here](https://huggingface.co/docs/transformers/perf_infer_gpu_one).

### Tips

Suno offers a library of voice presets in a number of languages [here](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c). These presets are also uploaded in the hub [here](https://huggingface.co/suno/bark-small/tree/main/speaker_embeddings) or [here](https://huggingface.co/suno/bark/tree/main/speaker_embeddings).

```
>>> from transformers import AutoProcessor, BarkModel

>>> processor = AutoProcessor.from_pretrained("suno/bark")
>>> model = BarkModel.from_pretrained("suno/bark")

>>> voice_preset = "v2/en_speaker_6"

>>> inputs = processor("Hello, my dog is cute", voice_preset=voice_preset)

>>> audio_array = model.generate(**inputs)
>>> audio_array = audio_array.cpu().numpy().squeeze()
```

Bark can generate highly realistic, **multilingual** speech as well as other audio - including music, background noise and simple sound effects.

```
>>> 
>>> inputs = processor("惊人的！我会说中文")

>>> 
>>> inputs = processor("Incroyable! Je peux générer du son.", voice_preset="fr_speaker_5")

>>> 
>>> inputs = processor("♪ Hello, my dog is cute ♪")

>>> audio_array = model.generate(**inputs)
>>> audio_array = audio_array.cpu().numpy().squeeze()
```

The model can also produce **nonverbal communications** like laughing, sighing and crying.

```
>>> 
>>> inputs = processor("Hello uh ... [clears throat], my dog is cute [laughter]")

>>> audio_array = model.generate(**inputs)
>>> audio_array = audio_array.cpu().numpy().squeeze()
```

To save the audio, simply take the sample rate from the model config and some scipy utility:

```
>>> from scipy.io.wavfile import write as write_wav

>>> 
>>> sample_rate = model.generation_config.sample_rate
>>> write_wav("bark_generation.wav", sample_rate, audio_array)
```

This model was contributed by [Yoach Lacombe (ylacombe)](https://huggingface.co/ylacombe) and [Sanchit Gandhi (sanchit-gandhi)](https://github.com/sanchit-gandhi). The original code can be found [here](https://github.com/suno-ai/bark).

## BarkConfig

### class transformers.BarkConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bark/configuration_bark.py#L219)

( semantic\_config: typing.Dict = Nonecoarse\_acoustics\_config: typing.Dict = Nonefine\_acoustics\_config: typing.Dict = Nonecodec\_config: typing.Dict = Noneinitializer\_range = 0.02\*\*kwargs )

Parameters

-   **semantic\_config** ([BarkSemanticConfig](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkSemanticConfig), _optional_) — Configuration of the underlying semantic sub-model.
-   **coarse\_acoustics\_config** ([BarkCoarseConfig](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkCoarseConfig), _optional_) — Configuration of the underlying coarse acoustics sub-model.
-   **fine\_acoustics\_config** ([BarkFineConfig](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkFineConfig), _optional_) — Configuration of the underlying fine acoustics sub-model.
-   **codec\_config** ([AutoConfig](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoConfig), _optional_) — Configuration of the underlying codec sub-model.
    
    Example —
    

This is the configuration class to store the configuration of a [BarkModel](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkModel). It is used to instantiate a Bark model according to the specified sub-models configurations, defining the model architecture.

Instantiating a configuration with the defaults will yield a similar configuration to that of the Bark [suno/bark](https://huggingface.co/suno/bark) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

#### from\_sub\_model\_configs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bark/configuration_bark.py#L309)

( semantic\_config: BarkSemanticConfigcoarse\_acoustics\_config: BarkCoarseConfigfine\_acoustics\_config: BarkFineConfigcodec\_config: PretrainedConfig\*\*kwargs ) → [BarkConfig](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkConfig)

An instance of a configuration object

Instantiate a [BarkConfig](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkConfig) (or a derived class) from bark sub-models configuration.

## BarkProcessor

### class transformers.BarkProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bark/processing_bark.py#L34)

( tokenizerspeaker\_embeddings = None )

Parameters

-   **tokenizer** ([PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)) — An instance of [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer).
-   **speaker\_embeddings** (`Dict[Dict[str]]`, _optional_, defaults to `None`) — Optional nested speaker embeddings dictionary. The first level contains voice preset names (e.g `"en_speaker_4"`). The second level contains `"semantic_prompt"`, `"coarse_prompt"` and `"fine_prompt"` embeddings. The values correspond to the path of the corresponding `np.ndarray`. See [here](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c) for a list of `voice_preset_names`.

Constructs a Bark processor which wraps a text tokenizer and optional Bark voice presets into a single processor.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bark/processing_bark.py#L218)

( text = Nonevoice\_preset = Nonereturn\_tensors = 'pt'max\_length = 256add\_special\_tokens = Falsereturn\_attention\_mask = Truereturn\_token\_type\_ids = False\*\*kwargs ) → Tuple([BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding), [BatchFeature](/docs/transformers/v4.34.0/en/main_classes/image_processor#transformers.BatchFeature))

Main method to prepare for the model one or several sequences(s). This method forwards the `text` and `kwargs` arguments to the AutoTokenizer’s `__call__()` to encode the text. The method also proposes a voice preset which is a dictionary of arrays that conditions `Bark`’s output. `kwargs` arguments are forwarded to the tokenizer and to `cached_file` method if `voice_preset` is a valid filename.

#### from\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bark/processing_bark.py#L63)

( pretrained\_processor\_name\_or\_pathspeaker\_embeddings\_dict\_path = 'speaker\_embeddings\_path.json'\*\*kwargs )

Parameters

-   **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) — This can be either:
    
    -   a string, the _model id_ of a pretrained [BarkProcessor](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkProcessor) hosted inside a model repo on huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
    -   a path to a _directory_ containing a processor saved using the [save\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkProcessor.save_pretrained) method, e.g., `./my_model_directory/`.
    
-   **speaker\_embeddings\_dict\_path** (`str`, _optional_, defaults to `"speaker_embeddings_path.json"`) — The name of the `.json` file containing the speaker\_embeddings dictionnary located in `pretrained_model_name_or_path`. If `None`, no speaker\_embeddings is loaded. \*\*kwargs — Additional keyword arguments passed along to both `~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`.

Instantiate a Bark processor associated with a pretrained model.

#### save\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bark/processing_bark.py#L117)

( save\_directoryspeaker\_embeddings\_dict\_path = 'speaker\_embeddings\_path.json'speaker\_embeddings\_directory = 'speaker\_embeddings'push\_to\_hub: bool = False\*\*kwargs )

Parameters

-   **save\_directory** (`str` or `os.PathLike`) — Directory where the tokenizer files and the speaker embeddings will be saved (directory will be created if it does not exist).
-   **speaker\_embeddings\_dict\_path** (`str`, _optional_, defaults to `"speaker_embeddings_path.json"`) — The name of the `.json` file that will contains the speaker\_embeddings nested path dictionnary, if it exists, and that will be located in `pretrained_model_name_or_path/speaker_embeddings_directory`.
-   **speaker\_embeddings\_directory** (`str`, _optional_, defaults to `"speaker_embeddings/"`) — The name of the folder in which the speaker\_embeddings arrays will be saved.
-   **push\_to\_hub** (`bool`, _optional_, defaults to `False`) — Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with `repo_id` (will default to the name of `save_directory` in your namespace). kwargs — Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/v4.34.0/en/main_classes/processors#transformers.ProcessorMixin.push_to_hub) method.

Saves the attributes of this processor (tokenizer…) in the specified directory so that it can be reloaded using the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkProcessor.from_pretrained) method.

## BarkModel

### class transformers.BarkModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bark/modeling_bark.py#L1426)

( config )

Parameters

-   **config** ([BarkConfig](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The full Bark model, a text-to-speech model composed of 4 sub-models:

-   [BarkSemanticModel](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkSemanticModel) (also referred to as the ‘text’ model): a causal auto-regressive transformer model that takes as input tokenized text, and predicts semantic text tokens that capture the meaning of the text.
-   [BarkCoarseModel](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkCoarseModel) (also refered to as the ‘coarse acoustics’ model), also a causal autoregressive transformer, that takes into input the results of the last model. It aims at regressing the first two audio codebooks necessary to `encodec`.
-   [BarkFineModel](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkFineModel) (the ‘fine acoustics’ model), this time a non-causal autoencoder transformer, which iteratively predicts the last codebooks based on the sum of the previous codebooks embeddings.
-   having predicted all the codebook channels from the [EncodecModel](/docs/transformers/v4.34.0/en/model_doc/encodec#transformers.EncodecModel), Bark uses it to decode the output audio array.

It should be noted that each of the first three modules can support conditional speaker embeddings to condition the output sound according to specific predefined voice.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### generate

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bark/modeling_bark.py#L1507)

( input\_ids: typing.Optional\[torch.Tensor\] = Nonehistory\_prompt: typing.Union\[typing.Dict\[str, torch.Tensor\], NoneType\] = None\*\*kwargs ) → torch.LongTensor

Generates audio from an input prompt and an additional optional `Bark` speaker prompt.

Example:

```
>>> from transformers import AutoProcessor, BarkModel

>>> processor = AutoProcessor.from_pretrained("suno/bark-small")
>>> model = BarkModel.from_pretrained("suno/bark-small")

>>> 
>>> voice_preset = "v2/en_speaker_6"

>>> inputs = processor("Hello, my dog is cute, I need him in my life", voice_preset=voice_preset)

>>> audio_array = model.generate(**inputs, semantic_max_new_tokens=100)
>>> audio_array = audio_array.cpu().numpy().squeeze()
```

#### enable\_cpu\_offload

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bark/modeling_bark.py#L1458)

( gpu\_id: typing.Optional\[int\] = 0 )

Parameters

-   **gpu\_id** (`int`, _optional_, defaults to 0) — GPU id on which the sub-models will be loaded and offloaded.

Offloads all sub-models to CPU using accelerate, reducing memory usage with a low impact on performance. This method moves one whole sub-model at a time to the GPU when it is used, and the sub-model remains in GPU until the next sub-model runs.

## BarkSemanticModel

### class transformers.BarkSemanticModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bark/modeling_bark.py#L714)

( config )

Parameters

-   **config** ([BarkSemanticConfig](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkSemanticConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Bark semantic (or text) model. It shares the same architecture as the coarse model. It is a GPT-2 like autoregressive model with a language modeling head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bark/modeling_bark.py#L531)

( input\_ids: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneinput\_embeds: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None )

The [BarkCausalModel](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkCausalModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## BarkCoarseModel

### class transformers.BarkCoarseModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bark/modeling_bark.py#L819)

( config )

Parameters

-   **config** ([BarkCoarseConfig](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkCoarseConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Bark coarse acoustics model. It shares the same architecture as the semantic (or text) model. It is a GPT-2 like autoregressive model with a language modeling head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bark/modeling_bark.py#L531)

( input\_ids: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneinput\_embeds: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None )

The [BarkCausalModel](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkCausalModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## BarkFineModel

### class transformers.BarkFineModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bark/modeling_bark.py#L1029)

( config )

Parameters

-   **config** ([BarkFineConfig](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkFineConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Bark fine acoustics model. It is a non-causal GPT-like model with `config.n_codes_total` embedding layers and language modeling heads, one for each codebook. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bark/modeling_bark.py#L1161)

( codebook\_idx: intinput\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneinput\_embeds: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None )

The [BarkFineModel](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkFineModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## BarkCausalModel

### class transformers.BarkCausalModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bark/modeling_bark.py#L450)

( config )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bark/modeling_bark.py#L531)

( input\_ids: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneinput\_embeds: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None )

The [BarkCausalModel](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkCausalModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## BarkCoarseConfig

### class transformers.BarkCoarseConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bark/configuration_bark.py#L182)

( block\_size = 1024input\_vocab\_size = 10048output\_vocab\_size = 10048num\_layers = 12num\_heads = 12hidden\_size = 768dropout = 0.0bias = Trueinitializer\_range = 0.02use\_cache = True\*\*kwargs )

This is the configuration class to store the configuration of a [BarkCoarseModel](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkCoarseModel). It is used to instantiate the model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Bark [suno/bark](https://huggingface.co/suno/bark) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import BarkCoarseConfig, BarkCoarseModel

>>> 
>>> configuration = BarkCoarseConfig()

>>> 
>>> model = BarkCoarseModel(configuration)

>>> 
>>> configuration = model.config
```

## BarkFineConfig

### class transformers.BarkFineConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bark/configuration_bark.py#L209)

( tie\_word\_embeddings = Truen\_codes\_total = 8n\_codes\_given = 1\*\*kwargs )

This is the configuration class to store the configuration of a [BarkFineModel](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkFineModel). It is used to instantiate the model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Bark [suno/bark](https://huggingface.co/suno/bark) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import BarkFineConfig, BarkFineModel

>>> 
>>> configuration = BarkFineConfig()

>>> 
>>> model = BarkFineModel(configuration)

>>> 
>>> configuration = model.config
```

## BarkSemanticConfig

### class transformers.BarkSemanticConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bark/configuration_bark.py#L160)

( block\_size = 1024input\_vocab\_size = 10048output\_vocab\_size = 10048num\_layers = 12num\_heads = 12hidden\_size = 768dropout = 0.0bias = Trueinitializer\_range = 0.02use\_cache = True\*\*kwargs )

This is the configuration class to store the configuration of a [BarkSemanticModel](/docs/transformers/v4.34.0/en/model_doc/bark#transformers.BarkSemanticModel). It is used to instantiate the model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Bark [suno/bark](https://huggingface.co/suno/bark) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import BarkSemanticConfig, BarkSemanticModel

>>> 
>>> configuration = BarkSemanticConfig()

>>> 
>>> model = BarkSemanticModel(configuration)

>>> 
>>> configuration = model.config
```