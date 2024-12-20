# Pix2Struct

## Overview

The Pix2Struct model was proposed in [Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding](https://arxiv.org/abs/2210.03347) by Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu, Julian Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, Kristina Toutanova.

The abstract from the paper is the following:

> Visually-situated language is ubiquitous — sources range from textbooks with diagrams to web pages with images and tables, to mobile apps with buttons and forms. Perhaps due to this diversity, previous work has typically relied on domain-specific recipes with limited sharing of the underlying data, model architectures, and objectives. We present Pix2Struct, a pretrained image-to-text model for purely visual language understanding, which can be finetuned on tasks containing visually-situated language. Pix2Struct is pretrained by learning to parse masked screenshots of web pages into simplified HTML. The web, with its richness of visual elements cleanly reflected in the HTML structure, provides a large source of pretraining data well suited to the diversity of downstream tasks. Intuitively, this objective subsumes common pretraining signals such as OCR, language modeling, image captioning. In addition to the novel pretraining strategy, we introduce a variable-resolution input representation and a more flexible integration of language and vision inputs, where language prompts such as questions are rendered directly on top of the input image. For the first time, we show that a single pretrained model can achieve state-of-the-art results in six out of nine tasks across four domains: documents, illustrations, user interfaces, and natural images.

Tips:

Pix2Struct has been fine tuned on a variety of tasks and datasets, ranging from image captioning, visual question answering (VQA) over different inputs (books, charts, science diagrams), captioning UI components etc. The full list can be found in Table 1 of the paper. We therefore advise you to use these models for the tasks they have been fine tuned on. For instance, if you want to use Pix2Struct for UI captioning, you should use the model fine tuned on the UI dataset. If you want to use Pix2Struct for image captioning, you should use the model fine tuned on the natural images captioning dataset and so on.

If you want to use the model to perform conditional text captioning, make sure to use the processor with `add_special_tokens=False`.

This model was contributed by [ybelkada](https://huggingface.co/ybelkada). The original code can be found [here](https://github.com/google-research/pix2struct).

## Resources

-   [Fine-tuning Notebook](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_pix2struct.ipynb)
-   [All models](https://huggingface.co/models?search=pix2struct)

## Pix2StructConfig

### class transformers.Pix2StructConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pix2struct/configuration_pix2struct.py#L291)

( text\_config = Nonevision\_config = Noneinitializer\_factor = 1.0initializer\_range = 0.02is\_vqa = Falsetie\_word\_embeddings = Falseis\_encoder\_decoder = True\*\*kwargs )

Parameters

-   **text\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [Pix2StructTextConfig](/docs/transformers/v4.34.0/en/model_doc/pix2struct#transformers.Pix2StructTextConfig).
-   **vision\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [Pix2StructVisionConfig](/docs/transformers/v4.34.0/en/model_doc/pix2struct#transformers.Pix2StructVisionConfig).
-   **initializer\_factor** (`float`, _optional_, defaults to 1.0) — Factor to multiply the initialization range with.
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **is\_vqa** (`bool`, _optional_, defaults to `False`) — Whether the model has been fine-tuned for VQA or not.
-   **kwargs** (_optional_) — Dictionary of keyword arguments.

[Pix2StructConfig](/docs/transformers/v4.34.0/en/model_doc/pix2struct#transformers.Pix2StructConfig) is the configuration class to store the configuration of a [Pix2StructForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/pix2struct#transformers.Pix2StructForConditionalGeneration). It is used to instantiate a Pix2Struct model according to the specified arguments, defining the text model and vision model configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the Pix2Struct-base [google/pix2struct-base](https://huggingface.co/google/pix2struct-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import Pix2StructConfig, Pix2StructForConditionalGeneration

>>> 
>>> configuration = Pix2StructConfig()

>>> 
>>> model = Pix2StructForConditionalGeneration(configuration)

>>> 
>>> configuration = model.config

>>> 

>>> 
>>> config_text = Pix2StructTextConfig()
>>> config_vision = Pix2StructVisionConfig()

>>> config = Pix2StructConfig.from_text_vision_configs(config_text, config_vision)
```

#### from\_text\_vision\_configs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pix2struct/configuration_pix2struct.py#L377)

( text\_config: Pix2StructTextConfigvision\_config: Pix2StructVisionConfig\*\*kwargs ) → [Pix2StructConfig](/docs/transformers/v4.34.0/en/model_doc/pix2struct#transformers.Pix2StructConfig)

An instance of a configuration object

Instantiate a [Pix2StructConfig](/docs/transformers/v4.34.0/en/model_doc/pix2struct#transformers.Pix2StructConfig) (or a derived class) from pix2struct text model configuration and pix2struct vision model configuration.

## Pix2StructTextConfig

### class transformers.Pix2StructTextConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pix2struct/configuration_pix2struct.py#L33)

( vocab\_size = 50244hidden\_size = 768d\_kv = 64d\_ff = 2048num\_layers = 12num\_heads = 12relative\_attention\_num\_buckets = 32relative\_attention\_max\_distance = 128dropout\_rate = 0.1layer\_norm\_epsilon = 1e-06initializer\_factor = 1.0dense\_act\_fn = 'gelu\_new'decoder\_start\_token\_id = 0use\_cache = Falsepad\_token\_id = 0eos\_token\_id = 1tie\_word\_embeddings = Falseis\_decoder = True\*\*kwargs )

This is the configuration class to store the configuration of a [Pix2StructTextModel](/docs/transformers/v4.34.0/en/model_doc/pix2struct#transformers.Pix2StructTextModel). It is used to instantiate a Pix2Struct text model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Pix2Struct text decoder used by the [google/pix2struct-base](https://huggingface.co/google/pix2struct-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import Pix2StructTextConfig, Pix2StructTextModel

>>> 
>>> configuration = Pix2StructTextConfig()

>>> 
>>> model = Pix2StructTextModel(configuration)

>>> 
>>> configuration = model.config
```

## Pix2StructVisionConfig

### class transformers.Pix2StructVisionConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pix2struct/configuration_pix2struct.py#L172)

( hidden\_size = 768patch\_embed\_hidden\_size = 768d\_ff = 2048d\_kv = 64num\_hidden\_layers = 12num\_attention\_heads = 12dense\_act\_fn = 'gelu\_new'layer\_norm\_eps = 1e-06dropout\_rate = 0.0attention\_dropout = 0.0initializer\_range = 1e-10initializer\_factor = 1.0seq\_len = 4096relative\_attention\_num\_buckets = 32relative\_attention\_max\_distance = 128\*\*kwargs )

This is the configuration class to store the configuration of a [Pix2StructVisionModel](/docs/transformers/v4.34.0/en/model_doc/pix2struct#transformers.Pix2StructVisionModel). It is used to instantiate a Pix2Struct vision model according to the specified arguments, defining the model architecture. Instantiating a configuration defaults will yield a similar configuration to that of the Pix2Struct-base [google/pix2struct-base](https://huggingface.co/google/pix2struct-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import Pix2StructVisionConfig, Pix2StructVisionModel

>>> 
>>> configuration = Pix2StructVisionConfig()

>>> 
>>> model = Pix2StructVisionModel(configuration)

>>> 
>>> configuration = model.config
```

## Pix2StructProcessor

### class transformers.Pix2StructProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pix2struct/processing_pix2struct.py#L26)

( image\_processortokenizer )

Parameters

-   **image\_processor** (`Pix2StructImageProcessor`) — An instance of [Pix2StructImageProcessor](/docs/transformers/v4.34.0/en/model_doc/pix2struct#transformers.Pix2StructImageProcessor). The image processor is a required input.
-   **tokenizer** (Union\[`T5TokenizerFast`, `T5Tokenizer`\]) — An instance of \[‘T5TokenizerFast\`\] or \[‘T5Tokenizer\`\]. The tokenizer is a required input.

Constructs a PIX2STRUCT processor which wraps a BERT tokenizer and PIX2STRUCT image processor into a single processor.

[Pix2StructProcessor](/docs/transformers/v4.34.0/en/model_doc/pix2struct#transformers.Pix2StructProcessor) offers all the functionalities of [Pix2StructImageProcessor](/docs/transformers/v4.34.0/en/model_doc/pix2struct#transformers.Pix2StructImageProcessor) and [T5TokenizerFast](/docs/transformers/v4.34.0/en/model_doc/mt5#transformers.T5TokenizerFast). See the docstring of `__call__()` and [decode()](/docs/transformers/v4.34.0/en/model_doc/pix2struct#transformers.Pix2StructProcessor.decode) for more information.

This method forwards all its arguments to Pix2StructTokenizerFast’s [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.batch_decode). Please refer to the docstring of this method for more information.

This method forwards all its arguments to Pix2StructTokenizerFast’s [decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.decode). Please refer to the docstring of this method for more information.

## Pix2StructImageProcessor

### class transformers.Pix2StructImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pix2struct/image_processing_pix2struct.py#L202)

( do\_convert\_rgb: bool = Truedo\_normalize: bool = Truepatch\_size: typing.Dict\[str, int\] = Nonemax\_patches: int = 2048is\_vqa: bool = False\*\*kwargs )

Parameters

-   **do\_convert\_rgb** (`bool`, _optional_, defaults to `True`) — Whether to convert the image to RGB.
-   **do\_normalize** (`bool`, _optional_, defaults to `True`) — Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess` method. According to Pix2Struct paper and code, the image is normalized with its own mean and standard deviation.
-   **patch\_size** (`Dict[str, int]`, _optional_, defaults to `{"height" -- 16, "width": 16}`): The patch size to use for the image. According to Pix2Struct paper and code, the patch size is 16x16.
-   **max\_patches** (`int`, _optional_, defaults to 2048) — The maximum number of patches to extract from the image as per the [Pix2Struct paper](https://arxiv.org/pdf/2210.03347.pdf).
-   **is\_vqa** (`bool`, _optional_, defaults to `False`) — Whether or not the image processor is for the VQA task. If `True` and `header_text` is passed in, text is rendered onto the input images.

Constructs a Pix2Struct image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pix2struct/image_processing_pix2struct.py#L362)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]header\_text: typing.Optional\[str\] = Nonedo\_convert\_rgb: bool = Nonedo\_normalize: typing.Optional\[bool\] = Nonemax\_patches: typing.Optional\[int\] = Nonepatch\_size: typing.Union\[typing.Dict\[str, int\], NoneType\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs )

Preprocess an image or batch of images. The processor first computes the maximum possible number of aspect-ratio preserving patches of size `patch_size` that can be extracted from the image. It then pads the image with zeros to make the image respect the constraint of `max_patches`. Before extracting the patches the images are standardized following the tensorflow implementation of `per_image_standardization` ([https://www.tensorflow.org/api\_docs/python/tf/image/per\_image\_standardization](https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization)).

## Pix2StructTextModel

### class transformers.Pix2StructTextModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pix2struct/modeling_pix2struct.py#L1317)

( config )

Parameters

-   **config** (Union\[`Pix2StructConfig`, `Pix2StructTextConfig`\]) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The standalone text decoder of Pix2Struct

The Pix2Struct model was proposed in [Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding](https://arxiv.org/abs/2210.03347) by Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu, Julian Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, Kristina Toutanova. It’s an encoder decoder transformer pre-trained in a image-to-text setting.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pix2struct/modeling_pix2struct.py#L1386)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Nonereturn\_dict: typing.Optional\[bool\] = None\*\*kwargs ) → [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

The [Pix2StructTextModel](/docs/transformers/v4.34.0/en/model_doc/pix2struct#transformers.Pix2StructTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoProcessor, Pix2StructTextModel

>>> processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
>>> model = Pix2StructTextModel.from_pretrained("google/pix2struct-textcaps-base")

>>> inputs = processor(text="Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)
>>> loss = outputs.loss
```

## Pix2StructVisionModel

### class transformers.Pix2StructVisionModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pix2struct/modeling_pix2struct.py#L548)

( config: Pix2StructConfig )

Parameters

-   **config** ([Pix2StructConfig](/docs/transformers/v4.34.0/en/model_doc/pix2struct#transformers.Pix2StructConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Pix2StructVision Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pix2struct/modeling_pix2struct.py#L581)

( flattened\_patches: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

The [Pix2StructVisionModel](/docs/transformers/v4.34.0/en/model_doc/pix2struct#transformers.Pix2StructVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import requests
>>> from PIL import Image
>>> from transformers import AutoProcessor, Pix2StructVisionModel

>>> image_processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
>>> model = Pix2StructVisionModel.from_pretrained("google/pix2struct-textcaps-base")

>>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = image_processor(images=image, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 2048, 768]
```

## Pix2StructForConditionalGeneration

### class transformers.Pix2StructForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pix2struct/modeling_pix2struct.py#L1598)

( config: Pix2StructConfig )

Parameters

-   **config** (Union\[`Pix2StructConfig`, `Pix2StructTextConfig`\]) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

A conditional generation model with a language modeling head. Can be used for sequence generation tasks.

The Pix2Struct model was proposed in [Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding](https://arxiv.org/abs/2210.03347) by Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu, Julian Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, Kristina Toutanova. It’s an encoder decoder transformer pre-trained in a image-to-text setting.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pix2struct/modeling_pix2struct.py#L1640)

( flattened\_patches: typing.Optional\[torch.FloatTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.BoolTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.FloatTensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

The [Pix2StructForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/pix2struct#transformers.Pix2StructForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

Inference:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Pix2StructForConditionalGeneration

>>> processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
>>> model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-base")

>>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> 
>>> generated_ids = model.generate(**inputs, max_new_tokens=50)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print(generated_text)
A stop sign is on a street corner.

>>> 
>>> text = "A picture of"
>>> inputs = processor(text=text, images=image, return_tensors="pt", add_special_tokens=False)

>>> generated_ids = model.generate(**inputs, max_new_tokens=50)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print(generated_text)
A picture of a stop sign with a red stop sign
```

Training:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Pix2StructForConditionalGeneration

>>> processor = AutoProcessor.from_pretrained("google/pix2struct-base")
>>> model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-base")

>>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> text = "A stop sign is on the street corner."

>>> inputs = processor(images=image, return_tensors="pt")
>>> labels = processor(text=text, return_tensors="pt").input_ids

>>> 
>>> outputs = model(**inputs, labels=labels)
>>> loss = outputs.loss
>>> print(f"{loss.item():.5f}")
5.94282
```