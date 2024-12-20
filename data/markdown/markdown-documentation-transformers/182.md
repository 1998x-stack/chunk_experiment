# IDEFICS

## Overview

The IDEFICS model was proposed in [OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents](https://huggingface.co/papers/2306.16527) by Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, Victor Sanh

The abstract from the paper is the following:

_Large multimodal models trained on natural documents, which interleave images and text, outperform models trained on image-text pairs on various multimodal benchmarks that require reasoning over one or multiple images to generate a text. However, the datasets used to train these models have not been released, and the collection process has not been fully specified. We introduce the OBELICS dataset, an open web-scale filtered dataset of interleaved image-text documents comprising 141 million web pages extracted from Common Crawl, 353 million associated images, and 115 billion text tokens. We describe the dataset creation process, present comprehensive filtering rules, and provide an analysis of the dataset’s content. To show the viability of OBELISC, we train an 80 billion parameters vision and language model on the dataset and obtain competitive performance on various multimodal benchmarks. We release the code to reproduce the dataset along with the dataset itself._

This model was contributed by [HuggingFaceM4](https://huggingface.co/HuggingFaceM4). The original code can be found [here](INSERT%20LINK%20TO%20GITHUB%20REPO%20HERE). (TODO: don’t have a public link yet).

Idefics modeling code in Transformers is for finetuning and inferencing the pre-trained Idefics models.

To train a new Idefics model from scratch use the m4 codebase (a link will be provided once it’s made public)

## IdeficsConfig

### class transformers.IdeficsConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/idefics/configuration_idefics.py#L159)

( vocab\_size = 32000 additional\_vocab\_size = 0 hidden\_size = 4096 intermediate\_size = 11008 num\_hidden\_layers = 32 num\_attention\_heads = 32 dropout = 0.0 hidden\_act = 'silu' initializer\_range = 0.02 alpha\_initializer = 'zeros' alphas\_initializer\_range = 0.0 alpha\_type = 'float' rms\_norm\_eps = 1e-06 use\_cache = True pad\_token\_id = 0 bos\_token\_id = 1 eos\_token\_id = 2 tie\_word\_embeddings = False cross\_layer\_interval = 1 qk\_layer\_norms = False freeze\_text\_layers = True freeze\_text\_module\_exceptions = \[\] freeze\_lm\_head = False freeze\_vision\_layers = True freeze\_vision\_module\_exceptions = \[\] use\_resampler = False vision\_config = None perceiver\_config = None \*\*kwargs )

Parameters

-   **additional\_vocab\_size** (`int`, \*optional\`, defaults to 0) — Additional vocabulary size of the model, typically for the special ”” token. Additional vocab tokens are always trainable whereas regular vocab tokens can be frozen or not.
-   **vocab\_size** (`int`, _optional_, defaults to 32000) — Vocabulary size of the Idefics model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [~IdeficsModel](/docs/transformers/v4.34.0/en/model_doc/idefics#transformers.IdeficsModel)
-   **hidden\_size** (`int`, _optional_, defaults to 4096) — Dimension of the hidden representations.
-   **intermediate\_size** (`int`, _optional_, defaults to 11008) — Dimension of the MLP representations.
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 32) — Number of hidden layers in the Transformer encoder.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 32) — Number of attention heads for each attention layer in the Transformer encoder.
-   **dropout** (`float`, _optional_, defaults to 0.0) — The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
-   **hidden\_act** (`str` or `function`, _optional_, defaults to `"silu"`) — The non-linear activation function (function or string) in the decoder.
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **alpha\_initializer** (`str`, _optional_, defaults to `"zeros"`) — Initialization type for the alphas.
-   **alphas\_initializer\_range** (`float`, _optional_, defaults to 0.0) — The standard deviation of the truncated\_normal\_initializer for initializing the alphas in the Gated Cross Attention.
-   **alpha\_type** (`str`, _optional_, defaults to `"float"`) — Whether the gating alphas should be vectors or single floats.
-   **rms\_norm\_eps** (`float`, _optional_, defaults to 1e-6) — The epsilon used by the rms normalization layers.
-   **use\_cache** (`bool`, _optional_, defaults to `True`) — Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True`.
-   **pad\_token\_id** (`int`, _optional_, defaults to 0) — Padding token id.
-   **bos\_token\_id** (`int`, _optional_, defaults to 1) — Beginning of stream token id.
-   **eos\_token\_id** (`int`, _optional_, defaults to 2) — End of stream token id.
-   **tie\_word\_embeddings(`bool`,** _optional_, defaults to `False`) — Whether to tie weight embeddings
-   **cross\_layer\_interval** (`int`, _optional_, default to 1) — Interval for cross attention (from text to image) layers.
-   **qk\_layer\_norms** (`bool`, _optional_, defaults to `False`) — Whether to add layer norm after q and k
-   **freeze\_text\_layers** (`bool`, _optional_, defaults to `True`) — Whether to freeze text layers
-   **freeze\_text\_module\_exceptions** (`bool`, _optional_, defaults to `[]`) — Exceptions to freezing text layers when `freeze_text_layers` is `True`
-   **freeze\_lm\_head** (`bool`, _optional_, defaults to `False`) — Whether to freeze lm head
-   **freeze\_vision\_layers** (`bool`, _optional_, defaults to `True`) — Whether to freeze vision layers
-   **freeze\_vision\_module\_exceptions** (`bool`, _optional_, defaults to `[]`) — Exceptions to freezing vision layers when `freeze_vision_layers` is `True`
-   **use\_resampler** (`bool`, _optional_, defaults to `False`) — Whether to use the Resampler
-   **vision\_config** (`IdeficsVisionConfig`, _optional_) — Custom vision config or dict
-   **perceiver\_config** (`IdeficsPerceiverConfig`, _optional_) — Custom perceiver config or dict

This is the configuration class to store the configuration of a [IdeficsModel](/docs/transformers/v4.34.0/en/model_doc/idefics#transformers.IdeficsModel). It is used to instantiate an Idefics model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Idefics-9B.

e.g. [HuggingFaceM4/idefics-9b](https://huggingface.co/HuggingFaceM4/idefics-9b)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import IdeficsModel, IdeficsConfig

>>> 
>>> configuration = IdeficsConfig()

>>> 
>>> model = IdeficsModel(configuration)

>>> 
>>> configuration = model.config
```

## IdeficsModel

### class transformers.IdeficsModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/idefics/modeling_idefics.py#L1049)

( config: IdeficsConfig )

Parameters

-   **config** ([IdeficsConfig](/docs/transformers/v4.34.0/en/model_doc/idefics#transformers.IdeficsConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights. config — IdeficsConfig

The bare LLaMA Model outputting raw hidden-states without any specific head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

Transformer decoder consisting of `config.num_hidden_layers` layers. Each layer is a `IdeficsDecoderLayer`

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/idefics/modeling_idefics.py#L1151)

( input\_ids: LongTensor = None attention\_mask: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None past\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None pixel\_values: typing.Optional\[torch.FloatTensor\] = None image\_encoder\_embeddings: typing.Optional\[torch.FloatTensor\] = None perceiver\_embeddings: typing.Optional\[torch.FloatTensor\] = None image\_attention\_mask: typing.Optional\[torch.Tensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None interpolate\_pos\_encoding: typing.Optional\[bool\] = False return\_dict: typing.Optional\[bool\] = None )

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
    
    If you want to change padding behavior, you should read `modeling_opt._prepare_decoder_attention_mask` and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
    If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

The [IdeficsModel](/docs/transformers/v4.34.0/en/model_doc/idefics#transformers.IdeficsModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## IdeficsForVisionText2Text

### class transformers.IdeficsForVisionText2Text

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/idefics/modeling_idefics.py#L1405)

( config vision\_model = None )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/idefics/modeling_idefics.py#L1463)

( input\_ids: LongTensor = None attention\_mask: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None past\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None pixel\_values: typing.Optional\[torch.FloatTensor\] = None image\_encoder\_embeddings: typing.Optional\[torch.FloatTensor\] = None perceiver\_embeddings: typing.Optional\[torch.FloatTensor\] = None image\_attention\_mask: typing.Optional\[torch.Tensor\] = None labels: typing.Optional\[torch.LongTensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None interpolate\_pos\_encoding: typing.Optional\[bool\] = False return\_dict: typing.Optional\[bool\] = None ) → `transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
    
    If you want to change padding behavior, you should read `modeling_opt._prepare_decoder_attention_mask` and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
    If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
    
    Args — labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_): Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    

Returns

`transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([IdeficsConfig](/docs/transformers/v4.34.0/en/model_doc/idefics#transformers.IdeficsConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **image\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_) — Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images, sequence_length, hidden_size)`.
    
    image\_hidden\_states of the model produced by the vision encoder, and optionally by the perceiver
    

The [IdeficsForVisionText2Text](/docs/transformers/v4.34.0/en/model_doc/idefics#transformers.IdeficsForVisionText2Text) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, IdeficsForVisionText2Text

>>> model = IdeficsForVisionText2Text.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
>>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

>>> prompt = "Hey, are you consciours? Can you talk to me?"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> 
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
```

## IdeficsImageProcessor

### class transformers.IdeficsImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/idefics/image_processing_idefics.py#L51)

( image\_size: int = 224 image\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = None image\_std: typing.Union\[float, typing.List\[float\], NoneType\] = None image\_num\_channels: typing.Optional\[int\] = 3 \*\*kwargs )

Parameters

-   **image\_size** (`int`, _optional_, defaults to `224`) — Resize to image size
-   **image\_num\_channels** (`int`, _optional_, defaults to `3`) — Number of image channels.
-   **image\_mean** (`float` or `List[float]`, _optional_, defaults to `IDEFICS_STANDARD_MEAN`) — Mean to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be overridden by the `image_mean` parameter in the `preprocess` method.
-   **image\_std** (`float` or `List[float]`, _optional_, defaults to `IDEFICS_STANDARD_STD`) — Standard deviation to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method. Can be overridden by the `image_std` parameter in the `preprocess` method.

Constructs a Idefics image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/idefics/image_processing_idefics.py#L87)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\] image\_num\_channels: typing.Optional\[int\] = 3 image\_size: typing.Union\[typing.Dict\[str, int\], NoneType\] = None image\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = None image\_std: typing.Union\[float, typing.List\[float\], NoneType\] = None transform: typing.Callable = None \*\*kwargs )

Parameters

-   **images** (`ImageInput`) — A list of images to preprocess.
-   **image\_size** (`int`, _optional_, defaults to `self.image_size`) — Resize to image size
-   **image\_num\_channels** (`int`, _optional_, defaults to `self.image_num_channels`) — Number of image channels.
-   **image\_mean** (`float` or `List[float]`, _optional_, defaults to `IDEFICS_STANDARD_MEAN`) — Mean to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be overridden by the `image_mean` parameter in the `preprocess` method.
-   **image\_std** (`float` or `List[float]`, _optional_, defaults to `IDEFICS_STANDARD_STD`) — Standard deviation to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method. Can be overridden by the `image_std` parameter in the `preprocess` method.
-   **transform** (`Callable`, _optional_, defaults to `None`) — A custom transform function that accepts a single image can be passed for training. For example, `torchvision.Compose` can be used to compose multiple transforms. If `None` - an inference mode is assumed - and then a preset of inference-specific transforms will be applied to the images

Preprocess a batch of images.

## IdeficsProcessor

### class transformers.IdeficsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/idefics/processing_idefics.py#L108)

( image\_processor tokenizer = None image\_size = 224 add\_end\_of\_utterance\_token = None \*\*kwargs )

Parameters

-   **image\_processor** (`IdeficsImageProcessor`) — An instance of [IdeficsImageProcessor](/docs/transformers/v4.34.0/en/model_doc/idefics#transformers.IdeficsImageProcessor). The image processor is a required input.
-   **tokenizer** (`LlamaTokenizerFast`) — An instance of [LlamaTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/llama2#transformers.LlamaTokenizerFast). The tokenizer is a required input.
-   **image\_size** (`int`, _optional_, defaults to 224) — Image size (assuming a square image)

Constructs a IDEFICS processor which wraps a LLama tokenizer and IDEFICS image processor into a single processor.

[IdeficsProcessor](/docs/transformers/v4.34.0/en/model_doc/idefics#transformers.IdeficsProcessor) offers all the functionalities of [IdeficsImageProcessor](/docs/transformers/v4.34.0/en/model_doc/idefics#transformers.IdeficsImageProcessor) and [LlamaTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/llama2#transformers.LlamaTokenizerFast). See the docstring of [**call**()](/docs/transformers/v4.34.0/en/model_doc/idefics#transformers.IdeficsProcessor.__call__) and `decode()` for more information.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/idefics/processing_idefics.py#L148)

( prompts: typing.Union\[typing.List\[str\], typing.List\[typing.List\[str\]\]\] padding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = False truncation: typing.Union\[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy\] = None max\_length: typing.Optional\[int\] = None transform: typing.Callable = None add\_eos\_token = False add\_end\_of\_utterance\_token = None debug = False return\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = <TensorType.PYTORCH: 'pt'> ) → a dict with entries

Parameters

-   **prompts** (`Union[List[TextInput], [List[List[TextInput]]]]`) — either a single prompt or a batched list of prompts - see the detailed description immediately after the end of the arguments doc section.
-   **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.utils.PaddingStrategy), _optional_, defaults to `False`) — Select a strategy to pad the returned sequences (according to the model’s padding side and padding index) among:
    
    -   `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
    -   `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided.
    -   `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).
    
-   **max\_length** (`int`, _optional_) — Maximum length of the returned list and optionally padding length (see above).
-   **truncation** (`bool`, _optional_) — Activates truncation to cut input sequences longer than `max_length` to `max_length`.
-   **transform** (`Callable`, _optional_) — A custom transform function that accepts a single image can be passed for training. For example, `torchvision.Compose` can be used to compose multiple functions. If `None` a preset inference-specific set of transforms will be applied to the images
-   **add\_eos\_token** (`bool`, _optional_, defaults to `False`) — Adds `eos_token` at the end of the final prompt if True\`
-   **add\_end\_of\_utterance\_token** (`bool`, _optional_) — Whether to automatically add `<end_of_utterance>` after each prompt’s text input (unless followed by an image). If `None` the tokenizer will be checked instead and if this token is found in `additional_special_tokens` then the value will be `True`.
-   **debug** (`bool`, _optional_, defaults to `False`) — `True` value will help debug prompt generation by dumping useful information
-   **return\_tensors** (`str` or `TensorType`, _optional_, defaults to `TensorType.PYTORCH`) — The type of tensors to return. Can be one of:
    
    -   `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
    

Returns

a dict with entries

`input_ids`, `attention_mask`, `pixel_values`, `image_attention_mask` which can be directly passed to `model.generate`

This method takes batched or non-batched prompts made of text and images and converts them into prompts that the model was trained on and prepares the image pixel values for the model to process.

Detailed explanation:

Each entry in `prompts` is either a text to be passed as is or an image that will be processed.

An image can be either an image object (`PIL.Image`) or a url from which the image can be retrieved.

When the processor encounters an image it’ll inject `<fake_token_around_image><image><fake_token_around_image>` entry into the prompt.

Example:

```
checkpoint = "HuggingFaceM4/idefics-9b"
processor = AutoProcessor.from_pretrained(checkpoint)
url = "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg"
img = processor.image_processor.fetch_images([url])[0]

prompts = [
    "User:",
    img,
    "Describe this image.
t: An image of two kittens in grass.

    "User:",
    "https://hips.hearstapps.com/hmg-prod/images/dog-puns-1581708208.jpg",
    "Describe this image.
t:",
]

inputs = processor(prompts, return_tensors="pt")
generated_ids = model.generate(**inputs, max_length=100)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

In this example the `prompts` will be converted into:

```
<s>User:<fake_token_around_image><image><fake_token_around_image>Describe this image.
Assistant: An image of two kittens in grass.
User:<fake_token_around_image><image><fake_token_around_image>Describe this image.
Assistant:'
```

and the two images will be massaged using [IdeficsImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) method and placed inside the `pixel_values` dict entry of the return value.

This example also examplifies that images can be passed as objects or as text urls. It can be seen that the first image is passed as object and the second one as a url.

To do training do:

```
image_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            (w, h), scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=self.image_mean, std=self.image_std),
    ]
)
inputs = processor(prompts, transform=image_transform, return_tensors="pt")
```

In order to help debug prompt generation enable `debug=True` which will show you what’s happening.