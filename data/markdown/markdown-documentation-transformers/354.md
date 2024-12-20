# SEW-D

## Overview

SEW-D (Squeezed and Efficient Wav2Vec with Disentangled attention) was proposed in [Performance-Efficiency Trade-offs in Unsupervised Pre-training for Speech Recognition](https://arxiv.org/abs/2109.06870) by Felix Wu, Kwangyoun Kim, Jing Pan, Kyu Han, Kilian Q. Weinberger, Yoav Artzi.

The abstract from the paper is the following:

_This paper is a study of performance-efficiency trade-offs in pre-trained models for automatic speech recognition (ASR). We focus on wav2vec 2.0, and formalize several architecture designs that influence both the model performance and its efficiency. Putting together all our observations, we introduce SEW (Squeezed and Efficient Wav2vec), a pre-trained model architecture with significant improvements along both performance and efficiency dimensions across a variety of training setups. For example, under the 100h-960h semi-supervised setup on LibriSpeech, SEW achieves a 1.9x inference speedup compared to wav2vec 2.0, with a 13.5% relative reduction in word error rate. With a similar inference time, SEW reduces word error rate by 25-50% across different model sizes._

Tips:

-   SEW-D is a speech model that accepts a float array corresponding to the raw waveform of the speech signal.
-   SEWDForCTC is fine-tuned using connectionist temporal classification (CTC) so the model output has to be decoded using [Wav2Vec2CTCTokenizer](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer).

This model was contributed by [anton-l](https://huggingface.co/anton-l).

## Documentation resources

-   [Audio classification task guide](../tasks/audio_classification)
-   [Automatic speech recognition task guide](../tasks/asr)

## SEWDConfig

### class transformers.SEWDConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sew_d/configuration_sew_d.py#L32)

( vocab\_size = 32 hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 squeeze\_factor = 2 max\_position\_embeddings = 512 position\_buckets = 256 share\_att\_key = True relative\_attention = True pos\_att\_type = ('p2c', 'c2p') norm\_rel\_ebd = 'layer\_norm' hidden\_act = 'gelu\_python' hidden\_dropout = 0.1 activation\_dropout = 0.1 attention\_dropout = 0.1 feat\_proj\_dropout = 0.0 final\_dropout = 0.1 initializer\_range = 0.02 layer\_norm\_eps = 1e-07 feature\_layer\_norm\_eps = 1e-05 feat\_extract\_norm = 'group' feat\_extract\_activation = 'gelu' conv\_dim = (64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512) conv\_stride = (5, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1) conv\_kernel = (10, 3, 1, 3, 1, 3, 1, 3, 1, 2, 1, 2, 1) conv\_bias = False num\_conv\_pos\_embeddings = 128 num\_conv\_pos\_embedding\_groups = 16 apply\_spec\_augment = True mask\_time\_prob = 0.05 mask\_time\_length = 10 mask\_time\_min\_masks = 2 mask\_feature\_prob = 0.0 mask\_feature\_length = 10 mask\_feature\_min\_masks = 0 ctc\_loss\_reduction = 'mean' ctc\_zero\_infinity = False use\_weighted\_layer\_sum = False classifier\_proj\_size = 256 pad\_token\_id = 0 bos\_token\_id = 1 eos\_token\_id = 2 \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 32) — Vocabulary size of the SEW-D model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `SEWD`.
-   **hidden\_size** (`int`, _optional_, defaults to 768) — Dimensionality of the encoder layers and the pooler layer.
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 12) — Number of hidden layers in the Transformer encoder.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 12) — Number of attention heads for each attention layer in the Transformer encoder.
-   **intermediate\_size** (`int`, _optional_, defaults to 3072) — Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
-   **squeeze\_factor** (`int`, _optional_, defaults to 2) — Sequence length downsampling factor after the encoder and upsampling factor after the transformer.
-   **max\_position\_embeddings** (`int`, _optional_, defaults to 512) — The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
-   **position\_buckets** (`int`, _optional_, defaults to 256) — The maximum size of relative position embeddings.
-   **share\_att\_key** (`bool`, _optional_, defaults to `True`) — Whether to share attention key with c2p and p2c.
-   **relative\_attention** (`bool`, _optional_, defaults to `True`) — Whether to use relative position encoding.
-   **pos\_att\_type** (`Tuple[str]`, _optional_, defaults to `("p2c", "c2p")`) — The type of relative position attention, it can be a combination of `("p2c", "c2p")`, e.g. `("p2c")`, `("p2c", "c2p")`, `("p2c", "c2p")`.
-   **norm\_rel\_ebd** (`str`, _optional_, defaults to `"layer_norm"`) — Whether to use layer norm in relative embedding (`"layer_norm"` if yes)
-   **hidden\_act** (`str` or `function`, _optional_, defaults to `"gelu_python"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"`, `"gelu_python"` and `"gelu_new"` are supported.
-   **hidden\_dropout** (`float`, _optional_, defaults to 0.1) — Deprecated. Not used by the model and will be removed in a future version.
-   **activation\_dropout** (`float`, _optional_, defaults to 0.1) — The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
-   **attention\_dropout** (`float`, _optional_, defaults to 0.1) — The dropout ratio for the attention probabilities.
-   **final\_dropout** (`float`, _optional_, defaults to 0.1) — The dropout probability for the final projection layer of [SEWDForCTC](/docs/transformers/v4.34.0/en/model_doc/sew-d#transformers.SEWDForCTC).
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-7) — The epsilon used by the layer normalization layers in the transformer encoder.
-   **feature\_layer\_norm\_eps** (`float`, _optional_, defaults to 1e-5) — The epsilon used by the layer normalization after the feature encoder.
-   **feat\_extract\_norm** (`str`, _optional_, defaults to `"group"`) — The norm to be applied to 1D convolutional layers in feature encoder. One of `"group"` for group normalization of only the first 1D convolutional layer or `"layer"` for layer normalization of all 1D convolutional layers.
-   **feat\_proj\_dropout** (`float`, _optional_, defaults to 0.0) — The dropout probability for output of the feature encoder.
-   **feat\_extract\_activation** (`str,` optional`, defaults to` “gelu”`) -- The non-linear activation function (function or string) in the 1D convolutional layers of the feature extractor. If string,` “gelu”`,` “relu”`,` “selu”`and`“gelu\_new”\` are supported.
-   **conv\_dim** (`Tuple[int]` or `List[int]`, _optional_, defaults to `(64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512)`) — A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the feature encoder. The length of _conv\_dim_ defines the number of 1D convolutional layers.
-   **conv\_stride** (`Tuple[int]` or `List[int]`, _optional_, defaults to `(5, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1)`) — A tuple of integers defining the stride of each 1D convolutional layer in the feature encoder. The length of _conv\_stride_ defines the number of convolutional layers and has to match the length of _conv\_dim_.
-   **conv\_kernel** (`Tuple[int]` or `List[int]`, _optional_, defaults to `(10, 3, 1, 3, 1, 3, 1, 3, 1, 2, 1, 2, 1)`) — A tuple of integers defining the kernel size of each 1D convolutional layer in the feature encoder. The length of _conv\_kernel_ defines the number of convolutional layers and has to match the length of _conv\_dim_.
-   **conv\_bias** (`bool`, _optional_, defaults to `False`) — Whether the 1D convolutional layers have a bias.
-   **num\_conv\_pos\_embeddings** (`int`, _optional_, defaults to 128) — Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional embeddings layer.
-   **num\_conv\_pos\_embedding\_groups** (`int`, _optional_, defaults to 16) — Number of groups of 1D convolutional positional embeddings layer.
-   **apply\_spec\_augment** (`bool`, _optional_, defaults to `True`) — Whether to apply _SpecAugment_ data augmentation to the outputs of the feature encoder. For reference see [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779).
-   **mask\_time\_prob** (`float`, _optional_, defaults to 0.05) — Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. The masking procecure generates ”mask\_time\_prob_len(time\_axis)/mask\_time\_length” independent masks over the axis. If reasoning from the propability of each feature vector to be chosen as the start of the vector span to be masked,_ mask\_time\_prob _should be \`prob\_vector\_start_mask\_time\_length`. Note that overlap may decrease the actual percentage of masked vectors. This is only relevant if` apply\_spec\_augment is True\`.
-   **mask\_time\_length** (`int`, _optional_, defaults to 10) — Length of vector span along the time axis.
-   **mask\_time\_min\_masks** (`int`, _optional_, defaults to 2), — The minimum number of masks of length `mask_feature_length` generated along the time axis, each time step, irrespectively of `mask_feature_prob`. Only relevant if ”mask\_time\_prob\*len(time\_axis)/mask\_time\_length < mask\_time\_min\_masks”
-   **mask\_feature\_prob** (`float`, _optional_, defaults to 0.0) — Percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The masking procecure generates ”mask\_feature\_prob_len(feature\_axis)/mask\_time\_length” independent masks over the axis. If reasoning from the propability of each feature vector to be chosen as the start of the vector span to be masked,_ mask\_feature\_prob _should be \`prob\_vector\_start_mask\_feature\_length`. Note that overlap may decrease the actual percentage of masked vectors. This is only relevant if` apply\_spec\_augment is True\`.
-   **mask\_feature\_length** (`int`, _optional_, defaults to 10) — Length of vector span along the feature axis.
-   **mask\_feature\_min\_masks** (`int`, _optional_, defaults to 0), — The minimum number of masks of length `mask_feature_length` generated along the feature axis, each time step, irrespectively of `mask_feature_prob`. Only relevant if ”mask\_feature\_prob\*len(feature\_axis)/mask\_feature\_length < mask\_feature\_min\_masks”
-   **diversity\_loss\_weight** (`int`, _optional_, defaults to 0.1) — The weight of the codebook diversity loss component.
-   **ctc\_loss\_reduction** (`str`, _optional_, defaults to `"sum"`) — Specifies the reduction to apply to the output of `torch.nn.CTCLoss`. Only relevant when training an instance of [SEWDForCTC](/docs/transformers/v4.34.0/en/model_doc/sew-d#transformers.SEWDForCTC).
-   **ctc\_zero\_infinity** (`bool`, _optional_, defaults to `False`) — Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance of [SEWDForCTC](/docs/transformers/v4.34.0/en/model_doc/sew-d#transformers.SEWDForCTC).
-   **use\_weighted\_layer\_sum** (`bool`, _optional_, defaults to `False`) — Whether to use a weighted average of layer outputs with learned weights. Only relevant when using an instance of [Wav2Vec2ForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2ForSequenceClassification).
-   **classifier\_proj\_size** (`int`, _optional_, defaults to 256) — Dimensionality of the projection before token mean-pooling for classification.

This is the configuration class to store the configuration of a [SEWDModel](/docs/transformers/v4.34.0/en/model_doc/sew-d#transformers.SEWDModel). It is used to instantiate a SEW-D model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the SEW-D [asapp/sew-d-tiny-100k](https://huggingface.co/asapp/sew-d-tiny-100k) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import SEWDConfig, SEWDModel

>>> 
>>> configuration = SEWDConfig()

>>> 
>>> model = SEWDModel(configuration)

>>> 
>>> configuration = model.config
```

Serializes this instance to a Python dictionary.

## SEWDModel

### class transformers.SEWDModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sew_d/modeling_sew_d.py#L1381)

( config: SEWDConfig )

Parameters

-   **config** ([SEWDConfig](/docs/transformers/v4.34.0/en/model_doc/sew-d#transformers.SEWDConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare SEW-D Model transformer outputting raw hidden-states without any specific head on top. SEW-D was proposed in [Performance-Efficiency Trade-offs in Unsupervised Pre-training for Speech Recognition](https://arxiv.org/abs/2109.06870) by Felix Wu, Kwangyoun Kim, Jing Pan, Kyu Han, Kilian Q. Weinberger, Yoav Artzi.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving etc.).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sew_d/modeling_sew_d.py#L1448)

( input\_values: typing.Optional\[torch.Tensor\] attention\_mask: typing.Optional\[torch.Tensor\] = None mask\_time\_indices: typing.Optional\[torch.FloatTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, _e.g._ via the soundfile library (`pip install soundfile`). To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion into a tensor of type `torch.FloatTensor`. See [Wav2Vec2Processor.**call**()](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) for details.
-   **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([SEWDConfig](/docs/transformers/v4.34.0/en/model_doc/sew-d#transformers.SEWDConfig)) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [SEWDModel](/docs/transformers/v4.34.0/en/model_doc/sew-d#transformers.SEWDModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoProcessor, SEWDModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = AutoProcessor.from_pretrained("asapp/sew-d-tiny-100k-ft-ls100h")
>>> model = SEWDModel.from_pretrained("asapp/sew-d-tiny-100k-ft-ls100h")

>>> 
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 292, 384]
```

## SEWDForCTC

### class transformers.SEWDForCTC

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sew_d/modeling_sew_d.py#L1510)

( config target\_lang: typing.Optional\[str\] = None )

Parameters

-   **config** ([SEWDConfig](/docs/transformers/v4.34.0/en/model_doc/sew-d#transformers.SEWDConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

SEW-D Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC). SEW-D was proposed in [Performance-Efficiency Trade-offs in Unsupervised Pre-training for Speech Recognition](https://arxiv.org/abs/2109.06870) by Felix Wu, Kwangyoun Kim, Jing Pan, Kyu Han, Kilian Q. Weinberger, Yoav Artzi.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving etc.).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sew_d/modeling_sew_d.py#L1582)

( input\_values: typing.Optional\[torch.Tensor\] attention\_mask: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None labels: typing.Optional\[torch.Tensor\] = None ) → [transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, _e.g._ via the soundfile library (`pip install soundfile`). To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion into a tensor of type `torch.FloatTensor`. See [Wav2Vec2Processor.**call**()](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) for details.
-   **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size, target_length)`, _optional_) — Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size - 1]`.

A [transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([SEWDConfig](/docs/transformers/v4.34.0/en/model_doc/sew-d#transformers.SEWDConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [SEWDForCTC](/docs/transformers/v4.34.0/en/model_doc/sew-d#transformers.SEWDForCTC) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoProcessor, SEWDForCTC
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = AutoProcessor.from_pretrained("asapp/sew-d-tiny-100k-ft-ls100h")
>>> model = SEWDForCTC.from_pretrained("asapp/sew-d-tiny-100k-ft-ls100h")

>>> 
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
>>> predicted_ids = torch.argmax(logits, dim=-1)

>>> 
>>> transcription = processor.batch_decode(predicted_ids)
>>> transcription[0]
'MISTER QUILTER IS THE APOSTIL OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'

>>> inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="pt").input_ids

>>> 
>>> loss = model(**inputs).loss
>>> round(loss.item(), 2)
0.21
```

## SEWDForSequenceClassification

### class transformers.SEWDForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sew_d/modeling_sew_d.py#L1670)

( config )

Parameters

-   **config** ([SEWDConfig](/docs/transformers/v4.34.0/en/model_doc/sew-d#transformers.SEWDConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

SEWD Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like SUPERB Keyword Spotting.

SEW-D was proposed in [Performance-Efficiency Trade-offs in Unsupervised Pre-training for Speech Recognition](https://arxiv.org/abs/2109.06870) by Felix Wu, Kwangyoun Kim, Jing Pan, Kyu Han, Kilian Q. Weinberger, Yoav Artzi.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving etc.).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/sew_d/modeling_sew_d.py#L1715)

( input\_values: typing.Optional\[torch.Tensor\] attention\_mask: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None labels: typing.Optional\[torch.Tensor\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, _e.g._ via the soundfile library (`pip install soundfile`). To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion into a tensor of type `torch.FloatTensor`. See [Wav2Vec2Processor.**call**()](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) for details.
-   **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

A [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([SEWDConfig](/docs/transformers/v4.34.0/en/model_doc/sew-d#transformers.SEWDConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [SEWDForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/sew-d#transformers.SEWDForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoFeatureExtractor, SEWDForSequenceClassification
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("anton-l/sew-d-mid-400k-ft-keyword-spotting")
>>> model = SEWDForSequenceClassification.from_pretrained("anton-l/sew-d-mid-400k-ft-keyword-spotting")

>>> 
>>> inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.argmax(logits, dim=-1).item()
>>> predicted_label = model.config.id2label[predicted_class_ids]
>>> predicted_label
'_unknown_'

>>> 
>>> target_label = model.config.id2label[0]
>>> inputs["labels"] = torch.tensor([model.config.label2id[target_label]])
>>> loss = model(**inputs).loss
>>> round(loss.item(), 2)
3.16
```