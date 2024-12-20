# Pop2Piano

[![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/sweetcocoa/pop2piano)

## Overview

The Pop2Piano model was proposed in [Pop2Piano : Pop Audio-based Piano Cover Generation](https://arxiv.org/abs/2211.00895) by Jongho Choi and Kyogu Lee.

Piano covers of pop music are widely enjoyed, but generating them from music is not a trivial task. It requires great expertise with playing piano as well as knowing different characteristics and melodies of a song. With Pop2Piano you can directly generate a cover from a song’s audio waveform. It is the first model to directly generate a piano cover from pop audio without melody and chord extraction modules.

Pop2Piano is an encoder-decoder Transformer model based on [T5](https://arxiv.org/pdf/1910.10683.pdf). The input audio is transformed to its waveform and passed to the encoder, which transforms it to a latent representation. The decoder uses these latent representations to generate token ids in an autoregressive way. Each token id corresponds to one of four different token types: time, velocity, note and ‘special’. The token ids are then decoded to their equivalent MIDI file.

The abstract from the paper is the following:

_Piano covers of pop music are enjoyed by many people. However, the task of automatically generating piano covers of pop music is still understudied. This is partly due to the lack of synchronized {Pop, Piano Cover} data pairs, which made it challenging to apply the latest data-intensive deep learning-based methods. To leverage the power of the data-driven approach, we make a large amount of paired and synchronized {Pop, Piano Cover} data using an automated pipeline. In this paper, we present Pop2Piano, a Transformer network that generates piano covers given waveforms of pop music. To the best of our knowledge, this is the first model to generate a piano cover directly from pop audio without using melody and chord extraction modules. We show that Pop2Piano, trained with our dataset, is capable of producing plausible piano covers._

Tips:

1.  To use Pop2Piano, you will need to install the 🤗 Transformers library, as well as the following third party modules:

```
pip install pretty-midi==0.2.9 essentia==2.1b6.dev1034 librosa scipy
```

Please note that you may need to restart your runtime after installation. 2. Pop2Piano is an Encoder-Decoder based model like T5. 3. Pop2Piano can be used to generate midi-audio files for a given audio sequence. 4. Choosing different composers in `Pop2PianoForConditionalGeneration.generate()` can lead to variety of different results. 5. Setting the sampling rate to 44.1 kHz when loading the audio file can give good performance. 6. Though Pop2Piano was mainly trained on Korean Pop music, it also does pretty well on other Western Pop or Hip Hop songs.

This model was contributed by [Susnato Dhar](https://huggingface.co/susnato). The original code can be found [here](https://github.com/sweetcocoa/pop2piano).

## Examples

-   Example using HuggingFace Dataset:

```
>>> from datasets import load_dataset
>>> from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

>>> model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
>>> processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")
>>> ds = load_dataset("sweetcocoa/pop2piano_ci", split="test")

>>> inputs = processor(
...     audio=ds["audio"][0]["array"], sampling_rate=ds["audio"][0]["sampling_rate"], return_tensors="pt"
... )
>>> model_output = model.generate(input_features=inputs["input_features"], composer="composer1")
>>> tokenizer_output = processor.batch_decode(
...     token_ids=model_output, feature_extractor_output=inputs
... )["pretty_midi_objects"][0]
>>> tokenizer_output.write("./Outputs/midi_output.mid")
```

-   Example using your own audio file:

```
>>> import librosa
>>> from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

>>> audio, sr = librosa.load("<your_audio_file_here>", sr=44100)  
>>> model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
>>> processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")

>>> inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt")
>>> model_output = model.generate(input_features=inputs["input_features"], composer="composer1")
>>> tokenizer_output = processor.batch_decode(
...     token_ids=model_output, feature_extractor_output=inputs
... )["pretty_midi_objects"][0]
>>> tokenizer_output.write("./Outputs/midi_output.mid")
```

-   Example of processing multiple audio files in batch:

```
>>> import librosa
>>> from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

>>> 
>>> audio1, sr1 = librosa.load("<your_first_audio_file_here>", sr=44100)  
>>> audio2, sr2 = librosa.load("<your_second_audio_file_here>", sr=44100)
>>> model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
>>> processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")

>>> inputs = processor(audio=[audio1, audio2], sampling_rate=[sr1, sr2], return_attention_mask=True, return_tensors="pt")
>>> 
>>> model_output = model.generate(
...     input_features=inputs["input_features"],
...     attention_mask=inputs["attention_mask"],
...     composer="composer1",
... )
>>> tokenizer_output = processor.batch_decode(
...     token_ids=model_output, feature_extractor_output=inputs
... )["pretty_midi_objects"]

>>> 
>>> tokenizer_output[0].write("./Outputs/midi_output1.mid")
>>> tokenizer_output[1].write("./Outputs/midi_output2.mid")
```

-   Example of processing multiple audio files in batch (Using `Pop2PianoFeatureExtractor` and `Pop2PianoTokenizer`):

```
>>> import librosa
>>> from transformers import Pop2PianoForConditionalGeneration, Pop2PianoFeatureExtractor, Pop2PianoTokenizer

>>> 
>>> audio1, sr1 = librosa.load("<your_first_audio_file_here>", sr=44100)  
>>> audio2, sr2 = librosa.load("<your_second_audio_file_here>", sr=44100)
>>> model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
>>> feature_extractor = Pop2PianoFeatureExtractor.from_pretrained("sweetcocoa/pop2piano")
>>> tokenizer = Pop2PianoTokenizer.from_pretrained("sweetcocoa/pop2piano")

>>> inputs = feature_extractor(
...     audio=[audio1, audio2], 
...     sampling_rate=[sr1, sr2], 
...     return_attention_mask=True, 
...     return_tensors="pt",
... )
>>> 
>>> model_output = model.generate(
...     input_features=inputs["input_features"],
...     attention_mask=inputs["attention_mask"],
...     composer="composer1",
... )
>>> tokenizer_output = tokenizer.batch_decode(
...     token_ids=model_output, feature_extractor_output=inputs
... )["pretty_midi_objects"]

>>> 
>>> tokenizer_output[0].write("./Outputs/midi_output1.mid")
>>> tokenizer_output[1].write("./Outputs/midi_output2.mid")
```

## Pop2PianoConfig

### class transformers.Pop2PianoConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pop2piano/configuration_pop2piano.py#L29)

( vocab\_size = 2400 composer\_vocab\_size = 21 d\_model = 512 d\_kv = 64 d\_ff = 2048 num\_layers = 6 num\_decoder\_layers = None num\_heads = 8 relative\_attention\_num\_buckets = 32 relative\_attention\_max\_distance = 128 dropout\_rate = 0.1 layer\_norm\_epsilon = 1e-06 initializer\_factor = 1.0 feed\_forward\_proj = 'gated-gelu' is\_encoder\_decoder = True use\_cache = True pad\_token\_id = 0 eos\_token\_id = 1 dense\_act\_fn = 'relu' \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 2400) — Vocabulary size of the `Pop2PianoForConditionalGeneration` model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [Pop2PianoForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/pop2piano#transformers.Pop2PianoForConditionalGeneration).
-   **composer\_vocab\_size** (`int`, _optional_, defaults to 21) — Denotes the number of composers.
-   **d\_model** (`int`, _optional_, defaults to 512) — Size of the encoder layers and the pooler layer.
-   **d\_kv** (`int`, _optional_, defaults to 64) — Size of the key, query, value projections per attention head. The `inner_dim` of the projection layer will be defined as `num_heads * d_kv`.
-   **d\_ff** (`int`, _optional_, defaults to 2048) — Size of the intermediate feed forward layer in each `Pop2PianoBlock`.
-   **num\_layers** (`int`, _optional_, defaults to 6) — Number of hidden layers in the Transformer encoder.
-   **num\_decoder\_layers** (`int`, _optional_) — Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
-   **num\_heads** (`int`, _optional_, defaults to 8) — Number of attention heads for each attention layer in the Transformer encoder.
-   **relative\_attention\_num\_buckets** (`int`, _optional_, defaults to 32) — The number of buckets to use for each attention layer.
-   **relative\_attention\_max\_distance** (`int`, _optional_, defaults to 128) — The maximum distance of the longer sequences for the bucket separation.
-   **dropout\_rate** (`float`, _optional_, defaults to 0.1) — The ratio for all dropout layers.
-   **layer\_norm\_epsilon** (`float`, _optional_, defaults to 1e-6) — The epsilon used by the layer normalization layers.
-   **initializer\_factor** (`float`, _optional_, defaults to 1.0) — A factor for initializing all weight matrices (should be kept to 1.0, used internally for initialization testing).
-   **feed\_forward\_proj** (`string`, _optional_, defaults to `"gated-gelu"`) — Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`.
-   **use\_cache** (`bool`, _optional_, defaults to `True`) — Whether or not the model should return the last key/values attentions (not used by all models).
-   **dense\_act\_fn** (`string`, _optional_, defaults to `"relu"`) — Type of Activation Function to be used in `Pop2PianoDenseActDense` and in `Pop2PianoDenseGatedActDense`.

This is the configuration class to store the configuration of a [Pop2PianoForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/pop2piano#transformers.Pop2PianoForConditionalGeneration). It is used to instantiate a Pop2PianoForConditionalGeneration model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Pop2Piano [sweetcocoa/pop2piano](https://huggingface.co/sweetcocoa/pop2piano) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## Pop2PianoFeatureExtractor

## Pop2PianoForConditionalGeneration

### class transformers.Pop2PianoForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pop2piano/modeling_pop2piano.py#L1020)

( config: Pop2PianoConfig )

Parameters

-   **config** ([Pop2PianoConfig](/docs/transformers/v4.34.0/en/model_doc/pop2piano#transformers.Pop2PianoConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Pop2Piano Model with a `language modeling` head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pop2piano/modeling_pop2piano.py#L1119)

( input\_ids: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None decoder\_input\_ids: typing.Optional\[torch.LongTensor\] = None decoder\_attention\_mask: typing.Optional\[torch.BoolTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None decoder\_head\_mask: typing.Optional\[torch.FloatTensor\] = None cross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = None encoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = None past\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None input\_features: typing.Optional\[torch.FloatTensor\] = None decoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Pop2Piano is a model with relative position embeddings so you should be able to pad the inputs on both the right and the left. Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for detail. [What are input IDs?](../glossary#input-ids) To know more on how to prepare `input_ids` for pretraining take a look a [Pop2Pianp Training](./Pop2Piano#training).
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**. [What are attention masks?](../glossary#attention-mask)
    
-   **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, _optional_) — Indices of decoder input sequence tokens in the vocabulary. Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details. [What are decoder input IDs?](../glossary#decoder-input-ids) Pop2Piano uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`). To know more on how to prepare
-   **decoder\_attention\_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, _optional_) — Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also be used by default.
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **decoder\_head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **encoder\_outputs** (`tuple(tuple(torch.FloatTensor)`, _optional_) — Tuple consists of (`last_hidden_state`, `optional`: _hidden\_states_, `optional`: _attentions_) `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`) — Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **input\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Does the same task as `inputs_embeds`. If `inputs_embeds` is not present but `input_features` is present then `input_features` will be considered as `inputs_embeds`.
-   **decoder\_inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be input (see `past_key_values`). This is useful if you want more control over how to convert `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix. If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value of `inputs_embeds`.
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ..., config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`

A [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([Pop2PianoConfig](/docs/transformers/v4.34.0/en/model_doc/pop2piano#transformers.Pop2PianoConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Language modeling loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
    
-   **decoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
    
-   **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
    
-   **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
    
-   **encoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [Pop2PianoForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/pop2piano#transformers.Pop2PianoForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

#### generate

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pop2piano/modeling_pop2piano.py#L1227)

( input\_features attention\_mask = None composer = 'composer1' generation\_config = None \*\*kwargs ) → [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) or `torch.LongTensor`

Parameters

-   **input\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — This is the featurized version of audio generated by `Pop2PianoFeatureExtractor`. attention\_mask — For batched generation `input_features` are padded to have the same shape across all examples. `attention_mask` helps to determine which areas were padded and which were not.
    
    -   1 for tokens that are **not padded**,
    -   0 for tokens that are **padded**.
    
-   **composer** (`str`, _optional_, defaults to `"composer1"`) — This value is passed to `Pop2PianoConcatEmbeddingToMel` to generate different embeddings for each `"composer"`. Please make sure that the composet value is present in `composer_to_feature_token` in `generation_config`. For an example please see [https://huggingface.co/sweetcocoa/pop2piano/blob/main/generation\_config.json](https://huggingface.co/sweetcocoa/pop2piano/blob/main/generation_config.json) .
-   **generation\_config** (`~generation.GenerationConfig`, _optional_) — The generation configuration to be used as base parametrization for the generation call. `**kwargs` passed to generate matching the attributes of `generation_config` will override them. If `generation_config` is not provided, the default will be used, which had the following loading priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model configuration. Please note that unspecified parameters will inherit [GenerationConfig](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationConfig)’s default values, whose documentation should be checked to parameterize generation. kwargs — Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with _decoder\__.

Returns

[ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) or `torch.LongTensor`

A [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) (if `return_dict_in_generate=True` or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`. Since Pop2Piano is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) types are:

-   [GreedySearchEncoderDecoderOutput](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.GreedySearchEncoderDecoderOutput),
-   [SampleEncoderDecoderOutput](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.SampleEncoderDecoderOutput),
-   [BeamSearchEncoderDecoderOutput](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.BeamSearchEncoderDecoderOutput),
-   [BeamSampleEncoderDecoderOutput](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.BeamSampleEncoderDecoderOutput)

Generates token ids for midi outputs.

Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the model’s default generation configuration. You can override any `generation_config` by passing the corresponding parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`. For an overview of generation strategies and code examples, check out the [following guide](./generation_strategies).

## Pop2PianoTokenizer

### class transformers.Pop2PianoTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects.py#L12)

( \*args \*\*kwargs )

#### \_\_call\_\_



( \*args \*\*kwargs )

Call self as a function.

## Pop2PianoProcessor

### class transformers.Pop2PianoProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects.py#L19)

( \*args \*\*kwargs )

#### \_\_call\_\_



( \*args \*\*kwargs )

Call self as a function.