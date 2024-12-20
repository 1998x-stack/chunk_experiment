Each framework has a generate method for text generation implemented in their respective `GenerationMixin` class:

-   PyTorch [generate()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.generate) is implemented in [GenerationMixin](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin).
-   TensorFlow [generate()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.TFGenerationMixin.generate) is implemented in [TFGenerationMixin](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.TFGenerationMixin).
-   Flax/JAX [generate()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.FlaxGenerationMixin.generate) is implemented in [FlaxGenerationMixin](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.FlaxGenerationMixin).

Regardless of your framework of choice, you can parameterize the generate method with a [GenerationConfig](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationConfig) class instance. Please refer to this class for the complete list of generation parameters, which control the behavior of the generation method.

To learn how to inspect a model’s generation configuration, what are the defaults, how to change the parameters ad hoc, and how to create and save a customized generation configuration, refer to the [text generation strategies guide](../generation_strategies). The guide also explains how to use related features, like token streaming.

# GenerationConfig

### class transformers.GenerationConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/configuration_utils.py#L40)

( \*\*kwargs )

Class that holds a configuration for a generation task. A `generate` call supports the following generation methods for text-decoder, text-to-text, speech-to-text, and vision-to-text models:

-   _greedy decoding_ by calling [greedy\_search()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.greedy_search) if `num_beams=1` and `do_sample=False`
-   _contrastive search_ by calling [contrastive\_search()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.contrastive_search) if `penalty_alpha>0.` and `top_k>1`
-   _multinomial sampling_ by calling [sample()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.sample) if `num_beams=1` and `do_sample=True`
-   _beam-search decoding_ by calling [beam\_search()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.beam_search) if `num_beams>1` and `do_sample=False`
-   _beam-search multinomial sampling_ by calling [beam\_sample()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.beam_sample) if `num_beams>1` and `do_sample=True`
-   _diverse beam-search decoding_ by calling [group\_beam\_search()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.group_beam_search), if `num_beams>1` and `num_beam_groups>1`
-   _constrained beam-search decoding_ by calling [constrained\_beam\_search()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.constrained_beam_search), if `constraints!=None` or `force_words_ids!=None`
-   _assisted decoding_ by calling `assisted_decoding()`, if `assistant_model` is passed to `.generate()`

You do not need to call any of the above methods directly. Pass custom parameter values to ‘.generate()‘. To learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).

#### from\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/configuration_utils.py#L555)

( pretrained\_model\_name: typing.Union\[str, os.PathLike\]config\_file\_name: typing.Union\[str, os.PathLike, NoneType\] = Nonecache\_dir: typing.Union\[str, os.PathLike, NoneType\] = Noneforce\_download: bool = Falselocal\_files\_only: bool = Falsetoken: typing.Union\[bool, str, NoneType\] = Nonerevision: str = 'main'\*\*kwargs ) → [GenerationConfig](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationConfig)

Instantiate a [GenerationConfig](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationConfig) from a generation configuration file.

Examples:

```
>>> from transformers import GenerationConfig

>>> 
>>> generation_config = GenerationConfig.from_pretrained("gpt2")

>>> 
>>> generation_config.save_pretrained("./test/saved_model/")
>>> generation_config = GenerationConfig.from_pretrained("./test/saved_model/")

>>> 
>>> generation_config.save_pretrained("./test/saved_model/", config_file_name="my_configuration.json")
>>> generation_config = GenerationConfig.from_pretrained("./test/saved_model/", "my_configuration.json")

>>> 
>>> 
>>> generation_config, unused_kwargs = GenerationConfig.from_pretrained(
...     "gpt2", top_k=1, foo=False, do_sample=True, return_unused_kwargs=True
... )
>>> generation_config.top_k
1

>>> unused_kwargs
{'foo': False}
```

#### from\_model\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/configuration_utils.py#L871)

( model\_config: PretrainedConfig ) → [GenerationConfig](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationConfig)

Parameters

-   **model\_config** (`PretrainedConfig`) — The model config that will be used to instantiate the generation config.

The configuration object instantiated from those parameters.

Instantiates a [GenerationConfig](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationConfig) from a [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig). This function is useful to convert legacy [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) objects, which may contain generation parameters, into a stand-alone [GenerationConfig](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationConfig).

#### save\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/configuration_utils.py#L477)

( save\_directory: typing.Union\[str, os.PathLike\]config\_file\_name: typing.Union\[str, os.PathLike, NoneType\] = Nonepush\_to\_hub: bool = False\*\*kwargs )

Parameters

-   **save\_directory** (`str` or `os.PathLike`) — Directory where the configuration JSON file will be saved (will be created if it does not exist).
-   **config\_file\_name** (`str` or `os.PathLike`, _optional_, defaults to `"generation_config.json"`) — Name of the generation configuration JSON file to be saved in `save_directory`.
-   **push\_to\_hub** (`bool`, _optional_, defaults to `False`) — Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with `repo_id` (will default to the name of `save_directory` in your namespace).
-   **kwargs** (`Dict[str, Any]`, _optional_) — Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/v4.34.0/en/main_classes/processors#transformers.ProcessorMixin.push_to_hub) method.

Save a generation configuration object to the directory `save_directory`, so that it can be re-loaded using the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationConfig.from_pretrained) class method.

## GenerationMixin

A class containing all functions for auto-regressive text generation, to be used as a mixin in [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel).

The class exposes [generate()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.generate), which can be used for:

-   _greedy decoding_ by calling [greedy\_search()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.greedy_search) if `num_beams=1` and `do_sample=False`
-   _contrastive search_ by calling [contrastive\_search()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.contrastive_search) if `penalty_alpha>0` and `top_k>1`
-   _multinomial sampling_ by calling [sample()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.sample) if `num_beams=1` and `do_sample=True`
-   _beam-search decoding_ by calling [beam\_search()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.beam_search) if `num_beams>1` and `do_sample=False`
-   _beam-search multinomial sampling_ by calling [beam\_sample()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.beam_sample) if `num_beams>1` and `do_sample=True`
-   _diverse beam-search decoding_ by calling [group\_beam\_search()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.group_beam_search), if `num_beams>1` and `num_beam_groups>1`
-   _constrained beam-search decoding_ by calling [constrained\_beam\_search()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.constrained_beam_search), if `constraints!=None` or `force_words_ids!=None`

You do not need to call any of the above methods directly. Pass custom parameter values to ‘generate’ instead. To learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).

#### generate

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/utils.py#L1300)

( inputs: typing.Optional\[torch.Tensor\] = Nonegeneration\_config: typing.Optional\[transformers.generation.configuration\_utils.GenerationConfig\] = Nonelogits\_processor: typing.Optional\[transformers.generation.logits\_process.LogitsProcessorList\] = Nonestopping\_criteria: typing.Optional\[transformers.generation.stopping\_criteria.StoppingCriteriaList\] = Noneprefix\_allowed\_tokens\_fn: typing.Union\[typing.Callable\[\[int, torch.Tensor\], typing.List\[int\]\], NoneType\] = Nonesynced\_gpus: typing.Optional\[bool\] = Noneassistant\_model: typing.Optional\[ForwardRef('PreTrainedModel')\] = Nonestreamer: typing.Optional\[ForwardRef('BaseStreamer')\] = Nonenegative\_prompt\_ids: typing.Optional\[torch.Tensor\] = Nonenegative\_prompt\_attention\_mask: typing.Optional\[torch.Tensor\] = None\*\*kwargs ) → [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) or `torch.LongTensor`

Generates sequences of token ids for models with a language modeling head.

Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the model’s default generation configuration. You can override any `generation_config` by passing the corresponding parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

For an overview of generation strategies and code examples, check out the [following guide](../generation_strategies).

#### compute\_transition\_scores

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/utils.py#L1058)

( sequences: Tensorscores: typing.Tuple\[torch.Tensor\]beam\_indices: typing.Optional\[torch.Tensor\] = Nonenormalize\_logits: bool = False ) → `torch.Tensor`

Computes the transition scores of sequences given the generation scores (and beam indices, if beam search was used). This is a convenient method to quicky obtain the scores of the selected tokens at generation time.

Examples:

```
>>> from transformers import GPT2Tokenizer, AutoModelForCausalLM
>>> import numpy as np

>>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("gpt2")
>>> tokenizer.pad_token_id = tokenizer.eos_token_id
>>> inputs = tokenizer(["Today is"], return_tensors="pt")

>>> 
>>> outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
>>> transition_scores = model.compute_transition_scores(
...     outputs.sequences, outputs.scores, normalize_logits=True
... )
>>> 
>>> 
>>> input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
>>> generated_tokens = outputs.sequences[:, input_length:]
>>> for tok, score in zip(generated_tokens[0], transition_scores[0]):
...     
...     print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
|   262 |  the     | -1.414 | 24.33%
|  1110 |  day     | -2.609 | 7.36%
|   618 |  when    | -2.010 | 13.40%
|   356 |  we      | -1.859 | 15.58%
|   460 |  can     | -2.508 | 8.14%

>>> 
>>> outputs = model.generate(
...     **inputs,
...     max_new_tokens=5,
...     num_beams=4,
...     num_return_sequences=4,
...     return_dict_in_generate=True,
...     output_scores=True,
... )
>>> transition_scores = model.compute_transition_scores(
...     outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
... )
>>> 
>>> 
>>> 
>>> output_length = input_length + np.sum(transition_scores.numpy() < 0, axis=1)
>>> length_penalty = model.generation_config.length_penalty
>>> reconstructed_scores = transition_scores.sum(axis=1) / (output_length**length_penalty)
>>> print(np.allclose(outputs.sequences_scores, reconstructed_scores))
True
```

#### greedy\_search

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/utils.py#L2286)

( input\_ids: LongTensorlogits\_processor: typing.Optional\[transformers.generation.logits\_process.LogitsProcessorList\] = Nonestopping\_criteria: typing.Optional\[transformers.generation.stopping\_criteria.StoppingCriteriaList\] = Nonemax\_length: typing.Optional\[int\] = Nonepad\_token\_id: typing.Optional\[int\] = Noneeos\_token\_id: typing.Union\[int, typing.List\[int\], NoneType\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneoutput\_scores: typing.Optional\[bool\] = Nonereturn\_dict\_in\_generate: typing.Optional\[bool\] = Nonesynced\_gpus: bool = Falsestreamer: typing.Optional\[ForwardRef('BaseStreamer')\] = None\*\*model\_kwargs )

Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

In most cases, you do not need to call [greedy\_search()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.greedy_search) directly. Use generate() instead. For an overview of generation strategies and code examples, check the [following guide](../generation_strategies).

Examples:

```
>>> from transformers import (
...     AutoTokenizer,
...     AutoModelForCausalLM,
...     LogitsProcessorList,
...     MinLengthLogitsProcessor,
...     StoppingCriteriaList,
...     MaxLengthCriteria,
... )

>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("gpt2")

>>> 
>>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

>>> input_prompt = "It might be possible to"
>>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

>>> 
>>> logits_processor = LogitsProcessorList(
...     [
...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
...     ]
... )
>>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

>>> outputs = model.greedy_search(
...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
... )

>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
["It might be possible to get a better understanding of the nature of the problem, but it's not"]
```

#### sample

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/utils.py#L2545)

( input\_ids: LongTensorlogits\_processor: typing.Optional\[transformers.generation.logits\_process.LogitsProcessorList\] = Nonestopping\_criteria: typing.Optional\[transformers.generation.stopping\_criteria.StoppingCriteriaList\] = Nonelogits\_warper: typing.Optional\[transformers.generation.logits\_process.LogitsProcessorList\] = Nonemax\_length: typing.Optional\[int\] = Nonepad\_token\_id: typing.Optional\[int\] = Noneeos\_token\_id: typing.Union\[int, typing.List\[int\], NoneType\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneoutput\_scores: typing.Optional\[bool\] = Nonereturn\_dict\_in\_generate: typing.Optional\[bool\] = Nonesynced\_gpus: bool = Falsestreamer: typing.Optional\[ForwardRef('BaseStreamer')\] = None\*\*model\_kwargs ) → [SampleDecoderOnlyOutput](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.SampleDecoderOnlyOutput), [SampleEncoderDecoderOutput](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.SampleEncoderDecoderOutput) or `torch.LongTensor`

Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

In most cases, you do not need to call [sample()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.sample) directly. Use generate() instead. For an overview of generation strategies and code examples, check the [following guide](../generation_strategies).

Examples:

```
>>> from transformers import (
...     AutoTokenizer,
...     AutoModelForCausalLM,
...     LogitsProcessorList,
...     MinLengthLogitsProcessor,
...     TopKLogitsWarper,
...     TemperatureLogitsWarper,
...     StoppingCriteriaList,
...     MaxLengthCriteria,
... )
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("gpt2")

>>> 
>>> model.config.pad_token_id = model.config.eos_token_id
>>> model.generation_config.pad_token_id = model.config.eos_token_id

>>> input_prompt = "Today is a beautiful day, and"
>>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

>>> 
>>> logits_processor = LogitsProcessorList(
...     [
...         MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
...     ]
... )
>>> 
>>> logits_warper = LogitsProcessorList(
...     [
...         TopKLogitsWarper(50),
...         TemperatureLogitsWarper(0.7),
...     ]
... )

>>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

>>> torch.manual_seed(0)
>>> outputs = model.sample(
...     input_ids,
...     logits_processor=logits_processor,
...     logits_warper=logits_warper,
...     stopping_criteria=stopping_criteria,
... )

>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Today is a beautiful day, and we must do everything possible to make it a day of celebration.']
```

#### beam\_search

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/utils.py#L2827)

( input\_ids: LongTensorbeam\_scorer: BeamScorerlogits\_processor: typing.Optional\[transformers.generation.logits\_process.LogitsProcessorList\] = Nonestopping\_criteria: typing.Optional\[transformers.generation.stopping\_criteria.StoppingCriteriaList\] = Nonemax\_length: typing.Optional\[int\] = Nonepad\_token\_id: typing.Optional\[int\] = Noneeos\_token\_id: typing.Union\[int, typing.List\[int\], NoneType\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneoutput\_scores: typing.Optional\[bool\] = Nonereturn\_dict\_in\_generate: typing.Optional\[bool\] = Nonesynced\_gpus: bool = False\*\*model\_kwargs )

Generates sequences of token ids for models with a language modeling head using **beam search decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

In most cases, you do not need to call [beam\_search()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.beam_search) directly. Use generate() instead. For an overview of generation strategies and code examples, check the [following guide](../generation_strategies).

Examples:

```
>>> from transformers import (
...     AutoTokenizer,
...     AutoModelForSeq2SeqLM,
...     LogitsProcessorList,
...     MinLengthLogitsProcessor,
...     BeamSearchScorer,
... )
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

>>> encoder_input_str = "translate English to German: How old are you?"
>>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


>>> 
>>> num_beams = 3
>>> 
>>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
>>> input_ids = input_ids * model.config.decoder_start_token_id

>>> 
>>> model_kwargs = {
...     "encoder_outputs": model.get_encoder()(
...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
...     )
... }

>>> 
>>> beam_scorer = BeamSearchScorer(
...     batch_size=1,
...     num_beams=num_beams,
...     device=model.device,
... )

>>> 
>>> logits_processor = LogitsProcessorList(
...     [
...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
...     ]
... )

>>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Wie alt bist du?']
```

#### beam\_sample

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/utils.py#L3150)

( input\_ids: LongTensorbeam\_scorer: BeamScorerlogits\_processor: typing.Optional\[transformers.generation.logits\_process.LogitsProcessorList\] = Nonestopping\_criteria: typing.Optional\[transformers.generation.stopping\_criteria.StoppingCriteriaList\] = Nonelogits\_warper: typing.Optional\[transformers.generation.logits\_process.LogitsProcessorList\] = Nonemax\_length: typing.Optional\[int\] = Nonepad\_token\_id: typing.Optional\[int\] = Noneeos\_token\_id: typing.Union\[int, typing.List\[int\], NoneType\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneoutput\_scores: typing.Optional\[bool\] = Nonereturn\_dict\_in\_generate: typing.Optional\[bool\] = Nonesynced\_gpus: bool = False\*\*model\_kwargs )

Generates sequences of token ids for models with a language modeling head using **beam search multinomial sampling** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

In most cases, you do not need to call [beam\_sample()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.beam_sample) directly. Use generate() instead. For an overview of generation strategies and code examples, check the [following guide](../generation_strategies).

Examples:

```
>>> from transformers import (
...     AutoTokenizer,
...     AutoModelForSeq2SeqLM,
...     LogitsProcessorList,
...     MinLengthLogitsProcessor,
...     TopKLogitsWarper,
...     TemperatureLogitsWarper,
...     BeamSearchScorer,
... )
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

>>> encoder_input_str = "translate English to German: How old are you?"
>>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

>>> 
>>> num_beams = 3
>>> 
>>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
>>> input_ids = input_ids * model.config.decoder_start_token_id

>>> 
>>> model_kwargs = {
...     "encoder_outputs": model.get_encoder()(
...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
...     )
... }

>>> 
>>> beam_scorer = BeamSearchScorer(
...     batch_size=1,
...     max_length=model.config.max_length,
...     num_beams=num_beams,
...     device=model.device,
... )

>>> 
>>> logits_processor = LogitsProcessorList(
...     [MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id)]
... )
>>> 
>>> logits_warper = LogitsProcessorList(
...     [
...         TopKLogitsWarper(50),
...         TemperatureLogitsWarper(0.7),
...     ]
... )

>>> outputs = model.beam_sample(
...     input_ids, beam_scorer, logits_processor=logits_processor, logits_warper=logits_warper, **model_kwargs
... )

>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Wie alt bist du?']
```

#### contrastive\_search

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/utils.py#L1842)

( input\_ids: LongTensortop\_k: typing.Optional\[int\] = 1penalty\_alpha: typing.Optional\[float\] = 0logits\_processor: typing.Optional\[transformers.generation.logits\_process.LogitsProcessorList\] = Nonelogits\_warper: typing.Optional\[transformers.generation.logits\_process.LogitsProcessorList\] = Nonestopping\_criteria: typing.Optional\[transformers.generation.stopping\_criteria.StoppingCriteriaList\] = Nonepad\_token\_id: typing.Optional\[int\] = Noneeos\_token\_id: typing.Union\[int, typing.List\[int\], NoneType\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneoutput\_scores: typing.Optional\[bool\] = Nonereturn\_dict\_in\_generate: typing.Optional\[bool\] = Nonesynced\_gpus: bool = Falsestreamer: typing.Optional\[ForwardRef('BaseStreamer')\] = Nonesequential: typing.Optional\[bool\] = None\*\*model\_kwargs )

Generates sequences of token ids for models with a language modeling head using **contrastive search** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

In most cases, you do not need to call [contrastive\_search()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.contrastive_search) directly. Use generate() instead. For an overview of generation strategies and code examples, check the [following guide](../generation_strategies).

Examples:

```
>>> from transformers import (
...     AutoTokenizer,
...     AutoModelForCausalLM,
...     StoppingCriteriaList,
...     MaxLengthCriteria,
... )

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
>>> model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
>>> 
>>> model.config.pad_token_id = model.config.eos_token_id
>>> input_prompt = "DeepMind Company is"
>>> input_ids = tokenizer(input_prompt, return_tensors="pt")
>>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=64)])
>>> outputs = model.contrastive_search(
...     **input_ids, penalty_alpha=0.6, top_k=4, stopping_criteria=stopping_criteria
... )
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['DeepMind Company is a company that focuses on the development and commercialization of artificial intelligence (AI). DeepMind’s mission is to help people understand and solve problems that are difficult to solve in the world today.\n\nIn this post, we talk about the benefits of deep learning in business and how it']
```

#### group\_beam\_search

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/utils.py#L3482)

( input\_ids: LongTensorbeam\_scorer: BeamScorerlogits\_processor: typing.Optional\[transformers.generation.logits\_process.LogitsProcessorList\] = Nonestopping\_criteria: typing.Optional\[transformers.generation.stopping\_criteria.StoppingCriteriaList\] = Nonemax\_length: typing.Optional\[int\] = Nonepad\_token\_id: typing.Optional\[int\] = Noneeos\_token\_id: typing.Union\[int, typing.List\[int\], NoneType\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneoutput\_scores: typing.Optional\[bool\] = Nonereturn\_dict\_in\_generate: typing.Optional\[bool\] = Nonesynced\_gpus: bool = False\*\*model\_kwargs )

Generates sequences of token ids for models with a language modeling head using **diverse beam search decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

In most cases, you do not need to call [group\_beam\_search()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.group_beam_search) directly. Use generate() instead. For an overview of generation strategies and code examples, check the [following guide](../generation_strategies).

Examples:

```
>>> from transformers import (
...     AutoTokenizer,
...     AutoModelForSeq2SeqLM,
...     LogitsProcessorList,
...     MinLengthLogitsProcessor,
...     HammingDiversityLogitsProcessor,
...     BeamSearchScorer,
... )
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

>>> encoder_input_str = "translate English to German: How old are you?"
>>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


>>> 
>>> num_beams = 6
>>> 
>>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
>>> input_ids = input_ids * model.config.decoder_start_token_id

>>> 
>>> model_kwargs = {
...     "encoder_outputs": model.get_encoder()(
...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
...     )
... }

>>> 
>>> beam_scorer = BeamSearchScorer(
...     batch_size=1,
...     max_length=model.config.max_length,
...     num_beams=num_beams,
...     device=model.device,
...     num_beam_groups=3,
... )

>>> 
>>> logits_processor = LogitsProcessorList(
...     [
...         HammingDiversityLogitsProcessor(5.5, num_beams=6, num_beam_groups=3),
...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
...     ]
... )

>>> outputs = model.group_beam_search(
...     input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs
... )

>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Wie alt bist du?']
```

#### constrained\_beam\_search

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/utils.py#L3861)

( input\_ids: LongTensorconstrained\_beam\_scorer: ConstrainedBeamSearchScorerlogits\_processor: typing.Optional\[transformers.generation.logits\_process.LogitsProcessorList\] = Nonestopping\_criteria: typing.Optional\[transformers.generation.stopping\_criteria.StoppingCriteriaList\] = Nonemax\_length: typing.Optional\[int\] = Nonepad\_token\_id: typing.Optional\[int\] = Noneeos\_token\_id: typing.Union\[int, typing.List\[int\], NoneType\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneoutput\_scores: typing.Optional\[bool\] = Nonereturn\_dict\_in\_generate: typing.Optional\[bool\] = Nonesynced\_gpus: typing.Optional\[bool\] = None\*\*model\_kwargs )

Generates sequences of token ids for models with a language modeling head using **constrained beam search decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

In most cases, you do not need to call [constrained\_beam\_search()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.constrained_beam_search) directly. Use generate() instead. For an overview of generation strategies and code examples, check the [following guide](../generation_strategies).

Examples:

```
>>> from transformers import (
...     AutoTokenizer,
...     AutoModelForSeq2SeqLM,
...     LogitsProcessorList,
...     MinLengthLogitsProcessor,
...     ConstrainedBeamSearchScorer,
...     PhrasalConstraint,
... )
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

>>> encoder_input_str = "translate English to German: How old are you?"
>>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


>>> 
>>> num_beams = 3
>>> 
>>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
>>> input_ids = input_ids * model.config.decoder_start_token_id

>>> 
>>> model_kwargs = {
...     "encoder_outputs": model.get_encoder()(
...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
...     )
... }

>>> constraint_str = "Sie"
>>> constraint_token_ids = tokenizer.encode(constraint_str)[:-1]  
>>> constraints = [PhrasalConstraint(token_ids=constraint_token_ids)]


>>> 
>>> beam_scorer = ConstrainedBeamSearchScorer(
...     batch_size=1, num_beams=num_beams, device=model.device, constraints=constraints
... )

>>> 
>>> logits_processor = LogitsProcessorList(
...     [
...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
...     ]
... )

>>> outputs = model.constrained_beam_search(
...     input_ids, beam_scorer, constraints=constraints, logits_processor=logits_processor, **model_kwargs
... )

>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Wie alt sind Sie?']
```

## TFGenerationMixin

### class transformers.TFGenerationMixin

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_utils.py#L444)

( )

A class containing all of the functions supporting generation, to be used as a mixin in [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel).

The class exposes [generate()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.TFGenerationMixin.generate), which can be used for:

-   _greedy decoding_ by calling `greedy_search()` if `num_beams=1` and `do_sample=False`
-   _contrastive search_ by calling `contrastive_search()` if `penalty_alpha>0` and `top_k>1`
-   _multinomial sampling_ by calling `sample()` if `num_beams=1` and `do_sample=True`
-   _beam-search decoding_ by calling `beam_search()` if `num_beams>1`

You do not need to call any of the above methods directly. Pass custom parameter values to ‘generate’ instead. To learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).

#### generate

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_utils.py#L645)

( inputs: typing.Optional\[tensorflow.python.framework.ops.Tensor\] = Nonegeneration\_config: typing.Optional\[transformers.generation.configuration\_utils.GenerationConfig\] = Nonelogits\_processor: typing.Optional\[transformers.generation.tf\_logits\_process.TFLogitsProcessorList\] = Noneseed = None\*\*kwargs ) → [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) or `tf.Tensor`

Generates sequences of token ids for models with a language modeling head.

Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the model’s default generation configuration. You can override any `generation_config` by passing the corresponding parameters to generate, e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

For an overview of generation strategies and code examples, check out the [following guide](../generation_strategies).

#### compute\_transition\_scores

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_utils.py#L477)

( sequences: Tensorscores: typing.Tuple\[tensorflow.python.framework.ops.Tensor\]beam\_indices: typing.Optional\[tensorflow.python.framework.ops.Tensor\] = Nonenormalize\_logits: bool = False ) → `tf.Tensor`

Computes the transition scores of sequences given the generation scores (and beam indices, if beam search was used). This is a convenient method to quicky obtain the scores of the selected tokens at generation time.

Examples:

```
>>> from transformers import GPT2Tokenizer, TFAutoModelForCausalLM
>>> import numpy as np

>>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
>>> model = TFAutoModelForCausalLM.from_pretrained("gpt2")
>>> tokenizer.pad_token_id = tokenizer.eos_token_id
>>> inputs = tokenizer(["Today is"], return_tensors="tf")

>>> 
>>> outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
>>> transition_scores = model.compute_transition_scores(
...     outputs.sequences, outputs.scores, normalize_logits=True
... )
>>> 
>>> 
>>> input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
>>> generated_tokens = outputs.sequences[:, input_length:]
>>> for tok, score in zip(generated_tokens[0], transition_scores[0]):
...     
...     print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
|   262 |  the     | -1.413 | 24.33%
|  1110 |  day     | -2.609 | 7.36%
|   618 |  when    | -2.009 | 13.41%
|   356 |  we      | -1.859 | 15.58%
|   460 |  can     | -2.508 | 8.14%

>>> 
>>> outputs = model.generate(
...     **inputs,
...     max_new_tokens=5,
...     num_beams=4,
...     num_return_sequences=4,
...     return_dict_in_generate=True,
...     output_scores=True,
... )
>>> transition_scores = model.compute_transition_scores(
...     outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
... )
>>> 
>>> 
>>> 
>>> output_length = input_length + np.sum(transition_scores.numpy() < 0, axis=1)
>>> length_penalty = model.generation_config.length_penalty
>>> reconstructed_scores = np.sum(transition_scores, axis=1) / (output_length**length_penalty)
>>> print(np.allclose(outputs.sequences_scores, reconstructed_scores))
True
```

## FlaxGenerationMixin

### class transformers.FlaxGenerationMixin

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_utils.py#L129)

( )

A class containing all functions for auto-regressive text generation, to be used as a mixin in [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel).

The class exposes [generate()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.FlaxGenerationMixin.generate), which can be used for:

-   _greedy decoding_ by calling `_greedy_search()` if `num_beams=1` and `do_sample=False`
-   _multinomial sampling_ by calling `_sample()` if `num_beams=1` and `do_sample=True`
-   _beam-search decoding_ by calling `_beam_search()` if `num_beams>1` and `do_sample=False`

You do not need to call any of the above methods directly. Pass custom parameter values to ‘generate’ instead. To learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).

#### generate

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_utils.py#L267)

( input\_ids: Arraygeneration\_config: typing.Optional\[transformers.generation.configuration\_utils.GenerationConfig\] = Noneprng\_key: typing.Optional\[jax.Array\] = Nonetrace: bool = Trueparams: typing.Union\[typing.Dict\[str, jax.Array\], NoneType\] = Nonelogits\_processor: typing.Optional\[transformers.generation.flax\_logits\_process.FlaxLogitsProcessorList\] = None\*\*kwargs )

Generates sequences of token ids for models with a language modeling head.