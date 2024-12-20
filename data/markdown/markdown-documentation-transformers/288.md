# UniSpeech

## Overview

The UniSpeech model was proposed in [UniSpeech: Unified Speech Representation Learning with Labeled and Unlabeled Data](https://arxiv.org/abs/2101.07597) by Chengyi Wang, Yu Wu, Yao Qian, Kenichi Kumatani, Shujie Liu, Furu Wei, Michael Zeng, Xuedong Huang .

The abstract from the paper is the following:

_In this paper, we propose a unified pre-training approach called UniSpeech to learn speech representations with both unlabeled and labeled data, in which supervised phonetic CTC learning and phonetically-aware contrastive self-supervised learning are conducted in a multi-task learning manner. The resultant representations can capture information more correlated with phonetic structures and improve the generalization across languages and domains. We evaluate the effectiveness of UniSpeech for cross-lingual representation learning on public CommonVoice corpus. The results show that UniSpeech outperforms self-supervised pretraining and supervised transfer learning for speech recognition by a maximum of 13.4% and 17.8% relative phone error rate reductions respectively (averaged over all testing languages). The transferability of UniSpeech is also demonstrated on a domain-shift speech recognition task, i.e., a relative word error rate reduction of 6% against the previous approach._

Tips:

-   UniSpeech is a speech model that accepts a float array corresponding to the raw waveform of the speech signal. Please use [Wav2Vec2Processor](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) for the feature extraction.
-   UniSpeech model can be fine-tuned using connectionist temporal classification (CTC) so the model output has to be decoded using [Wav2Vec2CTCTokenizer](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer).

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten). The Authors’ code can be found [here](https://github.com/microsoft/UniSpeech/tree/main/UniSpeech).

## Documentation resources

-   [Audio classification task guide](../tasks/audio_classification)
-   [Automatic speech recognition task guide](../tasks/asr)

## UniSpeechConfig

### class transformers.UniSpeechConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/unispeech/configuration_unispeech.py#L34)

( vocab\_size = 32hidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout = 0.1activation\_dropout = 0.1attention\_dropout = 0.1feat\_proj\_dropout = 0.0feat\_quantizer\_dropout = 0.0final\_dropout = 0.1layerdrop = 0.1initializer\_range = 0.02layer\_norm\_eps = 1e-05feat\_extract\_norm = 'group'feat\_extract\_activation = 'gelu'conv\_dim = (512, 512, 512, 512, 512, 512, 512)conv\_stride = (5, 2, 2, 2, 2, 2, 2)conv\_kernel = (10, 3, 3, 3, 3, 2, 2)conv\_bias = Falsenum\_conv\_pos\_embeddings = 128num\_conv\_pos\_embedding\_groups = 16do\_stable\_layer\_norm = Falseapply\_spec\_augment = Truemask\_time\_prob = 0.05mask\_time\_length = 10mask\_time\_min\_masks = 2mask\_feature\_prob = 0.0mask\_feature\_length = 10mask\_feature\_min\_masks = 0num\_codevectors\_per\_group = 320num\_codevector\_groups = 2contrastive\_logits\_temperature = 0.1num\_negatives = 100codevector\_dim = 256proj\_codevector\_dim = 256diversity\_loss\_weight = 0.1ctc\_loss\_reduction = 'mean'ctc\_zero\_infinity = Falseuse\_weighted\_layer\_sum = Falseclassifier\_proj\_size = 256num\_ctc\_classes = 80pad\_token\_id = 0bos\_token\_id = 1eos\_token\_id = 2replace\_prob = 0.5\*\*kwargs )

This is the configuration class to store the configuration of a [UniSpeechModel](/docs/transformers/v4.34.0/en/model_doc/unispeech#transformers.UniSpeechModel). It is used to instantiate an UniSpeech model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the UniSpeech [microsoft/unispeech-large-1500h-cv](https://huggingface.co/microsoft/unispeech-large-1500h-cv) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import UniSpeechConfig, UniSpeechModel

>>> 
>>> configuration = UniSpeechConfig()

>>> 
>>> model = UniSpeechModel(configuration)

>>> 
>>> configuration = model.config
```

## UniSpeech specific outputs

### class transformers.models.unispeech.modeling\_unispeech.UniSpeechForPreTrainingOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/unispeech/modeling_unispeech.py#L67)

( loss: typing.Optional\[torch.FloatTensor\] = Noneprojected\_states: FloatTensor = Noneprojected\_quantized\_states: FloatTensor = Nonecodevector\_perplexity: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Output type of `UniSpeechForPreTrainingOutput`, with potential hidden states and attentions.

## UniSpeechModel

### class transformers.UniSpeechModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/unispeech/modeling_unispeech.py#L1107)

( config: UniSpeechConfig )

Parameters

-   **config** ([UniSpeechConfig](/docs/transformers/v4.34.0/en/model_doc/unispeech#transformers.UniSpeechConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare UniSpeech Model transformer outputting raw hidden-states without any specific head on top. UniSpeech was proposed in [UniSpeech: Unified Speech Representation Learning with Labeled and Unlabeled Data](https://arxiv.org/abs/2101.07597) by Chengyi Wang, Yu Wu, Yao Qian, Kenichi Kumatani, Shujie Liu, Furu Wei, Michael Zeng, Xuedong Huang.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving etc.).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/unispeech/modeling_unispeech.py#L1172)

( input\_values: typing.Optional\[torch.Tensor\]attention\_mask: typing.Optional\[torch.Tensor\] = Nonemask\_time\_indices: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Wav2Vec2BaseModelOutput](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.modeling_outputs.Wav2Vec2BaseModelOutput) or `tuple(torch.FloatTensor)`

The [UniSpeechModel](/docs/transformers/v4.34.0/en/model_doc/unispeech#transformers.UniSpeechModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoProcessor, UniSpeechModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = AutoProcessor.from_pretrained("patrickvonplaten/unispeech-large-1500h-cv-timit")
>>> model = UniSpeechModel.from_pretrained("patrickvonplaten/unispeech-large-1500h-cv-timit")

>>> 
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 292, 1024]
```

## UniSpeechForCTC

### class transformers.UniSpeechForCTC

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/unispeech/modeling_unispeech.py#L1379)

( configtarget\_lang: typing.Optional\[str\] = None )

Parameters

-   **config** ([UniSpeechConfig](/docs/transformers/v4.34.0/en/model_doc/unispeech#transformers.UniSpeechConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

UniSpeech Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC). UniSpeech was proposed in [UniSpeech: Unified Speech Representation Learning with Labeled and Unlabeled Data](https://arxiv.org/abs/2101.07597) by Chengyi Wang, Yu Wu, Yao Qian, Kenichi Kumatani, Shujie Liu, Furu Wei, Michael Zeng, Xuedong Huang.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving etc.).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/unispeech/modeling_unispeech.py#L1451)

( input\_values: typing.Optional\[torch.Tensor\]attention\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.Tensor\] = None ) → [transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)`

The [UniSpeechForCTC](/docs/transformers/v4.34.0/en/model_doc/unispeech#transformers.UniSpeechForCTC) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoProcessor, UniSpeechForCTC
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = AutoProcessor.from_pretrained("patrickvonplaten/unispeech-large-1500h-cv-timit")
>>> model = UniSpeechForCTC.from_pretrained("patrickvonplaten/unispeech-large-1500h-cv-timit")

>>> 
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
>>> predicted_ids = torch.argmax(logits, dim=-1)

>>> 
>>> transcription = processor.batch_decode(predicted_ids)
>>> transcription[0]
'mister quilter is the apposl of the midle classes and weare glad to welcom his gosepl'

>>> inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="pt").input_ids

>>> 
>>> loss = model(**inputs).loss
>>> round(loss.item(), 2)
17.17
```

## UniSpeechForSequenceClassification

### class transformers.UniSpeechForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/unispeech/modeling_unispeech.py#L1538)

( config )

Parameters

-   **config** ([UniSpeechConfig](/docs/transformers/v4.34.0/en/model_doc/unispeech#transformers.UniSpeechConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

UniSpeech Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like SUPERB Keyword Spotting.

UniSpeech was proposed in [UniSpeech: Unified Speech Representation Learning with Labeled and Unlabeled Data](https://arxiv.org/abs/2101.07597) by Chengyi Wang, Yu Wu, Yao Qian, Kenichi Kumatani, Shujie Liu, Furu Wei, Michael Zeng, Xuedong Huang.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving etc.).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/unispeech/modeling_unispeech.py#L1586)

( input\_values: typing.Optional\[torch.Tensor\]attention\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.Tensor\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [UniSpeechForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/unispeech#transformers.UniSpeechForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoFeatureExtractor, UniSpeechForSequenceClassification
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("patrickvonplaten/unispeech-large-1500h-cv-timit")
>>> model = UniSpeechForSequenceClassification.from_pretrained("patrickvonplaten/unispeech-large-1500h-cv-timit")

>>> 
>>> inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.argmax(logits, dim=-1).item()
>>> predicted_label = model.config.id2label[predicted_class_ids]

>>> 
>>> target_label = model.config.id2label[0]
>>> inputs["labels"] = torch.tensor([model.config.label2id[target_label]])
>>> loss = model(**inputs).loss
```

## UniSpeechForPreTraining

### class transformers.UniSpeechForPreTraining

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/unispeech/modeling_unispeech.py#L1231)

( config: UniSpeechConfig )

Parameters

-   **config** ([UniSpeechConfig](/docs/transformers/v4.34.0/en/model_doc/unispeech#transformers.UniSpeechConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

UniSpeech Model with a vector-quantization module and ctc loss for pre-training. UniSpeech was proposed in [UniSpeech: Unified Speech Representation Learning with Labeled and Unlabeled Data](https://arxiv.org/abs/2101.07597) by Chengyi Wang, Yu Wu, Yao Qian, Kenichi Kumatani, Shujie Liu, Furu Wei, Michael Zeng, Xuedong Huang.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving etc.).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

The [UniSpeechForPreTraining](/docs/transformers/v4.34.0/en/model_doc/unispeech#transformers.UniSpeechForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoFeatureExtractor, UniSpeechForPreTraining

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/unispeech-large-1500h-cv")
>>> model = UniSpeechForPreTraining.from_pretrained("microsoft/unispeech-large-1500h-cv")
>>> 
```