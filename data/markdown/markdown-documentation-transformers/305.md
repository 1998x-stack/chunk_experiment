# Wav2Vec2-Conformer

## Overview

The Wav2Vec2-Conformer was added to an updated version of [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://arxiv.org/abs/2010.05171) by Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Sravya Popuri, Dmytro Okhonko, Juan Pino.

The official results of the model can be found in Table 3 and Table 4 of the paper.

The Wav2Vec2-Conformer weights were released by the Meta AI team within the [Fairseq library](https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/README.md#pre-trained-models).

Tips:

-   Wav2Vec2-Conformer follows the same architecture as Wav2Vec2, but replaces the _Attention_\-block with a _Conformer_\-block as introduced in [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100).
-   For the same number of layers, Wav2Vec2-Conformer requires more parameters than Wav2Vec2, but also yields an improved word error rate.
-   Wav2Vec2-Conformer uses the same tokenizer and feature extractor as Wav2Vec2.
-   Wav2Vec2-Conformer can use either no relative position embeddings, Transformer-XL-like position embeddings, or rotary position embeddings by setting the correct `config.position_embeddings_type`.

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten). The original code can be found [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec).

## Documentation resources

-   [Audio classification task guide](../tasks/audio_classification)
-   [Automatic speech recognition task guide](../tasks/asr)

## Wav2Vec2ConformerConfig

### class transformers.Wav2Vec2ConformerConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2_conformer/configuration_wav2vec2_conformer.py#L33)

( vocab\_size = Nonehidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout = 0.1activation\_dropout = 0.1attention\_dropout = 0.1feat\_proj\_dropout = 0.0feat\_quantizer\_dropout = 0.0final\_dropout = 0.1layerdrop = 0.1initializer\_range = 0.02layer\_norm\_eps = 1e-05feat\_extract\_norm = 'group'feat\_extract\_activation = 'gelu'conv\_dim = (512, 512, 512, 512, 512, 512, 512)conv\_stride = (5, 2, 2, 2, 2, 2, 2)conv\_kernel = (10, 3, 3, 3, 3, 2, 2)conv\_bias = Falsenum\_conv\_pos\_embeddings = 128num\_conv\_pos\_embedding\_groups = 16apply\_spec\_augment = Truemask\_time\_prob = 0.05mask\_time\_length = 10mask\_time\_min\_masks = 2mask\_feature\_prob = 0.0mask\_feature\_length = 10mask\_feature\_min\_masks = 0num\_codevectors\_per\_group = 320num\_codevector\_groups = 2contrastive\_logits\_temperature = 0.1num\_negatives = 100codevector\_dim = 256proj\_codevector\_dim = 256diversity\_loss\_weight = 0.1ctc\_loss\_reduction = 'sum'ctc\_zero\_infinity = Falseuse\_weighted\_layer\_sum = Falseclassifier\_proj\_size = 256tdnn\_dim = (512, 512, 512, 512, 1500)tdnn\_kernel = (5, 3, 3, 1, 1)tdnn\_dilation = (1, 2, 3, 1, 1)xvector\_output\_dim = 512pad\_token\_id = 0bos\_token\_id = 1eos\_token\_id = 2add\_adapter = Falseadapter\_kernel\_size = 3adapter\_stride = 2num\_adapter\_layers = 3output\_hidden\_size = Noneposition\_embeddings\_type = 'relative'rotary\_embedding\_base = 10000max\_source\_positions = 5000conv\_depthwise\_kernel\_size = 31conformer\_conv\_dropout = 0.1\*\*kwargs )

This is the configuration class to store the configuration of a [Wav2Vec2ConformerModel](/docs/transformers/v4.34.0/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerModel). It is used to instantiate an Wav2Vec2Conformer model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Wav2Vec2Conformer [facebook/wav2vec2-conformer-rel-pos-large](https://huggingface.co/facebook/wav2vec2-conformer-rel-pos-large) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import Wav2Vec2ConformerConfig, Wav2Vec2ConformerModel

>>> 
>>> configuration = Wav2Vec2ConformerConfig()

>>> 
>>> model = Wav2Vec2ConformerModel(configuration)

>>> 
>>> configuration = model.config
```

## Wav2Vec2Conformer specific outputs

### class transformers.models.wav2vec2\_conformer.modeling\_wav2vec2\_conformer.Wav2Vec2ConformerForPreTrainingOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py#L74)

( loss: typing.Optional\[torch.FloatTensor\] = Noneprojected\_states: FloatTensor = Noneprojected\_quantized\_states: FloatTensor = Nonecodevector\_perplexity: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecontrastive\_loss: typing.Optional\[torch.FloatTensor\] = Nonediversity\_loss: typing.Optional\[torch.FloatTensor\] = None )

Output type of [Wav2Vec2ConformerForPreTraining](/docs/transformers/v4.34.0/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerForPreTraining), with potential hidden states and attentions.

## Wav2Vec2ConformerModel

### class transformers.Wav2Vec2ConformerModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py#L1247)

( config: Wav2Vec2ConformerConfig )

Parameters

-   **config** ([Wav2Vec2ConformerConfig](/docs/transformers/v4.34.0/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Wav2Vec2Conformer Model transformer outputting raw hidden-states without any specific head on top. Wav2Vec2Conformer was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving etc.).

This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py#L1320)

( input\_values: typing.Optional\[torch.Tensor\]attention\_mask: typing.Optional\[torch.Tensor\] = Nonemask\_time\_indices: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Wav2Vec2BaseModelOutput](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.modeling_outputs.Wav2Vec2BaseModelOutput) or `tuple(torch.FloatTensor)`

The [Wav2Vec2ConformerModel](/docs/transformers/v4.34.0/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoProcessor, Wav2Vec2ConformerModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
>>> model = Wav2Vec2ConformerModel.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")

>>> 
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 292, 1024]
```

## Wav2Vec2ConformerForCTC

### class transformers.Wav2Vec2ConformerForCTC

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py#L1606)

( configtarget\_lang: typing.Optional\[str\] = None )

Parameters

-   **config** ([Wav2Vec2ConformerConfig](/docs/transformers/v4.34.0/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Wav2Vec2Conformer Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC). Wav2Vec2Conformer was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving etc.).

This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py#L1639)

( input\_values: typing.Optional\[torch.Tensor\]attention\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.Tensor\] = None ) → [transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)`

The [Wav2Vec2ConformerForCTC](/docs/transformers/v4.34.0/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerForCTC) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoProcessor, Wav2Vec2ConformerForCTC
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
>>> model = Wav2Vec2ConformerForCTC.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")

>>> 
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
>>> predicted_ids = torch.argmax(logits, dim=-1)

>>> 
>>> transcription = processor.batch_decode(predicted_ids)
>>> transcription[0]
'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'

>>> inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="pt").input_ids

>>> 
>>> loss = model(**inputs).loss
>>> round(loss.item(), 2)
64.21
```

## Wav2Vec2ConformerForSequenceClassification

### class transformers.Wav2Vec2ConformerForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py#L1727)

( config )

Parameters

-   **config** ([Wav2Vec2ConformerConfig](/docs/transformers/v4.34.0/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Wav2Vec2Conformer Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like SUPERB Keyword Spotting.

Wav2Vec2Conformer was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving etc.).

This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py#L1762)

( input\_values: typing.Optional\[torch.Tensor\]attention\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.Tensor\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [Wav2Vec2ConformerForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoFeatureExtractor, Wav2Vec2ConformerForSequenceClassification
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
>>> model = Wav2Vec2ConformerForSequenceClassification.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")

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

## Wav2Vec2ConformerForAudioFrameClassification

### class transformers.Wav2Vec2ConformerForAudioFrameClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py#L1838)

( config )

Parameters

-   **config** ([Wav2Vec2ConformerConfig](/docs/transformers/v4.34.0/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Wav2Vec2Conformer Model with a frame classification head on top for tasks like Speaker Diarization.

Wav2Vec2Conformer was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving etc.).

This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py#L1873)

( input\_values: typing.Optional\[torch.Tensor\]attention\_mask: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

The [Wav2Vec2ConformerForAudioFrameClassification](/docs/transformers/v4.34.0/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerForAudioFrameClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoFeatureExtractor, Wav2Vec2ConformerForAudioFrameClassification
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
>>> model = Wav2Vec2ConformerForAudioFrameClassification.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")

>>> 
>>> inputs = feature_extractor(dataset[0]["audio"]["array"], return_tensors="pt", sampling_rate=sampling_rate)
>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> probabilities = torch.sigmoid(logits[0])
>>> 
>>> labels = (probabilities > 0.5).long()
```

## Wav2Vec2ConformerForXVector

### class transformers.Wav2Vec2ConformerForXVector

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py#L1992)

( config )

Parameters

-   **config** ([Wav2Vec2ConformerConfig](/docs/transformers/v4.34.0/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Wav2Vec2Conformer Model with an XVector feature extraction head on top for tasks like Speaker Verification.

Wav2Vec2Conformer was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving etc.).

This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py#L2045)

( input\_values: typing.Optional\[torch.Tensor\]attention\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.Tensor\] = None ) → [transformers.modeling\_outputs.XVectorOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.XVectorOutput) or `tuple(torch.FloatTensor)`

The [Wav2Vec2ConformerForXVector](/docs/transformers/v4.34.0/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerForXVector) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoFeatureExtractor, Wav2Vec2ConformerForXVector
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
>>> model = Wav2Vec2ConformerForXVector.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")

>>> 
>>> inputs = feature_extractor(
...     [d["array"] for d in dataset[:2]["audio"]], sampling_rate=sampling_rate, return_tensors="pt", padding=True
... )
>>> with torch.no_grad():
...     embeddings = model(**inputs).embeddings

>>> embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

>>> 
>>> cosine_sim = torch.nn.CosineSimilarity(dim=-1)
>>> similarity = cosine_sim(embeddings[0], embeddings[1])
>>> threshold = 0.7  
>>> if similarity < threshold:
...     print("Speakers are not the same!")
```

## Wav2Vec2ConformerForPreTraining

### class transformers.Wav2Vec2ConformerForPreTraining

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py#L1385)

( config: Wav2Vec2ConformerConfig )

Parameters

-   **config** ([Wav2Vec2ConformerConfig](/docs/transformers/v4.34.0/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Wav2Vec2Conformer Model with a quantizer and `VQ` head on top. Wav2Vec2Conformer was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving etc.).

This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

The [Wav2Vec2ConformerForPreTraining](/docs/transformers/v4.34.0/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoFeatureExtractor, Wav2Vec2ConformerForPreTraining
>>> from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
...     _compute_mask_indices,
...     _sample_negative_indices,
... )
>>> from datasets import load_dataset

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large")
>>> model = Wav2Vec2ConformerForPreTraining.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large")

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> input_values = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt").input_values  

>>> 
>>> batch_size, raw_sequence_length = input_values.shape
>>> sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length).item()
>>> mask_time_indices = _compute_mask_indices(
...     shape=(batch_size, sequence_length), mask_prob=0.2, mask_length=2
... )
>>> sampled_negative_indices = _sample_negative_indices(
...     features_shape=(batch_size, sequence_length),
...     num_negatives=model.config.num_negatives,
...     mask_time_indices=mask_time_indices,
... )
>>> mask_time_indices = torch.tensor(data=mask_time_indices, device=input_values.device, dtype=torch.long)
>>> sampled_negative_indices = torch.tensor(
...     data=sampled_negative_indices, device=input_values.device, dtype=torch.long
... )

>>> with torch.no_grad():
...     outputs = model(input_values, mask_time_indices=mask_time_indices)

>>> 
>>> cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)

>>> 
>>> cosine_sim[mask_time_indices.to(torch.bool)].mean() > 0.5
tensor(True)

>>> 
>>> model = model.train()
>>> loss = model(
...     input_values, mask_time_indices=mask_time_indices, sampled_negative_indices=sampled_negative_indices
... ).loss
```