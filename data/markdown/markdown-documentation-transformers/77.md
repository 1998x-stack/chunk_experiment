# Model outputs

All models have outputs that are instances of subclasses of [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput). Those are data structures containing all the information returned by the model, but that can also be used as tuples or dictionaries.

Let’s see how this looks in an example:

```
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  
outputs = model(**inputs, labels=labels)
```

The `outputs` object is a [SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput), as we can see in the documentation of that class below, it means it has an optional `loss`, a `logits`, an optional `hidden_states` and an optional `attentions` attribute. Here we have the `loss` since we passed along `labels`, but we don’t have `hidden_states` and `attentions` because we didn’t pass `output_hidden_states=True` or `output_attentions=True`.

When passing `output_hidden_states=True` you may expect the `outputs.hidden_states[-1]` to match `outputs.last_hidden_states` exactly. However, this is not always the case. Some models apply normalization or subsequent process to the last hidden state when it’s returned.

You can access each attribute as you would usually do, and if that attribute has not been returned by the model, you will get `None`. Here for instance `outputs.loss` is the loss computed by the model, and `outputs.attentions` is `None`.

When considering our `outputs` object as tuple, it only considers the attributes that don’t have `None` values. Here for instance, it has two elements, `loss` then `logits`, so

will return the tuple `(outputs.loss, outputs.logits)` for instance.

When considering our `outputs` object as dictionary, it only considers the attributes that don’t have `None` values. Here for instance, it has two keys that are `loss` and `logits`.

We document here the generic model outputs that are used by more than one model type. Specific output types are documented on their corresponding model page.

## ModelOutput

### class transformers.utils.ModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/generic.py#L288)

( \*args\*\*kwargs )

Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular python dictionary.

You can’t unpack a `ModelOutput` directly. Use the [to\_tuple()](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput.to_tuple) method to convert it to a tuple before.

Convert self to a tuple containing all the attributes/keys that are not `None`.

## BaseModelOutput

### class transformers.modeling\_outputs.BaseModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L25)

( last\_hidden\_state: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Parameters

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

Base class for model’s outputs, with potential hidden states and attentions.

## BaseModelOutputWithPooling

### class transformers.modeling\_outputs.BaseModelOutputWithPooling

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L70)

( last\_hidden\_state: FloatTensor = Nonepooler\_output: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for model’s outputs that also contains a pooling of the last hidden states.

## BaseModelOutputWithCrossAttentions

### class transformers.modeling\_outputs.BaseModelOutputWithCrossAttentions

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L163)

( last\_hidden\_state: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for model’s outputs, with potential hidden states and attentions.

## BaseModelOutputWithPoolingAndCrossAttentions

### class transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L196)

( last\_hidden\_state: FloatTensor = Nonepooler\_output: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for model’s outputs that also contains a pooling of the last hidden states.

## BaseModelOutputWithPast

### class transformers.modeling\_outputs.BaseModelOutputWithPast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L124)

( last\_hidden\_state: FloatTensor = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for model’s outputs that may also contain a past key/values (to speed up sequential decoding).

## BaseModelOutputWithPastAndCrossAttentions

### class transformers.modeling\_outputs.BaseModelOutputWithPastAndCrossAttentions

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L245)

( last\_hidden\_state: FloatTensor = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for model’s outputs that may also contain a past key/values (to speed up sequential decoding).

## Seq2SeqModelOutput

### class transformers.modeling\_outputs.Seq2SeqModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L425)

( last\_hidden\_state: FloatTensor = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for model encoder’s outputs that also contains : pre-computed hidden states that can speed up sequential decoding.

## CausalLMOutput

### class transformers.modeling\_outputs.CausalLMOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L558)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for causal language model (or autoregressive) outputs.

## CausalLMOutputWithCrossAttentions

### class transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L623)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for causal language model (or autoregressive) outputs.

## CausalLMOutputWithPast

### class transformers.modeling\_outputs.CausalLMOutputWithPast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L587)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for causal language model (or autoregressive) outputs.

## MaskedLMOutput

### class transformers.modeling\_outputs.MaskedLMOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L703)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for masked language models outputs.

## Seq2SeqLMOutput

### class transformers.modeling\_outputs.Seq2SeqLMOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L732)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for sequence-to-sequence language models outputs.

## NextSentencePredictorOutput

### class transformers.modeling\_outputs.NextSentencePredictorOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L867)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for outputs of models predicting if two sentences are consecutive or not.

## SequenceClassifierOutput

### class transformers.modeling\_outputs.SequenceClassifierOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L897)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for outputs of sentence classification models.

## Seq2SeqSequenceClassifierOutput

### class transformers.modeling\_outputs.Seq2SeqSequenceClassifierOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L926)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for outputs of sequence-to-sequence sentence classification models.

## MultipleChoiceModelOutput

### class transformers.modeling\_outputs.MultipleChoiceModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L986)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for outputs of multiple choice models.

## TokenClassifierOutput

### class transformers.modeling\_outputs.TokenClassifierOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L1017)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for outputs of token classification models.

## QuestionAnsweringModelOutput

### class transformers.modeling\_outputs.QuestionAnsweringModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L1046)

( loss: typing.Optional\[torch.FloatTensor\] = Nonestart\_logits: FloatTensor = Noneend\_logits: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for outputs of question answering models.

## Seq2SeqQuestionAnsweringModelOutput

### class transformers.modeling\_outputs.Seq2SeqQuestionAnsweringModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L1078)

( loss: typing.Optional\[torch.FloatTensor\] = Nonestart\_logits: FloatTensor = Noneend\_logits: FloatTensor = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for outputs of sequence-to-sequence question answering models.

## Seq2SeqSpectrogramOutput

### class transformers.modeling\_outputs.Seq2SeqSpectrogramOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L1411)

( loss: typing.Optional\[torch.FloatTensor\] = Nonespectrogram: FloatTensor = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for sequence-to-sequence spectrogram outputs.

## SemanticSegmenterOutput

### class transformers.modeling\_outputs.SemanticSegmenterOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L1141)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for outputs of semantic segmentation models.

## ImageClassifierOutput

### class transformers.modeling\_outputs.ImageClassifierOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L1179)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Parameters

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the model at the output of each stage.
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

Base class for outputs of image classification models.

## ImageClassifierOutputWithNoAttention

### class transformers.modeling\_outputs.ImageClassifierOutputWithNoAttention

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L1207)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Parameters

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at the output of each stage.

Base class for outputs of image classification models.

## DepthEstimatorOutput

### class transformers.modeling\_outputs.DepthEstimatorOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L1228)

( loss: typing.Optional\[torch.FloatTensor\] = Nonepredicted\_depth: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for outputs of depth estimation models.

## Wav2Vec2BaseModelOutput

### class transformers.modeling\_outputs.Wav2Vec2BaseModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L1286)

( last\_hidden\_state: FloatTensor = Noneextract\_features: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for models that have been trained with the Wav2Vec2 loss objective.

## XVectorOutput

### class transformers.modeling\_outputs.XVectorOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L1315)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Noneembeddings: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Output type of [Wav2Vec2ForXVector](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2ForXVector).

## Seq2SeqTSModelOutput

### class transformers.modeling\_outputs.Seq2SeqTSModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L1471)

( last\_hidden\_state: FloatTensor = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneloc: typing.Optional\[torch.FloatTensor\] = Nonescale: typing.Optional\[torch.FloatTensor\] = Nonestatic\_features: typing.Optional\[torch.FloatTensor\] = None )

Base class for time series model’s encoder outputs that also contains pre-computed hidden states that can speed up sequential decoding.

## Seq2SeqTSPredictionOutput

### class transformers.modeling\_outputs.Seq2SeqTSPredictionOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L1543)

( loss: typing.Optional\[torch.FloatTensor\] = Noneparams: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneloc: typing.Optional\[torch.FloatTensor\] = Nonescale: typing.Optional\[torch.FloatTensor\] = Nonestatic\_features: typing.Optional\[torch.FloatTensor\] = None )

Base class for time series model’s decoder outputs that also contain the loss as well as the parameters of the chosen distribution.

## SampleTSPredictionOutput

### class transformers.modeling\_outputs.SampleTSPredictionOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L1615)

( sequences: FloatTensor = None )

Parameters

-   **sequences** (`torch.FloatTensor` of shape `(batch_size, num_samples, prediction_length)` or `(batch_size, num_samples, prediction_length, input_size)`) — Sampled values from the chosen distribution.

Base class for time series model’s predictions outputs that contains the sampled values from the chosen distribution.

## TFBaseModelOutput

### class transformers.modeling\_tf\_outputs.TFBaseModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_outputs.py#L27)

( last\_hidden\_state: tf.Tensor = Nonehidden\_states: Tuple\[tf.Tensor\] | None = Noneattentions: Tuple\[tf.Tensor\] | None = None )

Parameters

-   **last\_hidden\_state** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
-   **hidden\_states** (`tuple(tf.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

Base class for model’s outputs, with potential hidden states and attentions.

## TFBaseModelOutputWithPooling

### class transformers.modeling\_tf\_outputs.TFBaseModelOutputWithPooling

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_outputs.py#L72)

( last\_hidden\_state: tf.Tensor = Nonepooler\_output: tf.Tensor = Nonehidden\_states: Tuple\[tf.Tensor\] | None = Noneattentions: Tuple\[tf.Tensor\] | None = None )

Base class for model’s outputs that also contains a pooling of the last hidden states.

## TFBaseModelOutputWithPoolingAndCrossAttentions

### class transformers.modeling\_tf\_outputs.TFBaseModelOutputWithPoolingAndCrossAttentions

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_outputs.py#L128)

( last\_hidden\_state: tf.Tensor = Nonepooler\_output: tf.Tensor = Nonepast\_key\_values: List\[tf.Tensor\] | None = Nonehidden\_states: Tuple\[tf.Tensor\] | None = Noneattentions: Tuple\[tf.Tensor\] | None = Nonecross\_attentions: Tuple\[tf.Tensor\] | None = None )

Base class for model’s outputs that also contains a pooling of the last hidden states.

## TFBaseModelOutputWithPast

### class transformers.modeling\_tf\_outputs.TFBaseModelOutputWithPast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_outputs.py#L176)

( last\_hidden\_state: tf.Tensor = Nonepast\_key\_values: List\[tf.Tensor\] | None = Nonehidden\_states: Tuple\[tf.Tensor\] | None = Noneattentions: Tuple\[tf.Tensor\] | None = None )

Base class for model’s outputs that may also contain a past key/values (to speed up sequential decoding).

## TFBaseModelOutputWithPastAndCrossAttentions

### class transformers.modeling\_tf\_outputs.TFBaseModelOutputWithPastAndCrossAttentions

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_outputs.py#L245)

( last\_hidden\_state: tf.Tensor = Nonepast\_key\_values: List\[tf.Tensor\] | None = Nonehidden\_states: Tuple\[tf.Tensor\] | None = Noneattentions: Tuple\[tf.Tensor\] | None = Nonecross\_attentions: Tuple\[tf.Tensor\] | None = None )

Base class for model’s outputs that may also contain a past key/values (to speed up sequential decoding).

## TFSeq2SeqModelOutput

### class transformers.modeling\_tf\_outputs.TFSeq2SeqModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_outputs.py#L288)

( last\_hidden\_state: tf.Tensor = Nonepast\_key\_values: List\[tf.Tensor\] | None = Nonedecoder\_hidden\_states: Tuple\[tf.Tensor\] | None = Nonedecoder\_attentions: Tuple\[tf.Tensor\] | None = Nonecross\_attentions: Tuple\[tf.Tensor\] | None = Noneencoder\_last\_hidden\_state: tf.Tensor | None = Noneencoder\_hidden\_states: Tuple\[tf.Tensor\] | None = Noneencoder\_attentions: Tuple\[tf.Tensor\] | None = None )

Base class for model encoder’s outputs that also contains : pre-computed hidden states that can speed up sequential decoding.

## TFCausalLMOutput

### class transformers.modeling\_tf\_outputs.TFCausalLMOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_outputs.py#L348)

( loss: tf.Tensor | None = Nonelogits: tf.Tensor = Nonehidden\_states: Tuple\[tf.Tensor\] | None = Noneattentions: Tuple\[tf.Tensor\] | None = None )

Base class for causal language model (or autoregressive) outputs.

## TFCausalLMOutputWithCrossAttentions

### class transformers.modeling\_tf\_outputs.TFCausalLMOutputWithCrossAttentions

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_outputs.py#L413)

( loss: tf.Tensor | None = Nonelogits: tf.Tensor = Nonepast\_key\_values: List\[tf.Tensor\] | None = Nonehidden\_states: Tuple\[tf.Tensor\] | None = Noneattentions: Tuple\[tf.Tensor\] | None = Nonecross\_attentions: Tuple\[tf.Tensor\] | None = None )

Base class for causal language model (or autoregressive) outputs.

## TFCausalLMOutputWithPast

### class transformers.modeling\_tf\_outputs.TFCausalLMOutputWithPast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_outputs.py#L377)

( loss: tf.Tensor | None = Nonelogits: tf.Tensor = Nonepast\_key\_values: List\[tf.Tensor\] | None = Nonehidden\_states: Tuple\[tf.Tensor\] | None = Noneattentions: Tuple\[tf.Tensor\] | None = None )

Base class for causal language model (or autoregressive) outputs.

## TFMaskedLMOutput

### class transformers.modeling\_tf\_outputs.TFMaskedLMOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_outputs.py#L456)

( loss: tf.Tensor | None = Nonelogits: tf.Tensor = Nonehidden\_states: Tuple\[tf.Tensor\] | None = Noneattentions: Tuple\[tf.Tensor\] | None = None )

Base class for masked language models outputs.

## TFSeq2SeqLMOutput

### class transformers.modeling\_tf\_outputs.TFSeq2SeqLMOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_outputs.py#L485)

( loss: tf.Tensor | None = Nonelogits: tf.Tensor = Nonepast\_key\_values: List\[tf.Tensor\] | None = Nonedecoder\_hidden\_states: Tuple\[tf.Tensor\] | None = Nonedecoder\_attentions: Tuple\[tf.Tensor\] | None = Nonecross\_attentions: Tuple\[tf.Tensor\] | None = Noneencoder\_last\_hidden\_state: tf.Tensor | None = Noneencoder\_hidden\_states: Tuple\[tf.Tensor\] | None = Noneencoder\_attentions: Tuple\[tf.Tensor\] | None = None )

Base class for sequence-to-sequence language models outputs.

## TFNextSentencePredictorOutput

### class transformers.modeling\_tf\_outputs.TFNextSentencePredictorOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_outputs.py#L544)

( loss: tf.Tensor | None = Nonelogits: tf.Tensor = Nonehidden\_states: Tuple\[tf.Tensor\] | None = Noneattentions: Tuple\[tf.Tensor\] | None = None )

Base class for outputs of models predicting if two sentences are consecutive or not.

## TFSequenceClassifierOutput

### class transformers.modeling\_tf\_outputs.TFSequenceClassifierOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_outputs.py#L574)

( loss: tf.Tensor | None = Nonelogits: tf.Tensor = Nonehidden\_states: Tuple\[tf.Tensor\] | None = Noneattentions: Tuple\[tf.Tensor\] | None = None )

Base class for outputs of sentence classification models.

## TFSeq2SeqSequenceClassifierOutput

### class transformers.modeling\_tf\_outputs.TFSeq2SeqSequenceClassifierOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_outputs.py#L603)

( loss: tf.Tensor | None = Nonelogits: tf.Tensor = Nonepast\_key\_values: List\[tf.Tensor\] | None = Nonedecoder\_hidden\_states: Tuple\[tf.Tensor\] | None = Nonedecoder\_attentions: Tuple\[tf.Tensor\] | None = Nonecross\_attentions: Tuple\[tf.Tensor\] | None = Noneencoder\_last\_hidden\_state: tf.Tensor | None = Noneencoder\_hidden\_states: Tuple\[tf.Tensor\] | None = Noneencoder\_attentions: Tuple\[tf.Tensor\] | None = None )

Base class for outputs of sequence-to-sequence sentence classification models.

## TFMultipleChoiceModelOutput

### class transformers.modeling\_tf\_outputs.TFMultipleChoiceModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_outputs.py#L754)

( loss: tf.Tensor | None = Nonelogits: tf.Tensor = Nonehidden\_states: Tuple\[tf.Tensor\] | None = Noneattentions: Tuple\[tf.Tensor\] | None = None )

Base class for outputs of multiple choice models.

## TFTokenClassifierOutput

### class transformers.modeling\_tf\_outputs.TFTokenClassifierOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_outputs.py#L785)

( loss: tf.Tensor | None = Nonelogits: tf.Tensor = Nonehidden\_states: Tuple\[tf.Tensor\] | None = Noneattentions: Tuple\[tf.Tensor\] | None = None )

Base class for outputs of token classification models.

## TFQuestionAnsweringModelOutput

### class transformers.modeling\_tf\_outputs.TFQuestionAnsweringModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_outputs.py#L814)

( loss: tf.Tensor | None = Nonestart\_logits: tf.Tensor = Noneend\_logits: tf.Tensor = Nonehidden\_states: Tuple\[tf.Tensor\] | None = Noneattentions: Tuple\[tf.Tensor\] | None = None )

Base class for outputs of question answering models.

## TFSeq2SeqQuestionAnsweringModelOutput

### class transformers.modeling\_tf\_outputs.TFSeq2SeqQuestionAnsweringModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_outputs.py#L846)

( loss: tf.Tensor | None = Nonestart\_logits: tf.Tensor = Noneend\_logits: tf.Tensor = Nonepast\_key\_values: List\[tf.Tensor\] | None = Nonedecoder\_hidden\_states: Tuple\[tf.Tensor\] | None = Nonedecoder\_attentions: Tuple\[tf.Tensor\] | None = Noneencoder\_last\_hidden\_state: tf.Tensor | None = Noneencoder\_hidden\_states: Tuple\[tf.Tensor\] | None = Noneencoder\_attentions: Tuple\[tf.Tensor\] | None = None )

Base class for outputs of sequence-to-sequence question answering models.

## FlaxBaseModelOutput

### class transformers.modeling\_flax\_outputs.FlaxBaseModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_flax_outputs.py#L23)

( last\_hidden\_state: Array = Nonehidden\_states: typing.Optional\[typing.Tuple\[jax.Array\]\] = Noneattentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = None )

Parameters

-   **last\_hidden\_state** (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
-   **hidden\_states** (`tuple(jnp.ndarray)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

Base class for model’s outputs, with potential hidden states and attentions.

“Returns a new object replacing the specified fields with new values.

## FlaxBaseModelOutputWithPast

### class transformers.modeling\_flax\_outputs.FlaxBaseModelOutputWithPast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_flax_outputs.py#L107)

( last\_hidden\_state: Array = Nonepast\_key\_values: typing.Union\[typing.Dict\[str, jax.Array\], NoneType\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[jax.Array\]\] = Noneattentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = None )

Base class for model’s outputs, with potential hidden states and attentions.

“Returns a new object replacing the specified fields with new values.

## FlaxBaseModelOutputWithPooling

### class transformers.modeling\_flax\_outputs.FlaxBaseModelOutputWithPooling

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_flax_outputs.py#L137)

( last\_hidden\_state: Array = Nonepooler\_output: Array = Nonehidden\_states: typing.Optional\[typing.Tuple\[jax.Array\]\] = Noneattentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = None )

Base class for model’s outputs that also contains a pooling of the last hidden states.

“Returns a new object replacing the specified fields with new values.

## FlaxBaseModelOutputWithPastAndCrossAttentions

### class transformers.modeling\_flax\_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_flax_outputs.py#L217)

( last\_hidden\_state: Array = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[jax.Array\]\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[jax.Array\]\] = Noneattentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = None )

Base class for model’s outputs that may also contain a past key/values (to speed up sequential decoding).

“Returns a new object replacing the specified fields with new values.

## FlaxSeq2SeqModelOutput

### class transformers.modeling\_flax\_outputs.FlaxSeq2SeqModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_flax_outputs.py#L263)

( last\_hidden\_state: Array = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[jax.Array\]\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[jax.Array\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = Noneencoder\_last\_hidden\_state: typing.Optional\[jax.Array\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[jax.Array\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = None )

Base class for model encoder’s outputs that also contains : pre-computed hidden states that can speed up sequential decoding.

“Returns a new object replacing the specified fields with new values.

## FlaxCausalLMOutputWithCrossAttentions

### class transformers.modeling\_flax\_outputs.FlaxCausalLMOutputWithCrossAttentions

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_flax_outputs.py#L324)

( logits: Array = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[jax.Array\]\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[jax.Array\]\] = Noneattentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = None )

Base class for causal language model (or autoregressive) outputs.

“Returns a new object replacing the specified fields with new values.

## FlaxMaskedLMOutput

### class transformers.modeling\_flax\_outputs.FlaxMaskedLMOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_flax_outputs.py#L365)

( logits: Array = Nonehidden\_states: typing.Optional\[typing.Tuple\[jax.Array\]\] = Noneattentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = None )

Parameters

-   **logits** (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
-   **hidden\_states** (`tuple(jnp.ndarray)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

Base class for masked language models outputs.

“Returns a new object replacing the specified fields with new values.

## FlaxSeq2SeqLMOutput

### class transformers.modeling\_flax\_outputs.FlaxSeq2SeqLMOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_flax_outputs.py#L394)

( logits: Array = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[jax.Array\]\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[jax.Array\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = Noneencoder\_last\_hidden\_state: typing.Optional\[jax.Array\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[jax.Array\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = None )

Base class for sequence-to-sequence language models outputs.

“Returns a new object replacing the specified fields with new values.

## FlaxNextSentencePredictorOutput

### class transformers.modeling\_flax\_outputs.FlaxNextSentencePredictorOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_flax_outputs.py#L451)

( logits: Array = Nonehidden\_states: typing.Optional\[typing.Tuple\[jax.Array\]\] = Noneattentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = None )

Parameters

-   **logits** (`jnp.ndarray` of shape `(batch_size, 2)`) — Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
-   **hidden\_states** (`tuple(jnp.ndarray)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

Base class for outputs of models predicting if two sentences are consecutive or not.

“Returns a new object replacing the specified fields with new values.

## FlaxSequenceClassifierOutput

### class transformers.modeling\_flax\_outputs.FlaxSequenceClassifierOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_flax_outputs.py#L478)

( logits: Array = Nonehidden\_states: typing.Optional\[typing.Tuple\[jax.Array\]\] = Noneattentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = None )

Parameters

-   **logits** (`jnp.ndarray` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
-   **hidden\_states** (`tuple(jnp.ndarray)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

Base class for outputs of sentence classification models.

“Returns a new object replacing the specified fields with new values.

## FlaxSeq2SeqSequenceClassifierOutput

### class transformers.modeling\_flax\_outputs.FlaxSeq2SeqSequenceClassifierOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_flax_outputs.py#L504)

( logits: Array = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[jax.Array\]\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[jax.Array\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = Noneencoder\_last\_hidden\_state: typing.Optional\[jax.Array\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[jax.Array\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = None )

Base class for outputs of sequence-to-sequence sentence classification models.

“Returns a new object replacing the specified fields with new values.

## FlaxMultipleChoiceModelOutput

### class transformers.modeling\_flax\_outputs.FlaxMultipleChoiceModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_flax_outputs.py#L561)

( logits: Array = Nonehidden\_states: typing.Optional\[typing.Tuple\[jax.Array\]\] = Noneattentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = None )

Base class for outputs of multiple choice models.

“Returns a new object replacing the specified fields with new values.

## FlaxTokenClassifierOutput

### class transformers.modeling\_flax\_outputs.FlaxTokenClassifierOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_flax_outputs.py#L589)

( logits: Array = Nonehidden\_states: typing.Optional\[typing.Tuple\[jax.Array\]\] = Noneattentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = None )

Parameters

-   **logits** (`jnp.ndarray` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
-   **hidden\_states** (`tuple(jnp.ndarray)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

Base class for outputs of token classification models.

“Returns a new object replacing the specified fields with new values.

## FlaxQuestionAnsweringModelOutput

### class transformers.modeling\_flax\_outputs.FlaxQuestionAnsweringModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_flax_outputs.py#L615)

( start\_logits: Array = Noneend\_logits: Array = Nonehidden\_states: typing.Optional\[typing.Tuple\[jax.Array\]\] = Noneattentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = None )

Base class for outputs of question answering models.

“Returns a new object replacing the specified fields with new values.

## FlaxSeq2SeqQuestionAnsweringModelOutput

### class transformers.modeling\_flax\_outputs.FlaxSeq2SeqQuestionAnsweringModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_flax_outputs.py#L644)

( start\_logits: Array = Noneend\_logits: Array = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[jax.Array\]\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[jax.Array\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = Noneencoder\_last\_hidden\_state: typing.Optional\[jax.Array\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[jax.Array\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = None )

Base class for outputs of sequence-to-sequence question answering models.

“Returns a new object replacing the specified fields with new values.