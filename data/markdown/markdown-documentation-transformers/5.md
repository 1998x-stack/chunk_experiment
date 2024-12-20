# Load pretrained instances with an AutoClass

With so many different Transformer architectures, it can be challenging to create one for your checkpoint. As a part of 🤗 Transformers core philosophy to make the library easy, simple and flexible to use, an `AutoClass` automatically infers and loads the correct architecture from a given checkpoint. The `from_pretrained()` method lets you quickly load a pretrained model for any architecture so you don’t have to devote time and resources to train a model from scratch. Producing this type of checkpoint-agnostic code means if your code works for one checkpoint, it will work with another checkpoint - as long as it was trained for a similar task - even if the architecture is different.

Remember, architecture refers to the skeleton of the model and checkpoints are the weights for a given architecture. For example, [BERT](https://huggingface.co/bert-base-uncased) is an architecture, while `bert-base-uncased` is a checkpoint. Model is a general term that can mean either architecture or checkpoint.

In this tutorial, learn to:

-   Load a pretrained tokenizer.
-   Load a pretrained image processor
-   Load a pretrained feature extractor.
-   Load a pretrained processor.
-   Load a pretrained model.

## AutoTokenizer

Nearly every NLP task begins with a tokenizer. A tokenizer converts your input into a format that can be processed by the model.

Load a tokenizer with [AutoTokenizer.from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained):

```
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

Then tokenize your input as shown below:

```
>>> sequence = "In a hole in the ground there lived a hobbit."
>>> print(tokenizer(sequence))
{'input_ids': [101, 1999, 1037, 4920, 1999, 1996, 2598, 2045, 2973, 1037, 7570, 10322, 4183, 1012, 102], 
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

## AutoImageProcessor

For vision tasks, an image processor processes the image into the correct input format.

```
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

## AutoFeatureExtractor

For audio tasks, a feature extractor processes the audio signal the correct input format.

Load a feature extractor with [AutoFeatureExtractor.from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoFeatureExtractor.from_pretrained):

```
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained(
...     "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
... )
```

## AutoProcessor

Multimodal tasks require a processor that combines two types of preprocessing tools. For example, the [LayoutLMV2](model_doc/layoutlmv2) model requires an image processor to handle images and a tokenizer to handle text; a processor combines both of them.

Load a processor with [AutoProcessor.from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoProcessor.from_pretrained):

```
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
```

## AutoModel

Finally, the `AutoModelFor` classes let you load a pretrained model for a given task (see [here](model_doc/auto) for a complete list of available tasks). For example, load a model for sequence classification with [AutoModelForSequenceClassification.from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained):

```
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
```

Easily reuse the same checkpoint to load an architecture for a different task:

```
>>> from transformers import AutoModelForTokenClassification

>>> model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased")
```

For PyTorch models, the `from_pretrained()` method uses `torch.load()` which internally uses `pickle` and is known to be insecure. In general, never load a model that could have come from an untrusted source, or that could have been tampered with. This security risk is partially mitigated for public models hosted on the Hugging Face Hub, which are [scanned for malware](https://huggingface.co/docs/hub/security-malware) at each commit. See the [Hub documentation](https://huggingface.co/docs/hub/security) for best practices like [signed commit verification](https://huggingface.co/docs/hub/security-gpg#signing-commits-with-gpg) with GPG.

TensorFlow and Flax checkpoints are not affected, and can be loaded within PyTorch architectures using the `from_tf` and `from_flax` kwargs for the `from_pretrained` method to circumvent this issue.

Generally, we recommend using the `AutoTokenizer` class and the `AutoModelFor` class to load pretrained instances of models. This will ensure you load the correct architecture every time. In the next [tutorial](preprocessing), learn how to use your newly loaded tokenizer, image processor, feature extractor and processor to preprocess a dataset for fine-tuning.

Finally, the `TFAutoModelFor` classes let you load a pretrained model for a given task (see [here](model_doc/auto) for a complete list of available tasks). For example, load a model for sequence classification with [TFAutoModelForSequenceClassification.from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained):

```
>>> from transformers import TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
```

Easily reuse the same checkpoint to load an architecture for a different task:

```
>>> from transformers import TFAutoModelForTokenClassification

>>> model = TFAutoModelForTokenClassification.from_pretrained("distilbert-base-uncased")
```

Generally, we recommend using the `AutoTokenizer` class and the `TFAutoModelFor` class to load pretrained instances of models. This will ensure you load the correct architecture every time. In the next [tutorial](preprocessing), learn how to use your newly loaded tokenizer, image processor, feature extractor and processor to preprocess a dataset for fine-tuning.