# Multiple choice

A multiple choice task is similar to question answering, except several candidate answers are provided along with a context and the model is trained to select the correct answer.

This guide will show you how to:

1.  Finetune [BERT](https://huggingface.co/bert-base-uncased) on the `regular` configuration of the [SWAG](https://huggingface.co/datasets/swag) dataset to select the best answer given multiple options and some context.
2.  Use your finetuned model for inference.

The task illustrated in this tutorial is supported by the following model architectures:

[ALBERT](../model_doc/albert), [BERT](../model_doc/bert), [BigBird](../model_doc/big_bird), [CamemBERT](../model_doc/camembert), [CANINE](../model_doc/canine), [ConvBERT](../model_doc/convbert), [Data2VecText](../model_doc/data2vec-text), [DeBERTa-v2](../model_doc/deberta-v2), [DistilBERT](../model_doc/distilbert), [ELECTRA](../model_doc/electra), [ERNIE](../model_doc/ernie), [ErnieM](../model_doc/ernie_m), [FlauBERT](../model_doc/flaubert), [FNet](../model_doc/fnet), [Funnel Transformer](../model_doc/funnel), [I-BERT](../model_doc/ibert), [Longformer](../model_doc/longformer), [LUKE](../model_doc/luke), [MEGA](../model_doc/mega), [Megatron-BERT](../model_doc/megatron-bert), [MobileBERT](../model_doc/mobilebert), [MPNet](../model_doc/mpnet), [MRA](../model_doc/mra), [Nezha](../model_doc/nezha), [Nyströmformer](../model_doc/nystromformer), [QDQBert](../model_doc/qdqbert), [RemBERT](../model_doc/rembert), [RoBERTa](../model_doc/roberta), [RoBERTa-PreLayerNorm](../model_doc/roberta-prelayernorm), [RoCBert](../model_doc/roc_bert), [RoFormer](../model_doc/roformer), [SqueezeBERT](../model_doc/squeezebert), [XLM](../model_doc/xlm), [XLM-RoBERTa](../model_doc/xlm-roberta), [XLM-RoBERTa-XL](../model_doc/xlm-roberta-xl), [XLNet](../model_doc/xlnet), [X-MOD](../model_doc/xmod), [YOSO](../model_doc/yoso)

Before you begin, make sure you have all the necessary libraries installed:

```
pip install transformers datasets evaluate
```

We encourage you to login to your Hugging Face account so you can upload and share your model with the community. When prompted, enter your token to login:

```
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Load SWAG dataset

Start by loading the `regular` configuration of the SWAG dataset from the 🤗 Datasets library:

```
>>> from datasets import load_dataset

>>> swag = load_dataset("swag", "regular")
```

Then take a look at an example:

```
>>> swag["train"][0]
{'ending0': 'passes by walking down the street playing their instruments.',
 'ending1': 'has heard approaching them.',
 'ending2': "arrives and they're outside dancing and asleep.",
 'ending3': 'turns the lead singer watches the performance.',
 'fold-ind': '3416',
 'gold-source': 'gold',
 'label': 0,
 'sent1': 'Members of the procession walk down the street holding small horn brass instruments.',
 'sent2': 'A drum line',
 'startphrase': 'Members of the procession walk down the street holding small horn brass instruments. A drum line',
 'video-id': 'anetv_jkn6uvmqwh4'}
```

While it looks like there are a lot of fields here, it is actually pretty straightforward:

-   `sent1` and `sent2`: these fields show how a sentence starts, and if you put the two together, you get the `startphrase` field.
-   `ending`: suggests a possible ending for how a sentence can end, but only one of them is correct.
-   `label`: identifies the correct sentence ending.

## Preprocess

The next step is to load a BERT tokenizer to process the sentence starts and the four possible endings:

```
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

The preprocessing function you want to create needs to:

1.  Make four copies of the `sent1` field and combine each of them with `sent2` to recreate how a sentence starts.
2.  Combine `sent2` with each of the four possible sentence endings.
3.  Flatten these two lists so you can tokenize them, and then unflatten them afterward so each example has a corresponding `input_ids`, `attention_mask`, and `labels` field.

```
>>> ending_names = ["ending0", "ending1", "ending2", "ending3"]


>>> def preprocess_function(examples):
...     first_sentences = [[context] * 4 for context in examples["sent1"]]
...     question_headers = examples["sent2"]
...     second_sentences = [
...         [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
...     ]

...     first_sentences = sum(first_sentences, [])
...     second_sentences = sum(second_sentences, [])

...     tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
...     return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
```

To apply the preprocessing function over the entire dataset, use 🤗 Datasets [map](https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/main_classes#datasets.Dataset.map) method. You can speed up the `map` function by setting `batched=True` to process multiple elements of the dataset at once:

```
tokenized_swag = swag.map(preprocess_function, batched=True)
```

🤗 Transformers doesn’t have a data collator for multiple choice, so you’ll need to adapt the [DataCollatorWithPadding](/docs/transformers/v4.34.0/en/main_classes/data_collator#transformers.DataCollatorWithPadding) to create a batch of examples. It’s more efficient to _dynamically pad_ the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length.

`DataCollatorForMultipleChoice` flattens all the model inputs, applies padding, and then unflattens the results:

```
>>> from dataclasses import dataclass
>>> from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
>>> from typing import Optional, Union
>>> import torch


>>> @dataclass
... class DataCollatorForMultipleChoice:
...     """
...     Data collator that will dynamically pad the inputs for multiple choice received.
...     """

...     tokenizer: PreTrainedTokenizerBase
...     padding: Union[bool, str, PaddingStrategy] = True
...     max_length: Optional[int] = None
...     pad_to_multiple_of: Optional[int] = None

...     def __call__(self, features):
...         label_name = "label" if "label" in features[0].keys() else "labels"
...         labels = [feature.pop(label_name) for feature in features]
...         batch_size = len(features)
...         num_choices = len(features[0]["input_ids"])
...         flattened_features = [
...             [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
...         ]
...         flattened_features = sum(flattened_features, [])

...         batch = self.tokenizer.pad(
...             flattened_features,
...             padding=self.padding,
...             max_length=self.max_length,
...             pad_to_multiple_of=self.pad_to_multiple_of,
...             return_tensors="pt",
...         )

...         batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
...         batch["labels"] = torch.tensor(labels, dtype=torch.int64)
...         return batch
```

```
>>> from dataclasses import dataclass
>>> from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
>>> from typing import Optional, Union
>>> import tensorflow as tf


>>> @dataclass
... class DataCollatorForMultipleChoice:
...     """
...     Data collator that will dynamically pad the inputs for multiple choice received.
...     """

...     tokenizer: PreTrainedTokenizerBase
...     padding: Union[bool, str, PaddingStrategy] = True
...     max_length: Optional[int] = None
...     pad_to_multiple_of: Optional[int] = None

...     def __call__(self, features):
...         label_name = "label" if "label" in features[0].keys() else "labels"
...         labels = [feature.pop(label_name) for feature in features]
...         batch_size = len(features)
...         num_choices = len(features[0]["input_ids"])
...         flattened_features = [
...             [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
...         ]
...         flattened_features = sum(flattened_features, [])

...         batch = self.tokenizer.pad(
...             flattened_features,
...             padding=self.padding,
...             max_length=self.max_length,
...             pad_to_multiple_of=self.pad_to_multiple_of,
...             return_tensors="tf",
...         )

...         batch = {k: tf.reshape(v, (batch_size, num_choices, -1)) for k, v in batch.items()}
...         batch["labels"] = tf.convert_to_tensor(labels, dtype=tf.int64)
...         return batch
```

## Evaluate

Including a metric during training is often helpful for evaluating your model’s performance. You can quickly load a evaluation method with the 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) library. For this task, load the [accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy) metric (see the 🤗 Evaluate [quick tour](https://huggingface.co/docs/evaluate/a_quick_tour) to learn more about how to load and compute a metric):

```
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

Then create a function that passes your predictions and labels to [compute](https://huggingface.co/docs/evaluate/v0.4.0/en/package_reference/main_classes#evaluate.EvaluationModule.compute) to calculate the accuracy:

```
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     predictions = np.argmax(predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=labels)
```

Your `compute_metrics` function is ready to go now, and you’ll return to it when you setup your training.

## Train

If you aren’t familiar with finetuning a model with the [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer), take a look at the basic tutorial [here](../training#train-with-pytorch-trainer)!

You’re ready to start training your model now! Load BERT with [AutoModelForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoModelForMultipleChoice):

```
>>> from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

>>> model = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased")
```

At this point, only three steps remain:

1.  Define your training hyperparameters in [TrainingArguments](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.TrainingArguments). The only required parameter is `output_dir` which specifies where to save your model. You’ll push this model to the Hub by setting `push_to_hub=True` (you need to be signed in to Hugging Face to upload your model). At the end of each epoch, the [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) will evaluate the accuracy and save the training checkpoint.
2.  Pass the training arguments to [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) along with the model, dataset, tokenizer, data collator, and `compute_metrics` function.
3.  Call [train()](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer.train) to finetune your model.

```
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_swag_model",
...     evaluation_strategy="epoch",
...     save_strategy="epoch",
...     load_best_model_at_end=True,
...     learning_rate=5e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     weight_decay=0.01,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_swag["train"],
...     eval_dataset=tokenized_swag["validation"],
...     tokenizer=tokenizer,
...     data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

Once training is completed, share your model to the Hub with the [push\_to\_hub()](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer.push_to_hub) method so everyone can use your model:

```
>>> trainer.push_to_hub()
```

If you aren’t familiar with finetuning a model with Keras, take a look at the basic tutorial [here](../training#train-a-tensorflow-model-with-keras)!

To finetune a model in TensorFlow, start by setting up an optimizer function, learning rate schedule, and some training hyperparameters:

```
>>> from transformers import create_optimizer

>>> batch_size = 16
>>> num_train_epochs = 2
>>> total_train_steps = (len(tokenized_swag["train"]) // batch_size) * num_train_epochs
>>> optimizer, schedule = create_optimizer(init_lr=5e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
```

Then you can load BERT with [TFAutoModelForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.TFAutoModelForMultipleChoice):

```
>>> from transformers import TFAutoModelForMultipleChoice

>>> model = TFAutoModelForMultipleChoice.from_pretrained("bert-base-uncased")
```

Convert your datasets to the `tf.data.Dataset` format with [prepare\_tf\_dataset()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.prepare_tf_dataset):

```
>>> data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_swag["train"],
...     shuffle=True,
...     batch_size=batch_size,
...     collate_fn=data_collator,
... )

>>> tf_validation_set = model.prepare_tf_dataset(
...     tokenized_swag["validation"],
...     shuffle=False,
...     batch_size=batch_size,
...     collate_fn=data_collator,
... )
```

Configure the model for training with [`compile`](https://keras.io/api/models/model_training_apis/#compile-method). Note that Transformers models all have a default task-relevant loss function, so you don’t need to specify one unless you want to:

```
>>> model.compile(optimizer=optimizer)  
```

The last two things to setup before you start training is to compute the accuracy from the predictions, and provide a way to push your model to the Hub. Both are done by using [Keras callbacks](../main_classes/keras_callbacks).

Pass your `compute_metrics` function to [KerasMetricCallback](/docs/transformers/v4.34.0/en/main_classes/keras_callbacks#transformers.KerasMetricCallback):

```
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
```

Specify where to push your model and tokenizer in the [PushToHubCallback](/docs/transformers/v4.34.0/en/main_classes/keras_callbacks#transformers.PushToHubCallback):

```
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_model",
...     tokenizer=tokenizer,
... )
```

Then bundle your callbacks together:

```
>>> callbacks = [metric_callback, push_to_hub_callback]
```

Finally, you’re ready to start training your model! Call [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) with your training and validation datasets, the number of epochs, and your callbacks to finetune the model:

```
>>> model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=2, callbacks=callbacks)
```

Once training is completed, your model is automatically uploaded to the Hub so everyone can use it!

For a more in-depth example of how to finetune a model for multiple choice, take a look at the corresponding [PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb) or [TensorFlow notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb).

## Inference

Great, now that you’ve finetuned a model, you can use it for inference!

Come up with some text and two candidate answers:

```
>>> prompt = "France has a bread law, Le Décret Pain, with strict rules on what is allowed in a traditional baguette."
>>> candidate1 = "The law does not apply to croissants and brioche."
>>> candidate2 = "The law applies to baguettes."
```

Tokenize each prompt and candidate answer pair and return PyTorch tensors. You should also create some `labels`:

```
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_swag_model")
>>> inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="pt", padding=True)
>>> labels = torch.tensor(0).unsqueeze(0)
```

Pass your inputs and labels to the model and return the `logits`:

```
>>> from transformers import AutoModelForMultipleChoice

>>> model = AutoModelForMultipleChoice.from_pretrained("my_awesome_swag_model")
>>> outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
>>> logits = outputs.logits
```

Get the class with the highest probability:

```
>>> predicted_class = logits.argmax().item()
>>> predicted_class
'0'
```

Tokenize each prompt and candidate answer pair and return TensorFlow tensors:

```
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_swag_model")
>>> inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="tf", padding=True)
```

Pass your inputs to the model and return the `logits`:

```
>>> from transformers import TFAutoModelForMultipleChoice

>>> model = TFAutoModelForMultipleChoice.from_pretrained("my_awesome_swag_model")
>>> inputs = {k: tf.expand_dims(v, 0) for k, v in inputs.items()}
>>> outputs = model(inputs)
>>> logits = outputs.logits
```

Get the class with the highest probability:

```
>>> predicted_class = int(tf.math.argmax(logits, axis=-1)[0])
>>> predicted_class
'0'
```