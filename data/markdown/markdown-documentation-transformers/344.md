# Question answering

Question answering tasks return an answer given a question. If you’ve ever asked a virtual assistant like Alexa, Siri or Google what the weather is, then you’ve used a question answering model before. There are two common types of question answering tasks:

-   Extractive: extract the answer from the given context.
-   Abstractive: generate an answer from the context that correctly answers the question.

This guide will show you how to:

1.  Finetune [DistilBERT](https://huggingface.co/distilbert-base-uncased) on the [SQuAD](https://huggingface.co/datasets/squad) dataset for extractive question answering.
2.  Use your finetuned model for inference.

The task illustrated in this tutorial is supported by the following model architectures:

[ALBERT](../model_doc/albert), [BART](../model_doc/bart), [BERT](../model_doc/bert), [BigBird](../model_doc/big_bird), [BigBird-Pegasus](../model_doc/bigbird_pegasus), [BLOOM](../model_doc/bloom), [CamemBERT](../model_doc/camembert), [CANINE](../model_doc/canine), [ConvBERT](../model_doc/convbert), [Data2VecText](../model_doc/data2vec-text), [DeBERTa](../model_doc/deberta), [DeBERTa-v2](../model_doc/deberta-v2), [DistilBERT](../model_doc/distilbert), [ELECTRA](../model_doc/electra), [ERNIE](../model_doc/ernie), [ErnieM](../model_doc/ernie_m), [Falcon](../model_doc/falcon), [FlauBERT](../model_doc/flaubert), [FNet](../model_doc/fnet), [Funnel Transformer](../model_doc/funnel), [OpenAI GPT-2](../model_doc/gpt2), [GPT Neo](../model_doc/gpt_neo), [GPT NeoX](../model_doc/gpt_neox), [GPT-J](../model_doc/gptj), [I-BERT](../model_doc/ibert), [LayoutLMv2](../model_doc/layoutlmv2), [LayoutLMv3](../model_doc/layoutlmv3), [LED](../model_doc/led), [LiLT](../model_doc/lilt), [Longformer](../model_doc/longformer), [LUKE](../model_doc/luke), [LXMERT](../model_doc/lxmert), [MarkupLM](../model_doc/markuplm), [mBART](../model_doc/mbart), [MEGA](../model_doc/mega), [Megatron-BERT](../model_doc/megatron-bert), [MobileBERT](../model_doc/mobilebert), [MPNet](../model_doc/mpnet), [MPT](../model_doc/mpt), [MRA](../model_doc/mra), [MT5](../model_doc/mt5), [MVP](../model_doc/mvp), [Nezha](../model_doc/nezha), [Nyströmformer](../model_doc/nystromformer), [OPT](../model_doc/opt), [QDQBert](../model_doc/qdqbert), [Reformer](../model_doc/reformer), [RemBERT](../model_doc/rembert), [RoBERTa](../model_doc/roberta), [RoBERTa-PreLayerNorm](../model_doc/roberta-prelayernorm), [RoCBert](../model_doc/roc_bert), [RoFormer](../model_doc/roformer), [Splinter](../model_doc/splinter), [SqueezeBERT](../model_doc/squeezebert), [T5](../model_doc/t5), [UMT5](../model_doc/umt5), [XLM](../model_doc/xlm), [XLM-RoBERTa](../model_doc/xlm-roberta), [XLM-RoBERTa-XL](../model_doc/xlm-roberta-xl), [XLNet](../model_doc/xlnet), [X-MOD](../model_doc/xmod), [YOSO](../model_doc/yoso)

Before you begin, make sure you have all the necessary libraries installed:

```
pip install transformers datasets evaluate
```

We encourage you to login to your Hugging Face account so you can upload and share your model with the community. When prompted, enter your token to login:

```
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Load SQuAD dataset

Start by loading a smaller subset of the SQuAD dataset from the 🤗 Datasets library. This’ll give you a chance to experiment and make sure everything works before spending more time training on the full dataset.

```
>>> from datasets import load_dataset

>>> squad = load_dataset("squad", split="train[:5000]")
```

Split the dataset’s `train` split into a train and test set with the [train\_test\_split](https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/main_classes#datasets.Dataset.train_test_split) method:

```
>>> squad = squad.train_test_split(test_size=0.2)
```

Then take a look at an example:

```
>>> squad["train"][0]
{'answers': {'answer_start': [515], 'text': ['Saint Bernadette Soubirous']},
 'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
 'id': '5733be284776f41900661182',
 'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
 'title': 'University_of_Notre_Dame'
}
```

There are several important fields here:

-   `answers`: the starting location of the answer token and the answer text.
-   `context`: background information from which the model needs to extract the answer.
-   `question`: the question a model should answer.

## Preprocess

The next step is to load a DistilBERT tokenizer to process the `question` and `context` fields:

```
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
```

There are a few preprocessing steps particular to question answering tasks you should be aware of:

1.  Some examples in a dataset may have a very long `context` that exceeds the maximum input length of the model. To deal with longer sequences, truncate only the `context` by setting `truncation="only_second"`.
2.  Next, map the start and end positions of the answer to the original `context` by setting `return_offset_mapping=True`.
3.  With the mapping in hand, now you can find the start and end tokens of the answer. Use the `sequence_ids` method to find which part of the offset corresponds to the `question` and which corresponds to the `context`.

Here is how you can create a function to truncate and map the start and end tokens of the `answer` to the `context`:

```
>>> def preprocess_function(examples):
...     questions = [q.strip() for q in examples["question"]]
...     inputs = tokenizer(
...         questions,
...         examples["context"],
...         max_length=384,
...         truncation="only_second",
...         return_offsets_mapping=True,
...         padding="max_length",
...     )

...     offset_mapping = inputs.pop("offset_mapping")
...     answers = examples["answers"]
...     start_positions = []
...     end_positions = []

...     for i, offset in enumerate(offset_mapping):
...         answer = answers[i]
...         start_char = answer["answer_start"][0]
...         end_char = answer["answer_start"][0] + len(answer["text"][0])
...         sequence_ids = inputs.sequence_ids(i)

...         
...         idx = 0
...         while sequence_ids[idx] != 1:
...             idx += 1
...         context_start = idx
...         while sequence_ids[idx] == 1:
...             idx += 1
...         context_end = idx - 1

...         
...         if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
...             start_positions.append(0)
...             end_positions.append(0)
...         else:
...             
...             idx = context_start
...             while idx <= context_end and offset[idx][0] <= start_char:
...                 idx += 1
...             start_positions.append(idx - 1)

...             idx = context_end
...             while idx >= context_start and offset[idx][1] >= end_char:
...                 idx -= 1
...             end_positions.append(idx + 1)

...     inputs["start_positions"] = start_positions
...     inputs["end_positions"] = end_positions
...     return inputs
```

To apply the preprocessing function over the entire dataset, use 🤗 Datasets [map](https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/main_classes#datasets.Dataset.map) function. You can speed up the `map` function by setting `batched=True` to process multiple elements of the dataset at once. Remove any columns you don’t need:

```
>>> tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
```

Now create a batch of examples using [DefaultDataCollator](/docs/transformers/v4.34.0/en/main_classes/data_collator#transformers.DefaultDataCollator). Unlike other data collators in 🤗 Transformers, the [DefaultDataCollator](/docs/transformers/v4.34.0/en/main_classes/data_collator#transformers.DefaultDataCollator) does not apply any additional preprocessing such as padding.

```
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```

```
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator(return_tensors="tf")
```

## Train

If you aren’t familiar with finetuning a model with the [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer), take a look at the basic tutorial [here](../training#train-with-pytorch-trainer)!

You’re ready to start training your model now! Load DistilBERT with [AutoModelForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoModelForQuestionAnswering):

```
>>> from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

>>> model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
```

At this point, only three steps remain:

1.  Define your training hyperparameters in [TrainingArguments](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.TrainingArguments). The only required parameter is `output_dir` which specifies where to save your model. You’ll push this model to the Hub by setting `push_to_hub=True` (you need to be signed in to Hugging Face to upload your model).
2.  Pass the training arguments to [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) along with the model, dataset, tokenizer, and data collator.
3.  Call [train()](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer.train) to finetune your model.

```
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_qa_model",
...     evaluation_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     weight_decay=0.01,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_squad["train"],
...     eval_dataset=tokenized_squad["test"],
...     tokenizer=tokenizer,
...     data_collator=data_collator,
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
>>> num_epochs = 2
>>> total_train_steps = (len(tokenized_squad["train"]) // batch_size) * num_epochs
>>> optimizer, schedule = create_optimizer(
...     init_lr=2e-5,
...     num_warmup_steps=0,
...     num_train_steps=total_train_steps,
... )
```

Then you can load DistilBERT with [TFAutoModelForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.TFAutoModelForQuestionAnswering):

```
>>> from transformers import TFAutoModelForQuestionAnswering

>>> model = TFAutoModelForQuestionAnswering("distilbert-base-uncased")
```

Convert your datasets to the `tf.data.Dataset` format with [prepare\_tf\_dataset()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.prepare_tf_dataset):

```
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_squad["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_validation_set = model.prepare_tf_dataset(
...     tokenized_squad["test"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

Configure the model for training with [`compile`](https://keras.io/api/models/model_training_apis/#compile-method):

```
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)
```

The last thing to setup before you start training is to provide a way to push your model to the Hub. This can be done by specifying where to push your model and tokenizer in the [PushToHubCallback](/docs/transformers/v4.34.0/en/main_classes/keras_callbacks#transformers.PushToHubCallback):

```
>>> from transformers.keras_callbacks import PushToHubCallback

>>> callback = PushToHubCallback(
...     output_dir="my_awesome_qa_model",
...     tokenizer=tokenizer,
... )
```

Finally, you’re ready to start training your model! Call [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) with your training and validation datasets, the number of epochs, and your callback to finetune the model:

```
>>> model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3, callbacks=[callback])
```

Once training is completed, your model is automatically uploaded to the Hub so everyone can use it!

For a more in-depth example of how to finetune a model for question answering, take a look at the corresponding [PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb) or [TensorFlow notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb).

## Evaluate

Evaluation for question answering requires a significant amount of postprocessing. To avoid taking up too much of your time, this guide skips the evaluation step. The [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) still calculates the evaluation loss during training so you’re not completely in the dark about your model’s performance.

If have more time and you’re interested in how to evaluate your model for question answering, take a look at the [Question answering](https://huggingface.co/course/chapter7/7?fw=pt#postprocessing) chapter from the 🤗 Hugging Face Course!

## Inference

Great, now that you’ve finetuned a model, you can use it for inference!

Come up with a question and some context you’d like the model to predict:

```
>>> question = "How many programming languages does BLOOM support?"
>>> context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
```

The simplest way to try out your finetuned model for inference is to use it in a [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline). Instantiate a `pipeline` for question answering with your model, and pass your text to it:

```
>>> from transformers import pipeline

>>> question_answerer = pipeline("question-answering", model="my_awesome_qa_model")
>>> question_answerer(question=question, context=context)
{'score': 0.2058267742395401,
 'start': 10,
 'end': 95,
 'answer': '176 billion parameters and can generate text in 46 languages natural languages and 13'}
```

You can also manually replicate the results of the `pipeline` if you’d like:

Tokenize the text and return PyTorch tensors:

```
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_qa_model")
>>> inputs = tokenizer(question, context, return_tensors="pt")
```

Pass your inputs to the model and return the `logits`:

```
>>> import torch
>>> from transformers import AutoModelForQuestionAnswering

>>> model = AutoModelForQuestionAnswering.from_pretrained("my_awesome_qa_model")
>>> with torch.no_grad():
...     outputs = model(**inputs)
```

Get the highest probability from the model output for the start and end positions:

```
>>> answer_start_index = outputs.start_logits.argmax()
>>> answer_end_index = outputs.end_logits.argmax()
```

Decode the predicted tokens to get the answer:

```
>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens)
'176 billion parameters and can generate text in 46 languages natural languages and 13'
```

Tokenize the text and return TensorFlow tensors:

```
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_qa_model")
>>> inputs = tokenizer(question, text, return_tensors="tf")
```

Pass your inputs to the model and return the `logits`:

```
>>> from transformers import TFAutoModelForQuestionAnswering

>>> model = TFAutoModelForQuestionAnswering.from_pretrained("my_awesome_qa_model")
>>> outputs = model(**inputs)
```

Get the highest probability from the model output for the start and end positions:

```
>>> answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
>>> answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
```

Decode the predicted tokens to get the answer:

```
>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens)
'176 billion parameters and can generate text in 46 languages natural languages and 13'
```