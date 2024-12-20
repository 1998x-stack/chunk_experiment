# TAPAS

## Overview

The TAPAS model was proposed in [TAPAS: Weakly Supervised Table Parsing via Pre-training](https://www.aclweb.org/anthology/2020.acl-main.398) by Jonathan Herzig, Paweł Krzysztof Nowak, Thomas Müller, Francesco Piccinno and Julian Martin Eisenschlos. It’s a BERT-based model specifically designed (and pre-trained) for answering questions about tabular data. Compared to BERT, TAPAS uses relative position embeddings and has 7 token types that encode tabular structure. TAPAS is pre-trained on the masked language modeling (MLM) objective on a large dataset comprising millions of tables from English Wikipedia and corresponding texts.

For question answering, TAPAS has 2 heads on top: a cell selection head and an aggregation head, for (optionally) performing aggregations (such as counting or summing) among selected cells. TAPAS has been fine-tuned on several datasets:

-   [SQA](https://www.microsoft.com/en-us/download/details.aspx?id=54253) (Sequential Question Answering by Microsoft)
-   [WTQ](https://github.com/ppasupat/WikiTableQuestions) (Wiki Table Questions by Stanford University)
-   [WikiSQL](https://github.com/salesforce/WikiSQL) (by Salesforce).

It achieves state-of-the-art on both SQA and WTQ, while having comparable performance to SOTA on WikiSQL, with a much simpler architecture.

The abstract from the paper is the following:

_Answering natural language questions over tables is usually seen as a semantic parsing task. To alleviate the collection cost of full logical forms, one popular approach focuses on weak supervision consisting of denotations instead of logical forms. However, training semantic parsers from weak supervision poses difficulties, and in addition, the generated logical forms are only used as an intermediate step prior to retrieving the denotation. In this paper, we present TAPAS, an approach to question answering over tables without generating logical forms. TAPAS trains from weak supervision, and predicts the denotation by selecting table cells and optionally applying a corresponding aggregation operator to such selection. TAPAS extends BERT’s architecture to encode tables as input, initializes from an effective joint pre-training of text segments and tables crawled from Wikipedia, and is trained end-to-end. We experiment with three different semantic parsing datasets, and find that TAPAS outperforms or rivals semantic parsing models by improving state-of-the-art accuracy on SQA from 55.1 to 67.2 and performing on par with the state-of-the-art on WIKISQL and WIKITQ, but with a simpler model architecture. We additionally find that transfer learning, which is trivial in our setting, from WIKISQL to WIKITQ, yields 48.7 accuracy, 4.2 points above the state-of-the-art._

In addition, the authors have further pre-trained TAPAS to recognize **table entailment**, by creating a balanced dataset of millions of automatically created training examples which are learned in an intermediate step prior to fine-tuning. The authors of TAPAS call this further pre-training intermediate pre-training (since TAPAS is first pre-trained on MLM, and then on another dataset). They found that intermediate pre-training further improves performance on SQA, achieving a new state-of-the-art as well as state-of-the-art on [TabFact](https://github.com/wenhuchen/Table-Fact-Checking), a large-scale dataset with 16k Wikipedia tables for table entailment (a binary classification task). For more details, see their follow-up paper: [Understanding tables with intermediate pre-training](https://www.aclweb.org/anthology/2020.findings-emnlp.27/) by Julian Martin Eisenschlos, Syrine Krichene and Thomas Müller.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tapas_architecture.png) TAPAS architecture. Taken from the [original blog post](https://ai.googleblog.com/2020/04/using-neural-networks-to-find-answers.html).

This model was contributed by [nielsr](https://huggingface.co/nielsr). The Tensorflow version of this model was contributed by [kamalkraj](https://huggingface.co/kamalkraj). The original code can be found [here](https://github.com/google-research/tapas).

Tips:

-   TAPAS is a model that uses relative position embeddings by default (restarting the position embeddings at every cell of the table). Note that this is something that was added after the publication of the original TAPAS paper. According to the authors, this usually results in a slightly better performance, and allows you to encode longer sequences without running out of embeddings. This is reflected in the `reset_position_index_per_cell` parameter of [TapasConfig](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasConfig), which is set to `True` by default. The default versions of the models available on the [hub](https://huggingface.co/models?search=tapas) all use relative position embeddings. You can still use the ones with absolute position embeddings by passing in an additional argument `revision="no_reset"` when calling the `from_pretrained()` method. Note that it’s usually advised to pad the inputs on the right rather than the left.
-   TAPAS is based on BERT, so `TAPAS-base` for example corresponds to a `BERT-base` architecture. Of course, `TAPAS-large` will result in the best performance (the results reported in the paper are from `TAPAS-large`). Results of the various sized models are shown on the [original Github repository](https://github.com/google-research/tapas%3E).
-   TAPAS has checkpoints fine-tuned on SQA, which are capable of answering questions related to a table in a conversational set-up. This means that you can ask follow-up questions such as “what is his age?” related to the previous question. Note that the forward pass of TAPAS is a bit different in case of a conversational set-up: in that case, you have to feed every table-question pair one by one to the model, such that the `prev_labels` token type ids can be overwritten by the predicted `labels` of the model to the previous question. See “Usage” section for more info.
-   TAPAS is similar to BERT and therefore relies on the masked language modeling (MLM) objective. It is therefore efficient at predicting masked tokens and at NLU in general, but is not optimal for text generation. Models trained with a causal language modeling (CLM) objective are better in that regard. Note that TAPAS can be used as an encoder in the EncoderDecoderModel framework, to combine it with an autoregressive text decoder such as GPT-2.

## Usage: fine-tuning

Here we explain how you can fine-tune [TapasForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasForQuestionAnswering) on your own dataset.

**STEP 1: Choose one of the 3 ways in which you can use TAPAS - or experiment**

Basically, there are 3 different ways in which one can fine-tune [TapasForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasForQuestionAnswering), corresponding to the different datasets on which Tapas was fine-tuned:

1.  SQA: if you’re interested in asking follow-up questions related to a table, in a conversational set-up. For example if you first ask “what’s the name of the first actor?” then you can ask a follow-up question such as “how old is he?“. Here, questions do not involve any aggregation (all questions are cell selection questions).
2.  WTQ: if you’re not interested in asking questions in a conversational set-up, but rather just asking questions related to a table, which might involve aggregation, such as counting a number of rows, summing up cell values or averaging cell values. You can then for example ask “what’s the total number of goals Cristiano Ronaldo made in his career?“. This case is also called **weak supervision**, since the model itself must learn the appropriate aggregation operator (SUM/COUNT/AVERAGE/NONE) given only the answer to the question as supervision.
3.  WikiSQL-supervised: this dataset is based on WikiSQL with the model being given the ground truth aggregation operator during training. This is also called **strong supervision**. Here, learning the appropriate aggregation operator is much easier.

To summarize:

| **Task** | **Example dataset** | **Description** |
| --- | --- | --- |
| Conversational | SQA | Conversational, only cell selection questions |
| Weak supervision for aggregation | WTQ | Questions might involve aggregation, and the model must learn this given only the answer as supervision |
| Strong supervision for aggregation | WikiSQL-supervised | Questions might involve aggregation, and the model must learn this given the gold aggregation operator |

Initializing a model with a pre-trained base and randomly initialized classification heads from the hub can be done as shown below.

```
>>> from transformers import TapasConfig, TapasForQuestionAnswering

>>> 
>>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base")

>>> 
>>> config = TapasConfig.from_pretrained("google/tapas-base-finetuned-wtq")
>>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)

>>> 
>>> config = TapasConfig("google-base-finetuned-wikisql-supervised")
>>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)
```

Of course, you don’t necessarily have to follow one of these three ways in which TAPAS was fine-tuned. You can also experiment by defining any hyperparameters you want when initializing [TapasConfig](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasConfig), and then create a [TapasForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasForQuestionAnswering) based on that configuration. For example, if you have a dataset that has both conversational questions and questions that might involve aggregation, then you can do it this way. Here’s an example:

```
>>> from transformers import TapasConfig, TapasForQuestionAnswering

>>> 
>>> config = TapasConfig(num_aggregation_labels=3, average_logits_per_cell=True)
>>> 
>>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)
```

Initializing a model with a pre-trained base and randomly initialized classification heads from the hub can be done as shown below. Be sure to have installed the [tensorflow\_probability](https://github.com/tensorflow/probability) dependency:

```
>>> from transformers import TapasConfig, TFTapasForQuestionAnswering

>>> 
>>> model = TFTapasForQuestionAnswering.from_pretrained("google/tapas-base")

>>> 
>>> config = TapasConfig.from_pretrained("google/tapas-base-finetuned-wtq")
>>> model = TFTapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)

>>> 
>>> config = TapasConfig("google-base-finetuned-wikisql-supervised")
>>> model = TFTapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)
```

Of course, you don’t necessarily have to follow one of these three ways in which TAPAS was fine-tuned. You can also experiment by defining any hyperparameters you want when initializing [TapasConfig](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasConfig), and then create a [TFTapasForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TFTapasForQuestionAnswering) based on that configuration. For example, if you have a dataset that has both conversational questions and questions that might involve aggregation, then you can do it this way. Here’s an example:

```
>>> from transformers import TapasConfig, TFTapasForQuestionAnswering

>>> 
>>> config = TapasConfig(num_aggregation_labels=3, average_logits_per_cell=True)
>>> 
>>> model = TFTapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)
```

What you can also do is start from an already fine-tuned checkpoint. A note here is that the already fine-tuned checkpoint on WTQ has some issues due to the L2-loss which is somewhat brittle. See [here](https://github.com/google-research/tapas/issues/91#issuecomment-735719340) for more info.

For a list of all pre-trained and fine-tuned TAPAS checkpoints available on HuggingFace’s hub, see [here](https://huggingface.co/models?search=tapas).

**STEP 2: Prepare your data in the SQA format**

Second, no matter what you picked above, you should prepare your dataset in the [SQA](https://www.microsoft.com/en-us/download/details.aspx?id=54253) format. This format is a TSV/CSV file with the following columns:

-   `id`: optional, id of the table-question pair, for bookkeeping purposes.
-   `annotator`: optional, id of the person who annotated the table-question pair, for bookkeeping purposes.
-   `position`: integer indicating if the question is the first, second, third,… related to the table. Only required in case of conversational setup (SQA). You don’t need this column in case you’re going for WTQ/WikiSQL-supervised.
-   `question`: string
-   `table_file`: string, name of a csv file containing the tabular data
-   `answer_coordinates`: list of one or more tuples (each tuple being a cell coordinate, i.e. row, column pair that is part of the answer)
-   `answer_text`: list of one or more strings (each string being a cell value that is part of the answer)
-   `aggregation_label`: index of the aggregation operator. Only required in case of strong supervision for aggregation (the WikiSQL-supervised case)
-   `float_answer`: the float answer to the question, if there is one (np.nan if there isn’t). Only required in case of weak supervision for aggregation (such as WTQ and WikiSQL)

The tables themselves should be present in a folder, each table being a separate csv file. Note that the authors of the TAPAS algorithm used conversion scripts with some automated logic to convert the other datasets (WTQ, WikiSQL) into the SQA format. The author explains this [here](https://github.com/google-research/tapas/issues/50#issuecomment-705465960). A conversion of this script that works with HuggingFace’s implementation can be found [here](https://github.com/NielsRogge/tapas_utils). Interestingly, these conversion scripts are not perfect (the `answer_coordinates` and `float_answer` fields are populated based on the `answer_text`), meaning that WTQ and WikiSQL results could actually be improved.

**STEP 3: Convert your data into tensors using TapasTokenizer**

Third, given that you’ve prepared your data in this TSV/CSV format (and corresponding CSV files containing the tabular data), you can then use [TapasTokenizer](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasTokenizer) to convert table-question pairs into `input_ids`, `attention_mask`, `token_type_ids` and so on. Again, based on which of the three cases you picked above, [TapasForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasForQuestionAnswering) requires different inputs to be fine-tuned:

| **Task** | **Required inputs** |
| --- | --- |
| Conversational | `input_ids`, `attention_mask`, `token_type_ids`, `labels` |
| Weak supervision for aggregation | `input_ids`, `attention_mask`, `token_type_ids`, `labels`, `numeric_values`, `numeric_values_scale`, `float_answer` |
| Strong supervision for aggregation | `input ids`, `attention mask`, `token type ids`, `labels`, `aggregation_labels` |

[TapasTokenizer](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasTokenizer) creates the `labels`, `numeric_values` and `numeric_values_scale` based on the `answer_coordinates` and `answer_text` columns of the TSV file. The `float_answer` and `aggregation_labels` are already in the TSV file of step 2. Here’s an example:

```
>>> from transformers import TapasTokenizer
>>> import pandas as pd

>>> model_name = "google/tapas-base"
>>> tokenizer = TapasTokenizer.from_pretrained(model_name)

>>> data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
>>> queries = [
...     "What is the name of the first actor?",
...     "How many movies has George Clooney played in?",
...     "What is the total number of movies?",
... ]
>>> answer_coordinates = [[(0, 0)], [(2, 1)], [(0, 1), (1, 1), (2, 1)]]
>>> answer_text = [["Brad Pitt"], ["69"], ["209"]]
>>> table = pd.DataFrame.from_dict(data)
>>> inputs = tokenizer(
...     table=table,
...     queries=queries,
...     answer_coordinates=answer_coordinates,
...     answer_text=answer_text,
...     padding="max_length",
...     return_tensors="pt",
... )
>>> inputs
{'input_ids': tensor([[ ... ]]), 'attention_mask': tensor([[...]]), 'token_type_ids': tensor([[[...]]]),
'numeric_values': tensor([[ ... ]]), 'numeric_values_scale: tensor([[ ... ]]), labels: tensor([[ ... ]])}
```

Note that [TapasTokenizer](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasTokenizer) expects the data of the table to be **text-only**. You can use `.astype(str)` on a dataframe to turn it into text-only data. Of course, this only shows how to encode a single training example. It is advised to create a dataloader to iterate over batches:

```
>>> import torch
>>> import pandas as pd

>>> tsv_path = "your_path_to_the_tsv_file"
>>> table_csv_path = "your_path_to_a_directory_containing_all_csv_files"


>>> class TableDataset(torch.utils.data.Dataset):
...     def __init__(self, data, tokenizer):
...         self.data = data
...         self.tokenizer = tokenizer

...     def __getitem__(self, idx):
...         item = data.iloc[idx]
...         table = pd.read_csv(table_csv_path + item.table_file).astype(
...             str
...         )  
...         encoding = self.tokenizer(
...             table=table,
...             queries=item.question,
...             answer_coordinates=item.answer_coordinates,
...             answer_text=item.answer_text,
...             truncation=True,
...             padding="max_length",
...             return_tensors="pt",
...         )
...         
...         encoding = {key: val.squeeze(0) for key, val in encoding.items()}
...         
...         encoding["float_answer"] = torch.tensor(item.float_answer)
...         return encoding

...     def __len__(self):
...         return len(self.data)


>>> data = pd.read_csv(tsv_path, sep="\t")
>>> train_dataset = TableDataset(data, tokenizer)
>>> train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
```

Third, given that you’ve prepared your data in this TSV/CSV format (and corresponding CSV files containing the tabular data), you can then use [TapasTokenizer](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasTokenizer) to convert table-question pairs into `input_ids`, `attention_mask`, `token_type_ids` and so on. Again, based on which of the three cases you picked above, [TFTapasForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TFTapasForQuestionAnswering) requires different inputs to be fine-tuned:

| **Task** | **Required inputs** |
| --- | --- |
| Conversational | `input_ids`, `attention_mask`, `token_type_ids`, `labels` |
| Weak supervision for aggregation | `input_ids`, `attention_mask`, `token_type_ids`, `labels`, `numeric_values`, `numeric_values_scale`, `float_answer` |
| Strong supervision for aggregation | `input ids`, `attention mask`, `token type ids`, `labels`, `aggregation_labels` |

[TapasTokenizer](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasTokenizer) creates the `labels`, `numeric_values` and `numeric_values_scale` based on the `answer_coordinates` and `answer_text` columns of the TSV file. The `float_answer` and `aggregation_labels` are already in the TSV file of step 2. Here’s an example:

```
>>> from transformers import TapasTokenizer
>>> import pandas as pd

>>> model_name = "google/tapas-base"
>>> tokenizer = TapasTokenizer.from_pretrained(model_name)

>>> data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
>>> queries = [
...     "What is the name of the first actor?",
...     "How many movies has George Clooney played in?",
...     "What is the total number of movies?",
... ]
>>> answer_coordinates = [[(0, 0)], [(2, 1)], [(0, 1), (1, 1), (2, 1)]]
>>> answer_text = [["Brad Pitt"], ["69"], ["209"]]
>>> table = pd.DataFrame.from_dict(data)
>>> inputs = tokenizer(
...     table=table,
...     queries=queries,
...     answer_coordinates=answer_coordinates,
...     answer_text=answer_text,
...     padding="max_length",
...     return_tensors="tf",
... )
>>> inputs
{'input_ids': tensor([[ ... ]]), 'attention_mask': tensor([[...]]), 'token_type_ids': tensor([[[...]]]),
'numeric_values': tensor([[ ... ]]), 'numeric_values_scale: tensor([[ ... ]]), labels: tensor([[ ... ]])}
```

Note that [TapasTokenizer](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasTokenizer) expects the data of the table to be **text-only**. You can use `.astype(str)` on a dataframe to turn it into text-only data. Of course, this only shows how to encode a single training example. It is advised to create a dataloader to iterate over batches:

```
>>> import tensorflow as tf
>>> import pandas as pd

>>> tsv_path = "your_path_to_the_tsv_file"
>>> table_csv_path = "your_path_to_a_directory_containing_all_csv_files"


>>> class TableDataset:
...     def __init__(self, data, tokenizer):
...         self.data = data
...         self.tokenizer = tokenizer

...     def __iter__(self):
...         for idx in range(self.__len__()):
...             item = self.data.iloc[idx]
...             table = pd.read_csv(table_csv_path + item.table_file).astype(
...                 str
...             )  
...             encoding = self.tokenizer(
...                 table=table,
...                 queries=item.question,
...                 answer_coordinates=item.answer_coordinates,
...                 answer_text=item.answer_text,
...                 truncation=True,
...                 padding="max_length",
...                 return_tensors="tf",
...             )
...             
...             encoding = {key: tf.squeeze(val, 0) for key, val in encoding.items()}
...             
...             encoding["float_answer"] = tf.convert_to_tensor(item.float_answer, dtype=tf.float32)
...             yield encoding["input_ids"], encoding["attention_mask"], encoding["numeric_values"], encoding[
...                 "numeric_values_scale"
...             ], encoding["token_type_ids"], encoding["labels"], encoding["float_answer"]

...     def __len__(self):
...         return len(self.data)


>>> data = pd.read_csv(tsv_path, sep="\t")
>>> train_dataset = TableDataset(data, tokenizer)
>>> output_signature = (
...     tf.TensorSpec(shape=(512,), dtype=tf.int32),
...     tf.TensorSpec(shape=(512,), dtype=tf.int32),
...     tf.TensorSpec(shape=(512,), dtype=tf.float32),
...     tf.TensorSpec(shape=(512,), dtype=tf.float32),
...     tf.TensorSpec(shape=(512, 7), dtype=tf.int32),
...     tf.TensorSpec(shape=(512,), dtype=tf.int32),
...     tf.TensorSpec(shape=(512,), dtype=tf.float32),
... )
>>> train_dataloader = tf.data.Dataset.from_generator(train_dataset, output_signature=output_signature).batch(32)
```

Note that here, we encode each table-question pair independently. This is fine as long as your dataset is **not conversational**. In case your dataset involves conversational questions (such as in SQA), then you should first group together the `queries`, `answer_coordinates` and `answer_text` per table (in the order of their `position` index) and batch encode each table with its questions. This will make sure that the `prev_labels` token types (see docs of [TapasTokenizer](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasTokenizer)) are set correctly. See [this notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TAPAS/Fine_tuning_TapasForQuestionAnswering_on_SQA.ipynb) for more info. See [this notebook](https://github.com/kamalkraj/Tapas-Tutorial/blob/master/TAPAS/Fine_tuning_TapasForQuestionAnswering_on_SQA.ipynb) for more info regarding using the TensorFlow model.

\*\*STEP 4: Train (fine-tune) the model

You can then fine-tune [TapasForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasForQuestionAnswering) as follows (shown here for the weak supervision for aggregation case):

```
>>> from transformers import TapasConfig, TapasForQuestionAnswering, AdamW

>>> 
>>> config = TapasConfig(
...     num_aggregation_labels=4,
...     use_answer_as_supervision=True,
...     answer_loss_cutoff=0.664694,
...     cell_selection_preference=0.207951,
...     huber_loss_delta=0.121194,
...     init_cell_selection_weights_to_zero=True,
...     select_one_column=True,
...     allow_empty_column_selection=False,
...     temperature=0.0352513,
... )
>>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)

>>> optimizer = AdamW(model.parameters(), lr=5e-5)

>>> model.train()
>>> for epoch in range(2):  
...     for batch in train_dataloader:
...         
...         input_ids = batch["input_ids"]
...         attention_mask = batch["attention_mask"]
...         token_type_ids = batch["token_type_ids"]
...         labels = batch["labels"]
...         numeric_values = batch["numeric_values"]
...         numeric_values_scale = batch["numeric_values_scale"]
...         float_answer = batch["float_answer"]

...         
...         optimizer.zero_grad()

...         
...         outputs = model(
...             input_ids=input_ids,
...             attention_mask=attention_mask,
...             token_type_ids=token_type_ids,
...             labels=labels,
...             numeric_values=numeric_values,
...             numeric_values_scale=numeric_values_scale,
...             float_answer=float_answer,
...         )
...         loss = outputs.loss
...         loss.backward()
...         optimizer.step()
```

You can then fine-tune [TFTapasForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TFTapasForQuestionAnswering) as follows (shown here for the weak supervision for aggregation case):

```
>>> import tensorflow as tf
>>> from transformers import TapasConfig, TFTapasForQuestionAnswering

>>> 
>>> config = TapasConfig(
...     num_aggregation_labels=4,
...     use_answer_as_supervision=True,
...     answer_loss_cutoff=0.664694,
...     cell_selection_preference=0.207951,
...     huber_loss_delta=0.121194,
...     init_cell_selection_weights_to_zero=True,
...     select_one_column=True,
...     allow_empty_column_selection=False,
...     temperature=0.0352513,
... )
>>> model = TFTapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)

>>> optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

>>> for epoch in range(2):  
...     for batch in train_dataloader:
...         
...         input_ids = batch[0]
...         attention_mask = batch[1]
...         token_type_ids = batch[4]
...         labels = batch[-1]
...         numeric_values = batch[2]
...         numeric_values_scale = batch[3]
...         float_answer = batch[6]

...         
...         with tf.GradientTape() as tape:
...             outputs = model(
...                 input_ids=input_ids,
...                 attention_mask=attention_mask,
...                 token_type_ids=token_type_ids,
...                 labels=labels,
...                 numeric_values=numeric_values,
...                 numeric_values_scale=numeric_values_scale,
...                 float_answer=float_answer,
...             )
...         grads = tape.gradient(outputs.loss, model.trainable_weights)
...         optimizer.apply_gradients(zip(grads, model.trainable_weights))
```

## Usage: inference

Here we explain how you can use [TapasForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasForQuestionAnswering) or [TFTapasForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TFTapasForQuestionAnswering) for inference (i.e. making predictions on new data). For inference, only `input_ids`, `attention_mask` and `token_type_ids` (which you can obtain using [TapasTokenizer](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasTokenizer)) have to be provided to the model to obtain the logits. Next, you can use the handy `~models.tapas.tokenization_tapas.convert_logits_to_predictions` method to convert these into predicted coordinates and optional aggregation indices.

However, note that inference is **different** depending on whether or not the setup is conversational. In a non-conversational set-up, inference can be done in parallel on all table-question pairs of a batch. Here’s an example of that:

```
>>> from transformers import TapasTokenizer, TapasForQuestionAnswering
>>> import pandas as pd

>>> model_name = "google/tapas-base-finetuned-wtq"
>>> model = TapasForQuestionAnswering.from_pretrained(model_name)
>>> tokenizer = TapasTokenizer.from_pretrained(model_name)

>>> data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
>>> queries = [
...     "What is the name of the first actor?",
...     "How many movies has George Clooney played in?",
...     "What is the total number of movies?",
... ]
>>> table = pd.DataFrame.from_dict(data)
>>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
>>> outputs = model(**inputs)
>>> predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
...     inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
... )

>>> 
>>> id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
>>> aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

>>> answers = []
>>> for coordinates in predicted_answer_coordinates:
...     if len(coordinates) == 1:
...         
...         answers.append(table.iat[coordinates[0]])
...     else:
...         
...         cell_values = []
...         for coordinate in coordinates:
...             cell_values.append(table.iat[coordinate])
...         answers.append(", ".join(cell_values))

>>> display(table)
>>> print("")
>>> for query, answer, predicted_agg in zip(queries, answers, aggregation_predictions_string):
...     print(query)
...     if predicted_agg == "NONE":
...         print("Predicted answer: " + answer)
...     else:
...         print("Predicted answer: " + predicted_agg + " > " + answer)
What is the name of the first actor?
Predicted answer: Brad Pitt
How many movies has George Clooney played in?
Predicted answer: COUNT > 69
What is the total number of movies?
Predicted answer: SUM > 87, 53, 69
```

Here we explain how you can use [TFTapasForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TFTapasForQuestionAnswering) for inference (i.e. making predictions on new data). For inference, only `input_ids`, `attention_mask` and `token_type_ids` (which you can obtain using [TapasTokenizer](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasTokenizer)) have to be provided to the model to obtain the logits. Next, you can use the handy `~models.tapas.tokenization_tapas.convert_logits_to_predictions` method to convert these into predicted coordinates and optional aggregation indices.

However, note that inference is **different** depending on whether or not the setup is conversational. In a non-conversational set-up, inference can be done in parallel on all table-question pairs of a batch. Here’s an example of that:

```
>>> from transformers import TapasTokenizer, TFTapasForQuestionAnswering
>>> import pandas as pd

>>> model_name = "google/tapas-base-finetuned-wtq"
>>> model = TFTapasForQuestionAnswering.from_pretrained(model_name)
>>> tokenizer = TapasTokenizer.from_pretrained(model_name)

>>> data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
>>> queries = [
...     "What is the name of the first actor?",
...     "How many movies has George Clooney played in?",
...     "What is the total number of movies?",
... ]
>>> table = pd.DataFrame.from_dict(data)
>>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="tf")
>>> outputs = model(**inputs)
>>> predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
...     inputs, outputs.logits, outputs.logits_aggregation
... )

>>> 
>>> id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
>>> aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

>>> answers = []
>>> for coordinates in predicted_answer_coordinates:
...     if len(coordinates) == 1:
...         
...         answers.append(table.iat[coordinates[0]])
...     else:
...         
...         cell_values = []
...         for coordinate in coordinates:
...             cell_values.append(table.iat[coordinate])
...         answers.append(", ".join(cell_values))

>>> display(table)
>>> print("")
>>> for query, answer, predicted_agg in zip(queries, answers, aggregation_predictions_string):
...     print(query)
...     if predicted_agg == "NONE":
...         print("Predicted answer: " + answer)
...     else:
...         print("Predicted answer: " + predicted_agg + " > " + answer)
What is the name of the first actor?
Predicted answer: Brad Pitt
How many movies has George Clooney played in?
Predicted answer: COUNT > 69
What is the total number of movies?
Predicted answer: SUM > 87, 53, 69
```

In case of a conversational set-up, then each table-question pair must be provided **sequentially** to the model, such that the `prev_labels` token types can be overwritten by the predicted `labels` of the previous table-question pair. Again, more info can be found in [this notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TAPAS/Fine_tuning_TapasForQuestionAnswering_on_SQA.ipynb) (for PyTorch) and [this notebook](https://github.com/kamalkraj/Tapas-Tutorial/blob/master/TAPAS/Fine_tuning_TapasForQuestionAnswering_on_SQA.ipynb) (for TensorFlow).

## Documentation resources

-   [Text classification task guide](../tasks/sequence_classification)
-   [Masked language modeling task guide](../tasks/masked_language_modeling)

## TAPAS specific outputs

### class transformers.models.tapas.modeling\_tapas.TableQuestionAnsweringOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tapas/modeling_tapas.py#L98)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Nonelogits\_aggregation: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Output type of [TapasForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasForQuestionAnswering).

## TapasConfig

### class transformers.TapasConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tapas/configuration_tapas.py#L45)

( vocab\_size = 30522hidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout\_prob = 0.1attention\_probs\_dropout\_prob = 0.1max\_position\_embeddings = 1024type\_vocab\_sizes = \[3, 256, 256, 2, 256, 256, 10\]initializer\_range = 0.02layer\_norm\_eps = 1e-12pad\_token\_id = 0positive\_label\_weight = 10.0num\_aggregation\_labels = 0aggregation\_loss\_weight = 1.0use\_answer\_as\_supervision = Noneanswer\_loss\_importance = 1.0use\_normalized\_answer\_loss = Falsehuber\_loss\_delta = Nonetemperature = 1.0aggregation\_temperature = 1.0use\_gumbel\_for\_cells = Falseuse\_gumbel\_for\_aggregation = Falseaverage\_approximation\_function = 'ratio'cell\_selection\_preference = Noneanswer\_loss\_cutoff = Nonemax\_num\_rows = 64max\_num\_columns = 32average\_logits\_per\_cell = Falseselect\_one\_column = Trueallow\_empty\_column\_selection = Falseinit\_cell\_selection\_weights\_to\_zero = Falsereset\_position\_index\_per\_cell = Truedisable\_per\_token\_loss = Falseaggregation\_labels = Noneno\_aggregation\_label\_index = None\*\*kwargs )

This is the configuration class to store the configuration of a [TapasModel](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasModel). It is used to instantiate a TAPAS model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the TAPAS [google/tapas-base-finetuned-sqa](https://huggingface.co/google/tapas-base-finetuned-sqa) architecture.

Configuration objects inherit from `PreTrainedConfig` and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Hyperparameters additional to BERT are taken from run\_task\_main.py and hparam\_utils.py of the original implementation. Original implementation available at [https://github.com/google-research/tapas/tree/master](https://github.com/google-research/tapas/tree/master).

Example:

```
>>> from transformers import TapasModel, TapasConfig

>>> 
>>> configuration = TapasConfig()
>>> 
>>> model = TapasModel(configuration)
>>> 
>>> configuration = model.config
```

## TapasTokenizer

### class transformers.TapasTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tapas/tokenization_tapas.py#L237)

( vocab\_filedo\_lower\_case = Truedo\_basic\_tokenize = Truenever\_split = Noneunk\_token = '\[UNK\]'sep\_token = '\[SEP\]'pad\_token = '\[PAD\]'cls\_token = '\[CLS\]'mask\_token = '\[MASK\]'empty\_token = '\[EMPTY\]'tokenize\_chinese\_chars = Truestrip\_accents = Nonecell\_trim\_length: int = -1max\_column\_id: int = Nonemax\_row\_id: int = Nonestrip\_column\_names: bool = Falseupdate\_answer\_coordinates: bool = Falsemin\_question\_length = Nonemax\_question\_length = Nonemodel\_max\_length: int = 512additional\_special\_tokens: typing.Optional\[typing.List\[str\]\] = None\*\*kwargs )

Construct a TAPAS tokenizer. Based on WordPiece. Flattens a table and one or more related sentences to be used by TAPAS models.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods. [TapasTokenizer](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasTokenizer) creates several token type ids to encode tabular structure. To be more precise, it adds 7 token type ids, in the following order: `segment_ids`, `column_ids`, `row_ids`, `prev_labels`, `column_ranks`, `inv_column_ranks` and `numeric_relations`:

-   segment\_ids: indicate whether a token belongs to the question (0) or the table (1). 0 for special tokens and padding.
-   column\_ids: indicate to which column of the table a token belongs (starting from 1). Is 0 for all question tokens, special tokens and padding.
-   row\_ids: indicate to which row of the table a token belongs (starting from 1). Is 0 for all question tokens, special tokens and padding. Tokens of column headers are also 0.
-   prev\_labels: indicate whether a token was (part of) an answer to the previous question (1) or not (0). Useful in a conversational setup (such as SQA).
-   column\_ranks: indicate the rank of a table token relative to a column, if applicable. For example, if you have a column “number of movies” with values 87, 53 and 69, then the column ranks of these tokens are 3, 1 and 2 respectively. 0 for all question tokens, special tokens and padding.
-   inv\_column\_ranks: indicate the inverse rank of a table token relative to a column, if applicable. For example, if you have a column “number of movies” with values 87, 53 and 69, then the inverse column ranks of these tokens are 1, 3 and 2 respectively. 0 for all question tokens, special tokens and padding.
-   numeric\_relations: indicate numeric relations between the question and the tokens of the table. 0 for all question tokens, special tokens and padding.

[TapasTokenizer](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasTokenizer) runs end-to-end tokenization on a table and associated sentences: punctuation splitting and wordpiece.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tapas/tokenization_tapas.py#L588)

( table: pd.DataFramequeries: typing.Union\[str, typing.List\[str\], typing.List\[int\], typing.List\[typing.List\[str\]\], typing.List\[typing.List\[int\]\], NoneType\] = Noneanswer\_coordinates: typing.Union\[typing.List\[typing.Tuple\], typing.List\[typing.List\[typing.Tuple\]\], NoneType\] = Noneanswer\_text: typing.Union\[typing.List\[str\], typing.List\[typing.List\[str\]\], NoneType\] = Noneadd\_special\_tokens: bool = Truepadding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = Falsetruncation: typing.Union\[bool, str, transformers.models.tapas.tokenization\_tapas.TapasTruncationStrategy\] = Falsemax\_length: typing.Optional\[int\] = Nonepad\_to\_multiple\_of: typing.Optional\[int\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonereturn\_token\_type\_ids: typing.Optional\[bool\] = Nonereturn\_attention\_mask: typing.Optional\[bool\] = Nonereturn\_overflowing\_tokens: bool = Falsereturn\_special\_tokens\_mask: bool = Falsereturn\_offsets\_mapping: bool = Falsereturn\_length: bool = Falseverbose: bool = True\*\*kwargs )

Main method to tokenize and prepare for the model one or several sequence(s) related to a table.

#### convert\_logits\_to\_predictions

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tapas/tokenization_tapas.py#L1951)

( datalogitslogits\_agg = Nonecell\_classification\_threshold = 0.5 ) → `tuple` comprising various elements depending on the inputs

Converts logits of [TapasForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasForQuestionAnswering) to actual predicted answer coordinates and optional aggregation indices.

The original implementation, on which this function is based, can be found [here](https://github.com/google-research/tapas/blob/4908213eb4df7aa988573350278b44c4dbe3f71b/tapas/experiments/prediction_utils.py#L288).

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tapas/tokenization_tapas.py#L456)

( save\_directory: strfilename\_prefix: typing.Optional\[str\] = None )

## TapasModel

### class transformers.TapasModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tapas/modeling_tapas.py#L848)

( configadd\_pooling\_layer = True )

Parameters

-   **config** ([TapasConfig](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Tapas Model transformer outputting raw hidden-states without any specific head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

This class is a small change compared to [BertModel](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertModel), taking into account the additional token type ids.

The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of cross-attention is added between the self-attention layers, following the architecture described in [Attention is all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tapas/modeling_tapas.py#L885)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

The [TapasModel](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, TapasModel
>>> import pandas as pd

>>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base")
>>> model = TapasModel.from_pretrained("google/tapas-base")

>>> data = {
...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
...     "Age": ["56", "45", "59"],
...     "Number of movies": ["87", "53", "69"],
... }
>>> table = pd.DataFrame.from_dict(data)
>>> queries = ["How many movies has George Clooney played in?", "How old is Brad Pitt?"]

>>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## TapasForMaskedLM

### class transformers.TapasForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tapas/modeling_tapas.py#L1001)

( config )

Parameters

-   **config** ([TapasConfig](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Tapas Model with a `language modeling` head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tapas/modeling_tapas.py#L1021)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None\*\*kwargs ) → [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

The [TapasForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, TapasForMaskedLM
>>> import pandas as pd

>>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base")
>>> model = TapasForMaskedLM.from_pretrained("google/tapas-base")

>>> data = {
...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
...     "Age": ["56", "45", "59"],
...     "Number of movies": ["87", "53", "69"],
... }
>>> table = pd.DataFrame.from_dict(data)

>>> inputs = tokenizer(
...     table=table, queries="How many [MASK] has George [MASK] played in?", return_tensors="pt"
... )
>>> labels = tokenizer(
...     table=table, queries="How many movies has George Clooney played in?", return_tensors="pt"
... )["input_ids"]

>>> outputs = model(**inputs, labels=labels)
>>> logits = outputs.logits
```

## TapasForSequenceClassification

### class transformers.TapasForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tapas/modeling_tapas.py#L1462)

( config )

Parameters

-   **config** ([TapasConfig](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Tapas Model with a sequence classification head on top (a linear layer on top of the pooled output), e.g. for table entailment tasks, such as TabFact (Chen et al., 2020).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tapas/modeling_tapas.py#L1474)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [TapasForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, TapasForSequenceClassification
>>> import torch
>>> import pandas as pd

>>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-finetuned-tabfact")
>>> model = TapasForSequenceClassification.from_pretrained("google/tapas-base-finetuned-tabfact")

>>> data = {
...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
...     "Age": ["56", "45", "59"],
...     "Number of movies": ["87", "53", "69"],
... }
>>> table = pd.DataFrame.from_dict(data)
>>> queries = [
...     "There is only one actor who is 45 years old",
...     "There are 3 actors which played in more than 60 movies",
... ]

>>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
>>> labels = torch.tensor([1, 0])  

>>> outputs = model(**inputs, labels=labels)
>>> loss = outputs.loss
>>> logits = outputs.logits
```

## TapasForQuestionAnswering

### class transformers.TapasForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tapas/modeling_tapas.py#L1117)

( config: TapasConfig )

Parameters

-   **config** ([TapasConfig](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Tapas Model with a cell selection head and optional aggregation head on top for question-answering tasks on tables (linear layers on top of the hidden-states output to compute `logits` and optional `logits_aggregation`), e.g. for SQA, WTQ or WikiSQL-supervised tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tapas/modeling_tapas.py#L1152)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonetable\_mask: typing.Optional\[torch.LongTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneaggregation\_labels: typing.Optional\[torch.LongTensor\] = Nonefloat\_answer: typing.Optional\[torch.FloatTensor\] = Nonenumeric\_values: typing.Optional\[torch.FloatTensor\] = Nonenumeric\_values\_scale: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.models.tapas.modeling\_tapas.TableQuestionAnsweringOutput](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.models.tapas.modeling_tapas.TableQuestionAnsweringOutput) or `tuple(torch.FloatTensor)`

The [TapasForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, TapasForQuestionAnswering
>>> import pandas as pd

>>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
>>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")

>>> data = {
...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
...     "Age": ["56", "45", "59"],
...     "Number of movies": ["87", "53", "69"],
... }
>>> table = pd.DataFrame.from_dict(data)
>>> queries = ["How many movies has George Clooney played in?", "How old is Brad Pitt?"]

>>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
>>> outputs = model(**inputs)

>>> logits = outputs.logits
>>> logits_aggregation = outputs.logits_aggregation
```

## TFTapasModel

### class transformers.TFTapasModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tapas/modeling_tf_tapas.py#L974)

( \*args\*\*kwargs )

Parameters

-   **config** ([TapasConfig](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Tapas Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TensorFlow models and layers in `transformers` accept two formats as input:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional argument.

The reason the second format is supported is that Keras methods prefer this format when passing inputs to models and layers. Because of this support, when using methods like `model.fit()` things should “just work” for you - just pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first positional argument:

-   a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
-   a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
-   a dictionary with one or several input Tensors associated to the input names given in the docstring: `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

Note that when creating models and layers with [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don’t need to worry about any of this, as you can just pass inputs like you would to any other Python function!

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tapas/modeling_tf_tapas.py#L980)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonetoken\_type\_ids: np.ndarray | tf.Tensor | None = Noneposition\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFBaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling) or `tuple(tf.Tensor)`

The [TFTapasModel](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TFTapasModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, TapasModel
>>> import pandas as pd

>>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base")
>>> model = TapasModel.from_pretrained("google/tapas-base")

>>> data = {
...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
...     "Age": ["56", "45", "59"],
...     "Number of movies": ["87", "53", "69"],
... }
>>> table = pd.DataFrame.from_dict(data)
>>> queries = ["How many movies has George Clooney played in?", "How old is Brad Pitt?"]

>>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="tf")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## TFTapasForMaskedLM

### class transformers.TFTapasForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tapas/modeling_tf_tapas.py#L1038)

( \*args\*\*kwargs )

Parameters

-   **config** ([TapasConfig](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Tapas Model with a `language modeling` head on top.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TensorFlow models and layers in `transformers` accept two formats as input:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional argument.

The reason the second format is supported is that Keras methods prefer this format when passing inputs to models and layers. Because of this support, when using methods like `model.fit()` things should “just work” for you - just pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first positional argument:

-   a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
-   a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
-   a dictionary with one or several input Tensors associated to the input names given in the docstring: `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

Note that when creating models and layers with [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don’t need to worry about any of this, as you can just pass inputs like you would to any other Python function!

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tapas/modeling_tf_tapas.py#L1054)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonetoken\_type\_ids: np.ndarray | tf.Tensor | None = Noneposition\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonelabels: np.ndarray | tf.Tensor | None = Nonetraining: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFMaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFMaskedLMOutput) or `tuple(tf.Tensor)`

The [TFTapasForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TFTapasForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, TapasForMaskedLM
>>> import pandas as pd

>>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base")
>>> model = TapasForMaskedLM.from_pretrained("google/tapas-base")

>>> data = {
...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
...     "Age": ["56", "45", "59"],
...     "Number of movies": ["87", "53", "69"],
... }
>>> table = pd.DataFrame.from_dict(data)

>>> inputs = tokenizer(
...     table=table, queries="How many [MASK] has George [MASK] played in?", return_tensors="tf"
... )
>>> labels = tokenizer(
...     table=table, queries="How many movies has George Clooney played in?", return_tensors="tf"
... )["input_ids"]

>>> outputs = model(**inputs, labels=labels)
>>> logits = outputs.logits
```

## TFTapasForSequenceClassification

### class transformers.TFTapasForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tapas/modeling_tf_tapas.py#L1563)

( \*args\*\*kwargs )

Parameters

-   **config** ([TapasConfig](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Tapas Model with a sequence classification head on top (a linear layer on top of the pooled output), e.g. for table entailment tasks, such as TabFact (Chen et al., 2020).

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TensorFlow models and layers in `transformers` accept two formats as input:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional argument.

The reason the second format is supported is that Keras methods prefer this format when passing inputs to models and layers. Because of this support, when using methods like `model.fit()` things should “just work” for you - just pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first positional argument:

-   a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
-   a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
-   a dictionary with one or several input Tensors associated to the input names given in the docstring: `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

Note that when creating models and layers with [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don’t need to worry about any of this, as you can just pass inputs like you would to any other Python function!

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tapas/modeling_tf_tapas.py#L1574)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonetoken\_type\_ids: np.ndarray | tf.Tensor | None = Noneposition\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonelabels: np.ndarray | tf.Tensor | None = Nonetraining: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSequenceClassifierOutput) or `tuple(tf.Tensor)`

The [TFTapasForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TFTapasForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, TapasForSequenceClassification
>>> import tensorflow as tf
>>> import pandas as pd

>>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-finetuned-tabfact")
>>> model = TapasForSequenceClassification.from_pretrained("google/tapas-base-finetuned-tabfact")

>>> data = {
...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
...     "Age": ["56", "45", "59"],
...     "Number of movies": ["87", "53", "69"],
... }
>>> table = pd.DataFrame.from_dict(data)
>>> queries = [
...     "There is only one actor who is 45 years old",
...     "There are 3 actors which played in more than 60 movies",
... ]

>>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="tf")
>>> labels = tf.convert_to_tensor([1, 0])  

>>> outputs = model(**inputs, labels=labels)
>>> loss = outputs.loss
>>> logits = outputs.logits
```

## TFTapasForQuestionAnswering

### class transformers.TFTapasForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tapas/modeling_tf_tapas.py#L1238)

( \*args\*\*kwargs )

Parameters

-   **config** ([TapasConfig](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TapasConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Tapas Model with a cell selection head and optional aggregation head on top for question-answering tasks on tables (linear layers on top of the hidden-states output to compute `logits` and optional `logits_aggregation`), e.g. for SQA, WTQ or WikiSQL-supervised tasks.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TensorFlow models and layers in `transformers` accept two formats as input:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional argument.

The reason the second format is supported is that Keras methods prefer this format when passing inputs to models and layers. Because of this support, when using methods like `model.fit()` things should “just work” for you - just pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first positional argument:

-   a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
-   a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
-   a dictionary with one or several input Tensors associated to the input names given in the docstring: `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

Note that when creating models and layers with [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don’t need to worry about any of this, as you can just pass inputs like you would to any other Python function!

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tapas/modeling_tf_tapas.py#L1260)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonetoken\_type\_ids: np.ndarray | tf.Tensor | None = Noneposition\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Nonetable\_mask: np.ndarray | tf.Tensor | None = Noneaggregation\_labels: np.ndarray | tf.Tensor | None = Nonefloat\_answer: np.ndarray | tf.Tensor | None = Nonenumeric\_values: np.ndarray | tf.Tensor | None = Nonenumeric\_values\_scale: np.ndarray | tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonelabels: np.ndarray | tf.Tensor | None = Nonetraining: Optional\[bool\] = False ) → `transformers.models.tapas.modeling_tf_tapas.TFTableQuestionAnsweringOutput` or `tuple(tf.Tensor)`

The [TFTapasForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/tapas#transformers.TFTapasForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, TapasForQuestionAnswering
>>> import pandas as pd

>>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
>>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")

>>> data = {
...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
...     "Age": ["56", "45", "59"],
...     "Number of movies": ["87", "53", "69"],
... }
>>> table = pd.DataFrame.from_dict(data)
>>> queries = ["How many movies has George Clooney played in?", "How old is Brad Pitt?"]

>>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="tf")
>>> outputs = model(**inputs)

>>> logits = outputs.logits
>>> logits_aggregation = outputs.logits_aggregation
```