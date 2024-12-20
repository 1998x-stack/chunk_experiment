# RAG

[![Models](https://img.shields.io/badge/All_model_pages-rag-blueviolet)](https://huggingface.co/models?filter=rag)

## Overview

Retrieval-augmented generation (“RAG”) models combine the powers of pretrained dense retrieval (DPR) and sequence-to-sequence models. RAG models retrieve documents, pass them to a seq2seq model, then marginalize to generate outputs. The retriever and seq2seq modules are initialized from pretrained models, and fine-tuned jointly, allowing both retrieval and generation to adapt to downstream tasks.

It is based on the paper [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) by Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela.

The abstract from the paper is the following:

_Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely manipulate knowledge is still limited, and hence on knowledge-intensive tasks, their performance lags behind task-specific architectures. Additionally, providing provenance for their decisions and updating their world knowledge remain open research problems. Pre-trained models with a differentiable access mechanism to explicit nonparametric memory can overcome this issue, but have so far been only investigated for extractive downstream tasks. We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) — models which combine pre-trained parametric and non-parametric memory for language generation. We introduce RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever. We compare two RAG formulations, one which conditions on the same retrieved passages across the whole generated sequence, the other can use different passages per token. We fine-tune and evaluate our models on a wide range of knowledge-intensive NLP tasks and set the state-of-the-art on three open domain QA tasks, outperforming parametric seq2seq models and task-specific retrieve-and-extract architectures. For language generation tasks, we find that RAG models generate more specific, diverse and factual language than a state-of-the-art parametric-only seq2seq baseline._

This model was contributed by [ola13](https://huggingface.co/ola13).

Tips:

-   Retrieval-augmented generation (“RAG”) models combine the powers of pretrained dense retrieval (DPR) and Seq2Seq models. RAG models retrieve docs, pass them to a seq2seq model, then marginalize to generate outputs. The retriever and seq2seq modules are initialized from pretrained models, and fine-tuned jointly, allowing both retrieval and generation to adapt to downstream tasks.

## RagConfig

### class transformers.RagConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/configuration_rag.py#L81)

( vocab\_size = Noneis\_encoder\_decoder = Trueprefix = Nonebos\_token\_id = Nonepad\_token\_id = Noneeos\_token\_id = Nonedecoder\_start\_token\_id = Nonetitle\_sep = ' / 'doc\_sep = ' // 'n\_docs = 5max\_combined\_length = 300retrieval\_vector\_size = 768retrieval\_batch\_size = 8dataset = 'wiki\_dpr'dataset\_split = 'train'index\_name = 'compressed'index\_path = Nonepassages\_path = Noneuse\_dummy\_dataset = Falsereduce\_loss = Falselabel\_smoothing = 0.0do\_deduplication = Trueexclude\_bos\_score = Falsedo\_marginalize = Falseoutput\_retrieved = Falseuse\_cache = Trueforced\_eos\_token\_id = None\*\*kwargs )

[RagConfig](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagConfig) stores the configuration of a _RagModel_. Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

#### from\_question\_encoder\_generator\_configs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/configuration_rag.py#L169)

( question\_encoder\_config: PretrainedConfiggenerator\_config: PretrainedConfig\*\*kwargs ) → [EncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig)

An instance of a configuration object

Instantiate a [EncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig) (or a derived class) from a pre-trained encoder model configuration and decoder model configuration.

## RagTokenizer

### class transformers.RagTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/tokenization_rag.py#L28)

( question\_encodergenerator )

## Rag specific outputs

### class transformers.models.rag.modeling\_rag.RetrievAugLMMarginOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/modeling_rag.py#L39)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Nonedoc\_scores: FloatTensor = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneretrieved\_doc\_embeds: typing.Optional\[torch.FloatTensor\] = Noneretrieved\_doc\_ids: typing.Optional\[torch.LongTensor\] = Nonecontext\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonecontext\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonequestion\_encoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Nonequestion\_enc\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonequestion\_enc\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonegenerator\_enc\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Nonegenerator\_enc\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonegenerator\_enc\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonegenerator\_dec\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonegenerator\_dec\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonegenerator\_cross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for retriever augmented marginalized models outputs.

### class transformers.models.rag.modeling\_rag.RetrievAugLMOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/modeling_rag.py#L134)

( logits: FloatTensor = Nonedoc\_scores: FloatTensor = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneretrieved\_doc\_embeds: typing.Optional\[torch.FloatTensor\] = Noneretrieved\_doc\_ids: typing.Optional\[torch.LongTensor\] = Nonecontext\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonecontext\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonequestion\_encoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Nonequestion\_enc\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonequestion\_enc\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonegenerator\_enc\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Nonegenerator\_enc\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonegenerator\_enc\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonegenerator\_dec\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonegenerator\_dec\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonegenerator\_cross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

## RagRetriever

### class transformers.RagRetriever

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/retrieval_rag.py#L323)

( configquestion\_encoder\_tokenizergenerator\_tokenizerindex = Noneinit\_retrieval = True )

Parameters

-   **config** ([RagConfig](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagConfig)) — The configuration of the RAG model this Retriever is used with. Contains parameters indicating which `Index` to build. You can load your own custom dataset with `config.index_name="custom"` or use a canonical one (default) from the datasets library with `config.index_name="wiki_dpr"` for example.
-   **question\_encoder\_tokenizer** ([PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)) — The tokenizer that was used to tokenize the question. It is used to decode the question and then use the generator\_tokenizer.
-   **generator\_tokenizer** ([PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)) — The tokenizer used for the generator part of the RagModel.
-   **index** (`Index`, optional, defaults to the one defined by the configuration) — If specified, use this index instead of the one built using the configuration

Retriever used to get documents from vector queries. It retrieves the documents embeddings as well as the documents contents, and it formats them to be used with a RagModel.

Examples:

```
>>> 
>>> from transformers import RagRetriever

>>> retriever = RagRetriever.from_pretrained(
...     "facebook/dpr-ctx_encoder-single-nq-base", dataset="wiki_dpr", index_name="compressed"
... )

>>> 
>>> from transformers import RagRetriever

>>> dataset = (
...     ...
... )  
>>> retriever = RagRetriever.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", indexed_dataset=dataset)

>>> 
>>> from transformers import RagRetriever

>>> dataset_path = "path/to/my/dataset"  
>>> index_path = "path/to/my/index.faiss"  
>>> retriever = RagRetriever.from_pretrained(
...     "facebook/dpr-ctx_encoder-single-nq-base",
...     index_name="custom",
...     passages_path=dataset_path,
...     index_path=index_path,
... )

>>> 
>>> from transformers import RagRetriever

>>> retriever = RagRetriever.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", index_name="legacy")
```

Retriever initialization function. It loads the index into memory.

#### postprocess\_docs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/retrieval_rag.py#L465)

( docsinput\_stringsprefixn\_docsreturn\_tensors = None ) → `tuple(tensors)`

Parameters

-   **docs** (`dict`) — Retrieved documents.
-   **input\_strings** (`str`) — Input strings decoded by `preprocess_query`.
-   **prefix** (`str`) — Prefix added at the beginning of each input, typically used with T5-based models.

a tuple consisting of two elements: contextualized `input_ids` and a compatible `attention_mask`.

Postprocessing retrieved `docs` and combining them with `input_strings`.

#### retrieve

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/retrieval_rag.py#L537)

( question\_hidden\_states: ndarrayn\_docs: int ) → `Tuple[np.ndarray, np.ndarray, List[dict]]`

Parameters

-   **question\_hidden\_states** (`np.ndarray` of shape `(batch_size, vector_size)`) — A batch of query vectors to retrieve with.
-   **n\_docs** (`int`) — The number of docs retrieved per query.

Returns

`Tuple[np.ndarray, np.ndarray, List[dict]]`

A tuple with the following objects:

-   **retrieved\_doc\_embeds** (`np.ndarray` of shape `(batch_size, n_docs, dim)`) — The retrieval embeddings of the retrieved docs per query.
-   **doc\_ids** (`np.ndarray` of shape `(batch_size, n_docs)`) — The ids of the documents in the index
-   **doc\_dicts** (`List[dict]`): The `retrieved_doc_embeds` examples per query.

Retrieves documents for specified `question_hidden_states`.

## RagModel

### class transformers.RagModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/modeling_rag.py#L489)

( config: typing.Optional\[transformers.configuration\_utils.PretrainedConfig\] = Nonequestion\_encoder: typing.Optional\[transformers.modeling\_utils.PreTrainedModel\] = Nonegenerator: typing.Optional\[transformers.modeling\_utils.PreTrainedModel\] = Noneretriever: typing.Optional\[transformers.models.rag.retrieval\_rag.RagRetriever\] = None\*\*kwargs )

Parameters

-   **config** ([RagConfig](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
-   **question\_encoder** ([PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel)) — An encoder model compatible with the faiss index encapsulated by the `retriever`.
-   **generator** ([PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel)) — A seq2seq model used as the generator in the RAG architecture.
-   **retriever** ([RagRetriever](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagRetriever)) — A retriever class encapsulating a faiss index queried to obtain context documents for current inputs.

The [RagModel](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

RAG is a seq2seq model which encapsulates two core components: a question encoder and a generator. During a forward pass, we encode the input with the question encoder and pass it to the retriever to extract relevant context documents. The documents are then prepended to the input. Such contextualized inputs is passed to the generator.

The question encoder can be any _autoencoding_ model, preferably [DPRQuestionEncoder](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRQuestionEncoder), and the generator can be any _seq2seq_ model, preferably [BartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartForConditionalGeneration).

The model can be initialized with a [RagRetriever](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagRetriever) for end-to-end generation or used in combination with the outputs of a retriever in multiple steps---see examples for more details. The model is compatible any _autoencoding_ model as the `question_encoder` and any _seq2seq_ model with language model head as the `generator`. It has been tested with [DPRQuestionEncoder](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRQuestionEncoder) as the `question_encoder` and [BartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartForConditionalGeneration) or [T5ForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/t5#transformers.T5ForConditionalGeneration) as the `generator`.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/modeling_rag.py#L532)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.BoolTensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonedoc\_scores: typing.Optional\[torch.FloatTensor\] = Nonecontext\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonecontext\_attention\_mask: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneoutput\_retrieved: typing.Optional\[bool\] = Nonen\_docs: typing.Optional\[int\] = None ) → [transformers.models.rag.modeling\_rag.RetrievAugLMOutput](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.models.rag.modeling_rag.RetrievAugLMOutput) or `tuple(torch.FloatTensor)`

The [RagModel](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, RagRetriever, RagModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-base")
>>> retriever = RagRetriever.from_pretrained(
...     "facebook/rag-token-base", index_name="exact", use_dummy_dataset=True
... )
>>> 
>>> model = RagModel.from_pretrained("facebook/rag-token-base", retriever=retriever)

>>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
>>> outputs = model(input_ids=inputs["input_ids"])
```

## RagSequenceForGeneration

### class transformers.RagSequenceForGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/modeling_rag.py#L735)

( config: typing.Optional\[transformers.configuration\_utils.PretrainedConfig\] = Nonequestion\_encoder: typing.Optional\[transformers.modeling\_utils.PreTrainedModel\] = Nonegenerator: typing.Optional\[transformers.modeling\_utils.PreTrainedModel\] = Noneretriever: typing.Optional\[transformers.models.rag.retrieval\_rag.RagRetriever\] = None\*\*kwargs )

Parameters

-   **config** ([RagConfig](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
-   **question\_encoder** ([PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel)) — An encoder model compatible with the faiss index encapsulated by the `retriever`.
-   **generator** ([PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel)) — A seq2seq model used as the generator in the RAG architecture.
-   **retriever** ([RagRetriever](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagRetriever)) — A retriever class encapsulating a faiss index queried to obtain context documents for current inputs.

The [RagSequenceForGeneration](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagSequenceForGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

A RAG-sequence model implementation. It performs RAG-sequence specific marginalization in the forward pass.

RAG is a seq2seq model which encapsulates two core components: a question encoder and a generator. During a forward pass, we encode the input with the question encoder and pass it to the retriever to extract relevant context documents. The documents are then prepended to the input. Such contextualized inputs is passed to the generator.

The question encoder can be any _autoencoding_ model, preferably [DPRQuestionEncoder](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRQuestionEncoder), and the generator can be any _seq2seq_ model, preferably [BartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartForConditionalGeneration).

The model can be initialized with a [RagRetriever](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagRetriever) for end-to-end generation or used in combination with the outputs of a retriever in multiple steps---see examples for more details. The model is compatible any _autoencoding_ model as the `question_encoder` and any _seq2seq_ model with language model head as the `generator`. It has been tested with [DPRQuestionEncoder](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRQuestionEncoder) as the `question_encoder` and [BartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartForConditionalGeneration) or [T5ForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/t5#transformers.T5ForConditionalGeneration) as the `generator`.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/modeling_rag.py#L764)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.BoolTensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = Nonecontext\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonecontext\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonedoc\_scores: typing.Optional\[torch.FloatTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneoutput\_retrieved: typing.Optional\[bool\] = Noneexclude\_bos\_score: typing.Optional\[bool\] = Nonereduce\_loss: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Nonen\_docs: typing.Optional\[int\] = None\*\*kwargs ) → [transformers.models.rag.modeling\_rag.RetrievAugLMMarginOutput](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput) or `tuple(torch.FloatTensor)`

The [RagSequenceForGeneration](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagSequenceForGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, RagRetriever, RagSequenceForGeneration
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
>>> retriever = RagRetriever.from_pretrained(
...     "facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True
... )
>>> 
>>> model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

>>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
>>> targets = tokenizer(text_target="In Paris, there are 10 million people.", return_tensors="pt")
>>> input_ids = inputs["input_ids"]
>>> labels = targets["input_ids"]
>>> outputs = model(input_ids=input_ids, labels=labels)

>>> 
>>> model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=True)
>>> 
>>> question_hidden_states = model.question_encoder(input_ids)[0]
>>> 
>>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
>>> doc_scores = torch.bmm(
...     question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
... ).squeeze(1)
>>> 
>>> outputs = model(
...     context_input_ids=docs_dict["context_input_ids"],
...     context_attention_mask=docs_dict["context_attention_mask"],
...     doc_scores=doc_scores,
...     decoder_input_ids=labels,
... )
```

#### generate

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/modeling_rag.py#L905)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.LongTensor\] = Nonecontext\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonecontext\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonedoc\_scores: typing.Optional\[torch.FloatTensor\] = Nonedo\_deduplication: typing.Optional\[bool\] = Nonenum\_return\_sequences: typing.Optional\[int\] = Nonenum\_beams: typing.Optional\[int\] = Nonen\_docs: typing.Optional\[int\] = None\*\*model\_kwargs ) → `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`

Implements RAG sequence “thorough” decoding. Read the [generate()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.generate)\` documentation for more information on how to set other generate input parameters.

## RagTokenForGeneration

### class transformers.RagTokenForGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/modeling_rag.py#L1133)

( config: typing.Optional\[transformers.configuration\_utils.PretrainedConfig\] = Nonequestion\_encoder: typing.Optional\[transformers.modeling\_utils.PreTrainedModel\] = Nonegenerator: typing.Optional\[transformers.modeling\_utils.PreTrainedModel\] = Noneretriever: typing.Optional\[transformers.models.rag.retrieval\_rag.RagRetriever\] = None\*\*kwargs )

Parameters

-   **config** ([RagConfig](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
-   **question\_encoder** ([PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel)) — An encoder model compatible with the faiss index encapsulated by the `retriever`.
-   **generator** ([PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel)) — A seq2seq model used as the generator in the RAG architecture.
-   **retriever** ([RagRetriever](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagRetriever)) — A retriever class encapsulating a faiss index queried to obtain context documents for current inputs.

The [RagTokenForGeneration](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagTokenForGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

A RAG-token model implementation. It performs RAG-token specific marginalization in the forward pass.

RAG is a seq2seq model which encapsulates two core components: a question encoder and a generator. During a forward pass, we encode the input with the question encoder and pass it to the retriever to extract relevant context documents. The documents are then prepended to the input. Such contextualized inputs is passed to the generator.

The question encoder can be any _autoencoding_ model, preferably [DPRQuestionEncoder](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRQuestionEncoder), and the generator can be any _seq2seq_ model, preferably [BartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartForConditionalGeneration).

The model can be initialized with a [RagRetriever](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagRetriever) for end-to-end generation or used in combination with the outputs of a retriever in multiple steps---see examples for more details. The model is compatible any _autoencoding_ model as the `question_encoder` and any _seq2seq_ model with language model head as the `generator`. It has been tested with [DPRQuestionEncoder](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.DPRQuestionEncoder) as the `question_encoder` and [BartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartForConditionalGeneration) or [T5ForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/t5#transformers.T5ForConditionalGeneration) as the `generator`.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/modeling_rag.py#L1233)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.BoolTensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = Nonecontext\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonecontext\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonedoc\_scores: typing.Optional\[torch.FloatTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneoutput\_retrieved: typing.Optional\[bool\] = Nonedo\_marginalize: typing.Optional\[bool\] = Nonereduce\_loss: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Nonen\_docs: typing.Optional\[int\] = None\*\*kwargs ) → [transformers.models.rag.modeling\_rag.RetrievAugLMMarginOutput](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput) or `tuple(torch.FloatTensor)`

The [RagTokenForGeneration](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagTokenForGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, RagRetriever, RagTokenForGeneration
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
>>> retriever = RagRetriever.from_pretrained(
...     "facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True
... )
>>> 
>>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

>>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
>>> targets = tokenizer(text_target="In Paris, there are 10 million people.", return_tensors="pt")
>>> input_ids = inputs["input_ids"]
>>> labels = targets["input_ids"]
>>> outputs = model(input_ids=input_ids, labels=labels)

>>> 
>>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", use_dummy_dataset=True)
>>> 
>>> question_hidden_states = model.question_encoder(input_ids)[0]
>>> 
>>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
>>> doc_scores = torch.bmm(
...     question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
... ).squeeze(1)
>>> 
>>> outputs = model(
...     context_input_ids=docs_dict["context_input_ids"],
...     context_attention_mask=docs_dict["context_attention_mask"],
...     doc_scores=doc_scores,
...     decoder_input_ids=labels,
... )

>>> 
>>> generated = model.generate(
...     context_input_ids=docs_dict["context_input_ids"],
...     context_attention_mask=docs_dict["context_attention_mask"],
...     doc_scores=doc_scores,
... )
>>> generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
```

#### generate

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/modeling_rag.py#L1374)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.LongTensor\] = Nonecontext\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonecontext\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonedoc\_scores: typing.Optional\[torch.FloatTensor\] = Nonen\_docs: typing.Optional\[int\] = Nonegeneration\_config: typing.Optional\[transformers.generation.configuration\_utils.GenerationConfig\] = Noneprefix\_allowed\_tokens\_fn: typing.Callable\[\[int, torch.Tensor\], typing.List\[int\]\] = Nonelogits\_processor: typing.Optional\[transformers.generation.logits\_process.LogitsProcessorList\] = \[\]stopping\_criteria: typing.Optional\[transformers.generation.stopping\_criteria.StoppingCriteriaList\] = \[\]\*\*kwargs ) → `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`

Implements RAG token decoding.

## TFRagModel

### class transformers.TFRagModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/modeling_tf_rag.py#L496)

( \*args\*\*kwargs )

Parameters

-   **config** ([RagConfig](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.
-   **question\_encoder** ([TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel)) — An encoder model compatible with the faiss index encapsulated by the `retriever`.
-   **generator** ([TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel)) — A seq2seq model used as the generator in the RAG architecture.
-   **retriever** ([RagRetriever](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagRetriever)) — A retriever class encapsulating a faiss index queried to obtain context documents for current inputs.

The [TFRagModel](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.TFRagModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

RAG is a sequence-to-sequence model which encapsulates two core components: a question encoder and a generator. During a forward pass, we encode the input with the question encoder and pass it to the retriever to extract relevant context documents. The documents are then prepended to the input. Such contextualized inputs is passed to the generator.

The question encoder can be any _autoencoding_ model, preferably [TFDPRQuestionEncoder](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.TFDPRQuestionEncoder), and the generator can be any _seq2seq_ model, preferably [TFBartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.TFBartForConditionalGeneration).

The model can be initialized with a [RagRetriever](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagRetriever) for end-to-end generation or used in combination with the outputs of a retriever in multiple steps---see examples for more details. The model is compatible any _autoencoding_ model as the `question_encoder` and any _seq2seq_ model with language model head as the `generator`. It has been tested with [TFDPRQuestionEncoder](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.TFDPRQuestionEncoder) as the `question_encoder` and [TFBartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.TFBartForConditionalGeneration) as the `generator`.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Tensorflow [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

The model is in a developing state as it is now fully supports in eager-mode only, and may not be exported in SavedModel format.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/modeling_tf_rag.py#L546)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Noneencoder\_outputs: np.ndarray | tf.Tensor | None = Nonedecoder\_input\_ids: np.ndarray | tf.Tensor | None = Nonedecoder\_attention\_mask: np.ndarray | tf.Tensor | None = Nonepast\_key\_values: Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\] | None = Nonedoc\_scores: np.ndarray | tf.Tensor | None = Nonecontext\_input\_ids: np.ndarray | tf.Tensor | None = Nonecontext\_attention\_mask: np.ndarray | tf.Tensor | None = Noneuse\_cache: bool | None = Noneoutput\_attentions: bool | None = Noneoutput\_hidden\_states: bool | None = Noneoutput\_retrieved: bool | None = Nonen\_docs: int | None = Nonereturn\_dict: bool | None = Nonetraining: bool = False\*\*kwargs ) → `transformers.models.rag.modeling_tf_rag.TFRetrievAugLMOutput` or `tuple(tf.Tensor)`

The [TFRagModel](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.TFRagModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, RagRetriever, TFRagModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-base")
>>> retriever = RagRetriever.from_pretrained(
...     "facebook/rag-token-base", index_name="exact", use_dummy_dataset=True
... )
>>> 
>>> model = TFRagModel.from_pretrained("facebook/rag-token-base", retriever=retriever, from_pt=True)

>>> input_dict = tokenizer.prepare_seq2seq_batch(
...     "How many people live in Paris?", "In Paris, there are 10 million people.", return_tensors="tf"
... )
>>> input_ids = input_dict["input_ids"]
>>> outputs = model(input_ids)
```

## TFRagSequenceForGeneration

### class transformers.TFRagSequenceForGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/modeling_tf_rag.py#L1301)

( \*args\*\*kwargs )

Parameters

-   **config** ([RagConfig](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.
-   **question\_encoder** ([TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel)) — An encoder model compatible with the faiss index encapsulated by the `retriever`.
-   **generator** ([TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel)) — A seq2seq model used as the generator in the RAG architecture.
-   **retriever** ([RagRetriever](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagRetriever)) — A retriever class encapsulating a faiss index queried to obtain context documents for current inputs.

The [TFRagSequenceForGeneration](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.TFRagSequenceForGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

A TF RAG-sequence model implementation. It performs RAG-sequence specific marginalization in the forward pass.

RAG is a sequence-to-sequence model which encapsulates two core components: a question encoder and a generator. During a forward pass, we encode the input with the question encoder and pass it to the retriever to extract relevant context documents. The documents are then prepended to the input. Such contextualized inputs is passed to the generator.

The question encoder can be any _autoencoding_ model, preferably [TFDPRQuestionEncoder](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.TFDPRQuestionEncoder), and the generator can be any _seq2seq_ model, preferably [TFBartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.TFBartForConditionalGeneration).

The model can be initialized with a [RagRetriever](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagRetriever) for end-to-end generation or used in combination with the outputs of a retriever in multiple steps---see examples for more details. The model is compatible any _autoencoding_ model as the `question_encoder` and any _seq2seq_ model with language model head as the `generator`. It has been tested with [TFDPRQuestionEncoder](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.TFDPRQuestionEncoder) as the `question_encoder` and [TFBartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.TFBartForConditionalGeneration) as the `generator`.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Tensorflow [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

The model is in a developing state as it is now fully supports in eager-mode only, and may not be exported in SavedModel format.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/modeling_tf_rag.py#L1348)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_input\_ids: np.ndarray | tf.Tensor | None = Nonedecoder\_attention\_mask: np.ndarray | tf.Tensor | None = Noneencoder\_outputs: np.ndarray | tf.Tensor | None = Nonepast\_key\_values: Optional\[Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\]\] = Nonedoc\_scores: np.ndarray | tf.Tensor | None = Nonecontext\_input\_ids: np.ndarray | tf.Tensor | None = Nonecontext\_attention\_mask: np.ndarray | tf.Tensor | None = Noneuse\_cache: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Noneoutput\_retrieved: Optional\[bool\] = Nonen\_docs: Optional\[int\] = Noneexclude\_bos\_score: Optional\[bool\] = Nonelabels: np.ndarray | tf.Tensor | None = Nonereduce\_loss: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False\*\*kwargs ) → `transformers.models.rag.modeling_tf_rag.TFRetrievAugLMMarginOutput` or `tuple(tf.Tensor)`

The [TFRagSequenceForGeneration](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.TFRagSequenceForGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, RagRetriever, TFRagSequenceForGeneration

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
>>> retriever = RagRetriever.from_pretrained(
...     "facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True
... )
>>> 
>>> model = TFRagSequenceForGeneration.from_pretrained(
...     "facebook/rag-sequence-nq", retriever=retriever, from_pt=True
... )

>>> input_dict = tokenizer.prepare_seq2seq_batch(
...     "How many people live in Paris?", "In Paris, there are 10 million people.", return_tensors="tf"
... )
>>> outputs = model(input_dict, output_retrieved=True)

>>> 
>>> 
>>> input_ids = input_dict["input_ids"]
>>> question_hidden_states = model.question_encoder(input_ids)[0]
>>> 
>>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.numpy(), return_tensors="tf")
>>> doc_scores = tf.squeeze(
...     tf.matmul(
...         tf.expand_dims(question_hidden_states, axis=1), docs_dict["retrieved_doc_embeds"], transpose_b=True
...     ),
...     axis=1,
... )
>>> 
>>> outputs = model(
...     inputs=None,
...     context_input_ids=docs_dict["context_input_ids"],
...     context_attention_mask=docs_dict["context_attention_mask"],
...     doc_scores=doc_scores,
...     decoder_input_ids=input_dict["labels"],
... )

>>> 
>>> generated = model.generate(
...     context_input_ids=docs_dict["context_input_ids"],
...     context_attention_mask=docs_dict["context_attention_mask"],
...     doc_scores=doc_scores,
... )
>>> generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
```

#### generate

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/modeling_tf_rag.py#L1583)

( input\_ids: TFModelInputType | None = Noneattention\_mask: tf.Tensor | None = Nonecontext\_input\_ids = Nonecontext\_attention\_mask = Nonedoc\_scores = Nonedo\_deduplication = Nonenum\_return\_sequences = Nonenum\_beams = Nonen\_docs = None\*\*model\_kwargs ) → `tf.Tensor` of shape `(batch_size * num_return_sequences, sequence_length)`

Implements RAG sequence “thorough” decoding. Read the [generate()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.generate)\` documentation for more information on how to set other generate input parameters

## TFRagTokenForGeneration

### class transformers.TFRagTokenForGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/modeling_tf_rag.py#L729)

( \*args\*\*kwargs )

Parameters

-   **config** ([RagConfig](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.
-   **question\_encoder** ([TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel)) — An encoder model compatible with the faiss index encapsulated by the `retriever`.
-   **generator** ([TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel)) — A seq2seq model used as the generator in the RAG architecture.
-   **retriever** ([RagRetriever](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagRetriever)) — A retriever class encapsulating a faiss index queried to obtain context documents for current inputs.

The [TFRagTokenForGeneration](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.TFRagTokenForGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

A TF RAG-token model implementation. It performs RAG-token specific marginalization in the forward pass.

RAG is a sequence-to-sequence model which encapsulates two core components: a question encoder and a generator. During a forward pass, we encode the input with the question encoder and pass it to the retriever to extract relevant context documents. The documents are then prepended to the input. Such contextualized inputs is passed to the generator.

The question encoder can be any _autoencoding_ model, preferably [TFDPRQuestionEncoder](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.TFDPRQuestionEncoder), and the generator can be any _seq2seq_ model, preferably [TFBartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.TFBartForConditionalGeneration).

The model can be initialized with a [RagRetriever](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagRetriever) for end-to-end generation or used in combination with the outputs of a retriever in multiple steps---see examples for more details. The model is compatible any _autoencoding_ model as the `question_encoder` and any _seq2seq_ model with language model head as the `generator`. It has been tested with [TFDPRQuestionEncoder](/docs/transformers/v4.34.0/en/model_doc/dpr#transformers.TFDPRQuestionEncoder) as the `question_encoder` and [TFBartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.TFBartForConditionalGeneration) as the `generator`.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Tensorflow [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

The model is in a developing state as it is now fully supports in eager-mode only, and may not be exported in SavedModel format.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/modeling_tf_rag.py#L842)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_input\_ids: np.ndarray | tf.Tensor | None = Nonedecoder\_attention\_mask: np.ndarray | tf.Tensor | None = Noneencoder\_outputs: np.ndarray | tf.Tensor | None = Nonepast\_key\_values: Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\] | None = Nonedoc\_scores: np.ndarray | tf.Tensor | None = Nonecontext\_input\_ids: np.ndarray | tf.Tensor | None = Nonecontext\_attention\_mask: np.ndarray | tf.Tensor | None = Noneuse\_cache: bool | None = Noneoutput\_attentions: bool | None = Noneoutput\_hidden\_states: bool | None = Noneoutput\_retrieved: bool | None = Nonen\_docs: int | None = Nonedo\_marginalize: bool | None = Nonelabels: np.ndarray | tf.Tensor | None = Nonereduce\_loss: bool | None = Nonereturn\_dict: bool | None = Nonetraining: bool = False\*\*kwargs ) → `transformers.models.rag.modeling_tf_rag.TFRetrievAugLMMarginOutput` or `tuple(tf.Tensor)`

The [TFRagTokenForGeneration](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.TFRagTokenForGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import tensorflow as tf
>>> from transformers import AutoTokenizer, RagRetriever, TFRagTokenForGeneration

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
>>> retriever = RagRetriever.from_pretrained(
...     "facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True
... )
>>> 
>>> model = TFRagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever, from_pt=True)

>>> input_dict = tokenizer.prepare_seq2seq_batch(
...     "How many people live in Paris?", "In Paris, there are 10 million people.", return_tensors="tf"
... )
>>> outputs = model(input_dict, output_retrieved=True)

>>> 
>>> 
>>> input_ids = input_dict["input_ids"]
>>> question_hidden_states = model.question_encoder(input_ids)[0]
>>> 
>>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.numpy(), return_tensors="tf")
>>> doc_scores = tf.squeeze(
...     tf.matmul(
...         tf.expand_dims(question_hidden_states, axis=1), docs_dict["retrieved_doc_embeds"], transpose_b=True
...     ),
...     axis=1,
... )
>>> 
>>> outputs = model(
...     inputs=None,
...     context_input_ids=docs_dict["context_input_ids"],
...     context_attention_mask=docs_dict["context_attention_mask"],
...     doc_scores=doc_scores,
...     decoder_input_ids=input_dict["labels"],
... )

>>> 
>>> generated = model.generate(
...     context_input_ids=docs_dict["context_input_ids"],
...     context_attention_mask=docs_dict["context_attention_mask"],
...     doc_scores=doc_scores,
... )
>>> generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
```

#### generate

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/rag/modeling_tf_rag.py#L997)

( input\_ids: TFModelInputType | None = Noneattention\_mask: tf.Tensor | None = Nonecontext\_input\_ids = Nonecontext\_attention\_mask = Nonedoc\_scores = Nonen\_docs = Nonegeneration\_config = Nonelogits\_processor = \[\]\*\*kwargs ) → `tf.Tensor` of shape `(batch_size * num_return_sequences, sequence_length)`

Implements TFRAG token decoding.