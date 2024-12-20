# Reformer

[![Models](https://img.shields.io/badge/All_model_pages-reformer-blueviolet)](https://huggingface.co/models?filter=reformer) [![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/docs-demos/reformer-crime-and-punishment)

**DISCLAIMER:** This model is still a work in progress, if you see something strange, file a [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title).

## Overview

The Reformer model was proposed in the paper [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451.pdf) by Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.

The abstract from the paper is the following:

_Large Transformer models routinely achieve state-of-the-art results on a number of tasks but training these models can be prohibitively costly, especially on long sequences. We introduce two techniques to improve the efficiency of Transformers. For one, we replace dot-product attention by one that uses locality-sensitive hashing, changing its complexity from O(L^2) to O(Llog(L)), where L is the length of the sequence. Furthermore, we use reversible residual layers instead of the standard residuals, which allows storing activations only once in the training process instead of N times, where N is the number of layers. The resulting model, the Reformer, performs on par with Transformer models while being much more memory-efficient and much faster on long sequences._

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten). The Authors’ code can be found [here](https://github.com/google/trax/tree/master/trax/models/reformer).

Tips:

-   Reformer does **not** work with _torch.nn.DataParallel_ due to a bug in PyTorch, see [issue #36035](https://github.com/pytorch/pytorch/issues/36035).
-   Use Axial position encoding (see below for more details). It’s a mechanism to avoid having a huge positional encoding matrix (when the sequence length is very big) by factorizing it into smaller matrices.
-   Replace traditional attention by LSH (local-sensitive hashing) attention (see below for more details). It’s a technique to avoid computing the full product query-key in the attention layers.
-   Avoid storing the intermediate results of each layer by using reversible transformer layers to obtain them during the backward pass (subtracting the residuals from the input of the next layer gives them back) or recomputing them for results inside a given layer (less efficient than storing them but saves memory).
-   Compute the feedforward operations by chunks and not on the whole batch.

## Axial Positional Encodings

Axial Positional Encodings were first implemented in Google’s [trax library](https://github.com/google/trax/blob/4d99ad4965bab1deba227539758d59f0df0fef48/trax/layers/research/position_encodings.py#L29) and developed by the authors of this model’s paper. In models that are treating very long input sequences, the conventional position id encodings store an embedings vector of size dd being the `config.hidden_size` for every position i,…,nsi, \\ldots, n\_s, with nsn\_s being `config.max_embedding_size`. This means that having a sequence length of ns\=219≈0.5Mn\_s = 2^{19} \\approx 0.5M and a `config.hidden_size` of d\=210≈1000d = 2^{10} \\approx 1000 would result in a position encoding matrix:

Xi,j, with i∈\[1,…,d\] and j∈\[1,…,ns\]X\_{i,j}, \\text{ with } i \\in \\left\[1,\\ldots, d\\right\] \\text{ and } j \\in \\left\[1,\\ldots, n\_s\\right\]

which alone has over 500M parameters to store. Axial positional encodings factorize Xi,jX\_{i,j} into two matrices:

Xi,j1, with i∈\[1,…,d1\] and j∈\[1,…,ns1\]X^{1}\_{i,j}, \\text{ with } i \\in \\left\[1,\\ldots, d^1\\right\] \\text{ and } j \\in \\left\[1,\\ldots, n\_s^1\\right\]

and

Xi,j2, with i∈\[1,…,d2\] and j∈\[1,…,ns2\]X^{2}\_{i,j}, \\text{ with } i \\in \\left\[1,\\ldots, d^2\\right\] \\text{ and } j \\in \\left\[1,\\ldots, n\_s^2\\right\]

with:

d\=d1+d2 and ns\=ns1×ns2.d = d^1 + d^2 \\text{ and } n\_s = n\_s^1 \\times n\_s^2 .

Therefore the following holds:

Xi,j\={Xi,k1,if i<d1 with k\=jmod  ns1Xi−d1,l2,if i≥d1 with l\=⌊jns1⌋X\_{i,j} = \\begin{cases} X^{1}\_{i, k}, & \\text{if }\\ i < d^1 \\text{ with } k = j \\mod n\_s^1 \\\\ X^{2}\_{i - d^1, l}, & \\text{if } i \\ge d^1 \\text{ with } l = \\lfloor\\frac{j}{n\_s^1}\\rfloor \\end{cases}

Intuitively, this means that a position embedding vector xj∈Rdx\_j \\in \\mathbb{R}^{d} is now the composition of two factorized embedding vectors: xk,l1+xl,k2x^1\_{k, l} + x^2\_{l, k}, where as the `config.max_embedding_size` dimension jj is factorized into k and lk \\text{ and } l. This design ensures that each position embedding vector xjx\_j is unique.

Using the above example again, axial position encoding with d1\=29,d2\=29,ns1\=29,ns2\=210d^1 = 2^9, d^2 = 2^9, n\_s^1 = 2^9, n\_s^2 = 2^{10} can drastically reduced the number of parameters from 500 000 000 to 218+219≈7800002^{18} + 2^{19} \\approx 780 000 parameters, this means 85% less memory usage.

In practice, the parameter `config.axial_pos_embds_dim` is set to a tuple (d1,d2)(d^1, d^2) which sum has to be equal to `config.hidden_size` and `config.axial_pos_shape` is set to a tuple (ns1,ns2)(n\_s^1, n\_s^2) which product has to be equal to `config.max_embedding_size`, which during training has to be equal to the _sequence length_ of the `input_ids`.

## LSH Self Attention

In Locality sensitive hashing (LSH) self attention the key and query projection weights are tied. Therefore, the key query embedding vectors are also tied. LSH self attention uses the locality sensitive hashing mechanism proposed in [Practical and Optimal LSH for Angular Distance](https://arxiv.org/abs/1509.02897) to assign each of the tied key query embedding vectors to one of `config.num_buckets` possible buckets. The premise is that the more “similar” key query embedding vectors (in terms of _cosine similarity_) are to each other, the more likely they are assigned to the same bucket.

The accuracy of the LSH mechanism can be improved by increasing `config.num_hashes` or directly the argument `num_hashes` of the forward function so that the output of the LSH self attention better approximates the output of the “normal” full self attention. The buckets are then sorted and chunked into query key embedding vector chunks each of length `config.lsh_chunk_length`. For each chunk, the query embedding vectors attend to its key vectors (which are tied to themselves) and to the key embedding vectors of `config.lsh_num_chunks_before` previous neighboring chunks and `config.lsh_num_chunks_after` following neighboring chunks.

For more information, see the [original Paper](https://arxiv.org/abs/2001.04451) or this great [blog post](https://www.pragmatic.ml/reformer-deep-dive/).

Note that `config.num_buckets` can also be factorized into a list (nbuckets1,nbuckets2)(n\_{\\text{buckets}}^1, n\_{\\text{buckets}}^2). This way instead of assigning the query key embedding vectors to one of (1,…,nbuckets)(1,\\ldots, n\_{\\text{buckets}}) they are assigned to one of (1−1,…,nbuckets1−1,…,1−nbuckets2,…,nbuckets1−nbuckets2)(1-1,\\ldots, n\_{\\text{buckets}}^1-1, \\ldots, 1-n\_{\\text{buckets}}^2, \\ldots, n\_{\\text{buckets}}^1-n\_{\\text{buckets}}^2). This is crucial for very long sequences to save memory.

When training a model from scratch, it is recommended to leave `config.num_buckets=None`, so that depending on the sequence length a good value for `num_buckets` is calculated on the fly. This value will then automatically be saved in the config and should be reused for inference.

Using LSH self attention, the memory and time complexity of the query-key matmul operation can be reduced from O(ns×ns)\\mathcal{O}(n\_s \\times n\_s) to O(ns×log⁡(ns))\\mathcal{O}(n\_s \\times \\log(n\_s)), which usually represents the memory and time bottleneck in a transformer model, with nsn\_s being the sequence length.

## Local Self Attention

Local self attention is essentially a “normal” self attention layer with key, query and value projections, but is chunked so that in each chunk of length `config.local_chunk_length` the query embedding vectors only attends to the key embedding vectors in its chunk and to the key embedding vectors of `config.local_num_chunks_before` previous neighboring chunks and `config.local_num_chunks_after` following neighboring chunks.

Using Local self attention, the memory and time complexity of the query-key matmul operation can be reduced from O(ns×ns)\\mathcal{O}(n\_s \\times n\_s) to O(ns×log⁡(ns))\\mathcal{O}(n\_s \\times \\log(n\_s)), which usually represents the memory and time bottleneck in a transformer model, with nsn\_s being the sequence length.

## Training

During training, we must ensure that the sequence length is set to a value that can be divided by the least common multiple of `config.lsh_chunk_length` and `config.local_chunk_length` and that the parameters of the Axial Positional Encodings are correctly set as described above. Reformer is very memory efficient so that the model can easily be trained on sequences as long as 64000 tokens.

For training, the [ReformerModelWithLMHead](/docs/transformers/v4.34.0/en/model_doc/reformer#transformers.ReformerModelWithLMHead) should be used as follows:

```
input_ids = tokenizer.encode("This is a sentence from the training data", return_tensors="pt")
loss = model(input_ids, labels=input_ids)[0]
```

## Documentation resources

-   [Text classification task guide](../tasks/sequence_classification)
-   [Question answering task guide](../tasks/question_answering)
-   [Causal language modeling task guide](../tasks/language_modeling)
-   [Masked language modeling task guide](../tasks/masked_language_modeling)

## ReformerConfig

### class transformers.ReformerConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/reformer/configuration_reformer.py#L32)

( attention\_head\_size = 64attn\_layers = \['local', 'lsh', 'local', 'lsh', 'local', 'lsh'\]axial\_norm\_std = 1.0axial\_pos\_embds = Trueaxial\_pos\_shape = \[64, 64\]axial\_pos\_embds\_dim = \[64, 192\]chunk\_size\_lm\_head = 0eos\_token\_id = 2feed\_forward\_size = 512hash\_seed = Nonehidden\_act = 'relu'hidden\_dropout\_prob = 0.05hidden\_size = 256initializer\_range = 0.02is\_decoder = Falselayer\_norm\_eps = 1e-12local\_num\_chunks\_before = 1local\_num\_chunks\_after = 0local\_attention\_probs\_dropout\_prob = 0.05local\_attn\_chunk\_length = 64lsh\_attn\_chunk\_length = 64lsh\_attention\_probs\_dropout\_prob = 0.0lsh\_num\_chunks\_before = 1lsh\_num\_chunks\_after = 0max\_position\_embeddings = 4096num\_attention\_heads = 12num\_buckets = Nonenum\_hashes = 1pad\_token\_id = 0vocab\_size = 320tie\_word\_embeddings = Falseuse\_cache = Trueclassifier\_dropout = None\*\*kwargs )

This is the configuration class to store the configuration of a [ReformerModel](/docs/transformers/v4.34.0/en/model_doc/reformer#transformers.ReformerModel). It is used to instantiate a Reformer model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the ReFormer [google/reformer-crime-and-punishment](https://huggingface.co/google/reformer-crime-and-punishment) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import ReformerConfig, ReformerModel

>>> 
>>> configuration = ReformerConfig()

>>> 
>>> model = ReformerModel(configuration)

>>> 
>>> configuration = model.config
```

## ReformerTokenizer

### class transformers.ReformerTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/reformer/tokenization_reformer.py#L48)

( vocab\_fileeos\_token = '</s>'unk\_token = '<unk>'additional\_special\_tokens = \[\]sp\_model\_kwargs: typing.Union\[typing.Dict\[str, typing.Any\], NoneType\] = None\*\*kwargs )

Construct a Reformer tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece) .

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/reformer/tokenization_reformer.py#L173)

( save\_directory: strfilename\_prefix: typing.Optional\[str\] = None )

## ReformerTokenizerFast

### class transformers.ReformerTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/reformer/tokenization_reformer_fast.py#L57)

( vocab\_file = Nonetokenizer\_file = Noneeos\_token = '</s>'unk\_token = '<unk>'additional\_special\_tokens = \[\]\*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — [SentencePiece](https://github.com/google/sentencepiece) file (generally has a _.spm_ extension) that contains the vocabulary necessary to instantiate a tokenizer.
-   **eos\_token** (`str`, _optional_, defaults to `"</s>"`) — The end of sequence token.
    
    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.
    
-   **unk\_token** (`str`, _optional_, defaults to `"<unk>"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **pad\_token** (`str`, _optional_, defaults to `"<pad>"`) — The token used for padding, for example when batching sequences of different lengths.
-   **additional\_special\_tokens** (`List[str]`, _optional_) — Additional special tokens used by the tokenizer.

Construct a “fast” Reformer tokenizer (backed by HuggingFace’s _tokenizers_ library). Based on [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

## ReformerModel

### class transformers.ReformerModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/reformer/modeling_reformer.py#L1976)

( config )

Parameters

-   **config** ([ReformerConfig](/docs/transformers/v4.34.0/en/model_doc/reformer#transformers.ReformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Reformer Model transformer outputting raw hidden-stateswithout any specific head on top. Reformer was proposed in [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) by Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/reformer/modeling_reformer.py#L2004)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonenum\_hashes: typing.Optional\[int\] = Nonepast\_buckets\_states: typing.Optional\[typing.List\[typing.Tuple\[torch.Tensor\]\]\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.reformer.modeling_reformer.ReformerModelOutput` or `tuple(torch.FloatTensor)`

The [ReformerModel](/docs/transformers/v4.34.0/en/model_doc/reformer#transformers.ReformerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, ReformerModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("google/reformer-crime-and-punishment")
>>> model = ReformerModel.from_pretrained("google/reformer-crime-and-punishment")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## ReformerModelWithLMHead

### class transformers.ReformerModelWithLMHead

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/reformer/modeling_reformer.py#L2188)

( config )

Parameters

-   **config** ([ReformerConfig](/docs/transformers/v4.34.0/en/model_doc/reformer#transformers.ReformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Reformer Model with a `language modeling` head on top. Reformer was proposed in [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) by Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/reformer/modeling_reformer.py#L2215)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonenum\_hashes: typing.Optional\[int\] = Nonepast\_buckets\_states: typing.Optional\[typing.List\[typing.Tuple\[torch.Tensor\]\]\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.Tensor\] = None ) → [transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)`

The [ReformerModelWithLMHead](/docs/transformers/v4.34.0/en/model_doc/reformer#transformers.ReformerModelWithLMHead) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoTokenizer, ReformerModelWithLMHead

>>> tokenizer = AutoTokenizer.from_pretrained("google/reformer-crime-and-punishment")
>>> model = ReformerModelWithLMHead.from_pretrained("google/reformer-crime-and-punishment")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs, labels=inputs["input_ids"])
>>> loss = outputs.loss
>>> logits = outputs.logits
```

## ReformerForMaskedLM

### class transformers.ReformerForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/reformer/modeling_reformer.py#L2314)

( config )

Parameters

-   **config** ([ReformerConfig](/docs/transformers/v4.34.0/en/model_doc/reformer#transformers.ReformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Reformer Model with a `language modeling` head on top. Reformer was proposed in [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) by Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/reformer/modeling_reformer.py#L2335)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonenum\_hashes: typing.Optional\[int\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

The [ReformerForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/reformer#transformers.ReformerForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

This example uses a false checkpoint since we don’t have any available pretrained model for the masked language modeling task with the Reformer architecture.

Example:

```
>>> import torch
>>> from transformers import AutoTokenizer, ReformerForMaskedLM

>>> tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-reformer")
>>> model = ReformerForMaskedLM.from_pretrained("hf-internal-testing/tiny-random-reformer")

>>> 
>>> tokenizer.add_special_tokens({"mask_token": "[MASK]"})
>>> inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")

>>> 
>>> model.resize_token_embeddings(new_num_tokens=model.config.vocab_size + 1)
>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

>>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
>>> predicted_token = tokenizer.decode(predicted_token_id)
```

```
>>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
>>> 
>>> labels = torch.where(
...     inputs.input_ids == tokenizer.mask_token_id, labels[:, : inputs["input_ids"].shape[-1]], -100
... )

>>> outputs = model(**inputs, labels=labels)
>>> loss = round(outputs.loss.item(), 2)
```

## ReformerForSequenceClassification

### class transformers.ReformerForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/reformer/modeling_reformer.py#L2444)

( config )

Parameters

-   **config** ([ReformerConfig](/docs/transformers/v4.34.0/en/model_doc/reformer#transformers.ReformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Reformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

Reformer was proposed in [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) by Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/reformer/modeling_reformer.py#L2458)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonenum\_hashes: typing.Optional\[int\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [ReformerForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/reformer#transformers.ReformerForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example of single-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, ReformerForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("google/reformer-crime-and-punishment")
>>> model = ReformerForSequenceClassification.from_pretrained("google/reformer-crime-and-punishment")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> label = model.config.id2label[predicted_class_id]
```

```
>>> 
>>> num_labels = len(model.config.id2label)
>>> model = ReformerForSequenceClassification.from_pretrained(
...     "google/reformer-crime-and-punishment", num_labels=num_labels
... )

>>> labels = torch.tensor(1)
>>> loss = model(**inputs, labels=labels).loss
```

## ReformerForQuestionAnswering

### class transformers.ReformerForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/reformer/modeling_reformer.py#L2591)

( config )

Parameters

-   **config** ([ReformerConfig](/docs/transformers/v4.34.0/en/model_doc/reformer#transformers.ReformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Reformer Model with a span classification head on top for extractive question-answering tasks like SQuAD / TriviaQA ( a linear layer on top of hidden-states output to compute `span start logits` and `span end logits`.

Reformer was proposed in [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) by Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/reformer/modeling_reformer.py#L2603)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonenum\_hashes: typing.Optional\[int\] = Nonestart\_positions: typing.Optional\[torch.Tensor\] = Noneend\_positions: typing.Optional\[torch.Tensor\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

The [ReformerForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/reformer#transformers.ReformerForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, ReformerForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("google/reformer-crime-and-punishment")
>>> model = ReformerForQuestionAnswering.from_pretrained("google/reformer-crime-and-punishment")

>>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

>>> inputs = tokenizer(question, text, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> answer_start_index = outputs.start_logits.argmax()
>>> answer_end_index = outputs.end_logits.argmax()

>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]

>>> 
>>> target_start_index = torch.tensor([14])
>>> target_end_index = torch.tensor([15])

>>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
>>> loss = outputs.loss
```