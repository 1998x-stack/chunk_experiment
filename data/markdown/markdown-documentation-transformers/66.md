# Configuration

The base class [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) implements the common methods for loading/saving a configuration either from a local file or directory, or from a pretrained model configuration provided by the library (downloaded from HuggingFace’s AWS S3 repository).

Each derived config class implements model specific attributes. Common attributes present in all config classes are: `hidden_size`, `num_attention_heads`, and `num_hidden_layers`. Text models further implement: `vocab_size`.

### class transformers.PretrainedConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/configuration_utils.py#L49)

( \*\*kwargs )

Parameters

-   **name\_or\_path** (`str`, _optional_, defaults to `""`) — Store the string that was passed to [PreTrainedModel.from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) or [TFPreTrainedModel.from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) as `pretrained_model_name_or_path` if the configuration was created with such a method.
-   **output\_hidden\_states** (`bool`, _optional_, defaults to `False`) — Whether or not the model should return all hidden-states.
-   **output\_attentions** (`bool`, _optional_, defaults to `False`) — Whether or not the model should returns all attentions.
-   **return\_dict** (`bool`, _optional_, defaults to `True`) — Whether or not the model should return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **is\_encoder\_decoder** (`bool`, _optional_, defaults to `False`) — Whether the model is used as an encoder/decoder or not.
-   **is\_decoder** (`bool`, _optional_, defaults to `False`) — Whether the model is used as decoder or not (in which case it’s used as an encoder).
-   **cross\_attention\_hidden\_size\*\*** (`bool`, _optional_) — The hidden size of the cross-attention layer in case the model is used as a decoder in an encoder-decoder setting and the cross-attention hidden dimension differs from `self.config.hidden_size`.
-   **add\_cross\_attention** (`bool`, _optional_, defaults to `False`) — Whether cross-attention layers should be added to the model. Note, this option is only relevant for models that can be used as decoder models within the [EncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel) class, which consists of all models in `AUTO_MODELS_FOR_CAUSAL_LM`.
-   **tie\_encoder\_decoder** (`bool`, _optional_, defaults to `False`) — Whether all encoder weights should be tied to their equivalent decoder weights. This requires the encoder and decoder model to have the exact same parameter names.
-   **prune\_heads** (`Dict[int, List[int]]`, _optional_, defaults to `{}`) — Pruned heads of the model. The keys are the selected layer indices and the associated values, the list of heads to prune in said layer.
    
    For instance `{1: [0, 2], 2: [2, 3]}` will prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.
    
-   **chunk\_size\_feed\_forward** (`int`, _optional_, defaults to `0`) — The chunk size of all feed forward layers in the residual attention blocks. A chunk size of `0` means that the feed forward layer is not chunked. A chunk size of n means that the feed forward layer processes `n` < sequence\_length embeddings at a time. For more information on feed forward chunking, see [How does Feed Forward Chunking work?](../glossary.html#feed-forward-chunking).

Parameters for sequence generation

-   **max\_length** (`int`, _optional_, defaults to 20) — Maximum length that will be used by default in the `generate` method of the model.
-   **min\_length** (`int`, _optional_, defaults to 0) — Minimum length that will be used by default in the `generate` method of the model.
-   **do\_sample** (`bool`, _optional_, defaults to `False`) — Flag that will be used by default in the `generate` method of the model. Whether or not to use sampling ; use greedy decoding otherwise.
-   **early\_stopping** (`bool`, _optional_, defaults to `False`) — Flag that will be used by default in the `generate` method of the model. Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.
-   **num\_beams** (`int`, _optional_, defaults to 1) — Number of beams for beam search that will be used by default in the `generate` method of the model. 1 means no beam search.
-   **num\_beam\_groups** (`int`, _optional_, defaults to 1) — Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams that will be used by default in the `generate` method of the model. 1 means no group beam search.
-   **diversity\_penalty** (`float`, _optional_, defaults to 0.0) — Value to control diversity for group beam search. that will be used by default in the `generate` method of the model. 0 means no diversity penalty. The higher the penalty, the more diverse are the outputs.
-   **temperature** (`float`, _optional_, defaults to 1.0) — The value used to module the next token probabilities that will be used by default in the `generate` method of the model. Must be strictly positive.
-   **top\_k** (`int`, _optional_, defaults to 50) — Number of highest probability vocabulary tokens to keep for top-k-filtering that will be used by default in the `generate` method of the model.
-   **top\_p** (`float`, _optional_, defaults to 1) — Value that will be used by default in the `generate` method of the model for `top_p`. If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or higher are kept for generation.
-   **typical\_p** (`float`, _optional_, defaults to 1) — Local typicality measures how similar the conditional probability of predicting a target token next is to the expected conditional probability of predicting a random token next, given the partial text already generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that add up to `typical_p` or higher are kept for generation. See [this paper](https://arxiv.org/pdf/2202.00666.pdf) for more details.
-   **repetition\_penalty** (`float`, _optional_, defaults to 1) — Parameter for repetition penalty that will be used by default in the `generate` method of the model. 1.0 means no penalty.
-   **length\_penalty** (`float`, _optional_, defaults to 1) — Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while `length_penalty` < 0.0 encourages shorter sequences.
-   **no\_repeat\_ngram\_size** (`int`, _optional_, defaults to 0) — Value that will be used by default in the — `generate` method of the model for `no_repeat_ngram_size`. If set to int > 0, all ngrams of that size can only occur once.
-   **encoder\_no\_repeat\_ngram\_size** (`int`, _optional_, defaults to 0) — Value that will be used by — default in the `generate` method of the model for `encoder_no_repeat_ngram_size`. If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the `decoder_input_ids`.
-   **bad\_words\_ids** (`List[int]`, _optional_) — List of token ids that are not allowed to be generated that will be used by default in the `generate` method of the model. In order to get the tokens of the words that should not appear in the generated text, use `tokenizer.encode(bad_word, add_prefix_space=True)`.
-   **num\_return\_sequences** (`int`, _optional_, defaults to 1) — Number of independently computed returned sequences for each element in the batch that will be used by default in the `generate` method of the model.
-   **output\_scores** (`bool`, _optional_, defaults to `False`) — Whether the model should return the logits when used for generation.
-   **return\_dict\_in\_generate** (`bool`, _optional_, defaults to `False`) — Whether the model should return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a `torch.LongTensor`.
-   **forced\_bos\_token\_id** (`int`, _optional_) — The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful for multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be the target language token.
-   **forced\_eos\_token\_id** (`int`, _optional_) — The id of the token to force as the last generated token when `max_length` is reached.
-   **remove\_invalid\_values** (`bool`, _optional_) — Whether to remove possible _nan_ and _inf_ outputs of the model to prevent the generation method to crash. Note that using `remove_invalid_values` can slow down generation.

Parameters for fine-tuning tasks

-   **architectures** (`List[str]`, _optional_) — Model architectures that can be used with the model pretrained weights.
-   **finetuning\_task** (`str`, _optional_) — Name of the task used to fine-tune the model. This can be used when converting from an original (TensorFlow or PyTorch) checkpoint.
-   **id2label** (`Dict[int, str]`, _optional_) — A map from index (for instance prediction index, or target index) to label.
-   **label2id** (`Dict[str, int]`, _optional_) — A map from label to index for the model.
-   **num\_labels** (`int`, _optional_) — Number of labels to use in the last layer added to the model, typically for a classification task.
-   **task\_specific\_params** (`Dict[str, Any]`, _optional_) — Additional keyword arguments to store for the current task.
-   **problem\_type** (`str`, _optional_) — Problem type for `XxxForSequenceClassification` models. Can be one of `"regression"`, `"single_label_classification"` or `"multi_label_classification"`.

Parameters linked to the tokenizer

-   **tokenizer\_class** (`str`, _optional_) — The name of the associated tokenizer class to use (if none is set, will use the tokenizer associated to the model by default).
-   **prefix** (`str`, _optional_) — A specific prompt that should be added at the beginning of each text before calling the model.
-   **bos\_token\_id** (`int`, _optional_) — The id of the _beginning-of-stream_ token.
-   **pad\_token\_id** (`int`, _optional_) — The id of the _padding_ token.
-   **eos\_token\_id** (`int`, _optional_) — The id of the _end-of-stream_ token.
-   **decoder\_start\_token\_id** (`int`, _optional_) — If an encoder-decoder model starts decoding with a different token than _bos_, the id of that token.
-   **sep\_token\_id** (`int`, _optional_) — The id of the _separation_ token.

PyTorch specific parameters

-   **torchscript** (`bool`, _optional_, defaults to `False`) — Whether or not the model should be used with Torchscript.
-   **tie\_word\_embeddings** (`bool`, _optional_, defaults to `True`) — Whether the model’s input and output word embeddings should be tied. Note that this is only relevant if the model has a output word embedding layer.
-   **torch\_dtype** (`str`, _optional_) — The `dtype` of the weights. This attribute can be used to initialize the model to a non-default `dtype` (which is normally `float32`) and thus allow for optimal storage allocation. For example, if the saved model is `float16`, ideally we want to load it back using the minimal amount of memory needed to load `float16` weights. Since the config object is stored in plain text, this attribute contains just the floating type string without the `torch.` prefix. For example, for `torch.float16` \``torch_dtype` is the `"float16"` string.
    
    This attribute is currently not being used during model loading time, but this may change in the future versions. But we can already start preparing for the future by saving the dtype with save\_pretrained.
    

TensorFlow specific parameters

-   **use\_bfloat16** (`bool`, _optional_, defaults to `False`) — Whether or not the model should use BFloat16 scalars (only used by some TensorFlow models).
-   **tf\_legacy\_loss** (`bool`, _optional_, defaults to `False`) — Whether the model should use legacy TensorFlow losses. Legacy losses have variable output shapes and may not be XLA-compatible. This option is here for backward compatibility and will be removed in Transformers v5.

Base class for all configuration classes. Handles a few parameters common to all models’ configurations as well as methods for loading/downloading/saving configurations.

A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to initialize a model does **not** load the model weights. It only affects the model’s configuration.

Class attributes (overridden by derived classes):

-   **model\_type** (`str`) — An identifier for the model type, serialized into the JSON file, and used to recreate the correct object in [AutoConfig](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoConfig).
-   **is\_composition** (`bool`) — Whether the config class is composed of multiple sub-configs. In this case the config has to be initialized from two or more configs of type [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) like: [EncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig) or [~RagConfig](/docs/transformers/v4.34.0/en/model_doc/rag#transformers.RagConfig).
-   **keys\_to\_ignore\_at\_inference** (`List[str]`) — A list of keys to ignore by default when looking at dictionary outputs of the model during inference.
-   **attribute\_map** (`Dict[str, str]`) — A dict that maps model specific attribute names to the standardized naming of attributes.

Common attributes (present in all subclasses):

-   **vocab\_size** (`int`) — The number of tokens in the vocabulary, which is also the first dimension of the embeddings matrix (this attribute may be missing for models that don’t have a text modality like ViT).
-   **hidden\_size** (`int`) — The hidden size of the model.
-   **num\_attention\_heads** (`int`) — The number of attention heads used in the multi-head attention layers of the model.
-   **num\_hidden\_layers** (`int`) — The number of blocks in the model.

#### push\_to\_hub

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/hub.py#L786)

( repo\_id: str use\_temp\_dir: typing.Optional\[bool\] = None commit\_message: typing.Optional\[str\] = None private: typing.Optional\[bool\] = None token: typing.Union\[bool, str, NoneType\] = None max\_shard\_size: typing.Union\[int, str, NoneType\] = '10GB' create\_pr: bool = False safe\_serialization: bool = False revision: str = None \*\*deprecated\_kwargs )

Parameters

-   **repo\_id** (`str`) — The name of the repository you want to push your config to. It should contain your organization name when pushing to a given organization.
-   **use\_temp\_dir** (`bool`, _optional_) — Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub. Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
-   **commit\_message** (`str`, _optional_) — Message to commit while pushing. Will default to `"Upload config"`.
-   **private** (`bool`, _optional_) — Whether or not the repository created should be private.
-   **token** (`bool` or `str`, _optional_) — The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url` is not specified.
-   **max\_shard\_size** (`int` or `str`, _optional_, defaults to `"10GB"`) — Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
-   **create\_pr** (`bool`, _optional_, defaults to `False`) — Whether or not to create a PR with the uploaded files or directly commit.
-   **safe\_serialization** (`bool`, _optional_, defaults to `False`) — Whether or not to convert the model weights in safetensors format for safer serialization.
-   **revision** (`str`, _optional_) — Branch to push the uploaded files to.

Upload the configuration file to the 🤗 Model Hub.

Examples:

```
from transformers import AutoConfig

config = AutoConfig.from_pretrained("bert-base-cased")


config.push_to_hub("my-finetuned-bert")


config.push_to_hub("huggingface/my-finetuned-bert")
```

#### dict\_torch\_dtype\_to\_str

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/configuration_utils.py#L982)

( d: typing.Dict\[str, typing.Any\] )

Checks whether the passed dictionary and its nested dicts have a _torch\_dtype_ key and if it’s not None, converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into _“float32”_ string, which can then be stored in the json format.

#### from\_dict

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/configuration_utils.py#L723)

( config\_dict: typing.Dict\[str, typing.Any\] \*\*kwargs ) → [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)

Parameters

-   **config\_dict** (`Dict[str, Any]`) — Dictionary that will be used to instantiate the configuration object. Such a dictionary can be retrieved from a pretrained checkpoint by leveraging the [get\_config\_dict()](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig.get_config_dict) method.
-   **kwargs** (`Dict[str, Any]`) — Additional parameters from which to initialize the configuration object.

The configuration object instantiated from those parameters.

Instantiates a [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) from a Python dictionary of parameters.

#### from\_json\_file

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/configuration_utils.py#L781)

( json\_file: typing.Union\[str, os.PathLike\] ) → [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)

Parameters

-   **json\_file** (`str` or `os.PathLike`) — Path to the JSON file containing the parameters.

The configuration object instantiated from that JSON file.

Instantiates a [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) from the path to a JSON file of parameters.

#### from\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/configuration_utils.py#L497)

( pretrained\_model\_name\_or\_path: typing.Union\[str, os.PathLike\] cache\_dir: typing.Union\[str, os.PathLike, NoneType\] = None force\_download: bool = False local\_files\_only: bool = False token: typing.Union\[bool, str, NoneType\] = None revision: str = 'main' \*\*kwargs ) → [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)

Parameters

-   **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) — This can be either:
    
    -   a string, the _model id_ of a pretrained model configuration hosted inside a model repo on huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
    -   a path to a _directory_ containing a configuration file saved using the [save\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig.save_pretrained) method, e.g., `./my_model_directory/`.
    -   a path or url to a saved configuration JSON _file_, e.g., `./my_model_directory/configuration.json`.
    
-   **cache\_dir** (`str` or `os.PathLike`, _optional_) — Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used.
-   **force\_download** (`bool`, _optional_, defaults to `False`) — Whether or not to force to (re-)download the configuration files and override the cached versions if they exist.
-   **resume\_download** (`bool`, _optional_, defaults to `False`) — Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
-   **proxies** (`Dict[str, str]`, _optional_) — A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
-   **token** (`str` or `bool`, _optional_) — The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
-   **revision** (`str`, _optional_, defaults to `"main"`) — The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git.
    
    To test a pull request you made on the Hub, you can pass \`revision=“refs/pr/“.
    
-   **return\_unused\_kwargs** (`bool`, _optional_, defaults to `False`) — If `False`, then this function returns just the final configuration object.
    
    If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where _unused\_kwargs_ is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the part of `kwargs` which has not been used to update `config` and is otherwise ignored.
    
-   **subfolder** (`str`, _optional_, defaults to `""`) — In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can specify the folder name here.
-   **kwargs** (`Dict[str, Any]`, _optional_) — The values in kwargs of any keys which are configuration attributes will be used to override the loaded values. Behavior concerning key/value pairs whose keys are _not_ configuration attributes is controlled by the `return_unused_kwargs` keyword parameter.

The configuration object instantiated from this pretrained model.

Instantiate a [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) (or a derived class) from a pretrained model configuration.

Examples:

```

config = BertConfig.from_pretrained(
    "bert-base-uncased"
)  
config = BertConfig.from_pretrained(
    "./test/saved_model/"
)  
config = BertConfig.from_pretrained("./test/saved_model/my_configuration.json")
config = BertConfig.from_pretrained("bert-base-uncased", output_attentions=True, foo=False)
assert config.output_attentions == True
config, unused_kwargs = BertConfig.from_pretrained(
    "bert-base-uncased", output_attentions=True, foo=False, return_unused_kwargs=True
)
assert config.output_attentions == True
assert unused_kwargs == {"foo": False}
```

#### get\_config\_dict

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/configuration_utils.py#L600)

( pretrained\_model\_name\_or\_path: typing.Union\[str, os.PathLike\] \*\*kwargs ) → `Tuple[Dict, Dict]`

Parameters

-   **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) — The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

Returns

`Tuple[Dict, Dict]`

The dictionary(ies) that will be used to instantiate the configuration object.

From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) using `from_dict`.

#### register\_for\_auto\_class

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/configuration_utils.py#L994)

( auto\_class = 'AutoConfig' )

Parameters

-   **auto\_class** (`str` or `type`, _optional_, defaults to `"AutoConfig"`) — The auto class to register this new configuration with.

Register this class with a given auto class. This should only be used for custom configurations as the ones in the library are already mapped with `AutoConfig`.

This API is experimental and may have some slight breaking changes in the next releases.

#### save\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/configuration_utils.py#L423)

( save\_directory: typing.Union\[str, os.PathLike\] push\_to\_hub: bool = False \*\*kwargs )

Parameters

-   **save\_directory** (`str` or `os.PathLike`) — Directory where the configuration JSON file will be saved (will be created if it does not exist).
-   **push\_to\_hub** (`bool`, _optional_, defaults to `False`) — Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with `repo_id` (will default to the name of `save_directory` in your namespace).
-   **kwargs** (`Dict[str, Any]`, _optional_) — Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/v4.34.0/en/main_classes/processors#transformers.ProcessorMixin.push_to_hub) method.

Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained) class method.

#### to\_dict

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/configuration_utils.py#L863)

( ) → `Dict[str, Any]`

Dictionary of all the attributes that make up this configuration instance.

Serializes this instance to a Python dictionary.

#### to\_diff\_dict

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/configuration_utils.py#L809)

( ) → `Dict[str, Any]`

Dictionary of all the attributes that make up this configuration instance,

Removes all attributes from config which correspond to the default config attributes for better readability and serializes to a Python dictionary.

#### to\_json\_file

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/configuration_utils.py#L920)

( json\_file\_path: typing.Union\[str, os.PathLike\] use\_diff: bool = True )

Parameters

-   **json\_file\_path** (`str` or `os.PathLike`) — Path to the JSON file in which this configuration instance’s parameters will be saved.
-   **use\_diff** (`bool`, _optional_, defaults to `True`) — If set to `True`, only the difference between the config instance and the default `PretrainedConfig()` is serialized to JSON file.

Save this instance to a JSON file.

#### to\_json\_string

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/configuration_utils.py#L902)

( use\_diff: bool = True ) → `str`

Parameters

-   **use\_diff** (`bool`, _optional_, defaults to `True`) — If set to `True`, only the difference between the config instance and the default `PretrainedConfig()` is serialized to JSON string.

String containing all the attributes that make up this configuration instance in JSON format.

Serializes this instance to a JSON string.

#### update

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/configuration_utils.py#L934)

( config\_dict: typing.Dict\[str, typing.Any\] )

Parameters

-   **config\_dict** (`Dict[str, Any]`) — Dictionary of attributes that should be updated for this class.

Updates attributes of this class with attributes from `config_dict`.

#### update\_from\_string

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/configuration_utils.py#L944)

( update\_str: str )

Parameters

-   **update\_str** (`str`) — String with attributes that should be updated for this class.

Updates attributes of this class with attributes from `update_str`.

The expected format is ints, floats and strings as is, and for booleans use `true` or `false`. For example: “n\_embd=10,resid\_pdrop=0.2,scale\_attn\_weights=false,summary\_type=cls\_index”

The keys to change have to already exist in the config object.