# VITS

## Overview

The VITS model was proposed in [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103) by Jaehyeon Kim, Jungil Kong, Juhee Son.

VITS (**V**ariational **I**nference with adversarial learning for end-to-end **T**ext-to-**S**peech) is an end-to-end speech synthesis model that predicts a speech waveform conditional on an input text sequence. It is a conditional variational autoencoder (VAE) comprised of a posterior encoder, decoder, and conditional prior.

A set of spectrogram-based acoustic features are predicted by the flow-based module, which is formed of a Transformer-based text encoder and multiple coupling layers. The spectrogram is decoded using a stack of transposed convolutional layers, much in the same style as the HiFi-GAN vocoder. Motivated by the one-to-many nature of the TTS problem, where the same text input can be spoken in multiple ways, the model also includes a stochastic duration predictor, which allows the model to synthesise speech with different rhythms from the same input text.

The model is trained end-to-end with a combination of losses derived from variational lower bound and adversarial training. To improve the expressiveness of the model, normalizing flows are applied to the conditional prior distribution. During inference, the text encodings are up-sampled based on the duration prediction module, and then mapped into the waveform using a cascade of the flow module and HiFi-GAN decoder. Due to the stochastic nature of the duration predictor, the model is non-deterministic, and thus requires a fixed seed to generate the same speech waveform.

The abstract from the paper is the following:

_Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel sampling have been proposed, but their sample quality does not match that of two-stage TTS systems. In this work, we present a parallel end-to-end TTS method that generates more natural sounding audio than current two-stage models. Our method adopts variational inference augmented with normalizing flows and an adversarial training process, which improves the expressive power of generative modeling. We also propose a stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the natural one-to-many relationship in which a text input can be spoken in multiple ways with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS) on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly available TTS systems and achieves a MOS comparable to ground truth._

This model can also be used with TTS checkpoints from [Massively Multilingual Speech (MMS)](https://arxiv.org/abs/2305.13516) as these checkpoints use the same architecture and a slightly modified tokenizer.

This model was contributed by [Matthijs](https://huggingface.co/Matthijs) and [sanchit-gandhi](https://huggingface.co/sanchit-gandhi). The original code can be found [here](https://github.com/jaywalnut310/vits).

## Model Usage

Both the VITS and MMS-TTS checkpoints can be used with the same API. Since the flow-based model is non-deterministic, it is good practice to set a seed to ensure reproducibility of the outputs. For languages with a Roman alphabet, such as English or French, the tokenizer can be used directly to pre-process the text inputs. The following code example runs a forward pass using the MMS-TTS English checkpoint:

```
import torch
from transformers import VitsTokenizer, VitsModel, set_seed

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
model = VitsModel.from_pretrained("facebook/mms-tts-eng")

inputs = tokenizer(text="Hello - my dog is cute", return_tensors="pt")

set_seed(555)  

with torch.no_grad():
   outputs = model(**inputs)

waveform = outputs.waveform[0]
```

The resulting waveform can be saved as a `.wav` file:

```
import scipy

scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=waveform)
```

Or displayed in a Jupyter Notebook / Google Colab:

```
from IPython.display import Audio

Audio(waveform, rate=model.config.sampling_rate)
```

For certain languages with a non-Roman alphabet, such as Arabic, Mandarin or Hindi, the [`uroman`](https://github.com/isi-nlp/uroman) perl package is required to pre-process the text inputs to the Roman alphabet.

You can check whether you require the `uroman` package for your language by inspecting the `is_uroman` attribute of the pre-trained `tokenizer`:

```
from transformers import VitsTokenizer

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
print(tokenizer.is_uroman)
```

If required, you should apply the uroman package to your text inputs **prior** to passing them to the `VitsTokenizer`, since currently the tokenizer does not support performing the pre-processing itself.

To do this, first clone the uroman repository to your local machine and set the bash variable `UROMAN` to the local path:

```
git clone https://github.com/isi-nlp/uroman.git
cd uroman
export UROMAN=$(pwd)
```

You can then pre-process the text input using the following code snippet. You can either rely on using the bash variable `UROMAN` to point to the uroman repository, or you can pass the uroman directory as an argument to the `uromaize` function:

```
import torch
from transformers import VitsTokenizer, VitsModel, set_seed
import os
import subprocess

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-kor")
model = VitsModel.from_pretrained("facebook/mms-tts-kor")

def uromanize(input_string, uroman_path):
    """Convert non-Roman strings to Roman using the `uroman` perl package."""
    script_path = os.path.join(uroman_path, "bin", "uroman.pl")

    command = ["perl", script_path]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    stdout, stderr = process.communicate(input=input_string.encode())

    if process.returncode != 0:
        raise ValueError(f"Error {process.returncode}: {stderr.decode()}")

    
    return stdout.decode()[:-1]

text = "이봐 무슨 일이야"
uromaized_text = uromanize(text, uroman_path=os.environ["UROMAN"])

inputs = tokenizer(text=uromaized_text, return_tensors="pt")

set_seed(555)  
with torch.no_grad():
   outputs = model(inputs["input_ids"])

waveform = outputs.waveform[0]
```

## VitsConfig

### class transformers.VitsConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vits/configuration_vits.py#L29)

( vocab\_size = 38 hidden\_size = 192 num\_hidden\_layers = 6 num\_attention\_heads = 2 window\_size = 4 use\_bias = True ffn\_dim = 768 layerdrop = 0.1 ffn\_kernel\_size = 3 flow\_size = 192 spectrogram\_bins = 513 hidden\_act = 'relu' hidden\_dropout = 0.1 attention\_dropout = 0.1 activation\_dropout = 0.1 initializer\_range = 0.02 layer\_norm\_eps = 1e-05 use\_stochastic\_duration\_prediction = True num\_speakers = 1 speaker\_embedding\_size = 0 upsample\_initial\_channel = 512 upsample\_rates = \[8, 8, 2, 2\] upsample\_kernel\_sizes = \[16, 16, 4, 4\] resblock\_kernel\_sizes = \[3, 7, 11\] resblock\_dilation\_sizes = \[\[1, 3, 5\], \[1, 3, 5\], \[1, 3, 5\]\] leaky\_relu\_slope = 0.1 depth\_separable\_channels = 2 depth\_separable\_num\_layers = 3 duration\_predictor\_flow\_bins = 10 duration\_predictor\_tail\_bound = 5.0 duration\_predictor\_kernel\_size = 3 duration\_predictor\_dropout = 0.5 duration\_predictor\_num\_flows = 4 duration\_predictor\_filter\_channels = 256 prior\_encoder\_num\_flows = 4 prior\_encoder\_num\_wavenet\_layers = 4 posterior\_encoder\_num\_wavenet\_layers = 16 wavenet\_kernel\_size = 5 wavenet\_dilation\_rate = 1 wavenet\_dropout = 0.0 speaking\_rate = 1.0 noise\_scale = 0.667 noise\_scale\_duration = 0.8 sampling\_rate = 16000 \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 38) — Vocabulary size of the VITS model. Defines the number of different tokens that can be represented by the `inputs_ids` passed to the forward method of [VitsModel](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsModel).
-   **hidden\_size** (`int`, _optional_, defaults to 192) — Dimensionality of the text encoder layers.
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 6) — Number of hidden layers in the Transformer encoder.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 2) — Number of attention heads for each attention layer in the Transformer encoder.
-   **window\_size** (`int`, _optional_, defaults to 4) — Window size for the relative positional embeddings in the attention layers of the Transformer encoder.
-   **use\_bias** (`bool`, _optional_, defaults to `True`) — Whether to use bias in the key, query, value projection layers in the Transformer encoder.
-   **ffn\_dim** (`int`, _optional_, defaults to 768) — Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
-   **layerdrop** (`float`, _optional_, defaults to 0.1) — The LayerDrop probability for the encoder. See the \[LayerDrop paper\](see [https://arxiv.org/abs/1909.11556](https://arxiv.org/abs/1909.11556)) for more details.
-   **ffn\_kernel\_size** (`int`, _optional_, defaults to 3) — Kernel size of the 1D convolution layers used by the feed-forward network in the Transformer encoder.
-   **flow\_size** (`int`, _optional_, defaults to 192) — Dimensionality of the flow layers.
-   **spectrogram\_bins** (`int`, _optional_, defaults to 513) — Number of frequency bins in the target spectrogram.
-   **hidden\_act** (`str` or `function`, _optional_, defaults to `"relu"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
-   **hidden\_dropout** (`float`, _optional_, defaults to 0.1) — The dropout probability for all fully connected layers in the embeddings and encoder.
-   **attention\_dropout** (`float`, _optional_, defaults to 0.1) — The dropout ratio for the attention probabilities.
-   **activation\_dropout** (`float`, _optional_, defaults to 0.1) — The dropout ratio for activations inside the fully connected layer.
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-5) — The epsilon used by the layer normalization layers.
-   **use\_stochastic\_duration\_prediction** (`bool`, _optional_, defaults to `True`) — Whether to use the stochastic duration prediction module or the regular duration predictor.
-   **num\_speakers** (`int`, _optional_, defaults to 1) — Number of speakers if this is a multi-speaker model.
-   **speaker\_embedding\_size** (`int`, _optional_, defaults to 0) — Number of channels used by the speaker embeddings. Is zero for single-speaker models.
-   **upsample\_initial\_channel** (`int`, _optional_, defaults to 512) — The number of input channels into the HiFi-GAN upsampling network.
-   **upsample\_rates** (`Tuple[int]` or `List[int]`, _optional_, defaults to `[8, 8, 2, 2]`) — A tuple of integers defining the stride of each 1D convolutional layer in the HiFi-GAN upsampling network. The length of `upsample_rates` defines the number of convolutional layers and has to match the length of `upsample_kernel_sizes`.
-   **upsample\_kernel\_sizes** (`Tuple[int]` or `List[int]`, _optional_, defaults to `[16, 16, 4, 4]`) — A tuple of integers defining the kernel size of each 1D convolutional layer in the HiFi-GAN upsampling network. The length of `upsample_kernel_sizes` defines the number of convolutional layers and has to match the length of `upsample_rates`.
-   **resblock\_kernel\_sizes** (`Tuple[int]` or `List[int]`, _optional_, defaults to `[3, 7, 11]`) — A tuple of integers defining the kernel sizes of the 1D convolutional layers in the HiFi-GAN multi-receptive field fusion (MRF) module.
-   **resblock\_dilation\_sizes** (`Tuple[Tuple[int]]` or `List[List[int]]`, _optional_, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`) — A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the HiFi-GAN multi-receptive field fusion (MRF) module.
-   **leaky\_relu\_slope** (`float`, _optional_, defaults to 0.1) — The angle of the negative slope used by the leaky ReLU activation.
-   **depth\_separable\_channels** (`int`, _optional_, defaults to 2) — Number of channels to use in each depth-separable block.
-   **depth\_separable\_num\_layers** (`int`, _optional_, defaults to 3) — Number of convolutional layers to use in each depth-separable block.
-   **duration\_predictor\_flow\_bins** (`int`, _optional_, defaults to 10) — Number of channels to map using the unonstrained rational spline in the duration predictor model.
-   **duration\_predictor\_tail\_bound** (`float`, _optional_, defaults to 5.0) — Value of the tail bin boundary when computing the unconstrained rational spline in the duration predictor model.
-   **duration\_predictor\_kernel\_size** (`int`, _optional_, defaults to 3) — Kernel size of the 1D convolution layers used in the duration predictor model.
-   **duration\_predictor\_dropout** (`float`, _optional_, defaults to 0.5) — The dropout ratio for the duration predictor model.
-   **duration\_predictor\_num\_flows** (`int`, _optional_, defaults to 4) — Number of flow stages used by the duration predictor model.
-   **duration\_predictor\_filter\_channels** (`int`, _optional_, defaults to 256) — Number of channels for the convolution layers used in the duration predictor model.
-   **prior\_encoder\_num\_flows** (`int`, _optional_, defaults to 4) — Number of flow stages used by the prior encoder flow model.
-   **prior\_encoder\_num\_wavenet\_layers** (`int`, _optional_, defaults to 4) — Number of WaveNet layers used by the prior encoder flow model.
-   **posterior\_encoder\_num\_wavenet\_layers** (`int`, _optional_, defaults to 16) — Number of WaveNet layers used by the posterior encoder model.
-   **wavenet\_kernel\_size** (`int`, _optional_, defaults to 5) — Kernel size of the 1D convolution layers used in the WaveNet model.
-   **wavenet\_dilation\_rate** (`int`, _optional_, defaults to 1) — Dilation rates of the dilated 1D convolutional layers used in the WaveNet model.
-   **wavenet\_dropout** (`float`, _optional_, defaults to 0.0) — The dropout ratio for the WaveNet layers.
-   **speaking\_rate** (`float`, _optional_, defaults to 1.0) — Speaking rate. Larger values give faster synthesised speech.
-   **noise\_scale** (`float`, _optional_, defaults to 0.667) — How random the speech prediction is. Larger values create more variation in the predicted speech.
-   **noise\_scale\_duration** (`float`, _optional_, defaults to 0.8) — How random the duration prediction is. Larger values create more variation in the predicted durations.
-   **sampling\_rate** (`int`, _optional_, defaults to 16000) — The sampling rate at which the output audio waveform is digitalized expressed in hertz (Hz).

This is the configuration class to store the configuration of a [VitsModel](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsModel). It is used to instantiate a VITS model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the VITS [facebook/mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import VitsModel, VitsConfig

>>> 
>>> configuration = VitsConfig()

>>> 
>>> model = VitsModel(configuration)

>>> 
>>> configuration = model.config
```

## VitsTokenizer

### class transformers.VitsTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vits/tokenization_vits.py#L57)

( vocab\_file pad\_token = '<pad>' unk\_token = '<unk>' language = None add\_blank = True normalize = True phonemize = True is\_uroman = False \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — Path to the vocabulary file.
-   **language** (`str`, _optional_) — Language identifier.
-   **add\_blank** (`bool`, _optional_, defaults to `True`) — Whether to insert token id 0 in between the other tokens.
-   **normalize** (`bool`, _optional_, defaults to `True`) — Whether to normalize the input text by removing all casing and punctuation.
-   **phonemize** (`bool`, _optional_, defaults to `True`) — Whether to convert the input text into phonemes.
-   **is\_uroman** (`bool`, _optional_, defaults to `False`) — Whether the `uroman` Romanizer needs to be applied to the input text prior to tokenizing.

Construct a VITS tokenizer. Also supports MMS-TTS.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L2732)

( text: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\]\] = None text\_pair: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\], NoneType\] = None text\_target: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\]\] = None text\_pair\_target: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\], NoneType\] = None add\_special\_tokens: bool = True padding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = False truncation: typing.Union\[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy\] = None max\_length: typing.Optional\[int\] = None stride: int = 0 is\_split\_into\_words: bool = False pad\_to\_multiple\_of: typing.Optional\[int\] = None return\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None return\_token\_type\_ids: typing.Optional\[bool\] = None return\_attention\_mask: typing.Optional\[bool\] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True \*\*kwargs ) → [BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

-   **text** (`str`, `List[str]`, `List[List[str]]`, _optional_) — The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
-   **text\_pair** (`str`, `List[str]`, `List[List[str]]`, _optional_) — The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
-   **text\_target** (`str`, `List[str]`, `List[List[str]]`, _optional_) — The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
-   **text\_pair\_target** (`str`, `List[str]`, `List[List[str]]`, _optional_) — The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
-   **add\_special\_tokens** (`bool`, _optional_, defaults to `True`) — Whether or not to add special tokens when encoding the sequences. This will use the underlying `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are automatically added to the input ids. This is usefull if you want to add `bos` or `eos` tokens automatically.
-   **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.utils.PaddingStrategy), _optional_, defaults to `False`) — Activates and controls padding. Accepts the following values:
    
    -   `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
    -   `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided.
    -   `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).
    
-   **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), _optional_, defaults to `False`) — Activates and controls truncation. Accepts the following values:
    
    -   `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will truncate token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is provided.
    -   `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
    -   `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
    -   `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater than the model maximum admissible input size).
    
-   **max\_length** (`int`, _optional_) — Controls the maximum length to use by one of the truncation/padding parameters.
    
    If left unset or set to `None`, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.
    
-   **stride** (`int`, _optional_, defaults to 0) — If set to a number along with `max_length`, the overflowing tokens returned when `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence returned to provide some overlap between truncated and overflowing sequences. The value of this argument defines the number of overlapping tokens.
-   **is\_split\_into\_words** (`bool`, _optional_, defaults to `False`) — Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace) which it will tokenize. This is useful for NER or token classification.
-   **pad\_to\_multiple\_of** (`int`, _optional_) — If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
-   **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.TensorType), _optional_) — If set, will return tensors instead of list of python integers. Acceptable values are:
    
    -   `'tf'`: Return TensorFlow `tf.constant` objects.
    -   `'pt'`: Return PyTorch `torch.Tensor` objects.
    -   `'np'`: Return Numpy `np.ndarray` objects.
    
-   **return\_token\_type\_ids** (`bool`, _optional_) — Whether to return token type IDs. If left to the default, will return the token type IDs according to the specific tokenizer’s default, defined by the `return_outputs` attribute.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **return\_attention\_mask** (`bool`, _optional_) — Whether to return the attention mask. If left to the default, will return the attention mask according to the specific tokenizer’s default, defined by the `return_outputs` attribute.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **return\_overflowing\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead of returning overflowing tokens.
-   **return\_special\_tokens\_mask** (`bool`, _optional_, defaults to `False`) — Whether or not to return special tokens mask information.
-   **return\_offsets\_mapping** (`bool`, _optional_, defaults to `False`) — Whether or not to return `(char_start, char_end)` for each token.
    
    This is only available on fast tokenizers inheriting from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast), if using Python’s tokenizer, this method will raise `NotImplementedError`.
    
-   **return\_length** (`bool`, _optional_, defaults to `False`) — Whether or not to return the lengths of the encoded inputs.
-   **verbose** (`bool`, _optional_, defaults to `True`) — Whether or not to print more information and warnings. \*\*kwargs — passed to the `self.tokenize()` method

A [BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

-   **input\_ids** — List of token ids to be fed to a model.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **token\_type\_ids** — List of token type ids to be fed to a model (when `return_token_type_ids=True` or if _“token\_type\_ids”_ is in `self.model_input_names`).
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **attention\_mask** — List of indices specifying which tokens should be attended to by the model (when `return_attention_mask=True` or if _“attention\_mask”_ is in `self.model_input_names`).
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **overflowing\_tokens** — List of overflowing tokens sequences (when a `max_length` is specified and `return_overflowing_tokens=True`).
    
-   **num\_truncated\_tokens** — Number of tokens truncated (when a `max_length` is specified and `return_overflowing_tokens=True`).
    
-   **special\_tokens\_mask** — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
    
-   **length** — The length of the inputs (when `return_length=True`)
    

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of sequences.

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vits/tokenization_vits.py#L238)

( save\_directory: str filename\_prefix: typing.Optional\[str\] = None )

## VitsModel

### class transformers.VitsModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vits/modeling_vits.py#L1356)

( config: VitsConfig )

Parameters

-   **config** ([VitsConfig](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The complete VITS model, for text-to-speech synthesis. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vits/modeling_vits.py#L1386)

( input\_ids: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None speaker\_id: typing.Optional\[int\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None labels: typing.Optional\[torch.FloatTensor\] = None ) → `transformers.models.vits.modeling_vits.VitsModelOutput` or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **speaker\_id** (`int`, _optional_) — Which speaker embedding to use. Only used for multispeaker models.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.FloatTensor` of shape `(batch_size, config.spectrogram_bins, sequence_length)`, _optional_) — Float values of target spectrogram. Timesteps set to `-100.0` are ignored (masked) for the loss computation.

Returns

`transformers.models.vits.modeling_vits.VitsModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.vits.modeling_vits.VitsModelOutput` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([VitsConfig](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsConfig)) and inputs.

-   **waveform** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — The final audio waveform predicted by the model.
    
-   **sequence\_lengths** (`torch.FloatTensor` of shape `(batch_size,)`) — The length in samples of each element in the `waveform` batch.
    
-   **spectrogram** (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_bins)`) — The log-mel spectrogram predicted at the output of the flow model. This spectrogram is passed to the Hi-Fi GAN decoder model to obtain the final audio waveform.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attention weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [VitsModel](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import VitsTokenizer, VitsModel, set_seed
>>> import torch

>>> tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
>>> model = VitsModel.from_pretrained("facebook/mms-tts-eng")

>>> inputs = tokenizer(text="Hello - my dog is cute", return_tensors="pt")

>>> set_seed(555)  

>>> with torch.no_grad():
...     outputs = model(inputs["input_ids"])
>>> outputs.waveform.shape
torch.Size([1, 45824])
```