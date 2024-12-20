# Wav2Vec2Phoneme

Transformers documentation

Natural Language Processing

Performance and scalability

Reinforcement learning models

## Overview

The Wav2Vec2Phoneme model was proposed in [Simple and Effective Zero-shot Cross-lingual Phoneme Recognition (Xu et al., 2021](https://arxiv.org/abs/2109.11680) by Qiantong Xu, Alexei Baevski, Michael Auli.

The abstract from the paper is the following:

_Recent progress in self-training, self-supervised pretraining and unsupervised learning enabled well performing speech recognition systems without any labeled data. However, in many cases there is labeled data available for related languages which is not utilized by these methods. This paper extends previous work on zero-shot cross-lingual transfer learning by fine-tuning a multilingually pretrained wav2vec 2.0 model to transcribe unseen languages. This is done by mapping phonemes of the training languages to the target language using articulatory features. Experiments show that this simple method significantly outperforms prior work which introduced task-specific architectures and used only part of a monolingually pretrained model._

Tips:

-   Wav2Vec2Phoneme uses the exact same architecture as Wav2Vec2
-   Wav2Vec2Phoneme is a speech model that accepts a float array corresponding to the raw waveform of the speech signal.
-   Wav2Vec2Phoneme model was trained using connectionist temporal classification (CTC) so the model output has to be decoded using [Wav2Vec2PhonemeCTCTokenizer](/docs/transformers/v4.34.0/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer).
-   Wav2Vec2Phoneme can be fine-tuned on multiple language at once and decode unseen languages in a single forward pass to a sequence of phonemes
-   By default the model outputs a sequence of phonemes. In order to transform the phonemes to a sequence of words one should make use of a dictionary and language model.

Relevant checkpoints can be found under [https://huggingface.co/models?other=phoneme-recognition](https://huggingface.co/models?other=phoneme-recognition).

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten)

The original code can be found [here](https://github.com/pytorch/fairseq/tree/master/fairseq/models/wav2vec).

Wav2Vec2Phoneme’s architecture is based on the Wav2Vec2 model, so one can refer to `Wav2Vec2`’s documentation page except for the tokenizer.

## Wav2Vec2PhonemeCTCTokenizer

### class transformers.Wav2Vec2PhonemeCTCTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2_phoneme/tokenization_wav2vec2_phoneme.py#L94)

( vocab\_filebos\_token = '<s>'eos\_token = '</s>'unk\_token = '<unk>'pad\_token = '<pad>'phone\_delimiter\_token = ' 'word\_delimiter\_token = Nonedo\_phonemize = Truephonemizer\_lang = 'en-us'phonemizer\_backend = 'espeak'\*\*kwargs )

Constructs a Wav2Vec2PhonemeCTC tokenizer.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains some of the main methods. Users should refer to the superclass for more information regarding such methods.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L2732)

( text: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\]\] = Nonetext\_pair: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\], NoneType\] = Nonetext\_target: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\]\] = Nonetext\_pair\_target: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\], NoneType\] = Noneadd\_special\_tokens: bool = Truepadding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = Falsetruncation: typing.Union\[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy\] = Nonemax\_length: typing.Optional\[int\] = Nonestride: int = 0is\_split\_into\_words: bool = Falsepad\_to\_multiple\_of: typing.Optional\[int\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonereturn\_token\_type\_ids: typing.Optional\[bool\] = Nonereturn\_attention\_mask: typing.Optional\[bool\] = Nonereturn\_overflowing\_tokens: bool = Falsereturn\_special\_tokens\_mask: bool = Falsereturn\_offsets\_mapping: bool = Falsereturn\_length: bool = Falseverbose: bool = True\*\*kwargs ) → [BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding)

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of sequences.

#### batch\_decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2_phoneme/tokenization_wav2vec2_phoneme.py#L523)

( sequences: typing.Union\[typing.List\[int\], typing.List\[typing.List\[int\]\], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')\]skip\_special\_tokens: bool = Falseclean\_up\_tokenization\_spaces: bool = Noneoutput\_char\_offsets: bool = False\*\*kwargs ) → `List[str]` or `~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput`

Convert a list of lists of token ids into a list of strings by calling decode.

#### decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2_phoneme/tokenization_wav2vec2_phoneme.py#L467)

( token\_ids: typing.Union\[int, typing.List\[int\], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')\]skip\_special\_tokens: bool = Falseclean\_up\_tokenization\_spaces: bool = Noneoutput\_char\_offsets: bool = False\*\*kwargs ) → `str` or `~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput`

Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special tokens and clean up tokenization spaces.

Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

#### phonemize

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2_phoneme/tokenization_wav2vec2_phoneme.py#L268)

( text: strphonemizer\_lang: typing.Optional\[str\] = None )