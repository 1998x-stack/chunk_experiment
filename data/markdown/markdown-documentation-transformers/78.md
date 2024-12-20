# Pipelines

The pipelines are a great and easy way to use models for inference. These pipelines are objects that abstract most of the complex code from the library, offering a simple API dedicated to several tasks, including Named Entity Recognition, Masked Language Modeling, Sentiment Analysis, Feature Extraction and Question Answering. See the [task summary](../task_summary) for examples of use.

There are two categories of pipeline abstractions to be aware about:

-   The [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) which is the most powerful object encapsulating all other pipelines.
-   Task-specific pipelines are available for [audio](#audio), [computer vision](#computer-vision), [natural language processing](#natural-language-processing), and [multimodal](#multimodal) tasks.

## The pipeline abstraction

The _pipeline_ abstraction is a wrapper around all the other available pipelines. It is instantiated as any other pipeline but can provide additional quality of life.

Simple call on one item:

```
>>> pipe = pipeline("text-classification")
>>> pipe("This restaurant is awesome")
[{'label': 'POSITIVE', 'score': 0.9998743534088135}]
```

If you want to use a specific model from the [hub](https://huggingface.co/) you can ignore the task if the model on the hub already defines it:

```
>>> pipe = pipeline(model="roberta-large-mnli")
>>> pipe("This restaurant is awesome")
[{'label': 'NEUTRAL', 'score': 0.7313136458396912}]
```

To call a pipeline on many items, you can call it with a _list_.

```
>>> pipe = pipeline("text-classification")
>>> pipe(["This restaurant is awesome", "This restaurant is awful"])
[{'label': 'POSITIVE', 'score': 0.9998743534088135},
 {'label': 'NEGATIVE', 'score': 0.9996669292449951}]
```

To iterate over full datasets it is recommended to use a `dataset` directly. This means you don’t need to allocate the whole dataset at once, nor do you need to do batching yourself. This should work just as fast as custom loops on GPU. If it doesn’t don’t hesitate to create an issue.

```
import datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0)
dataset = datasets.load_dataset("superb", name="asr", split="test")



for out in tqdm(pipe(KeyDataset(dataset, "file"))):
    print(out)
    
    
    
```

For ease of use, a generator is also possible:

```
from transformers import pipeline

pipe = pipeline("text-classification")


def data():
    while True:
        
        
        
        
        
        yield "This is a test"


for out in pipe(data()):
    print(out)
    
    
    
```

#### transformers.pipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/__init__.py#L527)

( task: str = Nonemodel: typing.Union\[str, ForwardRef('PreTrainedModel'), ForwardRef('TFPreTrainedModel'), NoneType\] = Noneconfig: typing.Union\[str, transformers.configuration\_utils.PretrainedConfig, NoneType\] = Nonetokenizer: typing.Union\[str, transformers.tokenization\_utils.PreTrainedTokenizer, ForwardRef('PreTrainedTokenizerFast'), NoneType\] = Nonefeature\_extractor: typing.Union\[str, ForwardRef('SequenceFeatureExtractor'), NoneType\] = Noneimage\_processor: typing.Union\[str, transformers.image\_processing\_utils.BaseImageProcessor, NoneType\] = Noneframework: typing.Optional\[str\] = Nonerevision: typing.Optional\[str\] = Noneuse\_fast: bool = Truetoken: typing.Union\[bool, str, NoneType\] = Nonedevice: typing.Union\[int, str, ForwardRef('torch.device'), NoneType\] = Nonedevice\_map = Nonetorch\_dtype = Nonetrust\_remote\_code: typing.Optional\[bool\] = Nonemodel\_kwargs: typing.Dict\[str, typing.Any\] = Nonepipeline\_class: typing.Optional\[typing.Any\] = None\*\*kwargs ) → [Pipeline](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.Pipeline)

Utility factory method to build a [Pipeline](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.Pipeline).

Pipelines are made of:

-   A [tokenizer](tokenizer) in charge of mapping raw textual input to token.
-   A [model](model) to make predictions from the inputs.
-   Some (optional) post processing for enhancing model’s output.

Examples:

```
>>> from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

>>> 
>>> analyzer = pipeline("sentiment-analysis")

>>> 
>>> oracle = pipeline(
...     "question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="bert-base-cased"
... )

>>> 
>>> model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
>>> recognizer = pipeline("ner", model=model, tokenizer=tokenizer)
```

## Pipeline batching

All pipelines can use batching. This will work whenever the pipeline uses its streaming ability (so when passing lists or `Dataset` or `generator`).

```
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets

dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
pipe = pipeline("text-classification", device=0)
for out in pipe(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
    
    
    
```

However, this is not automatically a win for performance. It can be either a 10x speedup or 5x slowdown depending on hardware, data and the actual model being used.

Example where it’s mostly a speedup:

```
from transformers import pipeline
from torch.utils.data import Dataset
from tqdm.auto import tqdm

pipe = pipeline("text-classification", device=0)


class MyDataset(Dataset):
    def __len__(self):
        return 5000

    def __getitem__(self, i):
        return "This is a test"


dataset = MyDataset()

for batch_size in [1, 8, 64, 256]:
    print("-" * 30)
    print(f"Streaming batch_size={batch_size}")
    for out in tqdm(pipe(dataset, batch_size=batch_size), total=len(dataset)):
        pass
```

```
# On GTX 970
------------------------------
Streaming no batching
100%|██████████████████████████████████████████████████████████████████████| 5000/5000 [00:26<00:00, 187.52it/s]
------------------------------
Streaming batch_size=8
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:04<00:00, 1205.95it/s]
------------------------------
Streaming batch_size=64
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:02<00:00, 2478.24it/s]
------------------------------
Streaming batch_size=256
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:01<00:00, 2554.43it/s]
(diminishing returns, saturated the GPU)
```

Example where it’s most a slowdown:

```
class MyDataset(Dataset):
    def __len__(self):
        return 5000

    def __getitem__(self, i):
        if i % 64 == 0:
            n = 100
        else:
            n = 1
        return "This is a test" * n
```

This is a occasional very long sentence compared to the other. In that case, the **whole** batch will need to be 400 tokens long, so the whole batch will be \[64, 400\] instead of \[64, 4\], leading to the high slowdown. Even worse, on bigger batches, the program simply crashes.

```
Streaming no batching
100%|█████████████████████████████████████████████████████████████████████| 1000/1000 [00:05<00:00, 183.69it/s]

Streaming batch_size=8
100%|█████████████████████████████████████████████████████████████████████| 1000/1000 [00:03<00:00, 265.74it/s]

Streaming batch_size=64
100%|██████████████████████████████████████████████████████████████████████| 1000/1000 [00:26<00:00, 37.80it/s]

Streaming batch_size=256
  0%|                                                                                 | 0/1000 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/home/nicolas/src/transformers/test.py", line 42, in <module>
    for out in tqdm(pipe(dataset, batch_size=256), total=len(dataset)):
....
    q = q / math.sqrt(dim_per_head)  
RuntimeError: CUDA out of memory. Tried to allocate 376.00 MiB (GPU 0; 3.95 GiB total capacity; 1.72 GiB already allocated; 354.88 MiB free; 2.46 GiB reserved in total by PyTorch)
```

There are no good (general) solutions for this problem, and your mileage may vary depending on your use cases. Rule of thumb:

For users, a rule of thumb is:

-   **Measure performance on your load, with your hardware. Measure, measure, and keep measuring. Real numbers are the only way to go.**
    
-   If you are latency constrained (live product doing inference), don’t batch
    
-   If you are using CPU, don’t batch.
    
-   If you are using throughput (you want to run your model on a bunch of static data), on GPU, then:
    
    -   If you have no clue about the size of the sequence\_length (“natural” data), by default don’t batch, measure and try tentatively to add it, add OOM checks to recover when it will fail (and it will at some point if you don’t control the sequence\_length.)
    -   If your sequence\_length is super regular, then batching is more likely to be VERY interesting, measure and push it until you get OOMs.
    -   The larger the GPU the more likely batching is going to be more interesting
-   As soon as you enable batching, make sure you can handle OOMs nicely.
    

## Pipeline chunk batching

`zero-shot-classification` and `question-answering` are slightly specific in the sense, that a single input might yield multiple forward pass of a model. Under normal circumstances, this would yield issues with `batch_size` argument.

In order to circumvent this issue, both of these pipelines are a bit specific, they are `ChunkPipeline` instead of regular `Pipeline`. In short:

```
preprocessed = pipe.preprocess(inputs)
model_outputs = pipe.forward(preprocessed)
outputs = pipe.postprocess(model_outputs)
```

Now becomes:

```
all_model_outputs = []
for preprocessed in pipe.preprocess(inputs):
    model_outputs = pipe.forward(preprocessed)
    all_model_outputs.append(model_outputs)
outputs = pipe.postprocess(all_model_outputs)
```

This should be very transparent to your code because the pipelines are used in the same way.

This is a simplified view, since the pipeline can handle automatically the batch to ! Meaning you don’t have to care about how many forward passes you inputs are actually going to trigger, you can optimize the `batch_size` independently of the inputs. The caveats from the previous section still apply.

## Pipeline custom code

If you want to override a specific pipeline.

Don’t hesitate to create an issue for your task at hand, the goal of the pipeline is to be easy to use and support most cases, so `transformers` could maybe support your use case.

If you want to try simply you can:

-   Subclass your pipeline of choice

```
class MyPipeline(TextClassificationPipeline):
    def postprocess():
        
        scores = scores * 100
        


my_pipeline = MyPipeline(model=model, tokenizer=tokenizer, ...)

my_pipeline = pipeline(model="xxxx", pipeline_class=MyPipeline)
```

That should enable you to do all the custom code you want.

## Implementing a pipeline

[Implementing a new pipeline](../add_new_pipeline)

## Audio

Pipelines available for audio tasks include the following.

### AudioClassificationPipeline

### class transformers.AudioClassificationPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/audio_classification.py#L67)

( \*args\*\*kwargs )

Audio classification pipeline using any `AutoModelForAudioClassification`. This pipeline predicts the class of a raw waveform or an audio file. In case of an audio file, ffmpeg should be installed to support multiple audio formats.

Example:

```
>>> from transformers import pipeline

>>> classifier = pipeline(model="superb/wav2vec2-base-superb-ks")
>>> classifier("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
[{'score': 0.997, 'label': '_unknown_'}, {'score': 0.002, 'label': 'left'}, {'score': 0.0, 'label': 'yes'}, {'score': 0.0, 'label': 'down'}, {'score': 0.0, 'label': 'stop'}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifier: `"audio-classification"`.

See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=audio-classification).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/audio_classification.py#L103)

( inputs: typing.Union\[numpy.ndarray, bytes, str\]\*\*kwargs ) → A list of `dict` with the following keys

Classify the sequence(s) given as inputs. See the [AutomaticSpeechRecognitionPipeline](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline) documentation for more information.

### AutomaticSpeechRecognitionPipeline

### class transformers.AutomaticSpeechRecognitionPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/automatic_speech_recognition.py#L135)

( model: PreTrainedModelfeature\_extractor: typing.Union\[ForwardRef('SequenceFeatureExtractor'), str\] = Nonetokenizer: typing.Optional\[transformers.tokenization\_utils.PreTrainedTokenizer\] = Nonedecoder: typing.Union\[ForwardRef('BeamSearchDecoderCTC'), str, NoneType\] = Nonemodelcard: typing.Optional\[transformers.modelcard.ModelCard\] = Noneframework: typing.Optional\[str\] = Nonetask: str = ''args\_parser: ArgumentHandler = Nonedevice: typing.Union\[int, ForwardRef('torch.device')\] = Nonetorch\_dtype: typing.Union\[str, ForwardRef('torch.dtype'), NoneType\] = Nonebinary\_output: bool = False\*\*kwargs )

Pipeline that aims at extracting spoken text contained within some audio.

The input can be either a raw waveform or a audio file. In case of the audio file, ffmpeg should be installed for to support multiple audio formats

Example:

```
>>> from transformers import pipeline

>>> transcriber = pipeline(model="openai/whisper-base")
>>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
{'text': ' He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered flour-fatten sauce.'}
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/automatic_speech_recognition.py#L294)

( inputs: typing.Union\[numpy.ndarray, bytes, str\]\*\*kwargs ) → `Dict`

Transcribe the audio sequence(s) given as inputs to text. See the [AutomaticSpeechRecognitionPipeline](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline) documentation for more information.

### TextToAudioPipeline

### class transformers.TextToAudioPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/text_to_audio.py#L27)

( \*argsvocoder = Nonesampling\_rate = None\*\*kwargs )

Text-to-audio generation pipeline using any `AutoModelForTextToWaveform` or `AutoModelForTextToSpectrogram`. This pipeline generates an audio file from an input text and optional other conditional inputs.

Example:

```
>>> from transformers import pipeline

>>> pipe = pipeline(model="suno/bark-small")
>>> output = pipe("Hey it's HuggingFace on the phone!")

>>> audio = output["audio"]
>>> sampling_rate = output["sampling_rate"]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifiers: `"text-to-speech"` or `"text-to-audio"`.

See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=text-to-speech).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/text_to_audio.py#L122)

( text\_inputs: typing.Union\[str, typing.List\[str\]\]\*\*forward\_params ) → A `dict` or a list of `dict`

Parameters

-   **text\_inputs** (`str` or `List[str]`) — The text(s) to generate.
-   **forward\_params** (_optional_) — Parameters passed to the model generation/forward method.

Returns

A `dict` or a list of `dict`

The dictionaries have two keys:

-   **audio** (`np.ndarray` of shape `(nb_channels, audio_length)`) — The generated audio waveform.
-   **sampling\_rate** (`int`) — The sampling rate of the generated audio waveform.

Generates speech/audio from the inputs. See the [TextToAudioPipeline](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.TextToAudioPipeline) documentation for more information.

### ZeroShotAudioClassificationPipeline

### class transformers.ZeroShotAudioClassificationPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/zero_shot_audio_classification.py#L33)

( \*\*kwargs )

Zero shot audio classification pipeline using `ClapModel`. This pipeline predicts the class of an audio when you provide an audio and a set of `candidate_labels`.

Example:

```
>>> from transformers import pipeline
>>> from datasets import load_dataset

>>> dataset = load_dataset("ashraq/esc50")
>>> audio = next(iter(dataset["train"]["audio"]))["array"]
>>> classifier = pipeline(task="zero-shot-audio-classification", model="laion/clap-htsat-unfused")
>>> classifier(audio, candidate_labels=["Sound of a dog", "Sound of vaccum cleaner"])
[{'score': 0.9996, 'label': 'Sound of a dog'}, {'score': 0.0004, 'label': 'Sound of vaccum cleaner'}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial) This audio classification pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifier: `"zero-shot-audio-classification"`. See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=zero-shot-audio-classification).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/zero_shot_audio_classification.py#L64)

( audios: typing.Union\[numpy.ndarray, bytes, str\]\*\*kwargs )

Parameters

-   **audios** (`str`, `List[str]`, `np.array` or `List[np.array]`) — The pipeline handles three types of inputs:
    
    -   A string containing a http link pointing to an audio
    -   A string containing a local path to an audio
    -   An audio loaded in numpy
    
-   **candidate\_labels** (`List[str]`) — The candidate labels for this audio
-   **hypothesis\_template** (`str`, _optional_, defaults to `"This is a sound of {}"`) — The sentence used in cunjunction with _candidate\_labels_ to attempt the audio classification by replacing the placeholder with the candidate\_labels. Then likelihood is estimated by using logits\_per\_audio

Assign labels to the audio(s) passed as inputs.

## Computer vision

Pipelines available for computer vision tasks include the following.

### DepthEstimationPipeline

### class transformers.DepthEstimationPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/depth_estimation.py#L23)

( \*args\*\*kwargs )

Depth estimation pipeline using any `AutoModelForDepthEstimation`. This pipeline predicts the depth of an image.

Example:

```
>>> from transformers import pipeline

>>> depth_estimator = pipeline(task="depth-estimation", model="Intel/dpt-large")
>>> output = depth_estimator("http://images.cocodataset.org/val2017/000000039769.jpg")
>>> 
>>> output["predicted_depth"].shape
torch.Size([1, 384, 384])
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This depth estimation pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifier: `"depth-estimation"`.

See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=depth-estimation).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/depth_estimation.py#L53)

( images: typing.Union\[str, typing.List\[str\], ForwardRef('Image.Image'), typing.List\[ForwardRef('Image.Image')\]\]\*\*kwargs )

Parameters

-   **images** (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`) — The pipeline handles three types of images:
    
    -   A string containing a http link pointing to an image
    -   A string containing a local path to an image
    -   An image loaded in PIL directly
    
    The pipeline accepts either a single image or a batch of images, which must then be passed as a string. Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL images.
    
-   **top\_k** (`int`, _optional_, defaults to 5) — The number of top labels that will be returned by the pipeline. If the provided number is higher than the number of labels available in the model configuration, it will default to the number of labels.
-   **timeout** (`float`, _optional_, defaults to None) — The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and the call may block forever.

Assign labels to the image(s) passed as inputs.

### ImageClassificationPipeline

### class transformers.ImageClassificationPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/image_classification.py#L32)

( \*args\*\*kwargs )

Image classification pipeline using any `AutoModelForImageClassification`. This pipeline predicts the class of an image.

Example:

```
>>> from transformers import pipeline

>>> classifier = pipeline(model="microsoft/beit-base-patch16-224-pt22k-ft22k")
>>> classifier("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
[{'score': 0.442, 'label': 'macaw'}, {'score': 0.088, 'label': 'popinjay'}, {'score': 0.075, 'label': 'parrot'}, {'score': 0.073, 'label': 'parodist, lampooner'}, {'score': 0.046, 'label': 'poll, poll_parrot'}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This image classification pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifier: `"image-classification"`.

See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=image-classification).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/image_classification.py#L74)

( images: typing.Union\[str, typing.List\[str\], ForwardRef('Image.Image'), typing.List\[ForwardRef('Image.Image')\]\]\*\*kwargs )

Parameters

-   **images** (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`) — The pipeline handles three types of images:
    
    -   A string containing a http link pointing to an image
    -   A string containing a local path to an image
    -   An image loaded in PIL directly
    
    The pipeline accepts either a single image or a batch of images, which must then be passed as a string. Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL images.
    
-   **top\_k** (`int`, _optional_, defaults to 5) — The number of top labels that will be returned by the pipeline. If the provided number is higher than the number of labels available in the model configuration, it will default to the number of labels.
-   **timeout** (`float`, _optional_, defaults to None) — The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and the call may block forever.

Assign labels to the image(s) passed as inputs.

### ImageSegmentationPipeline

### class transformers.ImageSegmentationPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/image_segmentation.py#L31)

( \*args\*\*kwargs )

Image segmentation pipeline using any `AutoModelForXXXSegmentation`. This pipeline predicts masks of objects and their classes.

Example:

```
>>> from transformers import pipeline

>>> segmenter = pipeline(model="facebook/detr-resnet-50-panoptic")
>>> segments = segmenter("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
>>> len(segments)
2

>>> segments[0]["label"]
'bird'

>>> segments[1]["label"]
'bird'

>>> type(segments[0]["mask"])  
<class 'PIL.Image.Image'>

>>> segments[0]["mask"].size
(768, 512)
```

This image segmentation pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifier: `"image-segmentation"`.

See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=image-segmentation).

Perform segmentation (detect masks & classes) in the image(s) passed as inputs.

### ImageToImagePipeline

### class transformers.ImageToImagePipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/image_to_image.py#L40)

( \*args\*\*kwargs )

Image to Image pipeline using any `AutoModelForImageToImage`. This pipeline generates an image based on a previous image input.

Example:

```
>>> from PIL import Image
>>> import requests

>>> from transformers import pipeline

>>> upscaler = pipeline("image-to-image", model="caidas/swin2SR-classical-sr-x2-64")
>>> img = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
>>> img = img.resize((64, 64))
>>> upscaled_img = upscaler(img)
>>> img.size
(64, 64)

>>> upscaled_img.size
(144, 144)
```

This image to image pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifier: `"image-to-image"`.

See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=image-to-image).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/image_to_image.py#L87)

( images: typing.Union\[str, typing.List\[str\], ForwardRef('Image.Image'), typing.List\[ForwardRef('Image.Image')\]\]\*\*kwargs )

Parameters

-   **images** (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`) — The pipeline handles three types of images:
    
    -   A string containing a http link pointing to an image
    -   A string containing a local path to an image
    -   An image loaded in PIL directly
    
    The pipeline accepts either a single image or a batch of images, which must then be passed as a string. Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL images.
    
-   **timeout** (`float`, _optional_, defaults to None) — The maximum time in seconds to wait for fetching images from the web. If None, no timeout is used and the call may block forever.

Transform the image(s) passed as inputs.

### ObjectDetectionPipeline

### class transformers.ObjectDetectionPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/object_detection.py#L27)

( \*args\*\*kwargs )

Object detection pipeline using any `AutoModelForObjectDetection`. This pipeline predicts bounding boxes of objects and their classes.

Example:

```
>>> from transformers import pipeline

>>> detector = pipeline(model="facebook/detr-resnet-50")
>>> detector("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
[{'score': 0.997, 'label': 'bird', 'box': {'xmin': 69, 'ymin': 171, 'xmax': 396, 'ymax': 507}}, {'score': 0.999, 'label': 'bird', 'box': {'xmin': 398, 'ymin': 105, 'xmax': 767, 'ymax': 507}}]

>>> 
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This object detection pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifier: `"object-detection"`.

See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=object-detection).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/object_detection.py#L72)

( \*args\*\*kwargs )

Parameters

-   **images** (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`) — The pipeline handles three types of images:
    
    -   A string containing an HTTP(S) link pointing to an image
    -   A string containing a local path to an image
    -   An image loaded in PIL directly
    
    The pipeline accepts either a single image or a batch of images. Images in a batch must all be in the same format: all as HTTP(S) links, all as local paths, or all as PIL images.
    
-   **threshold** (`float`, _optional_, defaults to 0.9) — The probability necessary to make a prediction.
-   **timeout** (`float`, _optional_, defaults to None) — The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and the call may block forever.

Detect objects (bounding boxes & classes) in the image(s) passed as inputs.

### VideoClassificationPipeline

### class transformers.VideoClassificationPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/video_classification.py#L22)

( \*args\*\*kwargs )

Video classification pipeline using any `AutoModelForVideoClassification`. This pipeline predicts the class of a video.

This video classification pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifier: `"video-classification"`.

See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=video-classification).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/video_classification.py#L51)

( videos: typing.Union\[str, typing.List\[str\]\]\*\*kwargs )

Parameters

-   **videos** (`str`, `List[str]`) — The pipeline handles three types of videos:
    
    -   A string containing a http link pointing to a video
    -   A string containing a local path to a video
    
    The pipeline accepts either a single video or a batch of videos, which must then be passed as a string. Videos in a batch must all be in the same format: all as http links or all as local paths.
    
-   **top\_k** (`int`, _optional_, defaults to 5) — The number of top labels that will be returned by the pipeline. If the provided number is higher than the number of labels available in the model configuration, it will default to the number of labels.
-   **num\_frames** (`int`, _optional_, defaults to `self.model.config.num_frames`) — The number of frames sampled from the video to run the classification on. If not provided, will default to the number of frames specified in the model configuration.
-   **frame\_sampling\_rate** (`int`, _optional_, defaults to 1) — The sampling rate used to select frames from the video. If not provided, will default to 1, i.e. every frame will be used.

Assign labels to the video(s) passed as inputs.

### ZeroShotImageClassificationPipeline

### class transformers.ZeroShotImageClassificationPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/zero_shot_image_classification.py#L31)

( \*\*kwargs )

Zero shot image classification pipeline using `CLIPModel`. This pipeline predicts the class of an image when you provide an image and a set of `candidate_labels`.

Example:

```
>>> from transformers import pipeline

>>> classifier = pipeline(model="openai/clip-vit-large-patch14")
>>> classifier(
...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
...     candidate_labels=["animals", "humans", "landscape"],
... )
[{'score': 0.965, 'label': 'animals'}, {'score': 0.03, 'label': 'humans'}, {'score': 0.005, 'label': 'landscape'}]

>>> classifier(
...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
...     candidate_labels=["black and white", "photorealist", "painting"],
... )
[{'score': 0.996, 'label': 'black and white'}, {'score': 0.003, 'label': 'photorealist'}, {'score': 0.0, 'label': 'painting'}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This image classification pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifier: `"zero-shot-image-classification"`.

See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=zero-shot-image-classification).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/zero_shot_image_classification.py#L74)

( images: typing.Union\[str, typing.List\[str\], ForwardRef('Image'), typing.List\[ForwardRef('Image')\]\]\*\*kwargs )

Parameters

-   **images** (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`) — The pipeline handles three types of images:
    
    -   A string containing a http link pointing to an image
    -   A string containing a local path to an image
    -   An image loaded in PIL directly
    
-   **candidate\_labels** (`List[str]`) — The candidate labels for this image
-   **hypothesis\_template** (`str`, _optional_, defaults to `"This is a photo of {}"`) — The sentence used in cunjunction with _candidate\_labels_ to attempt the image classification by replacing the placeholder with the candidate\_labels. Then likelihood is estimated by using logits\_per\_image
-   **timeout** (`float`, _optional_, defaults to None) — The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and the call may block forever.

Assign labels to the image(s) passed as inputs.

### ZeroShotObjectDetectionPipeline

### class transformers.ZeroShotObjectDetectionPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/zero_shot_object_detection.py#L23)

( \*\*kwargs )

Zero shot object detection pipeline using `OwlViTForObjectDetection`. This pipeline predicts bounding boxes of objects when you provide an image and a set of `candidate_labels`.

Example:

```
>>> from transformers import pipeline

>>> detector = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection")
>>> detector(
...     "http://images.cocodataset.org/val2017/000000039769.jpg",
...     candidate_labels=["cat", "couch"],
... )
[{'score': 0.287, 'label': 'cat', 'box': {'xmin': 324, 'ymin': 20, 'xmax': 640, 'ymax': 373}}, {'score': 0.254, 'label': 'cat', 'box': {'xmin': 1, 'ymin': 55, 'xmax': 315, 'ymax': 472}}, {'score': 0.121, 'label': 'couch', 'box': {'xmin': 4, 'ymin': 0, 'xmax': 642, 'ymax': 476}}]

>>> detector(
...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
...     candidate_labels=["head", "bird"],
... )
[{'score': 0.119, 'label': 'bird', 'box': {'xmin': 71, 'ymin': 170, 'xmax': 410, 'ymax': 508}}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This object detection pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifier: `"zero-shot-object-detection"`.

See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=zero-shot-object-detection).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/zero_shot_object_detection.py#L65)

( image: typing.Union\[str, ForwardRef('Image.Image'), typing.List\[typing.Dict\[str, typing.Any\]\]\]candidate\_labels: typing.Union\[str, typing.List\[str\]\] = None\*\*kwargs )

Parameters

-   **image** (`str`, `PIL.Image` or `List[Dict[str, Any]]`) — The pipeline handles three types of images:
    
    -   A string containing an http url pointing to an image
    -   A string containing a local path to an image
    -   An image loaded in PIL directly
    
    You can use this parameter to send directly a list of images, or a dataset or a generator like so:
    

Detect objects (bounding boxes & classes) in the image(s) passed as inputs.

## Natural Language Processing

Pipelines available for natural language processing tasks include the following.

### ConversationalPipeline

### class transformers.Conversation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/conversational.py#L18)

( messages: typing.Union\[str, typing.List\[typing.Dict\[str, str\]\]\] = Noneconversation\_id: UUID = None\*\*deprecated\_kwargs )

Parameters

-   **messages** (Union\[str, List\[Dict\[str, str\]\]\], _optional_) — The initial messages to start the conversation, either a string, or a list of dicts containing “role” and “content” keys. If a string is passed, it is interpreted as a single message with the “user” role.
-   **conversation\_id** (`uuid.UUID`, _optional_) — Unique identifier for the conversation. If not provided, a random UUID4 id will be assigned to the conversation.

Utility class containing a conversation and its history. This class is meant to be used as an input to the [ConversationalPipeline](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.ConversationalPipeline). The conversation contains several utility functions to manage the addition of new user inputs and generated model responses.

Usage:

```
conversation = Conversation("Going to the movies tonight - any suggestions?")
conversation.add_message({"role": "assistant", "content": "The Big lebowski."})
conversation.add_message({"role": "user", "content": "Is it good?"})
```

#### add\_user\_input

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/conversational.py#L88)

( text: stroverwrite: bool = False )

Add a user input to the conversation for the next round. This is a legacy method that assumes that inputs must alternate user/assistant/user/assistant, and so will not add multiple user messages in succession. We recommend just using `add_message` with role “user” instead.

This is a legacy method. We recommend just using `add_message` with an appropriate role instead.

This is a legacy method that no longer has any effect, as the Conversation no longer distinguishes between processed and unprocessed user input.

### class transformers.ConversationalPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/conversational.py#L179)

( \*args\*\*kwargs )

Multi-turn conversational pipeline.

Example:

```
>>> from transformers import pipeline, Conversation

>>> chatbot = pipeline(model="microsoft/DialoGPT-medium")
>>> conversation = Conversation("Going to the movies tonight - any suggestions?")
>>> conversation = chatbot(conversation)
>>> conversation.generated_responses[-1]
'The Big Lebowski'

>>> conversation.add_user_input("Is it an action movie?")
>>> conversation = chatbot(conversation)
>>> conversation.generated_responses[-1]
"It's a comedy."
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This conversational pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifier: `"conversational"`.

The models that this pipeline can use are models that have been fine-tuned on a multi-turn conversational task, currently: _‘microsoft/DialoGPT-small’_, _‘microsoft/DialoGPT-medium’_, _‘microsoft/DialoGPT-large’_. See the up-to-date list of available models on [huggingface.co/models](https://huggingface.co/models?filter=conversational).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/conversational.py#L238)

( conversations: typing.Union\[transformers.pipelines.conversational.Conversation, typing.List\[transformers.pipelines.conversational.Conversation\]\]num\_workers = 0\*\*kwargs ) → [Conversation](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.Conversation) or a list of [Conversation](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.Conversation)

Parameters

-   **conversations** (a [Conversation](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.Conversation) or a list of [Conversation](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.Conversation)) — Conversations to generate responses for.
-   **clean\_up\_tokenization\_spaces** (`bool`, _optional_, defaults to `False`) — Whether or not to clean up the potential extra spaces in the text output. generate\_kwargs — Additional keyword arguments to pass along to the generate method of the model (see the generate method corresponding to your framework [here](./model#generative-models)).

Conversation(s) with updated generated responses for those containing a new user input.

Generate responses for the conversation(s) given as inputs.

### FillMaskPipeline

### class transformers.FillMaskPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/fill_mask.py#L34)

( model: typing.Union\[ForwardRef('PreTrainedModel'), ForwardRef('TFPreTrainedModel')\]tokenizer: typing.Optional\[transformers.tokenization\_utils.PreTrainedTokenizer\] = Nonefeature\_extractor: typing.Optional\[ForwardRef('SequenceFeatureExtractor')\] = Noneimage\_processor: typing.Optional\[transformers.image\_processing\_utils.BaseImageProcessor\] = Nonemodelcard: typing.Optional\[transformers.modelcard.ModelCard\] = Noneframework: typing.Optional\[str\] = Nonetask: str = ''args\_parser: ArgumentHandler = Nonedevice: typing.Union\[int, ForwardRef('torch.device')\] = Nonetorch\_dtype: typing.Union\[str, ForwardRef('torch.dtype'), NoneType\] = Nonebinary\_output: bool = False\*\*kwargs )

Masked language modeling prediction pipeline using any `ModelWithLMHead`. See the [masked language modeling examples](../task_summary#masked-language-modeling) for more information.

Example:

```
>>> from transformers import pipeline

>>> fill_masker = pipeline(model="bert-base-uncased")
>>> fill_masker("This is a simple [MASK].")
[{'score': 0.042, 'token': 3291, 'token_str': 'problem', 'sequence': 'this is a simple problem.'}, {'score': 0.031, 'token': 3160, 'token_str': 'question', 'sequence': 'this is a simple question.'}, {'score': 0.03, 'token': 8522, 'token_str': 'equation', 'sequence': 'this is a simple equation.'}, {'score': 0.027, 'token': 2028, 'token_str': 'one', 'sequence': 'this is a simple one.'}, {'score': 0.024, 'token': 3627, 'token_str': 'rule', 'sequence': 'this is a simple rule.'}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This mask filling pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifier: `"fill-mask"`.

The models that this pipeline can use are models that have been trained with a masked language modeling objective, which includes the bi-directional models in the library. See the up-to-date list of available models on [huggingface.co/models](https://huggingface.co/models?filter=fill-mask).

This pipeline only works for inputs with exactly one token masked. Experimental: We added support for multiple masks. The returned values are raw model output, and correspond to disjoint probabilities where one might expect joint probabilities (See [discussion](https://github.com/huggingface/transformers/pull/10222)).

This pipeline now supports tokenizer\_kwargs. For example try:

```
>>> from transformers import pipeline

>>> fill_masker = pipeline(model="bert-base-uncased")
>>> tokenizer_kwargs = {"truncation": True}
>>> fill_masker(
...     "This is a simple [MASK]. " + "...with a large amount of repeated text appended. " * 100,
...     tokenizer_kwargs=tokenizer_kwargs,
... )
```

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/fill_mask.py#L248)

( inputs\*args\*\*kwargs ) → A list or a list of list of `dict`

Fill the masked token in the text(s) given as inputs.

### NerPipeline

### class transformers.TokenClassificationPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/token_classification.py#L96)

( args\_parser = <transformers.pipelines.token\_classification.TokenClassificationArgumentHandler object at 0x7f0a524069d0>\*args\*\*kwargs )

Named Entity Recognition pipeline using any `ModelForTokenClassification`. See the [named entity recognition examples](../task_summary#named-entity-recognition) for more information.

Example:

```
>>> from transformers import pipeline

>>> token_classifier = pipeline(model="Jean-Baptiste/camembert-ner", aggregation_strategy="simple")
>>> sentence = "Je m'appelle jean-baptiste et je vis à montréal"
>>> tokens = token_classifier(sentence)
>>> tokens
[{'entity_group': 'PER', 'score': 0.9931, 'word': 'jean-baptiste', 'start': 12, 'end': 26}, {'entity_group': 'LOC', 'score': 0.998, 'word': 'montréal', 'start': 38, 'end': 47}]

>>> token = tokens[0]
>>> 
>>> sentence[token["start"] : token["end"]]
' jean-baptiste'

>>> 
>>> syntaxer = pipeline(model="vblagoje/bert-english-uncased-finetuned-pos", aggregation_strategy="simple")
>>> syntaxer("My name is Sarah and I live in London")
[{'entity_group': 'PRON', 'score': 0.999, 'word': 'my', 'start': 0, 'end': 2}, {'entity_group': 'NOUN', 'score': 0.997, 'word': 'name', 'start': 3, 'end': 7}, {'entity_group': 'AUX', 'score': 0.994, 'word': 'is', 'start': 8, 'end': 10}, {'entity_group': 'PROPN', 'score': 0.999, 'word': 'sarah', 'start': 11, 'end': 16}, {'entity_group': 'CCONJ', 'score': 0.999, 'word': 'and', 'start': 17, 'end': 20}, {'entity_group': 'PRON', 'score': 0.999, 'word': 'i', 'start': 21, 'end': 22}, {'entity_group': 'VERB', 'score': 0.998, 'word': 'live', 'start': 23, 'end': 27}, {'entity_group': 'ADP', 'score': 0.999, 'word': 'in', 'start': 28, 'end': 30}, {'entity_group': 'PROPN', 'score': 0.999, 'word': 'london', 'start': 31, 'end': 37}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This token recognition pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifier: `"ner"` (for predicting the classes of tokens in a sequence: person, organisation, location or miscellaneous).

The models that this pipeline can use are models that have been fine-tuned on a token classification task. See the up-to-date list of available models on [huggingface.co/models](https://huggingface.co/models?filter=token-classification).

#### aggregate\_words

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/token_classification.py#L470)

( entities: typing.List\[dict\]aggregation\_strategy: AggregationStrategy )

Override tokens from a given word that disagree to force agreement on word boundaries.

Example: micro|soft| com|pany| B-ENT I-NAME I-ENT I-ENT will be rewritten with first strategy as microsoft| company| B-ENT I-ENT

#### gather\_pre\_entities

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/token_classification.py#L356)

( sentence: strinput\_ids: ndarrayscores: ndarrayoffset\_mapping: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], NoneType\]special\_tokens\_mask: ndarrayaggregation\_strategy: AggregationStrategy )

Fuse various numpy arrays into dicts with all the information needed for aggregation

#### group\_entities

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/token_classification.py#L533)

( entities: typing.List\[dict\] )

Parameters

-   **entities** (`dict`) — The entities predicted by the pipeline.

Find and group together the adjacent tokens with the same entity predicted.

#### group\_sub\_entities

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/token_classification.py#L498)

( entities: typing.List\[dict\] )

Parameters

-   **entities** (`dict`) — The entities predicted by the pipeline.

Group together the adjacent tokens with the same entity predicted.

See [TokenClassificationPipeline](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.TokenClassificationPipeline) for all details.

### QuestionAnsweringPipeline

### class transformers.QuestionAnsweringPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/question_answering.py#L225)

( model: typing.Union\[ForwardRef('PreTrainedModel'), ForwardRef('TFPreTrainedModel')\]tokenizer: PreTrainedTokenizermodelcard: typing.Optional\[transformers.modelcard.ModelCard\] = Noneframework: typing.Optional\[str\] = Nonetask: str = ''\*\*kwargs )

Question Answering pipeline using any `ModelForQuestionAnswering`. See the [question answering examples](../task_summary#question-answering) for more information.

Example:

```
>>> from transformers import pipeline

>>> oracle = pipeline(model="deepset/roberta-base-squad2")
>>> oracle(question="Where do I live?", context="My name is Wolfgang and I live in Berlin")
{'score': 0.9191, 'start': 34, 'end': 40, 'answer': 'Berlin'}
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This question answering pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifier: `"question-answering"`.

The models that this pipeline can use are models that have been fine-tuned on a question answering task. See the up-to-date list of available models on [huggingface.co/models](https://huggingface.co/models?filter=question-answering).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/question_answering.py#L343)

( \*args\*\*kwargs ) → A `dict` or a list of `dict`

Answer the question(s) given as inputs by using the context(s).

#### create\_sample

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/question_answering.py#L278)

( question: typing.Union\[str, typing.List\[str\]\]context: typing.Union\[str, typing.List\[str\]\] ) → One or a list of `SquadExample`

Parameters

-   **question** (`str` or `List[str]`) — The question(s) asked.
-   **context** (`str` or `List[str]`) — The context(s) in which we will look for the answer.

Returns

One or a list of `SquadExample`

The corresponding `SquadExample` grouping question and context.

QuestionAnsweringPipeline leverages the `SquadExample` internally. This helper method encapsulate all the logic for converting question(s) and context(s) to `SquadExample`.

We currently support extractive question answering.

#### span\_to\_answer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/question_answering.py#L630)

( text: strstart: intend: int ) → Dictionary like \`{‘answer’

Parameters

-   **text** (`str`) — The actual context to extract the answer from.
-   **start** (`int`) — The answer starting token index.
-   **end** (`int`) — The answer end token index.

Returns

Dictionary like \`{‘answer’

str, ‘start’: int, ‘end’: int}\`

When decoding from token probabilities, this method maps token indexes to actual word in the initial context.

### SummarizationPipeline

### class transformers.SummarizationPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/text2text_generation.py#L217)

( \*args\*\*kwargs )

Summarize news articles and other documents.

This summarizing pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifier: `"summarization"`.

The models that this pipeline can use are models that have been fine-tuned on a summarization task, which is currently, ’_bart-large-cnn_’, ’_t5-small_’, ’_t5-base_’, ’_t5-large_’, ’_t5-3b_’, ’_t5-11b_’. See the up-to-date list of available models on [huggingface.co/models](https://huggingface.co/models?filter=summarization). For a list of available parameters, see the [following documentation](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.generation.GenerationMixin.generate)

Usage:

```
summarizer = pipeline("summarization")
summarizer("An apple a day, keeps the doctor away", min_length=5, max_length=20)


summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
summarizer("An apple a day, keeps the doctor away", min_length=5, max_length=20)
```

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/text2text_generation.py#L245)

( \*args\*\*kwargs ) → A list or a list of list of `dict`

Summarize the text(s) given as inputs.

### TableQuestionAnsweringPipeline

### class transformers.TableQuestionAnsweringPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/table_question_answering.py#L88)

( args\_parser = <transformers.pipelines.table\_question\_answering.TableQuestionAnsweringArgumentHandler object at 0x7f0a524e1a60>\*args\*\*kwargs )

Table Question Answering pipeline using a `ModelForTableQuestionAnswering`. This pipeline is only available in PyTorch.

Example:

```
>>> from transformers import pipeline

>>> oracle = pipeline(model="google/tapas-base-finetuned-wtq")
>>> table = {
...     "Repository": ["Transformers", "Datasets", "Tokenizers"],
...     "Stars": ["36542", "4512", "3934"],
...     "Contributors": ["651", "77", "34"],
...     "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
... }
>>> oracle(query="How many stars does the transformers repository have?", table=table)
{'answer': 'AVERAGE > 36542', 'coordinates': [(0, 1)], 'cells': ['36542'], 'aggregator': 'AVERAGE'}
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This tabular question answering pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifier: `"table-question-answering"`.

The models that this pipeline can use are models that have been fine-tuned on a tabular question answering task. See the up-to-date list of available models on [huggingface.co/models](https://huggingface.co/models?filter=table-question-answering).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/table_question_answering.py#L270)

( \*args\*\*kwargs ) → A dictionary or a list of dictionaries containing results

Answers queries according to a table. The pipeline accepts several types of inputs which are detailed below:

-   `pipeline(table, query)`
-   `pipeline(table, [query])`
-   `pipeline(table=table, query=query)`
-   `pipeline(table=table, query=[query])`
-   `pipeline({"table": table, "query": query})`
-   `pipeline({"table": table, "query": [query]})`
-   `pipeline([{"table": table, "query": query}, {"table": table, "query": query}])`

The `table` argument should be a dict or a DataFrame built from that dict, containing the whole table:

Example:

```
data = {
    "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
    "age": ["56", "45", "59"],
    "number of movies": ["87", "53", "69"],
    "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
}
```

This dictionary can be passed in as such, or can be converted to a pandas DataFrame:

Example:

```
import pandas as pd

table = pd.DataFrame.from_dict(data)
```

### TextClassificationPipeline

### class transformers.TextClassificationPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/text_classification.py#L49)

( \*\*kwargs )

Text classification pipeline using any `ModelForSequenceClassification`. See the [sequence classification examples](../task_summary#sequence-classification) for more information.

Example:

```
>>> from transformers import pipeline

>>> classifier = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")
>>> classifier("This movie is disgustingly good !")
[{'label': 'POSITIVE', 'score': 1.0}]

>>> classifier("Director tried too much.")
[{'label': 'NEGATIVE', 'score': 0.996}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This text classification pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifier: `"sentiment-analysis"` (for classifying sequences according to positive or negative sentiments).

If multiple classification labels are available (`model.config.num_labels >= 2`), the pipeline will run a softmax over the results. If there is a single label, the pipeline will run a sigmoid over the result.

The models that this pipeline can use are models that have been fine-tuned on a sequence classification task. See the up-to-date list of available models on [huggingface.co/models](https://huggingface.co/models?filter=text-classification).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/text_classification.py#L122)

( \*args\*\*kwargs ) → A list or a list of list of `dict`

Classify the text(s) given as inputs.

### TextGenerationPipeline

### class transformers.TextGenerationPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/text_generation.py#L24)

( \*args\*\*kwargs )

Language generation pipeline using any `ModelWithLMHead`. This pipeline predicts the words that will follow a specified text prompt.

Example:

```
>>> from transformers import pipeline

>>> generator = pipeline(model="gpt2")
>>> generator("I can't believe you did such a ", do_sample=False)
[{'generated_text': "I can't believe you did such a icky thing to me. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I"}]

>>> 
>>> outputs = generator("My tart needs some", num_return_sequences=4, return_full_text=False)
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial). You can pass text generation parameters to this pipeline to control stopping criteria, decoding strategy, and more. Learn more about text generation parameters in [Text generation strategies](../generation_strategies) and [Text generation](text_generation).

This language generation pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifier: `"text-generation"`.

The models that this pipeline can use are models that have been trained with an autoregressive language modeling objective, which includes the uni-directional models in the library (e.g. gpt2). See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=text-generation).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/text_generation.py#L167)

( text\_inputs\*\*kwargs ) → A list or a list of list of `dict`

Complete the prompt(s) given as inputs.

### Text2TextGenerationPipeline

### class transformers.Text2TextGenerationPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/text2text_generation.py#L26)

( \*args\*\*kwargs )

Pipeline for text to text generation using seq2seq models.

Example:

```
>>> from transformers import pipeline

>>> generator = pipeline(model="mrm8488/t5-base-finetuned-question-generation-ap")
>>> generator(
...     "answer: Manuel context: Manuel has created RuPERTa-base with the support of HF-Transformers and Google"
... )
[{'generated_text': 'question: Who created the RuPERTa-base?'}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial). You can pass text generation parameters to this pipeline to control stopping criteria, decoding strategy, and more. Learn more about text generation parameters in [Text generation strategies](../generation_strategies) and [Text generation](text_generation).

This Text2TextGenerationPipeline pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifier: `"text2text-generation"`.

The models that this pipeline can use are models that have been fine-tuned on a translation task. See the up-to-date list of available models on [huggingface.co/models](https://huggingface.co/models?filter=text2text-generation). For a list of available parameters, see the [following documentation](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.generation.GenerationMixin.generate)

Usage:

```
text2text_generator = pipeline("text2text-generation")
text2text_generator("question: What is 42 ? context: 42 is the answer to life, the universe and everything")
```

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/text2text_generation.py#L138)

( \*args\*\*kwargs ) → A list or a list of list of `dict`

Generate the output text(s) using text(s) given as inputs.

#### check\_inputs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/text2text_generation.py#L111)

( input\_length: intmin\_length: intmax\_length: int )

Checks whether there might be something wrong with given input with regard to the model.

### TokenClassificationPipeline

### class transformers.TokenClassificationPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/token_classification.py#L96)

( args\_parser = <transformers.pipelines.token\_classification.TokenClassificationArgumentHandler object at 0x7f0a524069d0>\*args\*\*kwargs )

Named Entity Recognition pipeline using any `ModelForTokenClassification`. See the [named entity recognition examples](../task_summary#named-entity-recognition) for more information.

Example:

```
>>> from transformers import pipeline

>>> token_classifier = pipeline(model="Jean-Baptiste/camembert-ner", aggregation_strategy="simple")
>>> sentence = "Je m'appelle jean-baptiste et je vis à montréal"
>>> tokens = token_classifier(sentence)
>>> tokens
[{'entity_group': 'PER', 'score': 0.9931, 'word': 'jean-baptiste', 'start': 12, 'end': 26}, {'entity_group': 'LOC', 'score': 0.998, 'word': 'montréal', 'start': 38, 'end': 47}]

>>> token = tokens[0]
>>> 
>>> sentence[token["start"] : token["end"]]
' jean-baptiste'

>>> 
>>> syntaxer = pipeline(model="vblagoje/bert-english-uncased-finetuned-pos", aggregation_strategy="simple")
>>> syntaxer("My name is Sarah and I live in London")
[{'entity_group': 'PRON', 'score': 0.999, 'word': 'my', 'start': 0, 'end': 2}, {'entity_group': 'NOUN', 'score': 0.997, 'word': 'name', 'start': 3, 'end': 7}, {'entity_group': 'AUX', 'score': 0.994, 'word': 'is', 'start': 8, 'end': 10}, {'entity_group': 'PROPN', 'score': 0.999, 'word': 'sarah', 'start': 11, 'end': 16}, {'entity_group': 'CCONJ', 'score': 0.999, 'word': 'and', 'start': 17, 'end': 20}, {'entity_group': 'PRON', 'score': 0.999, 'word': 'i', 'start': 21, 'end': 22}, {'entity_group': 'VERB', 'score': 0.998, 'word': 'live', 'start': 23, 'end': 27}, {'entity_group': 'ADP', 'score': 0.999, 'word': 'in', 'start': 28, 'end': 30}, {'entity_group': 'PROPN', 'score': 0.999, 'word': 'london', 'start': 31, 'end': 37}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This token recognition pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifier: `"ner"` (for predicting the classes of tokens in a sequence: person, organisation, location or miscellaneous).

The models that this pipeline can use are models that have been fine-tuned on a token classification task. See the up-to-date list of available models on [huggingface.co/models](https://huggingface.co/models?filter=token-classification).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/token_classification.py#L219)

( inputs: typing.Union\[str, typing.List\[str\]\]\*\*kwargs ) → A list or a list of list of `dict`

Classify each token of the text(s) given as inputs.

#### aggregate\_words

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/token_classification.py#L470)

( entities: typing.List\[dict\]aggregation\_strategy: AggregationStrategy )

Override tokens from a given word that disagree to force agreement on word boundaries.

Example: micro|soft| com|pany| B-ENT I-NAME I-ENT I-ENT will be rewritten with first strategy as microsoft| company| B-ENT I-ENT

#### gather\_pre\_entities

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/token_classification.py#L356)

( sentence: strinput\_ids: ndarrayscores: ndarrayoffset\_mapping: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], NoneType\]special\_tokens\_mask: ndarrayaggregation\_strategy: AggregationStrategy )

Fuse various numpy arrays into dicts with all the information needed for aggregation

#### group\_entities

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/token_classification.py#L533)

( entities: typing.List\[dict\] )

Parameters

-   **entities** (`dict`) — The entities predicted by the pipeline.

Find and group together the adjacent tokens with the same entity predicted.

#### group\_sub\_entities

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/token_classification.py#L498)

( entities: typing.List\[dict\] )

Parameters

-   **entities** (`dict`) — The entities predicted by the pipeline.

Group together the adjacent tokens with the same entity predicted.

### TranslationPipeline

### class transformers.TranslationPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/text2text_generation.py#L287)

( \*args\*\*kwargs )

Translates from one language to another.

This translation pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifier: `"translation_xx_to_yy"`.

The models that this pipeline can use are models that have been fine-tuned on a translation task. See the up-to-date list of available models on [huggingface.co/models](https://huggingface.co/models?filter=translation). For a list of available parameters, see the [following documentation](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.generation.GenerationMixin.generate)

Usage:

```
en_fr_translator = pipeline("translation_en_to_fr")
en_fr_translator("How old are you?")
```

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/text2text_generation.py#L341)

( \*args\*\*kwargs ) → A list or a list of list of `dict`

Translate the text(s) given as inputs.

### ZeroShotClassificationPipeline

### class transformers.ZeroShotClassificationPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/zero_shot_classification.py#L47)

( args\_parser = <transformers.pipelines.zero\_shot\_classification.ZeroShotClassificationArgumentHandler object at 0x7f0a523da580>\*args\*\*kwargs )

NLI-based zero-shot classification pipeline using a `ModelForSequenceClassification` trained on NLI (natural language inference) tasks. Equivalent of `text-classification` pipelines, but these models don’t require a hardcoded number of potential classes, they can be chosen at runtime. It usually means it’s slower but it is **much** more flexible.

Any combination of sequences and labels can be passed and each combination will be posed as a premise/hypothesis pair and passed to the pretrained model. Then, the logit for _entailment_ is taken as the logit for the candidate label being valid. Any NLI model can be used, but the id of the _entailment_ label must be included in the model config’s :attr:_~transformers.PretrainedConfig.label2id_.

Example:

```
>>> from transformers import pipeline

>>> oracle = pipeline(model="facebook/bart-large-mnli")
>>> oracle(
...     "I have a problem with my iphone that needs to be resolved asap!!",
...     candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
... )
{'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['urgent', 'phone', 'computer', 'not urgent', 'tablet'], 'scores': [0.504, 0.479, 0.013, 0.003, 0.002]}

>>> oracle(
...     "I have a problem with my iphone that needs to be resolved asap!!",
...     candidate_labels=["english", "german"],
... )
{'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['english', 'german'], 'scores': [0.814, 0.186]}
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This NLI pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifier: `"zero-shot-classification"`.

The models that this pipeline can use are models that have been fine-tuned on an NLI task. See the up-to-date list of available models on [huggingface.co/models](https://huggingface.co/models?search=nli).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/zero_shot_classification.py#L163)

( sequences: typing.Union\[str, typing.List\[str\]\]\*args\*\*kwargs ) → A `dict` or a list of `dict`

Classify the sequence(s) given as inputs. See the [ZeroShotClassificationPipeline](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.ZeroShotClassificationPipeline) documentation for more information.

## Multimodal

Pipelines available for multimodal tasks include the following.

### DocumentQuestionAnsweringPipeline

### class transformers.DocumentQuestionAnsweringPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/document_question_answering.py#L102)

( \*args\*\*kwargs )

Document Question Answering pipeline using any `AutoModelForDocumentQuestionAnswering`. The inputs/outputs are similar to the (extractive) question answering pipeline; however, the pipeline takes an image (and optional OCR’d words/boxes) as input instead of text context.

Example:

```
>>> from transformers import pipeline

>>> document_qa = pipeline(model="impira/layoutlm-document-qa")
>>> document_qa(
...     image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
...     question="What is the invoice number?",
... )
[{'score': 0.425, 'answer': 'us-001', 'start': 16, 'end': 16}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This document question answering pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifier: `"document-question-answering"`.

The models that this pipeline can use are models that have been fine-tuned on a document question answering task. See the up-to-date list of available models on [huggingface.co/models](https://huggingface.co/models?filter=document-question-answering).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/document_question_answering.py#L194)

( image: typing.Union\[ForwardRef('Image.Image'), str\]question: typing.Optional\[str\] = Noneword\_boxes: typing.Tuple\[str, typing.List\[float\]\] = None\*\*kwargs ) → A `dict` or a list of `dict`

Answer the question(s) given as inputs by using the document(s). A document is defined as an image and an optional list of (word, box) tuples which represent the text in the document. If the `word_boxes` are not provided, it will use the Tesseract OCR engine (if available) to extract the words and boxes automatically for LayoutLM-like models which require them as input. For Donut, no OCR is run.

You can invoke the pipeline several ways:

-   `pipeline(image=image, question=question)`
-   `pipeline(image=image, question=question, word_boxes=word_boxes)`
-   `pipeline([{"image": image, "question": question}])`
-   `pipeline([{"image": image, "question": question, "word_boxes": word_boxes}])`

### FeatureExtractionPipeline

( model: typing.Union\[ForwardRef('PreTrainedModel'), ForwardRef('TFPreTrainedModel')\]tokenizer: typing.Optional\[transformers.tokenization\_utils.PreTrainedTokenizer\] = Nonefeature\_extractor: typing.Optional\[ForwardRef('SequenceFeatureExtractor')\] = Noneimage\_processor: typing.Optional\[transformers.image\_processing\_utils.BaseImageProcessor\] = Nonemodelcard: typing.Optional\[transformers.modelcard.ModelCard\] = Noneframework: typing.Optional\[str\] = Nonetask: str = ''args\_parser: ArgumentHandler = Nonedevice: typing.Union\[int, ForwardRef('torch.device')\] = Nonetorch\_dtype: typing.Union\[str, ForwardRef('torch.dtype'), NoneType\] = Nonebinary\_output: bool = False\*\*kwargs )

Feature extraction pipeline using no model head. This pipeline extracts the hidden states from the base transformer, which can be used as features in downstream tasks.

Example:

```
>>> from transformers import pipeline

>>> extractor = pipeline(model="bert-base-uncased", task="feature-extraction")
>>> result = extractor("This is a simple test.", return_tensors=True)
>>> result.shape  
torch.Size([1, 8, 768])
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This feature extraction pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the task identifier: `"feature-extraction"`.

All models may be used for this pipeline. See a list of all models, including community-contributed models on [huggingface.co/models](https://huggingface.co/models).

( \*args\*\*kwargs ) → A nested list of `float`

Parameters

-   **args** (`str` or `List[str]`) — One or several texts (or one list of texts) to get the features of.

The features computed by the model.

Extract the features of the input(s).

### ImageToTextPipeline

### class transformers.ImageToTextPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/image_to_text.py#L31)

( \*args\*\*kwargs )

Image To Text pipeline using a `AutoModelForVision2Seq`. This pipeline predicts a caption for a given image.

Example:

```
>>> from transformers import pipeline

>>> captioner = pipeline(model="ydshieh/vit-gpt2-coco-en")
>>> captioner("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
[{'generated_text': 'two birds are standing next to each other '}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This image to text pipeline can currently be loaded from pipeline() using the following task identifier: “image-to-text”.

See the list of available models on [huggingface.co/models](https://huggingface.co/models?pipeline_tag=image-to-text).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/image_to_text.py#L83)

( images: typing.Union\[str, typing.List\[str\], ForwardRef('Image.Image'), typing.List\[ForwardRef('Image.Image')\]\]\*\*kwargs ) → A list or a list of list of `dict`

Assign labels to the image(s) passed as inputs.

### VisualQuestionAnsweringPipeline

### class transformers.VisualQuestionAnsweringPipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/visual_question_answering.py#L19)

( \*args\*\*kwargs )

Visual Question Answering pipeline using a `AutoModelForVisualQuestionAnswering`. This pipeline is currently only available in PyTorch.

Example:

```
>>> from transformers import pipeline

>>> oracle = pipeline(model="dandelin/vilt-b32-finetuned-vqa")
>>> image_url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/lena.png"
>>> oracle(question="What is she wearing ?", image=image_url)
[{'score': 0.948, 'answer': 'hat'}, {'score': 0.009, 'answer': 'fedora'}, {'score': 0.003, 'answer': 'clothes'}, {'score': 0.003, 'answer': 'sun hat'}, {'score': 0.002, 'answer': 'nothing'}]

>>> oracle(question="What is she wearing ?", image=image_url, top_k=1)
[{'score': 0.948, 'answer': 'hat'}]

>>> oracle(question="Is this a person ?", image=image_url, top_k=1)
[{'score': 0.993, 'answer': 'yes'}]

>>> oracle(question="Is this a man ?", image=image_url, top_k=1)
[{'score': 0.996, 'answer': 'no'}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This visual question answering pipeline can currently be loaded from [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline) using the following task identifiers: `"visual-question-answering", "vqa"`.

The models that this pipeline can use are models that have been fine-tuned on a visual question answering task. See the up-to-date list of available models on [huggingface.co/models](https://huggingface.co/models?filter=visual-question-answering).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/visual_question_answering.py#L70)

( image: typing.Union\[ForwardRef('Image.Image'), str\]question: str = None\*\*kwargs ) → A dictionary or a list of dictionaries containing the result. The dictionaries contain the following keys

Answers open-ended questions about images. The pipeline accepts several types of inputs which are detailed below:

-   `pipeline(image=image, question=question)`
-   `pipeline({"image": image, "question": question})`
-   `pipeline([{"image": image, "question": question}])`
-   `pipeline([{"image": image, "question": question}, {"image": image, "question": question}])`

## Parent class: `Pipeline`

### class transformers.Pipeline

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/base.py#L741)

( model: typing.Union\[ForwardRef('PreTrainedModel'), ForwardRef('TFPreTrainedModel')\]tokenizer: typing.Optional\[transformers.tokenization\_utils.PreTrainedTokenizer\] = Nonefeature\_extractor: typing.Optional\[ForwardRef('SequenceFeatureExtractor')\] = Noneimage\_processor: typing.Optional\[transformers.image\_processing\_utils.BaseImageProcessor\] = Nonemodelcard: typing.Optional\[transformers.modelcard.ModelCard\] = Noneframework: typing.Optional\[str\] = Nonetask: str = ''args\_parser: ArgumentHandler = Nonedevice: typing.Union\[int, ForwardRef('torch.device')\] = Nonetorch\_dtype: typing.Union\[str, ForwardRef('torch.dtype'), NoneType\] = Nonebinary\_output: bool = False\*\*kwargs )

The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across different pipelines.

Base class implementing pipelined operations. Pipeline workflow is defined as a sequence of the following operations:

Input -> Tokenization -> Model Inference -> Post-Processing (task dependent) -> Output

Pipeline supports running on CPU or GPU through the device argument (see below).

Some pipeline, like for instance [FeatureExtractionPipeline](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.FeatureExtractionPipeline) (`'feature-extraction'`) output large tensor object as nested-lists. In order to avoid dumping such large structure as textual data we provide the `binary_output` constructor argument. If set to `True`, the output will be stored in the pickle format.

#### check\_model\_type

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/base.py#L962)

( supported\_models: typing.Union\[typing.List\[str\], dict\] )

Parameters

-   **supported\_models** (`List[str]` or `dict`) — The list of models supported by the pipeline, or a dictionary with model class values.

Check if the model class is in supported by the pipeline.

Context Manager allowing tensor allocation on the user-specified device in framework agnostic way.

Examples:

```
pipe = pipeline(..., device=0)
with pipe.device_placement():
    
    output = pipe(...)
```

#### ensure\_tensor\_on\_device

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/base.py#L928)

( \*\*inputs ) → `Dict[str, torch.Tensor]`

Parameters

-   **inputs** (keyword arguments that should be `torch.Tensor`, the rest is ignored) — The tensors to place on `self.device`.
-   **Recursive** on lists **only**. —

Returns

`Dict[str, torch.Tensor]`

The same as `inputs` but on the proper device.

Ensure PyTorch tensors are on the specified device.

#### postprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/base.py#L1025)

( model\_outputs: ModelOutput\*\*postprocess\_parameters: typing.Dict )

Postprocess will receive the raw outputs of the `_forward` method, generally tensors, and reformat them into something more friendly. Generally it will output a list or a dict or results (containing just strings and numbers).

Scikit / Keras interface to transformers’ pipelines. This method will forward to **call**().

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/base.py#L1004)

( input\_: typing.Any\*\*preprocess\_parameters: typing.Dict )

Preprocess will take the `input_` of a specific pipeline and return a dictionary of everything necessary for `_forward` to run properly. It should contain at least one tensor, but might have arbitrary other items.

#### save\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/base.py#L839)

( save\_directory: strsafe\_serialization: bool = False )

Parameters

-   **save\_directory** (`str`) — A path to the directory where to saved. It will be created if it doesn’t exist.
-   **safe\_serialization** (`str`) — Whether to save the model using `safetensors` or the traditional way for PyTorch or Tensorflow

Save the pipeline’s model and tokenizer.

Scikit / Keras interface to transformers’ pipelines. This method will forward to **call**().