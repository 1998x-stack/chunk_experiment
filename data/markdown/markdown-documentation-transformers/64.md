# Agents & Tools

Transformers Agents is an experimental API which is subject to change at any time. Results returned by the agents can vary as the APIs or underlying models are prone to change.

To learn more about agents and tools make sure to read the [introductory guide](../transformers_agents). This page contains the API docs for the underlying classes.

## Agents

We provide three types of agents: [HfAgent](/docs/transformers/v4.34.0/en/main_classes/agent#transformers.HfAgent) uses inference endpoints for opensource models, [LocalAgent](/docs/transformers/v4.34.0/en/main_classes/agent#transformers.LocalAgent) uses a model of your choice locally and [OpenAiAgent](/docs/transformers/v4.34.0/en/main_classes/agent#transformers.OpenAiAgent) uses OpenAI closed models.

### HfAgent

### class transformers.HfAgent

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tools/agents.py#L588)

( url\_endpoint token = None chat\_prompt\_template = None run\_prompt\_template = None additional\_tools = None )

Parameters

-   **url\_endpoint** (`str`) — The name of the url endpoint to use.
-   **token** (`str`, _optional_) — The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
-   **chat\_prompt\_template** (`str`, _optional_) — Pass along your own prompt if you want to override the default template for the `chat` method. Can be the actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named `chat_prompt_template.txt` in this repo in this case.
-   **run\_prompt\_template** (`str`, _optional_) — Pass along your own prompt if you want to override the default template for the `run` method. Can be the actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named `run_prompt_template.txt` in this repo in this case.
-   **additional\_tools** ([Tool](/docs/transformers/v4.34.0/en/main_classes/agent#transformers.Tool), list of tools or dictionary with tool values, _optional_) — Any additional tools to include on top of the default ones. If you pass along a tool with the same name as one of the default tools, that default tool will be overridden.

Agent that uses an inference endpoint to generate code.

Example:

```
from transformers import HfAgent

agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
agent.run("Is the following `text` (in Spanish) positive or negative?", text="¡Este es un API muy agradable!")
```

### LocalAgent

### class transformers.LocalAgent

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tools/agents.py#L659)

( model tokenizer chat\_prompt\_template = None run\_prompt\_template = None additional\_tools = None )

Parameters

-   **model** ([PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel)) — The model to use for the agent.
-   **tokenizer** ([PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)) — The tokenizer to use for the agent.
-   **chat\_prompt\_template** (`str`, _optional_) — Pass along your own prompt if you want to override the default template for the `chat` method. Can be the actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named `chat_prompt_template.txt` in this repo in this case.
-   **run\_prompt\_template** (`str`, _optional_) — Pass along your own prompt if you want to override the default template for the `run` method. Can be the actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named `run_prompt_template.txt` in this repo in this case.
-   **additional\_tools** ([Tool](/docs/transformers/v4.34.0/en/main_classes/agent#transformers.Tool), list of tools or dictionary with tool values, _optional_) — Any additional tools to include on top of the default ones. If you pass along a tool with the same name as one of the default tools, that default tool will be overridden.

Agent that uses a local model and tokenizer to generate code.

Example:

```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LocalAgent

checkpoint = "bigcode/starcoder"
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

agent = LocalAgent(model, tokenizer)
agent.run("Draw me a picture of rivers and lakes.")
```

#### from\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tools/agents.py#L704)

( pretrained\_model\_name\_or\_path \*\*kwargs )

Parameters

-   **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) — The name of a repo on the Hub or a local path to a folder containing both model and tokenizer.
-   **kwargs** (`Dict[str, Any]`, _optional_) — Keyword arguments passed along to [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained).

Convenience method to build a `LocalAgent` from a pretrained checkpoint.

Example:

```
import torch
from transformers import LocalAgent

agent = LocalAgent.from_pretrained("bigcode/starcoder", device_map="auto", torch_dtype=torch.bfloat16)
agent.run("Draw me a picture of rivers and lakes.")
```

### OpenAiAgent

### class transformers.OpenAiAgent

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tools/agents.py#L364)

( model = 'text-davinci-003' api\_key = None chat\_prompt\_template = None run\_prompt\_template = None additional\_tools = None )

Parameters

-   **model** (`str`, _optional_, defaults to `"text-davinci-003"`) — The name of the OpenAI model to use.
-   **api\_key** (`str`, _optional_) — The API key to use. If unset, will look for the environment variable `"OPENAI_API_KEY"`.
-   **chat\_prompt\_template** (`str`, _optional_) — Pass along your own prompt if you want to override the default template for the `chat` method. Can be the actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named `chat_prompt_template.txt` in this repo in this case.
-   **run\_prompt\_template** (`str`, _optional_) — Pass along your own prompt if you want to override the default template for the `run` method. Can be the actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named `run_prompt_template.txt` in this repo in this case.
-   **additional\_tools** ([Tool](/docs/transformers/v4.34.0/en/main_classes/agent#transformers.Tool), list of tools or dictionary with tool values, _optional_) — Any additional tools to include on top of the default ones. If you pass along a tool with the same name as one of the default tools, that default tool will be overridden.

Agent that uses the openai API to generate code.

The openAI models are used in generation mode, so even for the `chat()` API, it’s better to use models like `"text-davinci-003"` over the chat-GPT variant. Proper support for chat-GPT models will come in a next version.

Example:

```
from transformers import OpenAiAgent

agent = OpenAiAgent(model="text-davinci-003", api_key=xxx)
agent.run("Is the following `text` (in Spanish) positive or negative?", text="¡Este es un API muy agradable!")
```

### AzureOpenAiAgent

### class transformers.AzureOpenAiAgent

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tools/agents.py#L462)

( deployment\_id api\_key = None resource\_name = None api\_version = '2022-12-01' is\_chat\_model = None chat\_prompt\_template = None run\_prompt\_template = None additional\_tools = None )

Parameters

-   **deployment\_id** (`str`) — The name of the deployed Azure openAI model to use.
-   **api\_key** (`str`, _optional_) — The API key to use. If unset, will look for the environment variable `"AZURE_OPENAI_API_KEY"`.
-   **resource\_name** (`str`, _optional_) — The name of your Azure OpenAI Resource. If unset, will look for the environment variable `"AZURE_OPENAI_RESOURCE_NAME"`.
-   **api\_version** (`str`, _optional_, default to `"2022-12-01"`) — The API version to use for this agent.
-   **is\_chat\_mode** (`bool`, _optional_) — Whether you are using a completion model or a chat model (see note above, chat models won’t be as efficient). Will default to `gpt` being in the `deployment_id` or not.
-   **chat\_prompt\_template** (`str`, _optional_) — Pass along your own prompt if you want to override the default template for the `chat` method. Can be the actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named `chat_prompt_template.txt` in this repo in this case.
-   **run\_prompt\_template** (`str`, _optional_) — Pass along your own prompt if you want to override the default template for the `run` method. Can be the actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named `run_prompt_template.txt` in this repo in this case.
-   **additional\_tools** ([Tool](/docs/transformers/v4.34.0/en/main_classes/agent#transformers.Tool), list of tools or dictionary with tool values, _optional_) — Any additional tools to include on top of the default ones. If you pass along a tool with the same name as one of the default tools, that default tool will be overridden.

Agent that uses Azure OpenAI to generate code. See the [official documentation](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/) to learn how to deploy an openAI model on Azure

The openAI models are used in generation mode, so even for the `chat()` API, it’s better to use models like `"text-davinci-003"` over the chat-GPT variant. Proper support for chat-GPT models will come in a next version.

Example:

```
from transformers import AzureOpenAiAgent

agent = AzureAiAgent(deployment_id="Davinci-003", api_key=xxx, resource_name=yyy)
agent.run("Is the following `text` (in Spanish) positive or negative?", text="¡Este es un API muy agradable!")
```

### Agent

### class transformers.Agent

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tools/agents.py#L196)

( chat\_prompt\_template = None run\_prompt\_template = None additional\_tools = None )

Parameters

-   **chat\_prompt\_template** (`str`, _optional_) — Pass along your own prompt if you want to override the default template for the `chat` method. Can be the actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named `chat_prompt_template.txt` in this repo in this case.
-   **run\_prompt\_template** (`str`, _optional_) — Pass along your own prompt if you want to override the default template for the `run` method. Can be the actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named `run_prompt_template.txt` in this repo in this case.
-   **additional\_tools** ([Tool](/docs/transformers/v4.34.0/en/main_classes/agent#transformers.Tool), list of tools or dictionary with tool values, _optional_) — Any additional tools to include on top of the default ones. If you pass along a tool with the same name as one of the default tools, that default tool will be overridden.

Base class for all agents which contains the main API methods.

#### chat

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tools/agents.py#L268)

( task return\_code = False remote = False \*\*kwargs )

Parameters

-   **task** (`str`) — The task to perform
-   **return\_code** (`bool`, _optional_, defaults to `False`) — Whether to just return code and not evaluate it.
-   **remote** (`bool`, _optional_, defaults to `False`) — Whether or not to use remote tools (inference endpoints) instead of local ones.
-   **kwargs** (additional keyword arguments, _optional_) — Any keyword argument to send to the agent when evaluating the code.

Sends a new request to the agent in a chat. Will use the previous ones in its history.

Example:

```
from transformers import HfAgent

agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
agent.chat("Draw me a picture of rivers and lakes")

agent.chat("Transform the picture so that there is a rock in there")
```

#### run

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tools/agents.py#L318)

( task return\_code = False remote = False \*\*kwargs )

Parameters

-   **task** (`str`) — The task to perform
-   **return\_code** (`bool`, _optional_, defaults to `False`) — Whether to just return code and not evaluate it.
-   **remote** (`bool`, _optional_, defaults to `False`) — Whether or not to use remote tools (inference endpoints) instead of local ones.
-   **kwargs** (additional keyword arguments, _optional_) — Any keyword argument to send to the agent when evaluating the code.

Sends a request to the agent.

Example:

```
from transformers import HfAgent

agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
agent.run("Draw me a picture of rivers and lakes")
```

Clears the history of prior calls to [chat()](/docs/transformers/v4.34.0/en/main_classes/agent#transformers.Agent.chat).

## Tools

### load\_tool

#### transformers.load\_tool

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tools/base.py#L639)

( task\_or\_repo\_id model\_repo\_id = None remote = False token = None \*\*kwargs )

Parameters

-   **task\_or\_repo\_id** (`str`) — The task for which to load the tool or a repo ID of a tool on the Hub. Tasks implemented in Transformers are:
    
    -   `"document-question-answering"`
    -   `"image-captioning"`
    -   `"image-question-answering"`
    -   `"image-segmentation"`
    -   `"speech-to-text"`
    -   `"summarization"`
    -   `"text-classification"`
    -   `"text-question-answering"`
    -   `"text-to-speech"`
    -   `"translation"`
    
-   **model\_repo\_id** (`str`, _optional_) — Use this argument to use a different model than the default one for the tool you selected.
-   **remote** (`bool`, _optional_, defaults to `False`) — Whether to use your tool by downloading the model or (if it is available) with an inference endpoint.
-   **token** (`str`, _optional_) — The token to identify you on hf.co. If unset, will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
-   **kwargs** (additional keyword arguments, _optional_) — Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your tool, and the others will be passed along to its init.

Main function to quickly load a tool, be it on the Hub or in the Transformers library.

### Tool

### class transformers.Tool

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tools/base.py#L81)

( \*args \*\*kwargs )

A base class for the functions used by the agent. Subclass this and implement the `__call__` method as well as the following class attributes:

-   **description** (`str`) — A short description of what your tool does, the inputs it expects and the output(s) it will return. For instance ‘This is a tool that downloads a file from a `url`. It takes the `url` as input, and returns the text contained in the file’.
-   **name** (`str`) — A performative name that will be used for your tool in the prompt to the agent. For instance `"text-classifier"` or `"image_generator"`.
-   **inputs** (`List[str]`) — The list of modalities expected for the inputs (in the same order as in the call). Modalitiies should be `"text"`, `"image"` or `"audio"`. This is only used by `launch_gradio_demo` or to make a nice space from your tool.
-   **outputs** (`List[str]`) — The list of modalities returned but the tool (in the same order as the return of the call method). Modalitiies should be `"text"`, `"image"` or `"audio"`. This is only used by `launch_gradio_demo` or to make a nice space from your tool.

You can also override the method [setup()](/docs/transformers/v4.34.0/en/main_classes/agent#transformers.Tool.setup) if your tool as an expensive operation to perform before being usable (such as loading a model). [setup()](/docs/transformers/v4.34.0/en/main_classes/agent#transformers.Tool.setup) will be called the first time you use your tool, but not at instantiation.

Creates a [Tool](/docs/transformers/v4.34.0/en/main_classes/agent#transformers.Tool) from a gradio tool.

#### from\_hub

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tools/base.py#L176)

( repo\_id: str model\_repo\_id: typing.Optional\[str\] = None token: typing.Optional\[str\] = None remote: bool = False \*\*kwargs )

Parameters

-   **repo\_id** (`str`) — The name of the repo on the Hub where your tool is defined.
-   **model\_repo\_id** (`str`, _optional_) — If your tool uses a model and you want to use a different model than the default, you can pass a second repo ID or an endpoint url to this argument.
-   **token** (`str`, _optional_) — The token to identify you on hf.co. If unset, will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
-   **remote** (`bool`, _optional_, defaults to `False`) — Whether to use your tool by downloading the model or (if it is available) with an inference endpoint.
-   **kwargs** (additional keyword arguments, _optional_) — Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your tool, and the others will be passed along to its init.

Loads a tool defined on the Hub.

#### push\_to\_hub

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tools/base.py#L286)

( repo\_id: str commit\_message: str = 'Upload tool' private: typing.Optional\[bool\] = None token: typing.Union\[bool, str, NoneType\] = None create\_pr: bool = False )

Parameters

-   **repo\_id** (`str`) — The name of the repository you want to push your tool to. It should contain your organization name when pushing to a given organization.
-   **commit\_message** (`str`, _optional_, defaults to `"Upload tool"`) — Message to commit while pushing.
-   **private** (`bool`, _optional_) — Whether or not the repository created should be private.
-   **token** (`bool` or `str`, _optional_) — The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
-   **create\_pr** (`bool`, _optional_, defaults to `False`) — Whether or not to create a PR with the uploaded files or directly commit.

Upload the tool to the Hub.

#### save

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tools/base.py#L122)

( output\_dir )

Parameters

-   **output\_dir** (`str`) — The folder in which you want to save your tool.

Saves the relevant code files for your tool so it can be pushed to the Hub. This will copy the code of your tool in `output_dir` as well as autogenerate:

-   a config file named `tool_config.json`
-   an `app.py` file so that your tool can be converted to a space
-   a `requirements.txt` containing the names of the module used by your tool (as detected when inspecting its code)

You should only use this method to save tools that are defined in a separate module (not `__main__`).

Overwrite this method here for any operation that is expensive and needs to be executed before you start using your tool. Such as loading a big model.

### PipelineTool

### class transformers.PipelineTool

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tools/base.py#L433)

( model = None pre\_processor = None post\_processor = None device = None device\_map = None model\_kwargs = None token = None \*\*hub\_kwargs )

Parameters

-   **model** (`str` or [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel), _optional_) — The name of the checkpoint to use for the model, or the instantiated model. If unset, will default to the value of the class attribute `default_checkpoint`.
-   **pre\_processor** (`str` or `Any`, _optional_) — The name of the checkpoint to use for the pre-processor, or the instantiated pre-processor (can be a tokenizer, an image processor, a feature extractor or a processor). Will default to the value of `model` if unset.
-   **post\_processor** (`str` or `Any`, _optional_) — The name of the checkpoint to use for the post-processor, or the instantiated pre-processor (can be a tokenizer, an image processor, a feature extractor or a processor). Will default to the `pre_processor` if unset.
-   **device** (`int`, `str` or `torch.device`, _optional_) — The device on which to execute the model. Will default to any accelerator available (GPU, MPS etc…), the CPU otherwise.
-   **device\_map** (`str` or `dict`, _optional_) — If passed along, will be used to instantiate the model.
-   **model\_kwargs** (`dict`, _optional_) — Any keyword argument to send to the model instantiation.
-   **token** (`str`, _optional_) — The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
-   **hub\_kwargs** (additional keyword arguments, _optional_) — Any additional keyword argument to send to the methods that will load the data from the Hub.

A [Tool](/docs/transformers/v4.34.0/en/main_classes/agent#transformers.Tool) tailored towards Transformer models. On top of the class attributes of the base class [Tool](/docs/transformers/v4.34.0/en/main_classes/agent#transformers.Tool), you will need to specify:

-   **model\_class** (`type`) — The class to use to load the model in this tool.
-   **default\_checkpoint** (`str`) — The default checkpoint that should be used when the user doesn’t specify one.
-   **pre\_processor\_class** (`type`, _optional_, defaults to [AutoProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoProcessor)) — The class to use to load the pre-processor
-   **post\_processor\_class** (`type`, _optional_, defaults to [AutoProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoProcessor)) — The class to use to load the post-processor (when different from the pre-processor).

Uses the `post_processor` to decode the model output.

Uses the `pre_processor` to prepare the inputs for the `model`.

Sends the inputs through the `model`.

Instantiates the `pre_processor`, `model` and `post_processor` if necessary.

### RemoteTool

### class transformers.RemoteTool

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tools/base.py#L346)

( endpoint\_url = None token = None tool\_class = None )

Parameters

-   **endpoint\_url** (`str`) — The url of the endpoint to use.
-   **token** (`str`, _optional_) — The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
-   **tool\_class** (`type`, _optional_) — The corresponding `tool_class` if this is a remote version of an existing tool. Will help determine when the output should be converted to another type (like images).

A [Tool](/docs/transformers/v4.34.0/en/main_classes/agent#transformers.Tool) that will make requests to an inference endpoint.

You can override this method in your custom class of [RemoteTool](/docs/transformers/v4.34.0/en/main_classes/agent#transformers.RemoteTool) to apply some custom post-processing of the outputs of the endpoint.

Prepare the inputs received for the HTTP client sending data to the endpoint. Positional arguments will be matched with the signature of the `tool_class` if it was provided at instantation. Images will be encoded into bytes.

You can override this method in your custom class of [RemoteTool](/docs/transformers/v4.34.0/en/main_classes/agent#transformers.RemoteTool).

### launch\_gradio\_demo

#### transformers.launch\_gradio\_demo

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tools/base.py#L573)

( tool\_class: Tool )

Parameters

-   **tool\_class** (`type`) — The class of the tool for which to launch the demo.

Launches a gradio demo for a tool. The corresponding tool class needs to properly implement the class attributes `inputs` and `outputs`.

## Agent Types

Agents can handle any type of object in-between tools; tools, being completely multimodal, can accept and return text, image, audio, video, among other types. In order to increase compatibility between tools, as well as to correctly render these returns in ipython (jupyter, colab, ipython notebooks, …), we implement wrapper classes around these types.

The wrapped objects should continue behaving as initially; a text object should still behave as a string, an image object should still behave as a `PIL.Image`.

These types have three specific purposes:

-   Calling `to_raw` on the type should return the underlying object
-   Calling `to_string` on the type should return the object as a string: that can be the string in case of an `AgentText` but will be the path of the serialized version of the object in other instances
-   Displaying it in an ipython kernel should display the object correctly

### AgentText

### class transformers.tools.agent\_types.AgentText

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tools/agent_types.py#L71)

( value )

Text type returned by the agent. Behaves as a string.

### AgentImage

### class transformers.tools.agent\_types.AgentImage

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tools/agent_types.py#L83)

( value )

Image type returned by the agent. Behaves as a PIL.Image.

Returns the “raw” version of that object. In the case of an AgentImage, it is a PIL.Image.

Returns the stringified version of that object. In the case of an AgentImage, it is a path to the serialized version of the image.

### AgentAudio

### class transformers.tools.agent\_types.AgentAudio

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tools/agent_types.py#L155)

( value samplerate = 16000 )

Audio type returned by the agent.

Returns the “raw” version of that object. It is a `torch.Tensor` object.

Returns the stringified version of that object. In the case of an AgentAudio, it is a path to the serialized version of the audio.