# Business AI Meeting Companion

A small Python project that turns meeting audio into concise key points. It uses OpenAI's `whisper-tiny.en` model through Hugging Face Transformers for speech-to-text, then sends the transcript to an IBM watsonx.ai Llama model for summarisation. The main experience is a Gradio web app where you upload an audio file and receive the meeting highlights.

## What is included

| File | Purpose |
| --- | --- |
| `speech_analyzer.py` | Full Gradio application: transcription followed by key-point generation. |
| `hello.py` | Gradio transcription-only application. |
| `simple_speech2text.py` | Command-line example that transcribes `download_audio.mp3`. |
| `simple_llm.py` | Command-line example that calls the IBM watsonx.ai LLM. |
| `download_audio.mp3` | Sample audio used by the standalone transcription example. |

## Prerequisites

- Python 3.9 or later
- [PyTorch](https://pytorch.org/) appropriate for your operating system and hardware
- An IBM watsonx.ai project and credentials with access to the configured model (required for `speech_analyzer.py` and `simple_llm.py`)

The first transcription run downloads the Whisper model, so an internet connection is needed and may take a moment.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the Python packages:

```bash
pip install torch transformers gradio langchain-core langchain-ibm ibm-watsonx-ai
```

Install PyTorch using the command recommended by the [official PyTorch selector](https://pytorch.org/get-started/locally/) if you need GPU support or a platform-specific build.

## Configure IBM watsonx.ai

Before running the LLM-powered scripts, update the `WatsonxLLM` configuration in `speech_analyzer.py` and `simple_llm.py` with values for your account:

```python
llm_model = WatsonxLLM(
    model_id="your-supported-model-id",
    url="your-watsonx-region-url",
    params=params,
    project_id="your-project-id",
)
```

The repository currently uses the US South endpoint, the `skills-network` project, and the `meta-llama/llama-4-maverick-17b-128e-instruct-fp8` model. Make sure those are available to your IBM watsonx.ai account. Authenticate `ibm-watsonx-ai` according to your environment—for example, with an IBM Cloud API key supported by the SDK.

## Run the apps

Start the full meeting companion:

```bash
python speech_analyzer.py
```

Open `http://localhost:7860` in a browser, upload a meeting-audio file, and wait for the generated key points.

For transcription only, without an IBM watsonx.ai call:

```bash
python hello.py
```

To transcribe the included sample audio in the terminal:

```bash
python simple_speech2text.py
```

To test only the LLM connection:

```bash
python simple_llm.py
```

## How it works

1. Gradio receives an uploaded audio file as a file path.
2. The Transformers automatic-speech-recognition pipeline uses `openai/whisper-tiny.en` to create a transcript in 30-second chunks.
3. The transcript is inserted into a prompt requesting detailed key points.
4. IBM watsonx.ai generates the final response with a low temperature (`0.1`) for more consistent output.

## Notes

- `whisper-tiny.en` is lightweight and intended for English audio. For better accuracy, change the model to a larger Whisper variant, considering the additional compute and download size.
- The ASR pipeline is constructed on every upload in the current implementation. Loading it once at startup can make repeated uploads faster.
- Never commit API keys or other credentials to the repository.
