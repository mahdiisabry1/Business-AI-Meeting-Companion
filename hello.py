import torch
from transformers import pipeline
import gradio as gr 
from transformers.utils import logging as hf_logging
import warnings

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def transcript_audio(audio_file):
    pipe = pipeline(
        "automatic-speech-recognition",
        model = "openai/whisper-tiny.en",
        chunk_length_s = 30
    )

    result = pipe(audio_file, batch_size=8)["text"]
    return result

# Setting up gradio interface
audio_input = gr.Audio(sources = "upload", type = "filepath") # Audio input 
output_text = gr.Textbox() # Text output


# Create the Gradio interface with the function, inputs, and outputs
iface = gr.Interface(fn = transcript_audio, inputs = audio_input, outputs = output_text, title = "Audio Transcription App", description = "Upload the audio file")

# Launch the Gradio app
iface.launch(server_name="0.0.0.0", server_port = 7860)


# def greet(name):
#     return "Hello " + name + "!"

# demo = gr.Interface(fn = greet, inputs = "text", outputs = "text")

# demo.launch(server_name = "0.0.0.0", server_port = 7860)