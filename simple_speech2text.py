# import requests

# # URL of the audio file to be downloaded 
# url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-GPXX04C6EN/Testing%20speech%20to%20text.mp3"

# # Send a GET request to the URL to download the file
# response = requests.get(url)

# # Define the local file path where the audio file will be saved
# audio_file_path = "download_audio.mp3"

# if response.status_code == 200:
#     with open(audio_file_path, "wb") as file:
#         file.write(response.content)
#     print("File downloaded successfully")
# else:
#     print("Failed to download the file")

import warnings
import torch
from transformers import pipeline
from transformers.utils import logging as hf_logging
import warnings

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# Initialize the speech-to-text pipeline from Hugging Face Transformers
# This uses the "openai/whisper-tiny.en" model for automatic speech recognition (ASR)
# The `chunk_length_s` parameter specifies the chunk length in seconds for processing

pipe = pipeline(
    "automatic-speech-recognition",
    model = "openai/whisper-tiny.en",
    chunk_length_s = 30
)

audio_file = 'download_audio.mp3'

# Perform speech recognition on the audio file
# The `batch_size=8` parameter indicates how many chunks are processed at a time
# The result is stored in `prediction` with the key "text" containing the transcribed text

result = pipe(audio_file, batch_size = 8)["text"]
# Print the transcribed text to the console

print(result)