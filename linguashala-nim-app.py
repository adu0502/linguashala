import base64
import math
import os
import re
import json
import requests
import textwrap
import time
import torch
import unicodedata
import streamlit as st
from openai import OpenAI
from pydub import AudioSegment
from gtts import gTTS
from math import ceil
from moviepy.editor import *
from mutagen.mp3 import MP3
from PIL import Image, ImageDraw, ImageFont

def generate_sd_image_using_nims(nim_api, nim_headers, prompt, output_file_name):

    payload = {
    "text_prompts": [{"text": prompt}],
    "seed": 0,
    "sampler": "K_EULER_ANCESTRAL",
    "steps": 2
    }

    response = requests.post(nim_api, headers = nim_headers, json = payload)
    response.raise_for_status()
    response_body = response.json()

    base64_image = response_body['artifacts'][0]['base64']
    # Decode the base64 image data
    image_data = base64.b64decode(base64_image)

    # Save the decoded image data to a file
    with open(output_file_name, "wb") as file:
        file.write(image_data)

    return 1


def generate_script_llama_nims(client, input_topic, input_model, input_system_prompt):

    messages=[{"role":"system","content":input_system_prompt}],

    messages.append(
            {"role": "user", "content": input_topic},
        )

    chat_completion = client.chat.completions.create(model= input_model, 
                                                     messages = messages)
    reply = chat_completion.choices[0].message.content
    return reply


def extract_string_between_markers(input_string):

    # Regex pattern to extract text between [FRAME] markers
    pattern = r'\[FRAME\] (.*?)(?=\[FRAME\]|\Z)'

    matches = re.findall(pattern, input_string, re.DOTALL)
    if matches:
        cleaned_list = [text.replace('\n', ' ').replace('**', '').strip() for text in matches]
        return cleaned_list
    else:
        return []

def generate_batched_prompts(cleaned_script, sd_prompt, batch_size=10):

    client = OpenAI(
        base_url = "https://integrate.api.nvidia.com/v1",
        api_key = os.getenv('NIM_API_KEY'))

    sd_prompt_list = []
    total_samples = len(cleaned_script)
    num_batches = ceil(total_samples / batch_size)

    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, total_samples)

        for index in range(start_idx, end_idx):
            sd_prompt_list.append(generate_script_llama_nims(client, cleaned_script[index], "meta/llama-3.1-8b-instruct", sd_prompt))
        # If this isn't the last batch, wait for 60 seconds
        if batch < num_batches - 1:
            time.sleep(60)

    return sd_prompt_list


def sarvam_translate(input_text, output_language, sarvam_api_key):

    language_key_mapping = {
      "Hindi" : "hi-IN",
      "Kannada" : "kn-IN",
      "Tamil" : "ta-IN",
      "Telugu" : "te-IN",
      "Punjabi" : "pa-IN",
      "Malayalam" : "ml-IN",
      "Odia" : "od-IN",
      "Gujarati" : "gu-IN",
      "Bengali" : "bn-IN",
      "Marathi" : "mr-IN"
  }

    url = "https://api.sarvam.ai/translate"

    payload = {
        "input": input_text,
        "source_language_code": "en-IN",
        "target_language_code": language_key_mapping[output_language],
        "speaker_gender": "Male",
        "mode": "formal",
        "model": "mayura:v1",
        "enable_preprocessing": True
    }

    headers = {"Content-Type": "application/json", 'API-Subscription-Key': sarvam_api_key}

    response = requests.request("POST", url, json=payload, headers=headers)

    return response.text


def generate_sarvam_audio(input_text, output_language, sarvam_api_key):

    url = "https://api.sarvam.ai/text-to-speech"

    language_key_mapping = {
      "Hindi" : "hi-IN",
      "Kannada" : "kn-IN",
      "Tamil" : "ta-IN",
      "Telugu" : "te-IN",
      "Punjabi" : "pa-IN",
      "Malayalam" : "ml-IN",
      "Odia" : "od-IN",
      "Gujarati" : "gu-IN",
      "Bengali" : "bn-IN",
      "Marathi" : "mr-IN"
      }

    payload = {
    "inputs": [input_text],
    "target_language_code": language_key_mapping[output_language],
    "speaker": "amol",
    "pitch": 0,
    "pace": 0.9,
    "loudness": 1.5,
    "speech_sample_rate": 8000,
    "enable_preprocessing": True,
    "model": "bulbul:v1"
    }

    headers = {"Content-Type": "application/json",
               'API-Subscription-Key': sarvam_api_key}

    response = requests.request("POST", url, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()["audios"][0]
    else:
        return ''


def generate_mp3_file(input, file_name):

    base64_audio = input
    audio_data = base64.b64decode(base64_audio)

    # Write the binary data to a .wav file
    with open("temp_audio.wav", "wb") as wav_file:
        wav_file.write(audio_data)

    # Convert the .wav file to .mp3 using pydub
    # Load the .wav file
    audio = AudioSegment.from_wav("temp_audio.wav")

    # Export the audio to .mp3 format
    audio.export(file_name, format="mp3")

    return 1


# Function to create a video clip from text
def create_scene(text, scene_number, output_language, sarvam_api_key):

    creative_path = f"creative_{scene_number}.png"
    image_path = f"scene_{scene_number}.png"
    audio_path = f"scene_{scene_number}.mp3"

    # Create text image
    # create_text_image(text, image_path)
    create_video_frame(text, creative_path, image_path, language)

    # Convert text to speech
    # text_to_speech(text, audio_path)
    wav_response = generate_sarvam_audio(text, output_language, sarvam_api_key)
    # print(wav_response)
    mp3_response = generate_mp3_file(wav_response, audio_path)
    # Load image and audio

    audio = MP3(audio_path)
    # out_duration = math.ceil(audio.info.length + 0.5)
    out_duration = math.ceil(audio.info.length + 1)

    image_clip = ImageClip(image_path).set_duration(out_duration)
    audio_clip = AudioFileClip(audio_path)

    # Combine image and audio into a video clip
    video_clip = image_clip.set_audio(audio_clip)

    return video_clip


def create_video_frame(text, image_path, output_image_path, language,
                       frame_size=(1280, 720),
                       background_color=(0, 0, 0),
                       font_color=(255, 255, 255),
                       padding=30,
                       min_font_size=10,
                       max_font_size=100,
                       ):  
    
    font_path = language + ".ttf"

    # font_path = os.path.join(current_app.root_path, 'static', 'fonts', f"{language}.ttf")
    
    # Create a new image with black background
    frame = Image.new('RGB', frame_size, color=background_color)
    draw = ImageDraw.Draw(frame)

    # Load and resize the input image
    input_image = Image.open(image_path)
    image_width = frame_size[0] // 2  # Half of the frame width
    image_height = int(image_width * (input_image.height / input_image.width))
    input_image = input_image.resize((image_width, image_height), Image.LANCZOS)

    # Calculate the position to paste the image (right side)
    image_x = frame_size[0] - image_width
    image_y = (frame_size[1] - image_height) // 2
    frame.paste(input_image, (image_x, image_y))

    # Prepare text area
    text_width = frame_size[0] // 2 - 2 * padding
    text_height = frame_size[1] - 2 * padding

    # Remove the text normalization step to preserve non-English characters
    if language == "English":
        text = unicodedata.normalize('NFKD', text.strip()).encode('ascii', 'ignore').decode('ascii')

    def get_text_size(text, font):
        return draw.multiline_textbbox((0, 0), text, font=font)[2:]

    def get_wrapped_text(text, font, max_width):
        words = text.split()
        lines = []
        current_line = []
        for word in words:
            test_line = ' '.join(current_line + [word])
            if get_text_size(test_line, font)[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
        if current_line:
            lines.append(' '.join(current_line))
        return '\n'.join(lines)

    # Binary search to find the largest font size that fits
    low, high = min_font_size, max_font_size
    best_font = None
    best_wrapped_text = None

    while low <= high:
        mid = (low + high) // 2
        try:
            font = ImageFont.truetype(font_path, mid)  # Use the provided font_path
        except OSError:
            print(f"Warning: Could not load font from {font_path}. Using default font.")
            font = ImageFont.load_default().font_variant(size=mid)

        wrapped_text = get_wrapped_text(text, font, text_width)
        text_size = get_text_size(wrapped_text, font)

        if text_size[0] <= text_width and text_size[1] <= text_height:
            best_font = font
            best_wrapped_text = wrapped_text
            low = mid + 1
        else:
            high = mid - 1

    if best_font is None:
        raise ValueError("Unable to fit text within the specified dimensions")

    # Calculate text position (left side, vertically centered)
    text_bbox = draw.multiline_textbbox((0, 0), best_wrapped_text, font=best_font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    text_x = padding
    text_y = (frame_size[1] - text_height) // 2

    # Draw text
    draw.multiline_text((text_x, text_y), best_wrapped_text, font=best_font, fill=font_color, align="left")

    # Save the frame
    frame.save(output_image_path)
    return 1

def generate_script(topic):
    input_script_prompt = """
    Write a video script for an online course on [TOPIC]. Divide the script into clearly marked frames using [FRAME] at the start of each new section. Generate the script from the narrator's perspective, speaking directly to the viewer. Follow this structure:

    [FRAME] Welcome viewers and introduce the topic
    [FRAME] Explain the topic's importance and relevance
    [FRAME] Outline what viewers will learn

    [FRAME] Break down key concepts or steps
    [FRAME] Use analogies or examples to clarify ideas
    [FRAME] Provide practical tips and actionable advice
    [FRAME] Ensure smooth transitions between sections

    [FRAME] Summarize key takeaways
    [FRAME] Encourage reflection and application
    [FRAME] Conclude with next steps or preview upcoming content

    Use simple, accessible language and a friendly, conversational tone. Keep sentences short and explanations clear for easy understanding and translation. Avoid jargon unless necessary. Address the viewer directly, using "you" and "we" to create engagement.
    """

    client = OpenAI(
        base_url = "https://integrate.api.nvidia.com/v1",
        api_key = os.getenv('NIM_API_KEY'))

    script = generate_script_llama_nims(client, topic, "meta/llama-3.1-8b-instruct", input_script_prompt)
    cleaned_script = extract_string_between_markers(script)
    return cleaned_script


def generate_image_prompts(script):
    
    sd_prompt = """
    Review the following text frame for a video script.
    Based on the text for each frame, determine if an image would enhance the viewer's engagement.
    If an image is needed, create a creative, descriptive prompt (120 words max) for generating the image using Google's Imagegen model.
    If no image is necessary, return an empty string.
    No extra text or explanations are needed.
    """
    sd_prompt_list = generate_batched_prompts(script, sd_prompt)
    return sd_prompt_list


def create_output_images(nim_api, nim_headers, prompts):

    creative_img_list = []
    for index, prompt in enumerate(prompts):
        out_file_name = "creative_" + str(index+1) + ".png"

        file_loc = generate_sd_image_using_nims(nim_api, 
                                                nim_headers, 
                                                prompt, 
                                                output_file_name)

        creative_img_list.append(str(file_loc))
    
    return creative_img_list


def translate_script(script, language, sarvam_api_key):
    translated_script = []
    for frame in script:
        sarvam_response = sarvam_translate(frame, language, sarvam_api_key)
        translated_script.append(sarvam_response)

    translated_text = [json.loads(item)['translated_text'] for item in translated_script]
    return translated_text


def generate_video_frames(images, input_scipt, sarvam_api_key, language):
    # Create video clips for each scene
    
    video_clips = []
    for i, line in enumerate(input_scipt):
        scene_clip = create_scene(line, i+1, language, sarvam_api_key)
        video_clips.append(scene_clip)
    return video_clips


def create_final_video(script, video_frames):

    # Concatenate all video clips
    final_video = concatenate_videoclips(video_frames)

    # Save the final video
    final_video.write_videofile("final_video.mp4", fps=24)

    # Cleanup: Remove temporary files
    for i in range(len(script)):
        os.remove(f"creative_{i+1}.png")
        os.remove(f"scene_{i+1}.png")
        os.remove(f"scene_{i+1}.mp3")
    return "final_video.mp4"


st.set_page_config(page_title="Linguashala", layout="wide")
st.title("Linguashala - Ek Vichaar Anant Vistaar 🇮🇳")
st.subheader("Effortless Multilingual Course Creation for All 🗣️ 🎓 📖 🧑‍🏫 🌐 📚")

nim_api_key = os.getenv('NIM_API_KEY')
nim_sd_api_url = os.getenv('NIM_SD_URL')

nim_headers = {
    "Authorization": "Bearer " + nim_api_key,
    "Accept": "application/json",
}


sarvam_api = os.getenv('SARVAM_API_TOKEN')
sarvam_api_key = os.getenv('SARVAM_API_TOKEN')
topic = st.text_input("Enter the topic for your video:")

language = st.selectbox("Select the language for video creation:", 
                        ["English", "Hindi", "Bengali", 
                        "Tamil", "Telugu", "Marathi",
                        "Kannada", "Punjabi", "Malayalam", 
                        "Odia", "Gujarati", ])

if st.button("Create Video"):
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Function 1: Generate script
    status_placeholder.text("Generating script...")
    progress_placeholder.progress(0)
    cleaned_script = generate_script(topic)
    progress_placeholder.progress(20)
    status_placeholder.text("Script generation completed.")
    time.sleep(0.5)
    
    # Function 2: Generate image prompts
    status_placeholder.text("Generating image prompts...")
    imagegen_prompts = generate_image_prompts(cleaned_script)
    progress_placeholder.progress(40)
    status_placeholder.text("Image prompt generation completed.")


    # Function 3: Create stable diffusion images
    status_placeholder.text("Creating images...")
    creative_image_list = create_output_images(nim_sd_api_url, nim_headers, imagegen_prompts)
    progress_placeholder.progress(60)
    status_placeholder.text("Image creation completed.")
    
    if language != "English":
    # Function 4: Translate script
        status_placeholder.text("Translating script ...")
        translated_script = translate_script(cleaned_script, language, sarvam_api_key)
        progress_placeholder.progress(80)
        status_placeholder.text("Script translation completed.")
    
        # Function 5: Generate video frames
        status_placeholder.text("Generating video frames...")
        video_frames = generate_video_frames(creative_image_list, translated_script, sarvam_api_key, language)
        progress_placeholder.progress(90)
        status_placeholder.text("Video frame generation completed.")

        # Function 6: Create final video
        status_placeholder.text("Creating final video...")
        final_video = create_final_video(translated_script, video_frames)
        progress_placeholder.progress(100)
        status_placeholder.text("Video creation process completed!")
        st.success("Video created successfully!")
    
    else:
        status_placeholder.text("Generating video frames...")
        video_frames = generate_video_frames(creative_image_list, cleaned_script, sarvam_api_key, language)
        progress_placeholder.progress(90)
        status_placeholder.text("Video frame generation completed.")

        status_placeholder.text("Creating final video...")
        final_video = create_final_video(cleaned_script, video_frames)
        progress_placeholder.progress(100)
        status_placeholder.text("Video creation process completed!")
        st.success("Video created successfully!")
    
    # Display the video
    st.video(final_video)

st.sidebar.header("About Linguashala")
st.sidebar.write("""
Linguashala is an innovative platform for creating multilingual educational content.
Our AI-powered system generates courses in various languages, making knowledge 
accessible to a diverse audience.
""")