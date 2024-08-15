import streamlit as st
import os
import yt_dlp
import whisper
import subprocess
import re
from transformers import BartTokenizer, BartForConditionalGeneration
from nltk.tokenize import sent_tokenize

# Load the BART model and tokenizer
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def download_audio(video_url, output_file):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
            'preferredquality': '192',
        }],
        'outtmpl': output_file + '.%(ext)s',
        'noplaylist': True,
        'ffmpeg_location': "C:/ffmpeg-n7.0-latest-win64-lgpl-7.0/bin"  # Replace with the actual path to ffmpeg if necessary
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

def extract_audio_from_local_video(video_file, output_file):
    command = [
        'ffmpeg',
        '-i', video_file,
        '-vn',  # No video
        '-acodec', 'aac',
        output_file
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        raise

def transcribe_audio(audio_file):
    model = whisper.load_model("medium").to("cuda")  # Load the model to GPU
    result = model.transcribe(audio_file, task="translate")
    return result['text']

def chunk_text(text, tokenizer, max_tokens=500):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_length = 0

    for sentence in sentences:
        tokens = tokenizer.encode(sentence, add_special_tokens=False)
        if current_length + len(tokens) < max_tokens:
            current_chunk += " " + sentence
            current_length += len(tokens)
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_length = len(tokens)
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def summarize_chunk(chunk, tokenizer):
    inputs = tokenizer.encode(chunk, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=250, min_length=60, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def extract_key_points(text):
    sentences = sent_tokenize(text)
    key_points = []
    
    for sentence in sentences:
        if len(sentence.split()) > 10 and ('.' in sentence or ':' in sentence):
            key_points.append(sentence)
    
    return key_points

def summarize_text(text, tokenizer):
    chunks = chunk_text(text, tokenizer)
    summarized_chunks = [summarize_chunk(chunk, tokenizer) for chunk in chunks]
    
    combined_summary = "\n\n".join(summarized_chunks)
    key_points = extract_key_points(combined_summary)
    
    return {
        'combined_summary': combined_summary,
        'key_points': key_points
    }

def main(input_value):
    audio_file = 'temp_audio.m4a'

    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  
        r'localhost|'  
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  
        r'(?::\d+)?'  
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    if re.match(url_pattern, input_value):
        download_audio(input_value, 'temp_audio')  
    elif os.path.isfile(input_value):
        extract_audio_from_local_video(input_value, audio_file)
    else:
        raise ValueError("The input must be a valid URL or an existing local file path.")

    transcribed_text = transcribe_audio(audio_file)
    os.remove(audio_file)  

    results = summarize_text(transcribed_text, tokenizer)
    
    return results

# Streamlit UI
st.title("Cilp Master : Automated Video Highlights using Artificial Intelligence")

input_value = st.text_input("Please provide a URL or a local file path:")
if st.button("Transcribe and Summarize"):
    if input_value:
        try:
            results = main(input_value)
            st.write("Combined Summary:")
            st.write(results['combined_summary'])
            st.write("Key Points:")
            for point in results['key_points']:
                st.write(f"- {point}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please provide a valid input.")