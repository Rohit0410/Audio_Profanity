# !pip install openai-whisper
# !pip install whisper
# !pip install googletrans
# !pip install torch transformers librosa googletrans==4.0.0-rc1 translate nltk

import whisper
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
from googletrans import Translator
import nltk
from nltk.corpus import stopwords
import re
from flask import jsonify,request,Flask
import os
from pydub import AudioSegment
from text import process_user_input
import os
# AudioSegment.converter = r"/usr/src/app/ffmpeg/bin/ffmpeg.exe"
# AudioSegment.ffmpeg = r"/usr/src/app/ffmpeg/bin/ffmpeg.exe"
# AudioSegment.ffprobe = r"/usr/src/app/ffmpeg/bin/ffprobe.exe"
app = Flask(__name__)


# Download the stopwords from nltk
nltk.download('stopwords')

dst = "/usr/src/app/audio_chunks/sample.wav"
src="/usr/src/app/audio_chunks/sample.mp3"

model_name = "parsawar/profanity_model_3.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load the Hindi Wav2Vec2 model and processor
model_name1 = "Harveenchadha/hindi_base_wav2vec2"
processor1 = Wav2Vec2Processor.from_pretrained(model_name1)
model1 = Wav2Vec2ForCTC.from_pretrained(model_name)

model2 = whisper.load_model("base")

def convert_to_wav(audio_file):
    # Determine the format of the input file
    file_extension = audio_file.split('.')[-1].lower()
    if file_extension in ['mp3', 'm4a',"flac"]:
        audio = AudioSegment.from_file(audio_file)
        wav_file = audio_file.replace(file_extension, 'wav')
        audio.export(wav_file, format='wav')
        return wav_file
    elif file_extension == 'wav':
        return audio_file
    else:
        raise ValueError("Unsupported file format")

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = re.findall(r'\w+', text)
    filtered_text = ' '.join([word for word in words if word.lower() not in stop_words])
    return filtered_text

def translate_to_english(text):
    translator = Translator()
    print('text',text)
    english_translation = translator.translate(text, src='hi', dest='en').text
    return english_translation

def speech_recognition(audio_file):
    wav_file = convert_to_wav(audio_file)
    # Load the Whisper model
    result = model2.transcribe(wav_file)
    language = result['language']
    print(f"Detected language: {language}")

    if language == 'en':
        transcription = result['text']
    elif language == "hi":
        # Load the audio file
        speech, original_sample_rate = librosa.load(wav_file)

        # Resample the audio to 16000 Hz if necessary
        if original_sample_rate != 16000:
            speech = librosa.resample(speech, orig_sr=original_sample_rate, target_sr=16000)

        # Preprocess the audio signal
        inputs = processor1(speech, sampling_rate=16000, return_tensors="pt", padding=True)

        # Perform speech recognition
        with torch.no_grad():
            logits = model1(inputs.input_values).logits

        # Get the recognized text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor1.batch_decode(predicted_ids)[0]

        print('trans',transcription)

        # Translate Hindi transcription to English
        transcription = translate_to_english(transcription)
    else:
        raise ValueError("Unsupported language")

    print(f"Transcription: {transcription}")
    return transcription

# def profanity_check(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)

#     logits = outputs.logits
#     predicted_class = torch.argmax(logits, dim=1).item()

#     if predicted_class == 0:
#         return True  # Profanity detected
#     else:
#         return False  # No profanity detected
    
# def audio_format_convert(audio):
    

def profanity_classification_pipeline(audio_file):
    transcription = speech_recognition(audio_file)
    transcription_no_stopwords = remove_stopwords(transcription)
    # is_profane = process_user_input(transcription)
    similarity_score, assigned_label_similarity, hf_label_score, assigned_based_on_similarity = process_user_input(transcription)
    if assigned_based_on_similarity:
        response = {
           'label': str(assigned_label_similarity),
            'score': str(similarity_score),
            'Model': 'vector similarity'
        }
    else:
        response = {
            'label': str(assigned_label_similarity),  # Assuming assigned_label_similarity is used for Hugging Face model
            'score': str(hf_label_score),
            'Model': 'hugging face model'
        } 
    print(f"Transcription without stopwords: {transcription_no_stopwords}")
    print(f"Profanity detected: {response}")
    return response

Folder_path ="/usr/src/app/video_temp"
@app.route('/audio_profanity', methods=['POST'])
def detect_profanity():
    if 'Audio' not in request.files:
        return jsonify({'error': 'No Audio file provided'}), 400

    audio_file = request.files['Audio']
    audio_path = os.path.join(Folder_path, audio_file.filename)
    audio_file.save(audio_path)

    prediction = profanity_classification_pipeline(audio_path)
    print({'Profanity':prediction})
    # final = {'Profanity':prediction}
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True,port=5001)

# # Example usage
# audio_file = '/content/abuse-hindi-made-with-Voicemod.wav'
# profanity_classification_pipeline(audio_file)

# # Example usage
# audio_file = '/content/trainingaudio-aadiksha-007-aadiksha-5_Bz0Rj98D.wav'
# profanity_classification_pipeline(audio_file)

# # Example usage
# audio_file = '/content/trainingaudio-aliabhatt-032-aliabhatt-5_QQw2j4Ak.wav'
# profanity_classification_pipeline(audio_file)