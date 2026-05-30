import gradio as gr
import librosa
import numpy as np
import joblib
import warnings
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

warnings.filterwarnings('ignore')

print("Завантаження моделей класифікації...")
model = joblib.load('audio_classifier_model.pkl')
scaler = joblib.load('audio_scaler.pkl')
classes = joblib.load('classes.pkl')

model_name = 'google/flan-t5-small'
print(f"Завантаження LLM ({model_name})...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_llm = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    llm_loaded = True
    print("LLM успішно завантажено!")
except Exception as e:
    print(f'Не вдалося завантажити LLM: {e}')
    llm_loaded = False

def preprocess_and_extract(file_path, target_sr=22050, duration=5.0, top_db=20):
    """Обробляє аудіо та витягує рівно 30 ознак MFCC."""
    try:
        y, sr = librosa.load(file_path, sr=target_sr)
        y, _ = librosa.effects.trim(y, top_db=top_db)
        
        target_length = int(target_sr * duration)
        if len(y) > target_length:
            y = y[:target_length]
        else:
            y = np.pad(y, (0, max(0, target_length - len(y))))
        
        # ВАЖЛИВО n_mfcc=30 
        mfcc = librosa.feature.mfcc(y=y, sr=target_sr, n_mfcc=30)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Помилка обробки аудіо: {e}")
        return None

def explain_prediction(label, probability):
    """Генерує пояснення через LLM із захистом від галюцинацій."""
    if not llm_loaded:
        return "LLM не завантажена. Пояснення недоступне."
    
    prompt = f"In one short sentence, explain the military danger of a {label}."
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model_llm.generate(
                **inputs, 
                max_new_tokens=40,
                repetition_penalty=2.5, # Жорстко забороняємо повторювати "ai ai ai"
                do_sample=True,         # Додаємо трохи гнучкості
                temperature=0.7         # Оптимальна температура для логіки
            )
        llm_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        final_text = (
            f" Виявлено загрозу: {label}\n"
            f" Аналіз LLM (Eng): {llm_response}"
        )
        return final_text
        
    except Exception as e:
        return f"Помилка генерації тексту: {e}"
def classify_audio(audio_path):
    """Головна функція для Gradio інтерфейсу."""
    if audio_path is None:
        return {"Помилка": 1.0}, "Будь ласка, завантажте аудіо."
    
    features = preprocess_and_extract(audio_path)
    if features is None:
        return {"Помилка": 1.0}, "Не вдалося витягти ознаки з аудіо."
    
    features_reshaped = features.reshape(1, -1)
    
    probabilities = model.predict_proba(features_reshaped)[0]
    
    result = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
    
    top_index = int(np.argmax(probabilities))
    explanation = explain_prediction(classes[top_index], probabilities[top_index])
    
    return result, explanation

interface = gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio(type="filepath", label="Завантажте військовий звук (.wav, .mp3)"),
    outputs=[
        gr.Label(num_top_classes=3, label="Результат класифікації"),
        gr.Textbox(label="Пояснення LLM", lines=3)
    ],
    title="Військовий аудіо-класифікатор",
    description="Завантажте аудіофайл, щоб модель визначила тип військової техніки (дрон, артилерія тощо)."
)

if __name__ == "__main__":
    interface.launch()