import os
import json
from datetime import datetime
from pydub import AudioSegment
import speech_recognition as sr
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# ==== CONFIGURACIÓN Y RUTAS ====
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

BASE_DIR = os.path.dirname(__file__)
TRANSCRIPCIONES_PATH = os.path.join(BASE_DIR, "transcripciones_temp.json")
ANALISIS_CACHE_PATH = os.path.join(BASE_DIR, "ultimo_analisis_ingles.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==== CONVERTIR A WAV SI ES NECESARIO ====
def convert_to_wav(input_path: str, output_path: str) -> None:
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="wav")

# ==== TRANSCRIBIR AUDIO ====
def transcribir_audio(audio_path: str) -> str:
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)

# ==== GUARDAR TRANSCRIPCIÓN ====
def guardar_transcripcion(transcripcion: str) -> None:
    data = []
    if os.path.exists(TRANSCRIPCIONES_PATH):
        with open(TRANSCRIPCIONES_PATH, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                pass

    data.append({
        "fecha": datetime.now().isoformat(),
        "transcripcion": transcripcion
    })

    with open(TRANSCRIPCIONES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ==== GUARDAR ANÁLISIS ====
def guardar_analisis_en_json(analisis: str) -> None:
    with open(ANALISIS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump({"analisis": analisis}, f, ensure_ascii=False, indent=2)

# ==== INSTRUCCIONES DEL PROMPT ====
def generar_instruccion_ingles() -> str:
    return (
        "Eres un experto en enseñanza del idioma inglés con enfoque en análisis fonético y comprensión oral.\n\n"
        "Te presento una transcripción de un audio que fue grabado por un estudiante mientras hablaba en inglés. "
        "Tu trabajo es evaluar el nivel aproximado del hablante, detectar errores comunes en la pronunciación o estructuras gramaticales, "
        "y ofrecer sugerencias detalladas para mejorar.\n\n"
        "Estructura tu análisis en:\n"
        "- Evaluación General del Nivel (A1, A2, B1, B2, C1, C2)\n"
        "- Errores detectados (con ejemplos y explicaciones)\n"
        "- Sugerencias personalizadas para mejorar el estudio del inglés\n"
        "- Recomendaciones de recursos o prácticas para reforzar esas áreas\n\n"
        "La respuesta debe ser clara, motivadora y útil, como si fueras un coach personalizado de aprendizaje de inglés.\n"
    )

# ==== PROMPT TEMPLATE ====
prompt_template = PromptTemplate(
    input_variables=["instrucciones", "contenido_transcripcion"],
    template="{instrucciones}\n\nEsta es la transcripción que dijo el estudiante:\n\n{contenido_transcripcion}\n\nRealiza el análisis solicitado:"
)

# ==== ANALIZAR TRANSCRIPCIÓN ====
def analizar_transcripcion_ingles() -> str:
    if not api_key:
        return "❌ GROQ_API_KEY no está configurada en el entorno."

    with open(TRANSCRIPCIONES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not data:
            return "❌ No se encontró ninguna transcripción."

    ultima = data[-1]["transcripcion"]
    instrucciones = generar_instruccion_ingles()

    prompt = prompt_template.format(
        instrucciones=instrucciones,
        contenido_transcripcion=ultima
    )

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.5,
        max_tokens=3000
    )

    respuesta = llm.invoke(prompt)
    contenido = respuesta.content if hasattr(respuesta, "content") else str(respuesta)

    guardar_analisis_en_json(contenido)
    return contenido

# ==== FLUJO PRINCIPAL ====
def procesar_audio_y_analizar(audio_subido_path: str):
    try:
        # 1. Convertir a WAV
        wav_path = os.path.join(UPLOAD_FOLDER, "temp.wav")
        convert_to_wav(audio_subido_path, wav_path)
        print("✅ Audio convertido a WAV")

        # 2. Transcribir
        texto = transcribir_audio(wav_path)
        print("✅ Transcripción completa")
        guardar_transcripcion(texto)

        # 3. Analizar
        resultado = analizar_transcripcion_ingles()
        print("✅ Análisis completo")

        return {
            "transcripcion": texto,
            "analisis": resultado
        }

    except Exception as e:
        return {"error": f"❌ Error general: {str(e)}"}

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("❌ Debes proporcionar el nombre del archivo de audio. Ejemplo: python transcribir.py Mauri.mp3")
        sys.exit(1)

    test_audio = sys.argv[1]

    if not os.path.exists(test_audio):
        print(f"❌ El archivo '{test_audio}' no existe.")
        sys.exit(1)

    resultado = procesar_audio_y_analizar(test_audio)
    print("\n=== TRANSCRIPCIÓN ===\n")
    print(resultado.get("transcripcion", ""))
    print("\n=== ANÁLISIS ===\n")
    print(resultado.get("analisis", ""))

