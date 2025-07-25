import os
import uuid
import sys
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Añadir path base y carpeta iaModels
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "iaModels"))

from transcribir import procesar_audio_y_analizar

# Inicializar la app
app = FastAPI()

# Configurar CORS para desarrollo y producción
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # desarrollo local
        "https://solvo-audio-ai.vercel.app",  # producción Vercel
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribir-audio/")
async def transcribir_audio_endpoint(file: UploadFile = File(...)):
    try:
        os.makedirs("uploads", exist_ok=True)

        # Guardar archivo con nombre único
        ext = file.filename.split('.')[-1]
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join("uploads", filename)

        with open(filepath, "wb") as f:
            f.write(await file.read())

        # Procesar: transcribir + analizar con el agente
        resultado = procesar_audio_y_analizar(filepath)

        # Limpiar archivo temporal
        os.remove(filepath)

        return {
            "transcripcion": resultado.get("transcripcion", ""),
            "analisis": resultado.get("analisis", ""),
            "error": resultado.get("error", "")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en servidor: {str(e)}")


# Para despliegue local
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
