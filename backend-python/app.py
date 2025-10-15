from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import os
import tempfile

from services.analizador import AnalizadorContratos
from services.chatbot import ChatbotService
from services.preprocessing import extract_pdf_to_txt
from models.schemas import (
    AnalizarContratoRequest,
    ClasificarClausulaRequest,
    AnalisisResponse,
    ClasificacionResponse,
    HealthResponse,
    ChatRequest
)
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Crear directorios necesarios
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

analizador = None
chatbot = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global analizador, chatbot
    try:
        analizador = AnalizadorContratos(model_path="modelo_clausulas/")
        chatbot = ChatbotService()
        print("✅ Analizador inicializado correctamente")
    except Exception as e:
        print(f"❌ Error al inicializar: {e}")
        raise

    yield

    print("🔄 Limpiando recursos...")
    del analizador
    del chatbot

# Inicializar FastAPI
app = FastAPI(
    title="API de Análisis de Contratos",
    description="Backend unificado para análisis de contratos con ML",
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== ENDPOINTS ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verificar estado del servicio"""
    return {
        "success": True,
        "message": "Servicio de análisis de contratos disponible",
        "data": {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "modelo_cargado": analizador is not None
        }
    }

@app.post("/api/analizar-contrato", response_model=AnalisisResponse)
async def analizar_contrato(request: AnalizarContratoRequest):
    """
    Analizar contrato desde texto directo

    POST /api/analizar-contrato
    Body: {
        "texto_contrato": "texto del contrato...",
        "max_tokens_por_clausula": 512
    }
    """
    try:
        if not request.texto_contrato or request.texto_contrato.strip() == "":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El texto del contrato es requerido"
            )

        print(f"Analizando contrato de {len(request.texto_contrato)} caracteres")

        resultado = await analizador.analizar_contrato_completo(
            request.texto_contrato,
            request.max_tokens_por_clausula
        )

        if 'error' in resultado:
            return {
                'error': resultado['error'],
                'success': False
            }

        return {
            "success": True,
            "message": "Contrato analizado exitosamente",
            "data": resultado,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"Error en analizar_contrato: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/api/analizar-contrato-pdf")
async def analizar_contrato_pdf(
    pdf_file: UploadFile = File(...),
    max_tokens_por_clausula: int = Form(512)
):
    """
    Analizar contrato desde archivo PDF

    POST /api/analizar-contrato-pdf
    Form-data:
        - pdf_file: archivo PDF
        - max_tokens_por_clausula: número (opcional, default 512)
    """
    try:
        # Validar tipo de archivo
        allowed_types = ["application/pdf", "text/plain"]
        if pdf_file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Solo se permiten archivos PDF y TXT"
            )

        print(f"Procesando archivo: {pdf_file.filename}")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(await pdf_file.read())
            temp_path = temp_file.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8') as temp_txt:
            txt_path = temp_txt.name

        # Usar tu función existente pdf_to_txt
        extract_pdf_to_txt(temp_path, txt_path)
        with open(txt_path, 'r', encoding='utf-8') as f:
            texto_contrato = f.read()

        print(f"Texto extraído: {len(texto_contrato)} caracteres")

        # Validar tamaño (512MB máximo)
        max_size = 512 * 1024 * 1024
        if len(texto_contrato) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="El archivo excede el tamaño máximo de 512MB"
            )

        if not texto_contrato or not texto_contrato.strip():
            return {
                'error': 'No se pudo extraer texto del PDF. Verifique que no esté protegido o corrupto.',
                'success': False
            }

        # Medir tiempo de análisis
        start_time = datetime.now()
        # Analizar el contrato usando tu analizador existente
        resultado = analizador.analizar_contrato_completo(
            texto_contrato,
            max_tokens_por_clausula
        )
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"Tiempo de análisis: {duration:.3f} segundos")

        if 'error' in resultado:
            return {
                'error': resultado['error'],
                'success': False
            }

        return {
            "success": True,
            "message": "Archivo procesado exitosamente",
            "data": resultado,
            "filename": pdf_file.filename,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error en analizar_contrato_pdf: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    finally:
        # Limpiar archivo temporal
        for temp_file in [temp_path, txt_path]:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

@app.post("/api/clasificar-clausula", response_model=ClasificacionResponse)
async def clasificar_clausula(request: ClasificarClausulaRequest):
    """
    Clasificar una cláusula individual

    POST /api/clasificar-clausula
    Body: {
        "texto_clausula": "texto de la cláusula..."
    }
    """
    try:
        if not request.texto_clausula or request.texto_clausula.strip() == "":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El texto de la cláusula es requerido"
            )

        resultado = await analizador.clasificar_clausula(request.texto_clausula)

        return {
            "success": True,
            "message": "Cláusula clasificada exitosamente",
            "data": resultado,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"Error en clasificar_clausula: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/historial")
async def obtener_historial():
    """
    Obtener historial de análisis
    (Por implementar con base de datos)
    """
    return {
        "success": True,
        "message": "Historial obtenido exitosamente",
        "data": {
            "total": 0,
            "historial": []
        }
    }

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint para interactuar con el chatbot

    POST /api/chat
    Form-data:
        - message: mensaje del usuario
    """
    try:
        if not request.mensaje or request.mensaje.strip() == "":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El mensaje es requerido"
            )

        respuesta = await chatbot.chat(
            request.mensaje,
            request.historial,
            request.contexto_contrato
        )

        return {
            "success": True,
            "message": "Respuesta generada exitosamente",
            "data": {
                "respuesta": respuesta.respuesta,
                "modelo": respuesta.modelo,
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"Error en chat_endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Manejo de errores global
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    print(f"Error no manejado: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Error interno del servidor"
        }
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)