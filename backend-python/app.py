from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from sqlalchemy.orm import Session
import os
import tempfile

from services.analizador import AnalizadorContratos
from services.chatbot import ChatbotService
from services.database_service import DatabaseService
from services.preprocessing import extract_pdf_to_txt
from config.database import get_db, init_database
from models.schemas import (
    AnalizarContratoRequest,
    ClasificarClausulaRequest,
    AnalisisResponse,
    ClasificacionResponse,
    HealthResponse,
    ChatRequest,
    ChatResponse,
    HistorialResponse,
    RecuperarAnalisisResponse,
    ActualizarResponsableRequest,
    ActualizarResponsableResponse
)
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Crear directorios necesarios
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

analizador = None
chatbot = None
db_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global analizador, chatbot, db_service
    try:
        # Inicializar base de datos
        if not init_database():
            raise Exception("No se pudo inicializar la base de datos")

        analizador = AnalizadorContratos(model_path="modelo_clausulas/")
        chatbot = ChatbotService()
        db_service = DatabaseService()
        print("✅ Servicios inicializados correctamente")
    except Exception as e:
        print(f"❌ Error al inicializar: {e}")
        raise

    yield

    print("🔄 Limpiando recursos...")
    del analizador
    del chatbot
    del db_service

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
    max_tokens_por_clausula: int = Form(512),
    db: Session = Depends(get_db)
):
    """b
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

        # Guardar automáticamente en base de datos
        try:
            analisis_id = db_service.guardar_analisis_completo(
                db=db,
                filename=pdf_file.filename,
                resultado_analisis=resultado
            )
            print(f"📄 Análisis auto-guardado con ID: {analisis_id}")
        except Exception as e:
            print(f"⚠️ No se pudo guardar automáticamente: {e}")
            # Continuar sin fallar si no se puede guardar

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

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint para interactuar con el chatbot

    POST /api/chat
    Body: {
        "mensaje": "pregunta del usuario",
        "historial": [{"role": "user", "content": "..."}],
        "contexto_contrato": {...}
    }
    """
    try:
        if not request.mensaje or request.mensaje.strip() == "":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El mensaje es requerido"
            )

        if not chatbot.disponible():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Servicio de chatbot no disponible. Verifique la configuración de OPENAI_API_KEY"
            )

        respuesta = await chatbot.chat(
            request.mensaje,
            request.historial,
            request.contexto_contrato
        )

        return {
            "success": True,
            "message": respuesta["respuesta"],  # Para compatibilidad con frontend
            "data": {
                "respuesta": respuesta["respuesta"],
                "modelo": respuesta["modelo"],
                "tokens_usados": respuesta.get("tokens_usados"),
                "tiempo_respuesta": respuesta.get("tiempo_respuesta"),
                "finish_reason": respuesta.get("finish_reason")
            },
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"❌ Error en chat_endpoint: {str(e)}")
        print(f"📝 Tipo de error: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor: {str(e)}"
        )

# ==================== ENDPOINTS DE HISTORIAL ====================

@app.post("/api/analisis/save")
async def guardar_analisis(
    filename: str,
    resultado_analisis: dict,
    db: Session = Depends(get_db)
):
    """
    Guardar análisis completo en base de datos

    Este endpoint se llama automáticamente después de completar un análisis
    """
    try:
        analisis_id = db_service.guardar_analisis_completo(
            db=db,
            filename=filename,
            resultado_analisis=resultado_analisis
        )

        return {
            "success": True,
            "message": "Análisis guardado exitosamente",
            "data": {"analisis_id": analisis_id},
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"❌ Error guardando análisis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error guardando análisis: {str(e)}"
        )

@app.get("/api/analisis/historial", response_model=HistorialResponse)
async def obtener_historial(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Obtener historial de análisis realizados

    GET /api/analisis/historial?limit=50
    """
    try:
        historial = db_service.obtener_historial_analisis(db=db, limit=limit)

        return {
            "success": True,
            "message": "Historial obtenido exitosamente",
            "data": historial,
            "total": len(historial),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"❌ Error obteniendo historial: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo historial: {str(e)}"
        )

@app.get("/api/analisis/{analisis_id}", response_model=RecuperarAnalisisResponse)
async def recuperar_analisis(
    analisis_id: int,
    db: Session = Depends(get_db)
):
    """
    Recuperar análisis específico por ID

    GET /api/analisis/123
    """
    try:
        resultado = db_service.obtener_analisis_por_id(db=db, analisis_id=analisis_id)

        if not resultado:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Análisis con ID {analisis_id} no encontrado"
            )

        return {
            "success": True,
            "message": "Análisis recuperado exitosamente",
            "data": resultado,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error recuperando análisis {analisis_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error recuperando análisis: {str(e)}"
        )

@app.put("/api/analisis/{analisis_id}/responsable", response_model=ActualizarResponsableResponse)
async def actualizar_responsable(
    analisis_id: int,
    request: ActualizarResponsableRequest,
    db: Session = Depends(get_db)
):
    """
    Actualizar responsable en matriz de riesgos

    PUT /api/analisis/123/responsable
    Body: {
        "matrix_id": "1.a",
        "responsable": "Juan Pérez"
    }
    """
    try:
        success = db_service.actualizar_responsable_matriz(
            db=db,
            analisis_id=analisis_id,
            matrix_id=request.matrix_id,
            responsable=request.responsable
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No se encontró la fila {request.matrix_id} en el análisis {analisis_id}"
            )

        return {
            "success": True,
            "message": f"Responsable actualizado para {request.matrix_id}",
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error actualizando responsable: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error actualizando responsable: {str(e)}"
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