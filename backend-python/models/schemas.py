from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# ==================== REQUEST MODELS ====================

class MensajeHistorial(BaseModel):
    """Mensaje individual del historial"""
    role: str = Field(..., description="Rol del mensaje: 'user' o 'assistant'")
    content: str = Field(..., description="Contenido del mensaje")

class ChatRequest(BaseModel):
    """Request para el chatbot"""
    mensaje: str = Field(..., description="Mensaje del usuario")
    historial: Optional[List[MensajeHistorial]] = Field(default=None, description="Historial de conversación")
    contexto_contrato: Optional[Dict[str, Any]] = Field(default=None, description="Contexto del contrato analizado")

    class Config:
        json_schema_extra = {
            "example": {
                "mensaje": "¿Qué cláusulas son problemáticas?",
                "historial": [
                    {"role": "user", "content": "Analicé mi contrato"},
                    {"role": "assistant", "content": "Perfecto, ¿en qué puedo ayudarte?"}
                ],
                "contexto_contrato": {
                    "total_clausulas": 10,
                    "clausulas_abusivas": 2
                }
            }
        }

class AnalizarContratoRequest(BaseModel):
    """Request para analizar contrato desde texto"""
    texto_contrato: str = Field(..., description="Texto completo del contrato")
    max_tokens_por_clausula: int = Field(default=512, ge=100, le=2048)

    class Config:
        json_schema_extra = {
            "example": {
                "texto_contrato": "CONTRATO DE PRESTACIÓN DE SERVICIOS...",
                "max_tokens_por_clausula": 512
            }
        }

class ClasificarClausulaRequest(BaseModel):
    """Request para clasificar una cláusula individual"""
    texto_clausula: str = Field(..., description="Texto de la cláusula a clasificar")

    class Config:
        json_schema_extra = {
            "example": {
                "texto_clausula": "El contratista se compromete a entregar el producto..."
            }
        }

# ==================== RESPONSE MODELS ====================

class ChatData(BaseModel):
    """Datos de respuesta del chatbot"""
    respuesta: str
    modelo: str
    tokens_usados: Optional[Dict[str, int]] = None
    tiempo_respuesta: Optional[float] = None
    finish_reason: Optional[str] = None

class ChatResponse(BaseModel):
    """Response del chatbot"""
    success: bool
    message: str  # Para compatibilidad con frontend (contiene la respuesta)
    data: ChatData
    timestamp: str

class ClausulaAnalizada(BaseModel):
    """Modelo de una cláusula analizada"""
    numero: int
    contenido: str
    truncado: bool
    clasificacion: str
    confianza: float
    nivel_riesgo: Optional[str] = None
    matched_phrases: Optional[List[Dict[str, Any]]] = None
    # Nota: el analizador actual no retorna campos como 'es_abusiva' o 'explicacion'
    # si más adelante se añaden, se pueden incluir aquí como opcionales.

class RiskMatrixRow(BaseModel):
    """Fila de la matriz de riesgos"""
    id: str
    categoria: str
    probabilidad: int
    impacto: str
    riesgo_afectacion: List[str]
    mitigacion: str
    responsable: str

class AnalisisData(BaseModel):
    """Datos del análisis completo"""
    # Se ajusta a la salida real de `analizar_contrato_completo`
    total_clausulas: int
    clausulas_analizadas: List[ClausulaAnalizada]
    riesgos_por_nivel: Optional[Dict[str, List[Dict[str, Any]]]] = None
    estadisticas_tipos: Optional[Dict[str, int]] = None
    resumen_riesgos: Optional[Dict[str, Any]] = None
    wordcloud: Optional[List[Dict[str, Any]]] = None
    risk_matrix: Optional[List[RiskMatrixRow]] = None

class AnalisisResponse(BaseModel):
    """Response del análisis de contrato"""
    success: bool
    message: str
    data: AnalisisData
    filename: Optional[str] = None
    duration_seconds: Optional[float] = None
    timestamp: str
    pdf_info: Optional[Dict[str, Any]] = None  # Metadata del PDF

class ClasificacionData(BaseModel):
    """Datos de clasificación de cláusula"""
    # El método `clasificar_clausula` devuelve actualmente:
    # {'etiqueta': ..., 'confianza': ..., 'probabilidades': {...}}
    etiqueta: str
    confianza: float
    probabilidades: Optional[Dict[str, float]] = None
    # Campos opcionales para compatibilidad con futuros cambios
    clasificacion: Optional[str] = None
    es_abusiva: Optional[bool] = None
    severidad: Optional[str] = None
    explicacion: Optional[str] = None

class ClasificacionResponse(BaseModel):
    """Response de clasificación de cláusula"""
    success: bool
    message: str
    data: ClasificacionData
    timestamp: str

class HealthData(BaseModel):
    """Datos de health check"""
    status: str
    timestamp: str
    modelo_cargado: bool

class HealthResponse(BaseModel):
    """Response de health check"""
    success: bool
    message: str
    data: HealthData

class ErrorResponse(BaseModel):
    """Response de error"""
    success: bool = False
    error: str

# ==================== HISTORIAL MODELS ====================

class AnalisisHistorialItem(BaseModel):
    """Item del historial de análisis"""
    id: int
    filename: str
    timestamp: str
    total_clausulas: int
    riesgos_alto: int
    riesgos_medio: int
    riesgos_bajo: int
    modo_utilizado: Optional[str] = None

class HistorialResponse(BaseModel):
    """Response del historial de análisis"""
    success: bool
    message: str
    data: List[AnalisisHistorialItem]
    total: int
    timestamp: str

class RecuperarAnalisisResponse(BaseModel):
    """Response para recuperar análisis específico"""
    success: bool
    message: str
    data: AnalisisData
    timestamp: str

class ActualizarResponsableRequest(BaseModel):
    """Request para actualizar responsable"""
    matrix_id: str = Field(..., description="ID de la matriz (ej: '1.a')")
    responsable: str = Field(..., description="Nombre del responsable")

    class Config:
        json_schema_extra = {
            "example": {
                "matrix_id": "1.a",
                "responsable": "Juan Pérez"
            }
        }

class ActualizarResponsableResponse(BaseModel):
    """Response de actualización de responsable"""
    success: bool
    message: str
    timestamp: str

class InfoModeloData(BaseModel):
    """Información del modelo"""
    nombre: str
    version: str
    tipo: str
    clases: List[str]
    metricas: Optional[Dict[str, float]] = None
    ultima_actualizacion: Optional[str] = None

class InfoModeloResponse(BaseModel):
    """Response de información del modelo"""
    success: bool
    message: str
    data: InfoModeloData
