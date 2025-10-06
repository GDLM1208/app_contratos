import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class Settings:
    """Configuración de la aplicación"""

    # Aplicación
    APP_NAME: str = "API de Análisis de Contratos"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Servidor
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))

    # CORS
    CORS_ORIGINS: list = os.getenv(
        "CORS_ORIGINS",
        "*"
    ).split(",")

    # Paths
    BASE_DIR: Path = Path(__file__).parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    MODEL_DIR: Path = BASE_DIR / "model"

    # Modelo (Transformers)
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH",
        "./modelo-clausulas"
    )

    # Límites
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", 512 * 1024 * 1024))  # 512MB
    MAX_TOKENS_POR_CLAUSULA: int = int(os.getenv("MAX_TOKENS_POR_CLAUSULA", 512))

    # Timeouts
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", 300))  # 5 minutos

    # OpenAI Chatbot
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    CHATBOT_MAX_TOKENS: int = int(os.getenv("CHATBOT_MAX_TOKENS", 1000))
    CHATBOT_TEMPERATURE: float = float(os.getenv("CHATBOT_TEMPERATURE", 0.7))

    # Base de datos (opcional - para futuro)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    def __init__(self):
        # Crear directorios necesarios
        self.UPLOAD_DIR.mkdir(exist_ok=True)
        self.MODEL_DIR.mkdir(exist_ok=True)

# Instancia global de configuración
settings = Settings()