"""
Configuración de la base de datos SQLite
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.database import Base

# Ruta de la base de datos
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./contratos_analysis.db")

# Motor de base de datos
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Crear todas las tablas en la base de datos"""
    Base.metadata.create_all(bind=engine)
    print("✅ Tablas de base de datos creadas correctamente")

def get_db():
    """Dependency para obtener sesión de base de datos"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Inicializar la base de datos"""
    try:
        create_tables()
        return True
    except Exception as e:
        print(f"❌ Error inicializando base de datos: {e}")
        return False