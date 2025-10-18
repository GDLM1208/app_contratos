"""
Modelos de base de datos para el análisis de contratos
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import json

Base = declarative_base()

class Analisis(Base):
    """Tabla principal de análisis de contratos"""
    __tablename__ = "analisis"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    total_clausulas = Column(Integer, nullable=False)
    resumen_riesgos = Column(Text, nullable=True)  # JSON serializado
    wordcloud_data = Column(Text, nullable=True)  # JSON serializado
    modo_utilizado = Column(String(50), nullable=True)

    # Relaciones
    clausulas = relationship("ClausulaAnalizada", back_populates="analisis", cascade="all, delete-orphan")
    risk_matrix = relationship("RiskMatrix", back_populates="analisis", cascade="all, delete-orphan")

    def set_resumen_riesgos(self, data):
        """Serializar resumen de riesgos como JSON"""
        self.resumen_riesgos = json.dumps(data) if data else None

    def get_resumen_riesgos(self):
        """Deserializar resumen de riesgos desde JSON"""
        return json.loads(self.resumen_riesgos) if self.resumen_riesgos else None

    def set_wordcloud_data(self, data):
        """Serializar datos de wordcloud como JSON"""
        self.wordcloud_data = json.dumps(data) if data else None

    def get_wordcloud_data(self):
        """Deserializar datos de wordcloud desde JSON"""
        return json.loads(self.wordcloud_data) if self.wordcloud_data else None


class ClausulaAnalizada(Base):
    """Tabla de cláusulas analizadas"""
    __tablename__ = "clausulas_analizadas"

    id = Column(Integer, primary_key=True, index=True)
    analisis_id = Column(Integer, ForeignKey("analisis.id"), nullable=False)
    numero = Column(Integer, nullable=False)
    contenido = Column(Text, nullable=False)
    clasificacion = Column(String(100), nullable=False)
    confianza = Column(Float, nullable=False)
    nivel_riesgo = Column(String(20), nullable=True)
    matched_phrases = Column(Text, nullable=True)  # JSON serializado
    truncado = Column(Integer, default=0)  # 0 = False, 1 = True

    # Relación
    analisis = relationship("Analisis", back_populates="clausulas")

    def set_matched_phrases(self, data):
        """Serializar matched_phrases como JSON"""
        self.matched_phrases = json.dumps(data) if data else None

    def get_matched_phrases(self):
        """Deserializar matched_phrases desde JSON"""
        return json.loads(self.matched_phrases) if self.matched_phrases else []


class RiskMatrix(Base):
    """Tabla de matriz de riesgos"""
    __tablename__ = "risk_matrix"

    id = Column(Integer, primary_key=True, index=True)
    analisis_id = Column(Integer, ForeignKey("analisis.id"), nullable=False)
    clausula_numero = Column(Integer, nullable=False)  # Referencia a la cláusula específica
    matrix_id = Column(String(10), nullable=False)  # "1.a", "1.b", "2.a", etc.
    categoria = Column(String(100), nullable=False)
    probabilidad = Column(Integer, nullable=False)
    impacto = Column(String(20), nullable=False)
    riesgo_afectacion = Column(Text, nullable=False)  # JSON array serializado
    mitigacion = Column(Text, nullable=True, default="")
    responsable = Column(String(100), nullable=True, default="")

    # Relación
    analisis = relationship("Analisis", back_populates="risk_matrix")

    def set_riesgo_afectacion(self, data):
        """Serializar riesgo_afectacion como JSON"""
        self.riesgo_afectacion = json.dumps(data) if data else "[]"

    def get_riesgo_afectacion(self):
        """Deserializar riesgo_afectacion desde JSON"""
        return json.loads(self.riesgo_afectacion) if self.riesgo_afectacion else []