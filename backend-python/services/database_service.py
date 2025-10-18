"""
Servicio para operaciones de base de datos del análisis de contratos
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from models.database import Analisis, ClausulaAnalizada, RiskMatrix
from datetime import datetime

class DatabaseService:
    """Servicio para manejar operaciones de base de datos"""

    def __init__(self):
        pass

    def guardar_analisis_completo(
        self,
        db: Session,
        filename: str,
        resultado_analisis: Dict[str, Any]
    ) -> int:
        """
        Guardar análisis completo en la base de datos

        Args:
            db: Sesión de base de datos
            filename: Nombre del archivo analizado
            resultado_analisis: Resultado completo del análisis

        Returns:
            int: ID del análisis guardado
        """
        try:
            # Crear registro principal de análisis
            analisis = Analisis(
                filename=filename,
                total_clausulas=resultado_analisis.get('total_clausulas', 0),
                modo_utilizado=resultado_analisis.get('modo_utilizado', 'balanced')
            )

            # Serializar datos complejos
            analisis.set_resumen_riesgos(resultado_analisis.get('resumen_riesgos'))
            analisis.set_wordcloud_data(resultado_analisis.get('wordcloud'))

            db.add(analisis)
            db.flush()  # Para obtener el ID

            # Guardar cláusulas analizadas
            clausulas_analizadas = resultado_analisis.get('clausulas_analizadas', [])
            for clausula_data in clausulas_analizadas:
                clausula = ClausulaAnalizada(
                    analisis_id=analisis.id,
                    numero=clausula_data.get('numero', 0),
                    contenido=clausula_data.get('contenido', ''),
                    clasificacion=clausula_data.get('clasificacion', ''),
                    confianza=clausula_data.get('confianza', 0.0),
                    nivel_riesgo=clausula_data.get('nivel_riesgo', ''),
                    truncado=1 if clausula_data.get('truncado', False) else 0
                )
                clausula.set_matched_phrases(clausula_data.get('matched_phrases', []))
                db.add(clausula)

            # Guardar matriz de riesgos
            risk_matrix = resultado_analisis.get('risk_matrix', [])
            for risk_data in risk_matrix:
                risk_row = RiskMatrix(
                    analisis_id=analisis.id,
                    clausula_numero=self._extract_clause_number(risk_data.get('id', '')),
                    matrix_id=risk_data.get('id', ''),
                    categoria=risk_data.get('categoria', ''),
                    probabilidad=risk_data.get('probabilidad', 1),
                    impacto=risk_data.get('impacto', 'bajo'),
                    mitigacion=risk_data.get('mitigacion', ''),
                    responsable=risk_data.get('responsable', '')
                )
                risk_row.set_riesgo_afectacion(risk_data.get('riesgo_afectacion', []))
                db.add(risk_row)

            db.commit()
            print(f"✅ Análisis guardado con ID: {analisis.id}")
            return analisis.id

        except Exception as e:
            db.rollback()
            print(f"❌ Error guardando análisis: {e}")
            raise e

    def obtener_historial_analisis(self, db: Session, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Obtener historial de análisis ordenado por fecha

        Args:
            db: Sesión de base de datos
            limit: Límite de registros a retornar

        Returns:
            List: Lista de análisis con información básica
        """
        try:
            analisis_list = db.query(Analisis).order_by(Analisis.timestamp.desc()).limit(limit).all()

            historial = []
            for analisis in analisis_list:
                resumen = analisis.get_resumen_riesgos() or {}

                historial.append({
                    'id': analisis.id,
                    'filename': analisis.filename,
                    'timestamp': analisis.timestamp.isoformat(),
                    'total_clausulas': analisis.total_clausulas,
                    'riesgos_alto': resumen.get('clausulas_riesgo_alto', 0),
                    'riesgos_medio': resumen.get('clausulas_riesgo_medio', 0),
                    'riesgos_bajo': resumen.get('clausulas_riesgo_bajo', 0),
                    'modo_utilizado': analisis.modo_utilizado
                })

            return historial

        except Exception as e:
            print(f"❌ Error obteniendo historial: {e}")
            return []

    def obtener_analisis_por_id(self, db: Session, analisis_id: int) -> Optional[Dict[str, Any]]:
        """
        Recuperar análisis completo por ID

        Args:
            db: Sesión de base de datos
            analisis_id: ID del análisis a recuperar

        Returns:
            Dict: Análisis completo reconstruido
        """
        try:
            # Obtener análisis principal
            analisis = db.query(Analisis).filter(Analisis.id == analisis_id).first()
            if not analisis:
                return None

            # Obtener cláusulas analizadas
            clausulas = db.query(ClausulaAnalizada).filter(
                ClausulaAnalizada.analisis_id == analisis_id
            ).order_by(ClausulaAnalizada.numero).all()

            # Obtener matriz de riesgos
            risk_matrix = db.query(RiskMatrix).filter(
                RiskMatrix.analisis_id == analisis_id
            ).order_by(RiskMatrix.matrix_id).all()

            # Reconstruir el formato original
            resultado = {
                'total_clausulas': analisis.total_clausulas,
                'clausulas_analizadas': [],
                'resumen_riesgos': analisis.get_resumen_riesgos(),
                'wordcloud': analisis.get_wordcloud_data(),
                'risk_matrix': [],
                'modo_utilizado': analisis.modo_utilizado,
                'timestamp': analisis.timestamp.isoformat(),
                'filename': analisis.filename
            }

            # Reconstruir cláusulas
            for clausula in clausulas:
                resultado['clausulas_analizadas'].append({
                    'numero': clausula.numero,
                    'contenido': clausula.contenido,
                    'clasificacion': clausula.clasificacion,
                    'confianza': clausula.confianza,
                    'nivel_riesgo': clausula.nivel_riesgo,
                    'matched_phrases': clausula.get_matched_phrases(),
                    'truncado': bool(clausula.truncado)
                })

            # Reconstruir matriz de riesgos
            for risk_row in risk_matrix:
                resultado['risk_matrix'].append({
                    'id': risk_row.matrix_id,
                    'categoria': risk_row.categoria,
                    'probabilidad': risk_row.probabilidad,
                    'impacto': risk_row.impacto,
                    'riesgo_afectacion': risk_row.get_riesgo_afectacion(),
                    'mitigacion': risk_row.mitigacion,
                    'responsable': risk_row.responsable
                })

            return resultado

        except Exception as e:
            print(f"❌ Error recuperando análisis {analisis_id}: {e}")
            return None

    def actualizar_responsable_matriz(
        self,
        db: Session,
        analisis_id: int,
        matrix_id: str,
        responsable: str
    ) -> bool:
        """
        Actualizar responsable en la matriz de riesgos

        Args:
            db: Sesión de base de datos
            analisis_id: ID del análisis
            matrix_id: ID de la matriz (ej: "1.a")
            responsable: Nuevo responsable

        Returns:
            bool: True si se actualizó correctamente
        """
        try:
            risk_row = db.query(RiskMatrix).filter(
                RiskMatrix.analisis_id == analisis_id,
                RiskMatrix.matrix_id == matrix_id
            ).first()

            if not risk_row:
                return False

            risk_row.responsable = responsable
            db.commit()

            print(f"✅ Responsable actualizado: {matrix_id} -> {responsable}")
            return True

        except Exception as e:
            db.rollback()
            print(f"❌ Error actualizando responsable: {e}")
            return False

    def _extract_clause_number(self, matrix_id: str) -> int:
        """Extraer número de cláusula desde matrix_id (ej: "1.a" -> 1)"""
        try:
            return int(matrix_id.split('.')[0])
        except (ValueError, IndexError):
            return 0