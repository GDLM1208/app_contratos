# filtro_clausulas_construccion.py
import re
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Optional

class FiltradorClausulasConstructor:
    """
    Filtro híbrido para detectar cláusulas relevantes en contratos de construcción
    Basado en el paper "Automated detection of contractual risk clauses" (Moon et al., 2022)
    """

    def __init__(self):
        # Exclusiones específicas para contratos de construcción
        self.exclusiones_administrativas = [
            # Secciones administrativas
            'índice', 'tabla', 'anexo', 'apéndice', 'introducción', 'antecedentes',
            'considerando', 'considerandos', 'firma', 'firmas', 'fecha', 'página', 'folio',
            'capítulo', 'sección', 'título', 'subtítulo',

            # Información de contacto/identificación
            'dirección', 'teléfono', 'email', 'correo', 'identificación', 'cédula', 'ruc',
            'nit', 'registro', 'domicilio', 'representante legal',

            # Numeración y referencias simples
            'numeral', 'literal', 'inciso', 'párrafo', 'punto', 'item',

            # Tablas y formatos
            'cuadro', 'gráfico', 'diagrama', 'esquema', 'formato', 'planilla',
            'formulario', 'matriz', 'listado'
        ]

        # Patrones de estructura legal (basado en análisis de contratos)
        self.patrones_juridicos = {
            'obligaciones': r'\b(deberá|será|tendrá|podrá|debe|puede|queda obligado|se compromete)\b',
            'condiciones': r'\b(en caso de|si|cuando|siempre que|a condición de|bajo la condición)\b',
            'penalizaciones': r'\b(multa|penalidad|sanción|incumplimiento|mora|retraso)\b',
            'garantias': r'\b(garantía|garantizar|asegurar|responder|caucionar|avalar)\b',
            'rescision': r'\b(rescisión|rescindir|terminar|cancelar|anular|resolver)\b'
        }

        self.palabras_clave_juridicas = [
            # Términos que indican obligaciones/derechos
            'deberá', 'será', 'tendrá', 'podrá', 'debe', 'puede', 'queda obligado',
            'se compromete', 'garantiza', 'asegura', 'responde',

            # Términos de riesgo general
            'pago', 'plazo', 'multa', 'penalidad', 'incumplimiento', 'responsabilidad',
            'garantía', 'seguridad', 'riesgo', 'obligación', 'terminación', 'rescisión',

            # Términos contractuales
            'contratista', 'contratante', 'cliente', 'proveedor', 'ejecutor',
            'obra', 'proyecto', 'entrega', 'cumplimiento', 'verificación'
        ]

        # Configuraciones de filtrado
        self.longitud_minima = 30  # palabras
        self.longitud_maxima = 500  # palabras
        self.umbral_relevancia = 0.25  # score mínimo para considerar relevante

    def preprocesar_texto(self, texto: str) -> str:
        """
        Preprocesamiento basado en Moon et al. (2022)
        Solo normalización básica para preservar estructura
        """
        # Convertir a minúsculas
        texto = texto.lower()

        # Reemplazar según paper: URLs, referencias, números
        texto = re.sub(r'https?://\S+|www\.\S+', 'URL', texto)
        texto = re.sub(r'\b\d+(\.\d+)?\b', 'NUM', texto)
        texto = re.sub(r'\b(art\.|artículo|inc\.|inciso|lit\.|literal)\s*\d+', 'REF', texto)

        # Limpiar caracteres especiales manteniendo estructura básica
        texto = re.sub(r'[^\w\s\.\,\;\:\(\)\-\/]', ' ', texto)
        texto = re.sub(r'\s+', ' ', texto)

        return texto.strip()

    def calcular_densidad_juridica(self, texto: str) -> Dict[str, float]:
        """Calcula densidad de términos jurídicos sin categorizar"""
        palabras_texto = texto.lower().split()
        total_palabras = len(palabras_texto)

        if total_palabras == 0:
            return 0.0

        coincidencias = sum(1 for palabra in palabras_texto
                        if palabra in self.palabras_clave_juridicas)

        # Normalizar: score entre 0 y 5
        densidad = (coincidencias / total_palabras) * 10
        return min(densidad, 5.0)

    def calcular_score_patrones_juridicos(self, texto: str) -> float:
        """Calcula score basado en patrones de estructura jurídica"""
        score = 0.0

        for patron_nombre, patron_regex in self.patrones_juridicos.items():
            matches = re.findall(patron_regex, texto, re.IGNORECASE)
            if matches:
                # Peso diferencial según importancia del patrón
                peso_patron = {
                    'obligaciones': 1.5,
                    'condiciones': 1.3,
                    'penalizaciones': 2.0,
                    'garantias': 1.4,
                    'rescision': 1.8
                }.get(patron_nombre, 1.0)

                score += len(matches) * peso_patron

        return min(score, 5.0)  # Limitar score máximo

    def es_exclusion_administrativa(self, texto: str) -> bool:
        """Verifica si el texto corresponde a sección administrativa"""
        texto_lower = texto.lower()

        # Verificar exclusiones directas
        primera_palabra = texto_lower.split()[0] if texto_lower.split() else ''
        if primera_palabra in self.exclusiones_administrativas:
            return True

        # Verificar patrones de exclusión
        patrones_exclusion = [
            r'^\s*(capítulo|sección|título)\s+[ivxlcdm\d]+',  # Numeración romana/arábiga
            r'^\s*página\s+\d+',
            r'^\s*anexo\s+[a-z\d]+',
            r'^\s*tabla\s+\d+',
            r'^\s*figura\s+\d+'
        ]

        for patron in patrones_exclusion:
            if re.match(patron, texto_lower):
                return True

        return False

    def calcular_score_longitud(self, texto: str) -> float:
        """Calcula score basado en longitud óptima de cláusulas"""
        palabras = texto.split()
        num_palabras = len(palabras)

        if num_palabras < self.longitud_minima:
            return 0.0
        elif num_palabras > self.longitud_maxima:
            return 0.0

        # Score óptimo entre 50-150 palabras
        if 50 <= num_palabras <= 150:
            return 1.0
        elif 30 <= num_palabras < 50:
            return 0.7
        elif 150 < num_palabras <= 300:
            return 0.8
        else:
            return 0.5

    def filtrar_clausula(self, texto_original: str) -> Dict:
        """
        Filtra una cláusula individual y retorna análisis completo
        """
        if not texto_original or not texto_original.strip():
            return {
                'es_relevante': False,
                'score_total': 0.0,
                'longitud_palabras': 0,
                'razon_exclusion': 'texto_vacio'
            }

        # Preprocesar texto
        texto = self.preprocesar_texto(texto_original)

        # Verificar exclusiones administrativas
        if self.es_exclusion_administrativa(texto):
            return {
                'es_relevante': False,
                'score_total': 0.0,
                'longitud_palabras': len(texto.split()),
                'razon_exclusion': 'seccion_administrativa'
            }

         # Verificar longitud
        score_longitud = self.calcular_score_longitud(texto)
        if score_longitud == 0.0:
            return {
                'es_relevante': False,
                'score_total': 0.0,
                'longitud_palabras': len(texto.split()),
                'razon_exclusion': 'longitud_invalida'
            }

        # Calcular scores (sin categorización)
        densidad_juridica = self.calcular_densidad_juridica(texto)
        score_patrones = self.calcular_score_patrones_juridicos(texto)

        # Score total simplificado
        score_total = (
            densidad_juridica * 0.35 +    # 35% densidad de términos jurídicos
            score_patrones * 0.40 +        # 40% patrones jurídicos
            score_longitud * 0.25          # 25% longitud apropiada
        )

        # Determinar relevancia
        es_relevante = score_total >= self.umbral_relevancia

        return {
            'es_relevante': es_relevante,
            'score_total': round(score_total, 3),
            'densidad_juridica': round(densidad_juridica, 3),
            'score_patrones_juridicos': round(score_patrones, 3),
            'score_longitud': round(score_longitud, 3),
            'longitud_palabras': len(texto.split()),
            'razon_exclusion': None if es_relevante else 'score_bajo'
        }

    def filtrar_multiples_clausulas(self, clausulas: List[str]) -> List[Dict]:
        """Filtra múltiples cláusulas y retorna análisis completo"""
        resultados = []

        for i, clausula in enumerate(clausulas):
            resultado = self.filtrar_clausula(clausula)
            resultado['indice_original'] = i
            resultados.append(resultado)

        return resultados

    def obtener_estadisticas(self, resultados: List[Dict]) -> Dict:
        """Genera estadísticas del filtrado"""
        total = len(resultados)
        relevantes = sum(1 for r in resultados if r['es_relevante'])

        exclusiones_count = Counter()
        for resultado in resultados:
            if not resultado['es_relevante']:
                exclusiones_count[resultado.get('razon_exclusion', 'desconocido')] += 1

        scores_relevantes = [r['score_total'] for r in resultados if r['es_relevante']]

        return {
            'total_clausulas': total,
            'clausulas_relevantes': relevantes,
            'porcentaje_relevantes': round((relevantes / total) * 100, 1) if total > 0 else 0,
            'razones_exclusion': dict(exclusiones_count),
            'score_promedio': round(np.mean(scores_relevantes), 3) if scores_relevantes else 0,
            'score_minimo': round(min(scores_relevantes), 3) if scores_relevantes else 0,
            'score_maximo': round(max(scores_relevantes), 3) if scores_relevantes else 0
        }

# Función de integración con el sistema existente
def integrar_con_analizador_contratos(analizador_existente):
    """
    Integra el filtro con el AnalizadorContratos existente
    """
    filtro = FiltradorClausulasConstructor()

    # Función para filtrar cláusulas antes del análisis BERT
    def filtrar_clausulas_relevantes(clausulas_extraidas):
        clausulas_texto = [c['contenido'] for c in clausulas_extraidas]
        resultados_filtro = filtro.filtrar_multiples_clausulas(clausulas_texto)

        # Mantener solo las relevantes
        clausulas_filtradas = []
        for i, resultado in enumerate(resultados_filtro):
            if resultado['es_relevante']:
                clausula_original = clausulas_extraidas[i].copy()
                clausula_original['filtro_info'] = resultado
                clausulas_filtradas.append(clausula_original)

        return clausulas_filtradas, filtro.obtener_estadisticas(resultados_filtro)

    return filtrar_clausulas_relevantes

# Ejemplo de uso
if __name__ == "__main__":
    filtro = FiltradorClausulasConstructor()

    clausulas_ejemplo = [
        "CLÁUSULA PRIMERA: OBJETO - El contratista se compromete a ejecutar la construcción del edificio conforme a las especificaciones técnicas.",
        "I. Introducción - Antecedentes generales del proyecto.",
        "CLÁUSULA QUINTA: PENALIDADES - El contratista deberá pagar multa por cada día de retraso.",
        "Tabla de contenidos: 1. Introducción 2. Cláusulas",
        "CLÁUSULA DÉCIMA: SEGURIDAD - El contratista será responsable de implementar medidas de seguridad."
    ]

    resultados = filtro.filtrar_multiples_clausulas(clausulas_ejemplo)
    estadisticas = filtro.obtener_estadisticas(resultados)

    print("=== RESULTADOS DEL FILTRADO ===")
    for i, resultado in enumerate(resultados, 1):
        print(f"\nCláusula {i}:")
        print(f"  Relevante: {resultado['es_relevante']}")
        print(f"  Score: {resultado['score_total']}")
        if resultado['es_relevante']:
            print(f"  Densidad jurídica: {resultado['densidad_juridica']}")
            print(f"  Patrones jurídicos: {resultado['score_patrones_juridicos']}")
        else:
            print(f"  Razón exclusión: {resultado['razon_exclusion']}")

    print(f"\n=== ESTADÍSTICAS ===")
    print(f"Relevantes: {estadisticas['clausulas_relevantes']}/{estadisticas['total_clausulas']} ({estadisticas['porcentaje_relevantes']}%)")
    print(f"Score promedio: {estadisticas['score_promedio']}")