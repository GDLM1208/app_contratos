import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter, defaultdict
from .filtrado_clasulas import FiltradorClausulasConstructor

class AnalizadorContratos:
    def __init__(self, model_path="./modelo_clausulas"):
        """Inicializar el analizador con el modelo personalizado"""

        resolved_model_path = self._resolve_model_path(model_path)

        self.model = AutoModelForSequenceClassification.from_pretrained(resolved_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_model_path)
        self.id2label = self.model.config.id2label

        # Definir qué etiquetas representan riesgos (ajusta según tu modelo)
        self.etiquetas_riesgo = {
            'Pago': 'bajo',
            'Cambios': 'medio',
            'Penalidades': 'alto',
            'Resolucion de Disputas': 'medio',
            'Indemnizacion': 'alto',
            'Plazos de Reclamo': 'medio',
            'Seguridad y Salud': 'medio',
            'Funciones y Responsabilidades (RNR)': 'bajo',
            'Procedimientos': 'bajo',
            'Legal o Referencia Normativa': 'medio',
            'Temporal': 'medio',
        }
        # print(f"Modelo cargado. Etiquetas disponibles: {list(self.id2label.values())}")


    def _resolve_model_path(self, model_path: str) -> str:
        """
        Resolver la ruta del modelo para compatibilidad local y Railway

        Orden de prioridad:
        1. Variable de entorno MODEL_PATH (Railway)
        2. Ruta absoluta si se proporciona
        3. Ruta relativa al archivo analizador.py (local)
        4. Ruta absoluta en Railway volume (/app/modelo-clausulas)
        """

        # Prioridad 1: Variable de entorno (Railway)
        env_model_path = os.getenv("MODEL_PATH")
        if env_model_path:
            print(f"📍 Usando MODEL_PATH de variable de entorno: {env_model_path}")
            clean_path = os.path.normpath(env_model_path)
            if os.path.exists(clean_path):
                return clean_path
            else:
                print(f"⚠️ Advertencia: MODEL_PATH no existe: {clean_path}")

        # Prioridad 2: Ruta absoluta
        if os.path.isabs(model_path):
            print(f"📍 Usando ruta absoluta: {model_path}")
            return model_path

        # Prioridad 3: Ruta relativa al archivo actual (desarrollo local)
        base_path = os.path.dirname(os.path.abspath(__file__))
        local_model_path = os.path.join(base_path, model_path)

        if os.path.exists(local_model_path):
            print(f"📍 Encontrado modelo en ruta local: {local_model_path}")
            return local_model_path

        # Prioridad 4: Ruta en Railway volume
        railway_volume_path = f"/app/{model_path}"
        if os.path.exists(railway_volume_path):
            print(f"📍 Encontrado modelo en Railway volume: {railway_volume_path}")
            return railway_volume_path

        # Fallback: retornar ruta local y dejar que falle con mensaje claro
        print(f"⚠️ No se encontró modelo, usando fallback: {local_model_path}")
        return local_model_path

    def extraer_parrafos_y_fragmentos(self, texto, max_tokens=512):
        """
        Separar el contrato en párrafos y fragmentarlos si exceden el límite de tokens.
        """
        # Separar en párrafos usando doble salto de línea
        parrafos = re.split(r'\n\n+', texto)

        filtro = FiltradorClausulasConstructor()
        parrafos_procesados = []
        estadisticas_filtro = {'total_original': len(parrafos), 'descartadas': 0}

        for parrafo in parrafos:
            parrafo = parrafo.strip()

            filtro_clausula = filtro.filtrar_clausula(parrafo)

            if filtro_clausula['es_relevante']:
                # Tokenizar el párrafo
                tokens = self.tokenizer.encode(parrafo, add_special_tokens=False)

                # Si el párrafo excede el límite de tokens, fragmentarlo
                if len(tokens) > max_tokens:
                    fragmentos = []
                    while len(tokens) > max_tokens:
                        # Cortar en un punto natural, aquí un ejemplo simple: en el punto de la mitad
                        fragmento = self.tokenizer.decode(tokens[:max_tokens])
                        fragmentos.append(fragmento)
                        tokens = tokens[max_tokens:]  # Cortamos los tokens procesados
                    # Añadir el último fragmento
                    if tokens:
                        fragmentos.append(self.tokenizer.decode(tokens))
                    parrafos_procesados.append({
                        'parrafo': fragmentos,
                        'truncado': True,
                        'filtro_info': filtro_clausula
                    })
                else:
                    parrafos_procesados.append({
                        'parrafo': [parrafo],  # No se fragmenta si está dentro del límite
                        'truncado': False
                    })
            else:
                estadisticas_filtro['descartadas'] += 1

        estadisticas_filtro['procesadas'] = len(parrafos_procesados)
        estadisticas_filtro['porcentaje_retenido'] = round(
            (estadisticas_filtro['procesadas'] / estadisticas_filtro['total_original']) * 100, 1
        ) if estadisticas_filtro['total_original'] > 0 else 0

        print(f"📊 Filtrado completado: {estadisticas_filtro['procesadas']}/{estadisticas_filtro['total_original']} cláusulas retenidas ({estadisticas_filtro['porcentaje_retenido']}%)")

        return parrafos_procesados


    def clasificar_clausula(self, texto):
        """Clasificar una cláusula individual"""
        inputs = self.tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Obtener probabilidades
        probs = torch.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(outputs.logits, dim=1).item()
        confidence = probs[0][pred_idx].item()

        return {
            'etiqueta': self.id2label[pred_idx],
            'confianza': confidence,
            'probabilidades': {self.id2label[i]: prob.item() for i, prob in enumerate(probs[0])}
        }

    def analizar_contrato_completo(self, texto_contrato, max_tokens_por_clausula=512):
        """
        Analizar todo el contrato y generar reporte de riesgos
        """
        clausulas = self.extraer_parrafos_y_fragmentos(texto_contrato, max_tokens_por_clausula)

        if not clausulas:
            return {
                'error': 'No se encontraron párrafos en el formato esperado',
                'clausulas_encontradas': 0
            }

        resultados = []
        riesgos_encontrados = defaultdict(list)
        estadisticas = Counter()

        for i, clausula in enumerate(clausulas, 1):
            """ print(f"Procesando parrafo {i}/{len(clausulas)}...") """

            clasificacion = self.clasificar_clausula(clausula['parrafo'])

            resultado_clausula = {
                'numero': i,
                'contenido': clausula['parrafo'][:200] + "..." if len(clausula['parrafo']) > 200 else clausula['parrafo'],
                'truncado': clausula['truncado'],
                'clasificacion': clasificacion['etiqueta'],
                'confianza': round(clasificacion['confianza'], 3)
            }

            # Identificar nivel de riesgo
            etiqueta = clasificacion['etiqueta']
            if etiqueta in self.etiquetas_riesgo:
                nivel_riesgo = self.etiquetas_riesgo[etiqueta]
                resultado_clausula['nivel_riesgo'] = nivel_riesgo
                riesgos_encontrados[nivel_riesgo].append(resultado_clausula)
            else:
                resultado_clausula['nivel_riesgo'] = 'desconocido'

            estadisticas[etiqueta] += 1
            resultados.append(resultado_clausula)

        return {
            'total_clausulas': len(clausulas),
            'clausulas_analizadas': resultados,
            'riesgos_por_nivel': dict(riesgos_encontrados),
            'estadisticas_tipos': dict(estadisticas),
            'resumen_riesgos': self._generar_resumen_riesgos(riesgos_encontrados, estadisticas)
        }

    def _generar_resumen_riesgos(self, riesgos_por_nivel, estadisticas):
        """Generar resumen ejecutivo de riesgos"""
        total_clausulas = sum(estadisticas.values())

        resumen = {
            'total_clausulas_analizadas': total_clausulas,
            'clausulas_riesgo_alto': len(riesgos_por_nivel.get('alto', [])),
            'clausulas_riesgo_medio': len(riesgos_por_nivel.get('medio', [])),
            'clausulas_riesgo_bajo': len(riesgos_por_nivel.get('bajo', [])),
        }

        # Calcular porcentajes
        if total_clausulas > 0:
            resumen['porcentaje_riesgo_alto'] = round((resumen['clausulas_riesgo_alto'] / total_clausulas) * 100, 1)
            resumen['porcentaje_riesgo_medio'] = round((resumen['clausulas_riesgo_medio'] / total_clausulas) * 100, 1)
            resumen['porcentaje_riesgo_bajo'] = round((resumen['clausulas_riesgo_bajo'] / total_clausulas) * 100, 1)

        # Generar recomendaciones
        recomendaciones = []
        if resumen['clausulas_riesgo_alto'] > 0:
            recomendaciones.append(f"⚠️ ATENCIÓN: Se encontraron {resumen['clausulas_riesgo_alto']} cláusulas de RIESGO ALTO que requieren revisión urgente.")

        if resumen['clausulas_riesgo_medio'] > 2:
            recomendaciones.append(f"⚡ Se identificaron {resumen['clausulas_riesgo_medio']} cláusulas de riesgo medio. Considere negociar términos más favorables.")

        if resumen['porcentaje_riesgo_alto'] > 20:
            recomendaciones.append("🔴 Más del 20% de las cláusulas presentan riesgo alto. Se recomienda revisión legal especializada.")

        resumen['recomendaciones'] = recomendaciones
        return resumen

    def generar_reporte_detallado(self, resultado_analisis):
        """Generar reporte legible para humanos"""
        if 'error' in resultado_analisis:
            return f"Error en el análisis: {resultado_analisis['error']}"

        reporte = []
        reporte.append("="*80)
        reporte.append("REPORTE DE ANÁLISIS DE CONTRATO")
        reporte.append("="*80)

        resumen = resultado_analisis['resumen_riesgos']
        reporte.append(f"\n📊 RESUMEN EJECUTIVO:")
        reporte.append(f"• Total de cláusulas analizadas: {resumen['total_clausulas_analizadas']}")
        """ reporte.append(f"• Cláusulas de riesgo ALTO: {resumen['clausulas_riesgo_alto']} ({resumen.get('porcentaje_riesgo_alto', 0)}%)")
        reporte.append(f"• Cláusulas de riesgo MEDIO: {resumen['clausulas_riesgo_medio']} ({resumen.get('porcentaje_riesgo_medio', 0)}%)")
        reporte.append(f"• Cláusulas de riesgo BAJO: {resumen['clausulas_riesgo_bajo']} ({resumen.get('porcentaje_riesgo_bajo', 0)}%)")
        """

        if resumen['recomendaciones']:
            reporte.append(f"\n🎯 RECOMENDACIONES:")
            for rec in resumen['recomendaciones']:
                reporte.append(f"• {rec}")

        # Detalle por nivel de riesgo
        for nivel in ['alto', 'medio', 'bajo']:
            clausulas_nivel = resultado_analisis['riesgos_por_nivel'].get(nivel, [])
            if clausulas_nivel:
                reporte.append(f"\n🔍 CLÁUSULAS DE RIESGO {nivel.upper()}:")
                for clausula in clausulas_nivel:
                    reporte.append(f"  • {clausula['titulo']}")
                    reporte.append(f"    Tipo: {clausula['clasificacion']} (Confianza: {clausula['confianza']})")
                    reporte.append(f"    Vista previa: {clausula['parrafo'][:150]}...")
                    reporte.append("")

        # Estadísticas por tipo
        reporte.append(f"\n📈 DISTRIBUCIÓN POR TIPO DE CLÁUSULA:")
        for tipo, cantidad in resultado_analisis['estadisticas_tipos'].items():
            porcentaje = (cantidad / resumen['total_clausulas_analizadas']) * 100
            reporte.append(f"• {tipo}: {cantidad} ({porcentaje:.1f}%)")

        return "\n".join(reporte)


# Función principal para usar el analizador
def main():
    # Ejemplo de uso
    texto_contrato = """
    Este es el preámbulo del contrato de servicios profesionales.

    CLÁUSULA PRIMERA: ANTECEDENTES
    Con fecha 1 de enero de 2025, las partes acuerdan celebrar el presente contrato de prestación de servicios. La empresa contratante requiere servicios especializados de consultoría y el contratista cuenta con la experiencia necesaria para desarrollar dichas actividades.

    CLÁUSULA SEGUNDA: OBJETO
    El presente contrato tiene por objeto la prestación de servicios de consultoría en tecnología. El contratista se compromete a entregar los servicios acordados en los tiempos establecidos y con la calidad requerida.

    CLÁUSULA TERCERA: PENALIDADES
    En caso de incumplimiento por parte del contratista, este deberá pagar una penalidad equivalente al 10% del valor total del contrato por cada día de retraso. Además, la empresa se reserva el derecho de terminar el contrato unilateralmente.

    CLÁUSULA CUARTA: DURACIÓN
    Este contrato tendrá una duración de 12 meses a partir de su firma y podrá renovarse por períodos adicionales mediante acuerdo mutuo por escrito.

    CLÁUSULA QUINTA: CONFIDENCIALIDAD
    El contratista se compromete a mantener absoluta confidencialidad sobre toda la información a la que tenga acceso durante la ejecución del contrato.
    """

    # Inicializar analizador
    analizador = AnalizadorContratos()

    # Analizar contrato
    resultado = analizador.analizar_contrato_completo(texto_contrato)

    # Generar y mostrar reporte
    reporte = analizador.generar_reporte_detallado(resultado)
    # print(reporte)

if __name__ == "__main__":
    main()