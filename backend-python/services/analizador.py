import os
import re
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter, defaultdict
from functools import lru_cache
from .filtrado_clasulas import FiltradorClausulasConstructor
from .cloud_word import match_phrases_for_clause, build_wordcloud_payload_from_clauses

class AnalizadorContratos:
    _instance = None
    _model = None
    _tokenizer = None

    # Configuraciones de velocidad
    MODOS_VELOCIDAD = {
        "fast": {
            "max_ngram": 3,
            "use_embeddings": False,
            "top_n_phrases": 3,
            "min_confidence": 0.95,
            "min_score": 0.5  # Aumentado para ser más estricto
        },
        "balanced": {
            "max_ngram": 3,
            "use_embeddings": True,
            "top_n_phrases": 3,
            "min_confidence": 0.5,
            "min_score": 0.4  # Aumentado para ser más estricto
        },
        "detailed": {
            "max_ngram": 4,
            "use_embeddings": True,
            "top_n_phrases": 5,
            "min_confidence": 0.3,
            "min_score": 0.3  # Aumentado para ser más estricto
        }
    }

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AnalizadorContratos, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path="./modelo_clausulas"):
        """Inicializar el analizador con el modelo personalizado"""
        if self._model is None:
            self._load_model(model_path)

    def _load_model(self, model_path="./modelo_clausulas"):
        resolved_model_path = self._resolve_model_path(model_path)

        self.model = AutoModelForSequenceClassification.from_pretrained(resolved_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_model_path)
        self.id2label = self.model.config.id2label

        # Cargar etiquetas para matched_phrases
        self._load_category_phrases()

        # Mapeo entre clasificaciones del modelo y categorías de etiquetas
        self.mapeo_clasificacion_categoria = {
            'Clausula de pago': 'Clausula de pago',
            'Clausula de cambios': 'Clausula de cambios',
            'Clausula de penalidades': 'Clausula de penalidades',
            'Clausula de terminacion': 'Clausula de terminacion',
            'Clausula de resolucion de disputas': 'Clausula de resolucion de disputas',
            'Clausula de indemnizacion': 'Clausula de indemnizacion',
            'clausula de plazos de reclamo': 'clausula de plazos de reclamo',
            'clausula de seguridad y salud': 'clausula de seguridad y salud',
            'clausula de funciones y responsabilidades': 'clausula de funciones y responsabilidades',
            'clausula de procedimientos': 'clausula de procedimientos',
            'clausula legal o de referencia normativa': 'clausula legal o de referencia normativa',
            'clausula temporal': 'clausula temporal',
            # Agregar mapeos alternativos en caso de variaciones
            'Pago': 'Clausula de pago',
            'Cambios': 'Clausula de cambios',
            'Penalidades': 'Clausula de penalidades',
            'Terminación': 'Clausula de terminacion',
            'Resolución de Disputas': 'Clausula de resolucion de disputas',
            'Indemnización': 'Clausula de indemnizacion',
            'Plazos de Reclamo': 'clausula de plazos de reclamo',
            'Seguridad y Salud': 'clausula de seguridad y salud',
            'Funciones y Responsabilidades (RNR)': 'clausula de funciones y responsabilidades',
            'Procedimientos': 'clausula de procedimientos',
            'Legal o Referencia Normativa': 'clausula legal o de referencia normativa',
            'Temporal': 'clausula temporal'
        }

        # Definir qué etiquetas representan riesgos (ajusta según tu modelo)
        self.etiquetas_riesgo = {
            'Clausula de pago': 'bajo',
            'Clausula de cambios': 'medio',
            'Clausula de penalidades': 'alto',
            'Clausula de terminacion': 'alto',
            'Clausula de resolucion de disputas': 'medio',
            'Clausula de indemnizacion': 'alto',
            'clausula de plazos de reclamo': 'medio',
            'clausula de seguridad y salud': 'medio',
            'clausula de funciones y responsabilidades': 'bajo',
            'clausula de procedimientos': 'bajo',
            'clausula legal o de referencia normativa': 'medio',
            'clausula temporal': 'medio',
            # Mapeos alternativos
            'Pago': 'bajo',
            'Cambios': 'medio',
            'Penalidades': 'alto',
            'Terminacion': 'alto',
            'Resolucion de Disputas': 'medio',
            'Indemnizacion': 'alto',
            'Plazos de Reclamo': 'medio',
            'Seguridad y Salud': 'medio',
            'Funciones y Responsabilidades (RNR)': 'bajo',
            'Procedimientos': 'bajo',
            'Legal o Referencia Normativa': 'medio',
            'Temporal': 'medio',
        }

        # Mapeo para matriz de riesgos con ID, probabilidad y riesgo/afectación
        self.matriz_riesgos_mapeo = {
            'Clausula de pago': {'id': 1, 'probabilidad': 4, 'riesgo_afectacion': ['alcance', 'costo']},
            'Clausula de cambios': {'id': 2, 'probabilidad': 4, 'riesgo_afectacion': ['costo', 'tiempo']},
            'Clausula de penalidades': {'id': 3, 'probabilidad': 3, 'riesgo_afectacion': ['alcance', 'tiempo']},
            'Clausula de terminacion': {'id': 4, 'probabilidad': 2, 'riesgo_afectacion': ['alcance', 'costo', 'tiempo']},
            'Clausula de resolucion de disputas': {'id': 5, 'probabilidad': 4, 'riesgo_afectacion': ['costo', 'tiempo']},
            'Clausula de indemnizacion': {'id': 6, 'probabilidad': 3, 'riesgo_afectacion': ['costo']},
            'clausula de plazos de reclamo': {'id': 7, 'probabilidad': 4, 'riesgo_afectacion': ['tiempo']},
            'clausula de seguridad y salud': {'id': 8, 'probabilidad': 2, 'riesgo_afectacion': ['costo']},
            'clausula de funciones y responsabilidades': {'id': 9, 'probabilidad': 3, 'riesgo_afectacion': ['alcance', 'tiempo']},
            'clausula de procedimientos': {'id': 10, 'probabilidad': 3, 'riesgo_afectacion': ['alcance', 'tiempo']},
            'clausula legal o de referencia normativa': {'id': 11, 'probabilidad': 4, 'riesgo_afectacion': ['alcance', 'costo', 'tiempo']},
            'clausula temporal': {'id': 12, 'probabilidad': 2, 'riesgo_afectacion': ['costo', 'tiempo']},
            # Mapeos alternativos
            'Pago': {'id': 1, 'probabilidad': 4, 'riesgo_afectacion': ['alcance', 'costo']},
            'Cambios': {'id': 2, 'probabilidad': 4, 'riesgo_afectacion': ['costo', 'tiempo']},
            'Penalidades': {'id': 3, 'probabilidad': 3, 'riesgo_afectacion': ['alcance', 'tiempo']},
            'Terminacion': {'id': 4, 'probabilidad': 2, 'riesgo_afectacion': ['alcance', 'costo', 'tiempo']},
            'Resolucion de Disputas': {'id': 5, 'probabilidad': 4, 'riesgo_afectacion': ['costo', 'tiempo']},
            'Indemnizacion': {'id': 6, 'probabilidad': 3, 'riesgo_afectacion': ['costo']},
            'Plazos de Reclamo': {'id': 7, 'probabilidad': 4, 'riesgo_afectacion': ['tiempo']},
            'Seguridad y Salud': {'id': 8, 'probabilidad': 2, 'riesgo_afectacion': ['costo']},
            'Funciones y Responsabilidades (RNR)': {'id': 9, 'probabilidad': 3, 'riesgo_afectacion': ['alcance', 'tiempo']},
            'Procedimientos': {'id': 10, 'probabilidad': 3, 'riesgo_afectacion': ['alcance', 'tiempo']},
            'Legal o Referencia Normativa': {'id': 11, 'probabilidad': 4, 'riesgo_afectacion': ['alcance', 'costo', 'tiempo']},
            'Temporal': {'id': 12, 'probabilidad': 2, 'riesgo_afectacion': ['costo', 'tiempo']},
        }

        # Diccionario de estrategias de mitigación por tipo de riesgo/afectación
        # Se selecciona basándose en la probabilidad: alta (4-5) = 3 opciones, media (3) = 2 opciones, baja (1-2) = 1 opción
        self.estrategias_mitigacion = {
            'tiempo': [
                'Compresión cronograma',
                'Fast-tracking actividades paralelizables',
                'Buffer management programado'
            ],
            'alcance/costo': [
                'Control integrado de cambios',
                'Análisis de valor ganado',
                'Revisión umbrales costo-alcance'
            ],
            'alcance/tiempo': [
                'Priorización valor entregable',
                'Análisis ruta crítica',
                'Entregas parciales escalonadas'
            ],
            'costo/tiempo': [
                'Nivelación recursos críticos',
                'Trade-offs tiempo-costo',
                'Estimación paramétrica optimizada'
            ],
            'alcance/costo/tiempo': [
                'Replanificación con EVA',
                'Rebaselining integral proyecto',
                'Planificación iterativa ágil'
            ],
            'costo': [
                'Control integrado de cambios',
                'Análisis de valor ganado',
                'Revisión umbrales costo-alcance'
            ]
        }

    def _load_category_phrases(self):
        """Cargar el diccionario de frases por categoría desde models/etiquetas.json"""
        try:
            # Buscar el archivo en varias ubicaciones posibles
            possible_paths = [
                "models/etiquetas.json",
                "./models/etiquetas.json",
                os.path.join(os.path.dirname(__file__), "..", "models", "etiquetas.json"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "etiquetas.json")
            ]

            etiquetas_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    etiquetas_path = path
                    break

            if etiquetas_path is None:
                print("⚠️ No se encontró models/etiquetas.json, usando etiquetas por defecto")
                self.category_phrases = {}
                return

            with open(etiquetas_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convertir formato de etiquetas.json a formato esperado por cloud_word
            self.category_phrases = {}
            for categoria, info in data.items():
                etiquetas = info.get('etiquetas', [])
                self.category_phrases[categoria] = etiquetas

            print(f"✅ Cargadas {len(self.category_phrases)} categorías de etiquetas")

        except Exception as e:
            print(f"⚠️ Error cargando etiquetas: {e}")
            self.category_phrases = {}

    def get_category_phrases_for_classification(self, clasificacion):
        """
        Obtener las frases de etiquetas específicas para una clasificación dada.

        Args:
            clasificacion (str): La clasificación obtenida del modelo

        Returns:
            dict: Diccionario con solo la categoría correspondiente a la clasificación
        """
        # Mapear la clasificación a la categoría de etiquetas
        categoria_objetivo = self.mapeo_clasificacion_categoria.get(clasificacion)

        if not categoria_objetivo:
            print(f"⚠️ No se encontró mapeo para clasificación: '{clasificacion}'")
            print(f"Mapeos disponibles: {list(self.mapeo_clasificacion_categoria.keys())}")
            return {}

        if categoria_objetivo not in self.category_phrases:
            print(f"⚠️ No se encontraron etiquetas para categoría: '{categoria_objetivo}'")
            print(f"Categorías disponibles en etiquetas.json: {list(self.category_phrases.keys())}")
            return {}

        # Retornar solo las frases de la categoría específica
        return {categoria_objetivo: self.category_phrases[categoria_objetivo]}

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
            clean_path = os.path.normpath(env_model_path)
            if os.path.exists(clean_path):
                return clean_path

        # Prioridad 2: Ruta absoluta
        if os.path.isabs(model_path):
            return model_path

        # Prioridad 3: Ruta relativa al archivo actual (desarrollo local)
        base_path = os.path.dirname(os.path.abspath(__file__))
        local_model_path = os.path.join(base_path, model_path)

        if os.path.exists(local_model_path):
            print(f"Modelo en ruta local: {local_model_path}")
            return local_model_path

        # Prioridad 4: Ruta en Railway volume
        railway_volume_path = f"/app/{model_path}"
        if os.path.exists(railway_volume_path):
            print(f"Modelo en Railway volume: {railway_volume_path}")
            return railway_volume_path

        # Fallback: retornar ruta local y dejar que falle con mensaje claro
        print(f"⚠️ No se encontró modelo, usando fallback: {local_model_path}")
        return local_model_path

    def _normalizar_y_separar_parrafos(self, texto):
        """
        Normaliza y separa párrafos de forma robusta para PDFs con texto directo y escaneados.

        Estrategia:
        1. Detectar patrones de cláusulas contractuales (CLÁUSULA, Artículo, etc.)
        2. Separar por puntos finales seguidos de mayúscula
        3. Manejar saltos de línea inconsistentes
        4. Unificar formato para ambos tipos de PDF
        """

        # Paso 1: Normalizar saltos de línea múltiples
        texto = re.sub(r'\n{3,}', '\n\n', texto)  # Máximo 2 saltos consecutivos

        # Paso 2: Unir líneas que claramente son parte del mismo párrafo
        # (líneas que terminan sin punto y la siguiente no empieza con mayúscula/número)
        lineas = texto.split('\n')
        parrafos_unidos = []
        buffer_actual = []

        for i, linea in enumerate(lineas):
            linea_limpia = linea.strip()

            # Saltar líneas vacías
            if not linea_limpia:
                if buffer_actual:
                    parrafos_unidos.append(' '.join(buffer_actual))
                    buffer_actual = []
                continue

            # Detectar inicio de nueva cláusula/sección (patrones contractuales)
            es_inicio_clausula = bool(re.match(
                r'^(CLÁUSULA|CLAUSULA|ARTÍCULO|ARTICULO|CAPÍTULO|CAPITULO|'
                r'\d+\.|[IVX]+\.|[a-z]\)|\(\d+\)|Artículo|Art\.)',
                linea_limpia,
                re.IGNORECASE
            ))

            # Si es inicio de cláusula, guardar buffer anterior
            if es_inicio_clausula and buffer_actual:
                parrafos_unidos.append(' '.join(buffer_actual))
                buffer_actual = [linea_limpia]
                continue

            # Detectar si línea termina con punto final (fin de oración completa)
            termina_con_punto = linea_limpia.endswith(('.', ':', ';', '?', '!'))

            # Siguiente línea empieza con mayúscula/número (posible nueva oración)
            siguiente_es_mayuscula = False
            if i + 1 < len(lineas):
                siguiente_limpia = lineas[i + 1].strip()
                siguiente_es_mayuscula = bool(
                    siguiente_limpia and
                    (siguiente_limpia[0].isupper() or siguiente_limpia[0].isdigit())
                )

            # Agregar línea al buffer
            buffer_actual.append(linea_limpia)

            # Decidir si cerrar párrafo
            if termina_con_punto and siguiente_es_mayuscula:
                parrafos_unidos.append(' '.join(buffer_actual))
                buffer_actual = []

        # Guardar último buffer
        if buffer_actual:
            parrafos_unidos.append(' '.join(buffer_actual))

        # Paso 3: Separar por patrones de cláusulas si aún hay párrafos muy largos
        parrafos_finales = []
        for parrafo in parrafos_unidos:
            # Si el párrafo tiene múltiples cláusulas juntas, separarlas
            if len(parrafo.split()) > 500:  # Párrafos muy largos
                # Buscar patrones de inicio de cláusula dentro del texto
                fragmentos = re.split(
                    r'(?=\b(?:CLÁUSULA|CLAUSULA|ARTÍCULO|ARTICULO)\s+[IVX\d]+)',
                    parrafo,
                    flags=re.IGNORECASE
                )
                parrafos_finales.extend([f.strip() for f in fragmentos if f.strip()])
            else:
                parrafos_finales.append(parrafo.strip())

        # Filtrar párrafos vacíos y muy cortos (menos de 3 palabras)
        parrafos_finales = [
            p for p in parrafos_finales
            if p and len(p.split()) >= 3
        ]

        return parrafos_finales

    def extraer_parrafos_y_fragmentos(self, texto, max_tokens=512):
        """
        Separar el contrato en párrafos y fragmentarlos si exceden el límite de tokens.
        """
        # Separar en párrafos usando doble salto de línea
        parrafos = self._normalizar_y_separar_parrafos(texto)

        filtro = FiltradorClausulasConstructor()
        parrafos_procesados = []
        estadisticas_filtro = {'total_original': len(parrafos), 'descartadas': 0, 'truncados': 0}

        for parrafo in parrafos:
            parrafo = parrafo.strip()

            # Filtrar primero
            filtro_clausula = filtro.filtrar_clausula(parrafo)

            if filtro_clausula['es_relevante']:
                # Tokenizar el párrafo
                tokens = self.tokenizer.encode(parrafo, add_special_tokens=False)

                # Si el párrafo excede el límite de tokens, fragmentarlo
                if len(tokens) > max_tokens:
                    estadisticas_filtro['truncados'] += 1
                    fragmentos = self._fragmentar_en_oraciones(parrafo, max_tokens)

                    parrafos_procesados.append({
                        'parrafo': fragmentos,
                        'truncado': True,
                        'num_fragmentos': len(fragmentos),
                        'filtro_info': filtro_clausula
                    })
                else:
                    parrafos_procesados.append({
                        'parrafo': [parrafo],
                        'truncado': False,
                        'num_fragmentos': 1
                    })
            else:
                estadisticas_filtro['descartadas'] += 1

        estadisticas_filtro['procesadas'] = len(parrafos_procesados)
        estadisticas_filtro['porcentaje_retenido'] = round(
            (estadisticas_filtro['procesadas'] / estadisticas_filtro['total_original']) * 100, 1
        ) if estadisticas_filtro['total_original'] > 0 else 0

        print(f"📊 Filtrado completado: {estadisticas_filtro['procesadas']}/{estadisticas_filtro['total_original']} cláusulas retenidas ({estadisticas_filtro['porcentaje_retenido']}%)")
        if estadisticas_filtro['truncados'] > 0:
            print(f"   ℹ️ {estadisticas_filtro['truncados']} párrafos fragmentados inteligentemente")

        return parrafos_procesados

    def _fragmentar_en_oraciones(self, texto, max_tokens=512):
        """
        MEJORADO: Fragmenta en puntos naturales (oraciones) en lugar de cortes simples
        Preserva contexto y semántica
        """
        # Separar por oraciones (puntos, signos de exclamación, interrogación)
        oraciones = re.split(r'(?<=[.!?])\s+', texto)

        fragmentos = []
        fragmento_actual = []
        tokens_actuales = 0

        for oracion in oraciones:
            # Contar tokens de esta oración
            tokens_oracion = len(self.tokenizer.encode(oracion, add_special_tokens=False))

            # Si agregar esta oración excedería el límite
            if tokens_actuales + tokens_oracion > max_tokens and fragmento_actual:
                # Guardar fragmento anterior
                fragmentos.append(' '.join(fragmento_actual))
                fragmento_actual = [oracion]
                tokens_actuales = tokens_oracion
            else:
                fragmento_actual.append(oracion)
                tokens_actuales += tokens_oracion

        # Agregar último fragmento
        if fragmento_actual:
            fragmentos.append(' '.join(fragmento_actual))

        # Si no pudo fragmentar (oración muy larga), usar decode del tokenizador
        if not fragmentos:
            tokens = self.tokenizer.encode(texto, add_special_tokens=False)
            fragmentos.append(self.tokenizer.decode(tokens[:max_tokens]))
            if len(tokens) > max_tokens:
                fragmentos.append(self.tokenizer.decode(tokens[max_tokens:]))

        return fragmentos


    def clasificar_clausula(self, texto):
        """Clasificar una cláusula individual - CON truncamiento explícito"""
        # Tokenizar CON truncamiento automático a 512 tokens
        inputs = self.tokenizer(
            texto,
            return_tensors="pt",
            truncation=True,           # AGREGADO: truncar automáticamente
            padding=True,
            max_length=512             # Aplicar límite aquí
        )

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

    def analizar_contrato_completo(self, texto_contrato, max_tokens_por_clausula=512, modo="fast"):
        """
        Analizar todo el contrato y generar reporte de riesgos

        Args:
            texto_contrato (str): Texto del contrato a analizar
            max_tokens_por_clausula (int): Máximo de tokens por cláusula
            modo (str): Modo de análisis ('fast', 'balanced', 'detailed')
        """
        # Obtener configuración del modo
        if modo not in self.MODOS_VELOCIDAD:
            print(f"⚠️ Modo '{modo}' no reconocido, usando 'balanced'")
            modo = "balanced"

        config = self.MODOS_VELOCIDAD[modo]
        print(f"🔧 Usando modo '{modo}': {config}")

        clausulas = self.extraer_parrafos_y_fragmentos(texto_contrato, max_tokens_por_clausula)

        if not clausulas:
            return {
                'error': 'No se encontraron párrafos en el formato esperado',
                'clausulas_encontradas': 0,
                'modo_utilizado': modo
            }

        resultados = []
        riesgos_encontrados = defaultdict(list)
        estadisticas = Counter()

        for i, clausula in enumerate(clausulas, 1):
            # Previo: clasificacion = self.clasificar_clausula(clausula['parrafo'])

            # Convertir clausula['parrafo'] (lista) a string para clasificación
            texto_clausula = ' '.join(clausula['parrafo']) if isinstance(clausula['parrafo'], list) else clausula['parrafo']

            clasificacion = self.clasificar_clausula(texto_clausula)

            # Usar la confianza mínima del modo seleccionado
            if clasificacion['confianza'] < config['min_confidence']:
                continue

            # Generar matched_phrases SOLO para la categoría clasificada
            matched_phrases = []
            if hasattr(self, 'category_phrases') and self.category_phrases:
                try:
                    # Obtener solo las etiquetas de la categoría específica
                    category_phrases_filtered = self.get_category_phrases_for_classification(clasificacion['etiqueta'])

                    if category_phrases_filtered:
                        matched_phrases = match_phrases_for_clause(
                            texto_clausula,
                            category_phrases_filtered,
                            top_n=config['top_n_phrases'],
                            min_score=config['min_score'],
                            max_ngram=config['max_ngram'],
                            use_embeddings=config['use_embeddings']
                        )

                        if matched_phrases:
                            print(f"   ✅ Etiquetas encontradas: {[mp['phrase'] for mp in matched_phrases]}")
                        else:
                            print(f"   ❌ No se encontraron etiquetas válidas")
                    else:
                        print(f"⚠️ No se encontraron etiquetas para clasificación: {clasificacion['etiqueta']}")

                except Exception as e:
                    print(f"⚠️ Error generando matched_phrases para cláusula {i}: {e}")
                    import traceback
                    traceback.print_exc()

            resultado_clausula = {
                'numero': i,
                'contenido': texto_clausula[:200] + "..." if len(texto_clausula) > 200 else texto_clausula,
                'truncado': clausula['truncado'],
                'clasificacion': clasificacion['etiqueta'],
                'confianza': round(clasificacion['confianza'], 3),
                'matched_phrases': matched_phrases
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

        # Generar payload para wordcloud
        wordcloud_payload = []
        try:
            wordcloud_payload = build_wordcloud_payload_from_clauses(resultados, phrase_key='matched_phrases', min_score=0.0)
        except Exception as e:
            print(f"⚠️ Error generando wordcloud payload: {e}")

        # Generar matriz de riesgos
        risk_matrix = self._generar_matriz_riesgos(resultados)

        return {
            'total_clausulas': len(clausulas),
            'clausulas_analizadas': resultados,
            'riesgos_por_nivel': dict(riesgos_encontrados),
            'estadisticas_tipos': dict(estadisticas),
            'resumen_riesgos': self._generar_resumen_riesgos(riesgos_encontrados, estadisticas),
            'wordcloud': wordcloud_payload,
            'risk_matrix': risk_matrix,
            'modo_utilizado': modo,
            'configuracion_utilizada': config
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

    def _seleccionar_mitigaciones(self, riesgo_afectacion, probabilidad):
        """
        Seleccionar estrategias de mitigación basadas en el tipo de riesgo/afectación y probabilidad.

        Args:
            riesgo_afectacion (list): Lista de tipos de riesgo (ej: ['alcance', 'costo'])
            probabilidad (int): Nivel de probabilidad (1-5)

        Returns:
            str: Estrategias de mitigación formateadas para mostrar
        """
        # Normalizar el tipo de riesgo/afectación a clave del diccionario
        if len(riesgo_afectacion) == 1:
            clave_riesgo = riesgo_afectacion[0]
        else:
            # Convertir lista a clave ordenada (ej: ['alcance', 'costo'] -> 'alcance/costo')
            clave_riesgo = '/'.join(sorted(riesgo_afectacion))

        # Obtener las estrategias para este tipo de riesgo
        estrategias = self.estrategias_mitigacion.get(clave_riesgo, [])

        if not estrategias:
            return 'Mitigación no disponible'

        # Determinar cantidad de estrategias según probabilidad
        if probabilidad >= 4:  # Alta (4-5)
            cantidad = 3
        elif probabilidad == 3:  # Media (3)
            cantidad = 2
        else:  # Baja (1-2)
            cantidad = 1

        # Seleccionar las primeras N estrategias
        mitigaciones_seleccionadas = estrategias[:cantidad]

        # Formatear para mostrar con numeración
        return '\n'.join([f"{i+1}. {estrategia}" for i, estrategia in enumerate(mitigaciones_seleccionadas)])

    def _generar_matriz_riesgos(self, clausulas_analizadas):
        """
        Generar matriz de riesgos basada en las cláusulas encontradas

        Args:
            clausulas_analizadas: Lista de cláusulas analizadas

        Returns:
            List: Matriz de riesgos con formato para frontend
        """
        # Agrupar cláusulas por clasificación
        clausulas_por_tipo = defaultdict(list)
        for clausula in clausulas_analizadas:
            clasificacion = clausula['clasificacion']
            clausulas_por_tipo[clasificacion].append(clausula)

        matriz_riesgos = []

        for clasificacion, clausulas_del_tipo in clausulas_por_tipo.items():
            # Obtener información del mapeo
            info_mapeo = self.matriz_riesgos_mapeo.get(clasificacion, {})

            if not info_mapeo:
                print(f"⚠️ No se encontró mapeo para clasificación: {clasificacion}")
                continue

            # Generar ID con enumeración por letras (a, b, c, etc.)
            for i, clausula in enumerate(clausulas_del_tipo):
                letra = chr(ord('a') + i)  # Convertir índice a letra (0->a, 1->b, etc.)

                # Seleccionar mitigaciones basadas en riesgo/afectación y probabilidad
                mitigacion = self._seleccionar_mitigaciones(
                    info_mapeo['riesgo_afectacion'],
                    info_mapeo['probabilidad']
                )

                matriz_riesgos.append({
                    'id': f"{info_mapeo['id']}.{letra}",
                    'categoria': clasificacion,
                    'probabilidad': info_mapeo['probabilidad'],
                    'impacto': clausula.get('nivel_riesgo', 'desconocido'),
                    'riesgo_afectacion': info_mapeo['riesgo_afectacion'],
                    'mitigacion': mitigacion,  # Ahora generado dinámicamente
                    'responsable': ''  # Campo editable vacío
                })

        # Ordenar por ID numérico para mantener orden lógico
        # matriz_riesgos.sort(key=lambda x: (int(x['id'].split('.')[0]), x['id'].split('.')[1]))

        return matriz_riesgos

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

    # Probar diferentes modos
    for modo in ["fast", "balanced", "detailed"]:
        print(f"\n{'='*50}")
        print(f"ANALIZANDO CON MODO: {modo.upper()}")
        print(f"{'='*50}")

        # Analizar contrato
        resultado = analizador.analizar_contrato_completo(texto_contrato, modo=modo)

        # Mostrar información del modo utilizado
        if 'configuracion_utilizada' in resultado:
            config = resultado['configuracion_utilizada']
            print(f"🔧 Configuración: {config}")

        print(f"📊 Cláusulas analizadas: {len(resultado['clausulas_analizadas'])}")

        # Mostrar matched_phrases por cláusula
        for clausula in resultado['clausulas_analizadas']:
            print(f"\n📄 Cláusula {clausula['numero']} ({clausula['clasificacion']}):")
            print(f"   Confianza: {clausula['confianza']}")
            if clausula['matched_phrases']:
                print("   Etiquetas encontradas:")
                for mp in clausula['matched_phrases']:
                    print(f"   • {mp['phrase']} (score: {mp['score']:.3f}, método: {mp['method']})")
            else:
                print("   • No se encontraron etiquetas específicas")

if __name__ == "__main__":
    main()