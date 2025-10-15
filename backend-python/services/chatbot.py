"""
Servicio de Chatbot usando OpenAI
"""
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from datetime import datetime

class ChatbotService:
    """Servicio para manejar conversaciones con OpenAI"""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            print("⚠️ OPENAI_API_KEY no configurada")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
            print("✓ Cliente OpenAI inicializado")

        # Configuración del modelo
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_tokens = int(os.getenv("CHATBOT_MAX_TOKENS", 1000))
        self.temperature = float(os.getenv("CHATBOT_TEMPERATURE", 0.7))

        # System prompt especializado en contratos
        self.system_prompt = self._get_system_prompt()

    def disponible(self) -> bool:
        """Verificar si el servicio está disponible"""
        return self.client is not None

    async def chat(
        self,
        mensaje: str,
        historial: Optional[List[Any]] = None,
        contexto_contrato: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enviar mensaje al chatbot y obtener respuesta

        Args:
            mensaje: Mensaje del usuario
            historial: Historial de conversación previo (opcional)
            contexto_contrato: Información del contrato analizado (opcional)

        Returns:
            Dict con respuesta y metadata
        """
        if not self.disponible():
            raise Exception("Servicio de chatbot no disponible. Configure OPENAI_API_KEY")

        try:
            # Construir mensajes (historial puede venir como dicts o como Pydantic models)
            messages = self._construir_mensajes(mensaje, historial, contexto_contrato)

            # Llamar a OpenAI
            inicio = datetime.now()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            tiempo_respuesta = (datetime.now() - inicio).total_seconds()

            # Extraer respuesta
            respuesta_bot = response.choices[0].message.content

            return {
                "respuesta": respuesta_bot,
                "modelo": self.model,
                "tokens_usados": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                },
                "tiempo_respuesta": round(tiempo_respuesta, 2),
                "finish_reason": response.choices[0].finish_reason
            }

        except Exception as e:
            raise Exception(f"Error en chatbot: {str(e)}")

    async def chat_streaming(
        self,
        mensaje: str,
        historial: Optional[List[Any]] = None,
        contexto_contrato: Optional[Dict[str, Any]] = None
    ):
        """
        Streaming de respuesta del chatbot (para respuestas en tiempo real)

        Yields:
            Chunks de texto de la respuesta
        """
        if not self.disponible():
            raise Exception("Servicio de chatbot no disponible")

        try:
            messages = self._construir_mensajes(mensaje, historial, contexto_contrato)

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise Exception(f"Error en streaming: {str(e)}")

    # ==================== MÉTODOS INTERNOS ====================

    def _construir_mensajes(
        self,
        mensaje: str,
        historial: Optional[List[Any]],
        contexto_contrato: Optional[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Construir lista de mensajes para OpenAI"""

        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Agregar contexto del contrato si existe
        if contexto_contrato:
            contexto_msg = self._formatear_contexto_contrato(contexto_contrato)
            messages.append({
                "role": "system",
                "content": f"Contexto del contrato analizado:\n{contexto_msg}"
            })

        # Agregar historial de conversación
        if historial:
            # Limitar historial a últimos N mensajes para no exceder tokens
            max_historial = 10
            historial_reciente = historial[-max_historial:] if len(historial) > max_historial else historial
            # Aceptar tanto dicts como objetos (p. ej. Pydantic MensajeHistorial)
            for msg in historial_reciente:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                else:
                    # objeto con atributos role/content (pydantic model)
                    role = getattr(msg, "role", "user")
                    content = getattr(msg, "content", "")

                messages.append({
                    "role": role,
                    "content": content
                })

        # Agregar mensaje actual
        messages.append({
            "role": "user",
            "content": mensaje
        })

        return messages

    def _formatear_contexto_contrato(self, contexto: Dict[str, Any]) -> str:
        """Formatear información del contrato para el contexto"""

        partes = []

        # Resumen general
        if "total_clausulas" in contexto:
            partes.append(f"Total de cláusulas analizadas: {contexto['total_clausulas']}")

        if "clausulas_abusivas" in contexto:
            partes.append(f"Cláusulas problemáticas: {contexto['clausulas_abusivas']}")

        if "porcentaje_abusivas" in contexto:
            partes.append(f"Porcentaje de riesgo: {contexto['porcentaje_abusivas']}%")

        # Resumen de riesgos
        if "resumen" in contexto and isinstance(contexto["resumen"], dict):
            resumen = contexto["resumen"]

            if "riesgos_por_nivel" in resumen:
                riesgos = resumen["riesgos_por_nivel"]
                partes.append(f"\nRiesgos detectados:")
                partes.append(f"- Alto: {riesgos.get('alto', 0)}")
                partes.append(f"- Medio: {riesgos.get('medio', 0)}")
                partes.append(f"- Bajo: {riesgos.get('bajo', 0)}")

            if "recomendaciones" in resumen and resumen["recomendaciones"]:
                partes.append(f"\nRecomendaciones principales:")
                for rec in resumen["recomendaciones"][:3]:  # Solo las 3 primeras
                    partes.append(f"- {rec}")

        # Cláusulas específicas de riesgo alto
        if "clausulas" in contexto:
            clausulas_alto_riesgo = [
                c for c in contexto["clausulas"]
                if c.get("severidad") == "alto"
            ]

            if clausulas_alto_riesgo:
                partes.append(f"\nCláusulas de RIESGO ALTO:")
                for clausula in clausulas_alto_riesgo[:3]:  # Máximo 3
                    tipo = clausula.get("clasificacion", "Desconocido")
                    texto = clausula.get("texto", "")[:150]
                    partes.append(f"- Tipo: {tipo}")
                    partes.append(f"  Extracto: {texto}...")

        return "\n".join(partes)

    def _get_system_prompt(self) -> str:
        """Obtener el system prompt especializado"""

        return """Eres un asistente legal experto especializado en análisis de contratos.

Tu rol es ayudar a los usuarios a:
1. Entender los resultados del análisis de contratos
2. Explicar términos legales complejos de forma clara
3. Proporcionar recomendaciones sobre cláusulas problemáticas
4. Responder preguntas sobre derechos y obligaciones en el contrato

IMPORTANTE:
- Proporciona información legal general, NO asesoramiento legal específico
- Recomienda consultar con un abogado para casos específicos
- Sé claro, conciso y profesional
- Usa lenguaje accesible evitando jerga legal innecesaria
- Si el usuario pregunta sobre una cláusula específica, referencia el contexto del contrato si está disponible
- Enfócate en riesgos y oportunidades de mejora en las cláusulas

Si no tienes contexto de un contrato analizado, puedes responder preguntas generales sobre contratos y términos legales."""

    def actualizar_system_prompt(self, nuevo_prompt: str):
        """Actualizar el system prompt (útil para personalización)"""
        self.system_prompt = nuevo_prompt