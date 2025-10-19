import { useState, useRef, useEffect } from 'react';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  isLoading?: boolean;
}

const Chatbot = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: '¡Hola! Soy tu asistente de análisis de contratos. Puedo ayudarte a identificar riesgos, explicar cláusulas y responder preguntas sobre tus documentos. ¿En qué puedo ayudarte hoy?',
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputText,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = inputText;
    setInputText('');
    setIsTyping(true);

    try {
      // Preparar historial de conversación para el backend
      const conversationHistory = messages.map(msg => ({
        role: msg.sender === 'user' ? 'user' : 'assistant',
        content: msg.text
      }));

      // Llamar al backend
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          mensaje: currentInput,
          historial: conversationHistory
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('📋 Respuesta completa del backend:', data);

      if (data.success && data.message) {
        const botMessage: Message = {
          id: (Date.now() + 1).toString(),
          text: data.message,
          sender: 'bot',
          timestamp: new Date()
        };

        // Log adicional para debug
        console.log('✅ Mensaje del bot creado:', botMessage);
        console.log('📊 Metadata adicional:', data.data);

        setMessages(prev => [...prev, botMessage]);
      } else {
        throw new Error(data.error || 'Respuesta inválida del servidor');
      }
    } catch (error) {
      console.error('❌ Error en chatbot:', error);

      let errorMessage = 'Lo siento, hubo un error al procesar tu mensaje.';

      if (error instanceof Error) {
        if (error.message.includes('HTTP 503')) {
          errorMessage = 'El servicio de chatbot no está disponible. Por favor, verifique la configuración.';
        } else if (error.message.includes('HTTP 400')) {
          errorMessage = 'Mensaje inválido. Por favor, intenta con un mensaje diferente.';
        } else if (error.message.includes('HTTP')) {
          errorMessage = `Error del servidor: ${error.message}`;
        }
      }

      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        text: errorMessage + ' Por favor, intenta nuevamente.',
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const suggestedQuestions = [
    '¿Qué tipos de riesgos debo revisar en mi contrato?',
    '¿Cómo evaluar las penalidades por retraso?',
    '¿Qué cláusulas de responsabilidad son importantes?',
    '¿Cómo proteger mis intereses en el contrato?'
  ];

  const handleSuggestedQuestion = (question: string) => {
    setInputText(question);
  };

  return (
    <div className="chatbot-container">
      {/* Header del Chat */}
      <div className="chatbot-header">
        <div className="chatbot-avatar">
          <span className="avatar-icon">🤖</span>
        </div>
        <div className="chatbot-info">
          <h3 className="chatbot-title">Asistente de Contratos</h3>
          <p className="chatbot-subtitle">
            {isTyping ? 'Escribiendo...' : 'En línea'}
          </p>
        </div>
        <div className="chatbot-status">
          <div className={`status-indicator ${isTyping ? 'typing' : 'online'}`}></div>
        </div>
      </div>

      {/* Área de Mensajes */}
      <div className="messages-container">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}
          >
            {message.sender === 'bot' && (
              <div className="message-avatar">
                <span className="avatar-icon">🤖</span>
              </div>
            )}
            <div className="message-content">
              <div className="message-bubble">
                <p className="message-text">{message.text}</p>
                <span className="message-time">
                  {message.timestamp.toLocaleTimeString('es-ES', {
                    hour: '2-digit',
                    minute: '2-digit'
                  })}
                </span>
              </div>
            </div>
            {message.sender === 'user' && (
              <div className="message-avatar">
                <span className="avatar-icon">👤</span>
              </div>
            )}
          </div>
        ))}

        {isTyping && (
          <div className="message bot-message">
            <div className="message-avatar">
              <span className="avatar-icon">🤖</span>
            </div>
            <div className="message-content">
              <div className="message-bubble typing">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Preguntas Sugeridas */}
      {messages.length === 1 && (
        <div className="suggested-questions">
          <p className="suggestions-title">Preguntas sugeridas:</p>
          <div className="suggestions-grid">
            {suggestedQuestions.map((question, index) => (
              <button
                key={index}
                className="suggestion-button"
                onClick={() => handleSuggestedQuestion(question)}
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Área de Input */}
      <div className="input-container">
        <div className="input-wrapper">
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Escribe tu pregunta sobre el contrato..."
            className="message-input"
            rows={1}
            disabled={isTyping}
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputText.trim() || isTyping}
            className="send-button"
          >
            <svg className="send-icon" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </button>
        </div>
        <p className="input-hint">
          Presiona Enter para enviar, Shift+Enter para nueva línea
        </p>
      </div>
    </div>
  );
};

export default Chatbot;
