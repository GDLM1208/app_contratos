import { useState } from 'react'
import UploadForm from './components/UploadForm.tsx'
import ResultsTable from './components/ResultsTable.tsx'
import RiskBarChart from './components/RiskBarChart.tsx'
import Chatbot from './components/Chatbot.tsx'

type Section = 'analyzer' | 'chatbot' | 'history'

function App() {
  const [extractedText, setExtractedText] = useState('')
  const [activeSection, setActiveSection] = useState<Section>('analyzer')

  const sections = [
    { id: 'analyzer', name: 'Analizador'},
    { id: 'chatbot', name: 'Chatbot'},
    { id: 'history', name: 'Historial'},
  ]

  const renderSection = () => {
    switch (activeSection) {
      case 'analyzer':
        return (
          <>
            <div className="section-card">
              <h2 className="section-header">
                <svg className="section-icon" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
                </svg>
                Subir Documento
              </h2>
              <UploadForm onExtractedText={setExtractedText} />
            </div>

            {extractedText && (
              <div className="section-card">
                <h2 className="section-header">
                  <svg className="section-icon" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                  </svg>
                  Texto Extraído del PDF
                </h2>
                <div className="extracted-text scrollbar-thin">
                  {extractedText}
                </div>
              </div>
            )}

            <div className="section-card">
              <h2 className="section-header">
                <svg className="section-icon" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M3 3a1 1 0 000 2v8a2 2 0 002 2h2.586l-1.293 1.293a1 1 0 101.414 1.414L10 15.414l2.293 2.293a1 1 0 001.414-1.414L12.414 15H15a2 2 0 002-2V5a1 1 0 100-2H3zm11.707 4.707a1 1 0 00-1.414-1.414L10 9.586 8.707 8.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                Análisis de Riesgos
              </h2>
              <div className="card" style={{ padding: '1.5rem' }}>
                <RiskBarChart />
              </div>
            </div>

            <div className="section-card">
              <h2 className="section-header">
                <svg className="section-icon" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
                </svg>
                Detalle de Riesgos Detectados
              </h2>
              <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
                <ResultsTable />
              </div>
            </div>
          </>
        )
      case 'chatbot':
        return (
          <div className="section-card">
            <h2 className="section-header">
              <svg className="section-icon" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.338-3.123C2.493 12.767 2 11.434 2 10c0-3.866 3.582-7 8-7s8 3.134 8 7zM7 9H5v2h2V9zm8 0h-2v2h2V9zM9 9h2v2H9V9z" clipRule="evenodd" />
              </svg>
              Chatbot de Análisis
            </h2>
            <Chatbot />
          </div>
        )
      case 'history':
        return (
          <div className="section-card">
            <h2 className="section-header">
              <svg className="section-icon" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
              </svg>
              Historial de Análisis
            </h2>
            <div className="placeholder-section">
              <span className="placeholder-icon">📊</span>
              <h3 className="placeholder-title">Historial</h3>
              <p className="placeholder-description">
                Ver todos los análisis anteriores y comparar resultados
              </p>
            </div>
          </div>
        )
      default:
        return null
    }
  }

  return (
    <div style={{ maxWidth: '72rem', margin: '0 auto', padding: '2rem', minHeight: '100vh' }}>
      {/* Header Moderno */}
      <div className="card" style={{ padding: '2rem', marginBottom: '2rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <h1 style={{ 
            background: 'linear-gradient(135deg, #2563eb, #1d4ed8)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            marginBottom: '0.5rem',
            fontSize: '2.5rem',
            fontWeight: 700
          }}>
            Análisis de Contratos
          </h1>
          <p style={{ color: '#64748b', fontSize: '1.125rem', marginBottom: '2rem' }}>
            Plataforma inteligente para detectar riesgos en contratos de construcción
          </p>
        </div>

        {/* Navegación */}
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          gap: '0.5rem',
          flexWrap: 'wrap'
        }}>
          {sections.map((section) => (
            <button
              key={section.id}
              onClick={() => setActiveSection(section.id as Section)}
              className={`nav-button ${activeSection === section.id ? 'active' : ''}`}
            >
              {section.name}
            </button>
          ))}
        </div>
      </div>

      {/* Contenido de la sección activa */}
      {renderSection()}
    </div>
  )
}

export default App
