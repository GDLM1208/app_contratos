import { useState, useEffect } from 'react'
import UploadForm from './components/UploadForm.tsx'
import ResultsTable from './components/ResultsTable.tsx'
import RiskBarChart from './components/RiskBarChart.tsx'
import WordCloudChart from './components/WordCloudChart.tsx'
import RiskMatrixTable from './components/RiskMatrixTable.tsx'
import Chatbot from './components/Chatbot.tsx'
import HistoryList from './components/HistoryList.tsx'
import AnalysisInfoBanner from './components/AnalysisInfoBanner.tsx'

type Section = 'analyzer' | 'chatbot' | 'history'

function App() {
  const [activeSection, setActiveSection] = useState<Section>('analyzer')
  type Clause = {
    numero: number;
    contenido: string;
    truncado?: boolean;
    clasificacion?: string;
    confianza?: number;
    nivel_riesgo?: string;
    matched_phrases?: Array<{phrase: string; score: number; method: string}>;
  }

  type RiskMatrixRow = {
    id: string;
    categoria: string;
    probabilidad: number;
    impacto: string;
    riesgo_afectacion: string[];
    mitigacion: string;
    responsable: string;
  }

  type AnalysisData = {
    total_clausulas?: number;
    clausulas_analizadas?: Clause[];
    wordcloud?: Array<{text: string; value: number}>;
    risk_matrix?: RiskMatrixRow[];
    [key: string]: unknown;
  }

  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null)
  const [documentInfo, setDocumentInfo] = useState<{filename?: string; timestamp?: string} | null>(null)
  const [analysisSource, setAnalysisSource] = useState<'history' | 'new' | null>(null)

  // Función para cargar análisis desde el historial
  const handleLoadFromHistory = (backendData: {
    total_clausulas: number
    clausulas_analizadas: Array<{numero: number; contenido: string; nivel_riesgo: string; clasificacion?: string; confianza?: number; matched_phrases?: Array<{phrase: string; score: number; method: string}>}>
    wordcloud?: Array<{text: string; value: number}>
    risk_matrix?: Array<{id: string; categoria: string; probabilidad: number; impacto: string; riesgo_afectacion?: string[]; mitigacion?: string; responsable?: string}>
    filename: string
    timestamp: string
  }, historyDocumentInfo: {filename: string; timestamp: string}) => {
    // Usar los datos directamente del backend con formato correcto
    const transformedData: AnalysisData = {
      total_clausulas: backendData.total_clausulas,
      clausulas_analizadas: backendData.clausulas_analizadas || [],
      wordcloud: backendData.wordcloud || [],
      risk_matrix: backendData.risk_matrix?.map((rm) => ({
        id: rm.id,
        categoria: rm.categoria,
        probabilidad: rm.probabilidad,
        impacto: rm.impacto,
        riesgo_afectacion: rm.riesgo_afectacion || [],
        mitigacion: rm.mitigacion || '',
        responsable: rm.responsable || ''
      })) || []
    }

    setAnalysisData(transformedData)
    setDocumentInfo(historyDocumentInfo)
    setAnalysisSource('history')
    setActiveSection('analyzer')

    console.log('✅ Análisis cargado desde historial')
  }

  useEffect(() => {
    const handler = (e: Event) => {
      const event = e as CustomEvent
      const detail = event.detail.data as AnalysisData
      const filename = event.detail.filename
      const timestamp = event.detail.timestamp

      setAnalysisData(detail)
      setDocumentInfo({ filename, timestamp })
      setAnalysisSource('new')
      setActiveSection('analyzer')
    }

    // Handler para reset cuando se inicie un nuevo análisis
    const resetHandler = () => {
      setAnalysisData(null)
      setDocumentInfo(null)
      setAnalysisSource(null)
    }

    window.addEventListener('analysis:completed', handler as EventListener)
    window.addEventListener('analysis:started', resetHandler as EventListener)
    return () => {
      window.removeEventListener('analysis:completed', handler as EventListener)
      window.removeEventListener('analysis:started', resetHandler as EventListener)
    }
  }, [])

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
              <UploadForm />

              {/* Mostrar información del análisis actual si existe */}
              {analysisData && documentInfo && analysisSource && (
                <AnalysisInfoBanner
                  analysisInfo={{
                    filename: documentInfo.filename || 'Archivo desconocido',
                    timestamp: documentInfo.timestamp || new Date().toISOString(),
                    total_clausulas: analysisData.total_clausulas || 0,
                    source: analysisSource
                  }}
                />
              )}
            </div>

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
                  <path fillRule="evenodd" d="M9.504 1.132a1 1 0 01.992 0l1.75 1a1 1 0 11-.992 1.736L10 3.152l-1.254.716a1 1 0 11-.992-1.736l1.75-1zM5.618 4.504a1 1 0 01-.372 1.364L5.016 6l.23.132a1 1 0 11-.992 1.736L3 7.723V8a1 1 0 01-2 0V6a.996.996 0 01.52-.878l1.734-.99a1 1 0 011.364.372zm8.764 0a1 1 0 011.364-.372l1.734.99A.996.996 0 0118 6v2a1 1 0 11-2 0v-.277l-1.254.145a1 1 0 11-.992-1.736L14.984 6l-.23-.132a1 1 0 01-.372-1.364zm-7 4a1 1 0 011.364-.372L10 8.848l1.254-.716a1 1 0 11.992 1.736L11 10.723V12a1 1 0 11-2 0v-1.277l-1.246-.855a1 1 0 01-.372-1.364zM3 11a1 1 0 011 1v1.277l1.254.716a1 1 0 11-.992 1.736l-1.75-1A1 1 0 012 14v-2a1 1 0 011-1zm14 0a1 1 0 011 1v2a1 1 0 01-.504.868l-1.75 1a1 1 0 11-.992-1.736L16 13.277V12a1 1 0 011-1zm-9.618 5.504a1 1 0 011.364-.372l.254.145V16a1 1 0 112 0v.277l.254-.145a1 1 0 11.992 1.736l-1.75 1a.996.996 0 01-.992 0l-1.75-1a1 1 0 01-.372-1.364z" clipRule="evenodd" />
                </svg>
                Nube de Palabras Clave
              </h2>
              <div className="card" style={{ padding: '1.5rem', display: 'flex', justifyContent: 'center' }}>
                <WordCloudChart
                  data={analysisData?.wordcloud}
                  width={600}
                  height={400}
                />
              </div>
            </div>

            <div className="section-card">
              <h2 className="section-header">
                <svg className="section-icon" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" clipRule="evenodd" />
                </svg>
                Check Risk Register (CRR)
              </h2>
              <div className="card" style={{ padding: '1.5rem' }}>
                <RiskMatrixTable
                  data={analysisData?.risk_matrix}
                />
              </div>
            </div>

            <div className="section-card">
              <h2 className="section-header">
                <svg className="section-icon" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
                </svg>
                RTAM (Risk and Term Assessment Module)
              </h2>
              <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
                <ResultsTable rows={
                  analysisData?.clausulas_analizadas?.map((c: Clause) => ({
                    id: c.numero,
                    descripcion: c.contenido,
                    impacto: (c.nivel_riesgo || 'Desconocido'),
                    etiquetas: c.matched_phrases?.map(mp => mp.phrase).join(', ') || '',
                    comentarios: `Clasificación: ${c.clasificacion || 'N/A'} (Confianza: ${c.confianza ?? 'N/A'})`
                  })) || undefined
                } />
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
            <div className="card" style={{ padding: '1.5rem' }}>
              <HistoryList onLoadAnalysis={handleLoadFromHistory} />
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
