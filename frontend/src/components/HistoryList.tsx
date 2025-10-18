import { useState, useEffect, useCallback } from 'react'

interface AnalisisHistorialItem {
  id: number
  filename: string
  timestamp: string
  total_clausulas: number
  riesgos_alto: number
  riesgos_medio: number
  riesgos_bajo: number
  modo_utilizado?: string
}

interface DocumentInfo {
  filename: string
  timestamp: string
}

interface AnalysisData {
  total_clausulas: number
  clausulas_analizadas: Array<{
    numero: number
    contenido: string
    nivel_riesgo: string
    clasificacion?: string
    confianza?: number
    matched_phrases?: Array<{phrase: string; score: number; method: string}>
  }>
  wordcloud?: Array<{
    text: string
    value: number
  }>
  risk_matrix?: Array<{
    id: string
    categoria: string
    probabilidad: number
    impacto: string
    riesgo_afectacion?: string[]
    mitigacion?: string
    responsable?: string
  }>
  filename: string
  timestamp: string
}

interface HistoryListProps {
  onLoadAnalysis: (analysisData: AnalysisData, documentInfo: DocumentInfo) => void
}

const HistoryList = ({ onLoadAnalysis }: HistoryListProps) => {
  const [historial, setHistorial] = useState<AnalisisHistorialItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [loadingAnalysis, setLoadingAnalysis] = useState<number | null>(null)

  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

  const cargarHistorial = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)

      const response = await fetch(`${API_URL}/api/analisis/historial?limit=50`)

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const data = await response.json()

      if (data.success) {
        setHistorial(data.data || [])
      } else {
        throw new Error(data.error || 'Error obteniendo historial')
      }
    } catch (err) {
      console.error('Error cargando historial:', err)
      setError(err instanceof Error ? err.message : 'Error desconocido')
    } finally {
      setLoading(false)
    }
  }, [API_URL])

  useEffect(() => {
    cargarHistorial()
  }, [cargarHistorial])

  const cargarAnalisis = async (analisisId: number) => {
    try {
      setLoadingAnalysis(analisisId)
      setError(null)

      const response = await fetch(`${API_URL}/api/analisis/${analisisId}`)

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const result = await response.json()

      if (result.success && result.data) {
        // Preparar la información del documento
        const documentInfo = {
          filename: result.data.filename,
          timestamp: result.data.timestamp
        }

        // Llamar al callback para cargar el análisis en el componente principal
        onLoadAnalysis(result.data, documentInfo)

        console.log('✅ Análisis cargado desde historial:', analisisId)
      } else {
        throw new Error(result.error || 'Error recuperando análisis')
      }
    } catch (err) {
      console.error('Error cargando análisis:', err)
      setError(err instanceof Error ? err.message : 'Error cargando análisis')
    } finally {
      setLoadingAnalysis(null)
    }
  }

  const formatearFecha = (timestamp: string) => {
    try {
      const fecha = new Date(timestamp)
      return fecha.toLocaleString('es-ES', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      })
    } catch {
      return timestamp
    }
  }

  const obtenerColorRiesgo = (alto: number, medio: number, bajo: number) => {
    const total = alto + medio + bajo
    if (total === 0) return '#6b7280'

    const porcentajeAlto = (alto / total) * 100
    if (porcentajeAlto > 30) return '#ef4444'
    if (porcentajeAlto > 15) return '#f59e0b'
    return '#10b981'
  }

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        padding: '3rem',
        flexDirection: 'column',
        gap: '1rem'
      }}>
        <div style={{
          width: '40px',
          height: '40px',
          border: '4px solid #f3f4f6',
          borderTop: '4px solid #3b82f6',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite'
        }}></div>
        <p style={{ color: '#6b7280', margin: 0 }}>Cargando historial...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div style={{
        padding: '2rem',
        textAlign: 'center',
        backgroundColor: '#fef2f2',
        borderRadius: '8px',
        border: '1px solid #fecaca'
      }}>
        <svg width="48" height="48" fill="#ef4444" viewBox="0 0 20 20" style={{ marginBottom: '1rem' }}>
          <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
        </svg>
        <h3 style={{ color: '#dc2626', margin: '0 0 0.5rem 0' }}>Error cargando historial</h3>
        <p style={{ color: '#991b1b', margin: '0 0 1rem 0' }}>{error}</p>
        <button
          onClick={cargarHistorial}
          style={{
            padding: '0.5rem 1rem',
            backgroundColor: '#3b82f6',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '14px'
          }}
        >
          Reintentar
        </button>
      </div>
    )
  }

  if (historial.length === 0) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '3rem',
        backgroundColor: '#f8fafc',
        borderRadius: '8px',
        border: '2px dashed #cbd5e1'
      }}>
        <div style={{ textAlign: 'center', color: '#64748b' }}>
          <svg
            width="64"
            height="64"
            fill="currentColor"
            viewBox="0 0 20 20"
            style={{ marginBottom: '1rem', opacity: 0.5 }}
          >
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
          </svg>
          <h3 style={{ margin: '0 0 0.5rem 0', fontSize: '18px' }}>
            No hay análisis anteriores
          </h3>
          <p style={{ margin: 0, fontSize: '14px' }}>
            Los contratos que analices aparecerán aquí para poder revisarlos después
          </p>
        </div>
      </div>
    )
  }

  return (
    <div style={{ padding: '1rem 0' }}>
      <div style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h3 style={{ margin: 0, color: '#1f2937', fontSize: '18px', fontWeight: 600 }}>
            Análisis Anteriores ({historial.length})
          </h3>
          <button
            onClick={cargarHistorial}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: '#f3f4f6',
              color: '#374151',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '14px',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}
          >
            <svg width="16" height="16" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd" />
            </svg>
            Actualizar
          </button>
        </div>
      </div>

      <div style={{ display: 'grid', gap: '1rem' }}>
        {historial.map((item) => {
          const colorRiesgo = obtenerColorRiesgo(item.riesgos_alto, item.riesgos_medio, item.riesgos_bajo)
          const isLoading = loadingAnalysis === item.id

          return (
            <div
              key={item.id}
              style={{
                backgroundColor: '#ffffff',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                padding: '1.5rem',
                cursor: isLoading ? 'not-allowed' : 'pointer',
                transition: 'all 0.2s',
                opacity: isLoading ? 0.7 : 1
              }}
              onClick={() => !isLoading && cargarAnalisis(item.id)}
              onMouseEnter={(e) => {
                if (!isLoading) {
                  e.currentTarget.style.borderColor = '#3b82f6'
                  e.currentTarget.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }
              }}
              onMouseLeave={(e) => {
                if (!isLoading) {
                  e.currentTarget.style.borderColor = '#e5e7eb'
                  e.currentTarget.style.boxShadow = 'none'
                }
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                <div style={{ flex: 1 }}>
                  <h4 style={{
                    margin: '0 0 0.5rem 0',
                    color: '#1f2937',
                    fontSize: '16px',
                    fontWeight: 600,
                    wordBreak: 'break-word'
                  }}>
                    {item.filename}
                  </h4>
                  <p style={{
                    margin: 0,
                    color: '#6b7280',
                    fontSize: '14px'
                  }}>
                    {formatearFecha(item.timestamp)}
                  </p>
                </div>

                {isLoading ? (
                  <div style={{
                    width: '20px',
                    height: '20px',
                    border: '2px solid #f3f4f6',
                    borderTop: '2px solid #3b82f6',
                    borderRadius: '50%',
                    animation: 'spin 1s linear infinite'
                  }}></div>
                ) : (
                  <div style={{
                    width: '12px',
                    height: '12px',
                    backgroundColor: colorRiesgo,
                    borderRadius: '50%',
                    flexShrink: 0
                  }}></div>
                )}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '1rem' }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '24px', fontWeight: 700, color: '#1f2937' }}>
                    {item.total_clausulas}
                  </div>
                  <div style={{ fontSize: '12px', color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Cláusulas
                  </div>
                </div>

                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '24px', fontWeight: 700, color: '#ef4444' }}>
                    {item.riesgos_alto}
                  </div>
                  <div style={{ fontSize: '12px', color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Alto
                  </div>
                </div>

                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '24px', fontWeight: 700, color: '#f59e0b' }}>
                    {item.riesgos_medio}
                  </div>
                  <div style={{ fontSize: '12px', color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Medio
                  </div>
                </div>

                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '24px', fontWeight: 700, color: '#10b981' }}>
                    {item.riesgos_bajo}
                  </div>
                  <div style={{ fontSize: '12px', color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Bajo
                  </div>
                </div>
              </div>

              {item.modo_utilizado && (
                <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid #f3f4f6' }}>
                  <span style={{
                    fontSize: '12px',
                    color: '#6b7280',
                    backgroundColor: '#f9fafb',
                    padding: '0.25rem 0.5rem',
                    borderRadius: '4px',
                    textTransform: 'capitalize'
                  }}>
                    Modo: {item.modo_utilizado}
                  </span>
                </div>
              )}
            </div>
          )
        })}
      </div>

      <style>
        {`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}
      </style>
    </div>
  )
}

export default HistoryList