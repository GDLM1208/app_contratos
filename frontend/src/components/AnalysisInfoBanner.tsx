interface AnalysisInfo {
  filename: string
  timestamp: string
  total_clausulas: number
  source: 'history' | 'new'
}

interface AnalysisInfoBannerProps {
  analysisInfo: AnalysisInfo | null
}

const AnalysisInfoBanner = ({ analysisInfo }: AnalysisInfoBannerProps) => {
  if (!analysisInfo) return null

  const isFromHistory = analysisInfo.source === 'history'

  return (
    <div style={{
      marginTop: '1rem',
      padding: '1rem',
      backgroundColor: isFromHistory ? '#f0f9ff' : '#f0fdf4',
      borderRadius: '8px',
      border: `1px solid ${isFromHistory ? '#e0f2fe' : '#dcfce7'}`
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        marginBottom: '0.5rem'
      }}>
        {isFromHistory ? (
          <svg width="16" height="16" fill="currentColor" viewBox="0 0 20 20" style={{ color: '#0284c7' }}>
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
          </svg>
        ) : (
          <svg width="16" height="16" fill="currentColor" viewBox="0 0 20 20" style={{ color: '#059669' }}>
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
        )}
        <span style={{
          fontWeight: 600,
          color: isFromHistory ? '#0284c7' : '#059669',
          fontSize: '0.875rem'
        }}>
          {isFromHistory ? 'Análisis cargado desde historial' : 'Análisis completado'}
        </span>
      </div>

      <div style={{ fontSize: '0.875rem', color: '#64748b' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.5rem' }}>
          <div>
            <strong>Archivo:</strong> {analysisInfo.filename}
          </div>
          <div>
            <strong>Cláusulas:</strong> {analysisInfo.total_clausulas}
          </div>
          <div>
            <strong>Fecha:</strong> {new Date(analysisInfo.timestamp).toLocaleString('es-ES', {
              year: 'numeric',
              month: 'short',
              day: 'numeric',
              hour: '2-digit',
              minute: '2-digit'
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

export default AnalysisInfoBanner