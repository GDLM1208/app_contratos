import React from 'react'

interface RiskMatrixRow {
  id: string
  categoria: string
  probabilidad: number
  impacto: string
  riesgo_afectacion: string[]
  mitigacion: string
  responsable: string
}

interface RiskMatrixTableProps {
  data?: RiskMatrixRow[]
}

const RiskMatrixTable = ({ data = [] }: RiskMatrixTableProps) => {
  const [editableResponsables, setEditableResponsables] = React.useState<{[key: string]: string}>({})

  // Mapeo de probabilidad a colores y texto
  const getProbabilityInfo = (probabilidad: number) => {
    const mapping = {
      1: { text: 'Muy Baja', color: '#10b981', bgColor: '#ecfdf5' },
      2: { text: 'Baja', color: '#059669', bgColor: '#d1fae5' },
      3: { text: 'Media', color: '#f59e0b', bgColor: '#fef3c7' },
      4: { text: 'Alta', color: '#ef4444', bgColor: '#fee2e2' },
      5: { text: 'Muy Alta', color: '#dc2626', bgColor: '#fecaca' }
    }
    return mapping[probabilidad as keyof typeof mapping] || { text: 'Desconocida', color: '#6b7280', bgColor: '#f9fafb' }
  }

  // Mapeo de impacto a colores
  const getImpactColor = (impacto: string) => {
    const mapping = {
      'alto': '#dc2626',
      'medio': '#f59e0b',
      'bajo': '#10b981'
    }
    return mapping[impacto.toLowerCase() as keyof typeof mapping] || '#6b7280'
  }

  // Renderizar badges de riesgo/afectación
  const renderRiskBadges = (riesgos: string[]) => {
    const colorMapping = {
      'alcance': '#3b82f6',
      'costo': '#ef4444',
      'tiempo': '#f59e0b'
    }

    return riesgos.map((riesgo, index) => (
      <span
        key={index}
        style={{
          backgroundColor: colorMapping[riesgo.toLowerCase() as keyof typeof colorMapping] || '#6b7280',
          color: 'white',
          padding: '2px 8px',
          borderRadius: '12px',
          fontSize: '12px',
          fontWeight: '500',
          marginRight: index < riesgos.length - 1 ? '4px' : '0'
        }}
      >
        {riesgo.charAt(0).toUpperCase() + riesgo.slice(1)}
      </span>
    ))
  }

  const handleResponsableChange = (id: string, value: string) => {
    setEditableResponsables(prev => ({
      ...prev,
      [id]: value
    }))
  }

  if (!data || data.length === 0) {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '3rem',
          backgroundColor: '#f8fafc',
          borderRadius: '8px',
          border: '2px dashed #cbd5e1'
        }}
      >
        <div style={{ textAlign: 'center', color: '#64748b' }}>
          <svg
            width="48"
            height="48"
            fill="currentColor"
            viewBox="0 0 20 20"
            style={{ marginBottom: '8px', opacity: 0.5 }}
          >
            <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
          </svg>
          <p style={{ margin: 0, fontSize: '16px', fontWeight: 500 }}>
            No hay datos para la matriz de riesgos
          </p>
          <p style={{ margin: '4px 0 0 0', fontSize: '14px' }}>
            Analiza un contrato para ver la matriz de riesgos
          </p>
        </div>
      </div>
    )
  }

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
        <thead>
          <tr style={{ backgroundColor: '#f8fafc' }}>
            <th style={{ padding: '12px', textAlign: 'left', fontWeight: 600, borderBottom: '2px solid #e2e8f0', minWidth: '60px' }}>
              ID
            </th>
            <th style={{ padding: '12px', textAlign: 'left', fontWeight: 600, borderBottom: '2px solid #e2e8f0', minWidth: '200px' }}>
              Categoría
            </th>
            <th style={{ padding: '12px', textAlign: 'center', fontWeight: 600, borderBottom: '2px solid #e2e8f0', minWidth: '120px' }}>
              Probabilidad
            </th>
            <th style={{ padding: '12px', textAlign: 'center', fontWeight: 600, borderBottom: '2px solid #e2e8f0', minWidth: '80px' }}>
              Impacto
            </th>
            <th style={{ padding: '12px', textAlign: 'left', fontWeight: 600, borderBottom: '2px solid #e2e8f0', minWidth: '150px' }}>
              Riesgo/Afectación
            </th>
            <th style={{ padding: '12px', textAlign: 'left', fontWeight: 600, borderBottom: '2px solid #e2e8f0', minWidth: '200px' }}>
              Mitigación
            </th>
            <th style={{ padding: '12px', textAlign: 'left', fontWeight: 600, borderBottom: '2px solid #e2e8f0', minWidth: '150px' }}>
              Responsable
            </th>
          </tr>
        </thead>
        <tbody>
          {data.map((row, index) => {
            const probInfo = getProbabilityInfo(row.probabilidad)
            const impactColor = getImpactColor(row.impacto)

            return (
              <tr
                key={row.id}
                style={{
                  backgroundColor: index % 2 === 0 ? '#ffffff' : '#f8fafc',
                  transition: 'background-color 0.2s'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = '#f1f5f9'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = index % 2 === 0 ? '#ffffff' : '#f8fafc'
                }}
              >
                <td style={{ padding: '12px', fontWeight: 600, color: '#1e40af', borderBottom: '1px solid #e2e8f0' }}>
                  {row.id}
                </td>
                <td style={{ padding: '12px', borderBottom: '1px solid #e2e8f0' }}>
                  <div style={{ fontWeight: 500, color: '#1f2937' }}>
                    {row.categoria}
                  </div>
                </td>
                <td style={{ padding: '12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>
                  <div
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '4px',
                      padding: '4px 8px',
                      borderRadius: '6px',
                      backgroundColor: probInfo.bgColor,
                      color: probInfo.color,
                      fontWeight: 600,
                      fontSize: '13px'
                    }}
                  >
                    {row.probabilidad}
                    <span style={{ fontSize: '11px', fontWeight: 500 }}>
                      ({probInfo.text})
                    </span>
                  </div>
                </td>
                <td style={{ padding: '12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>
                  <span
                    style={{
                      padding: '4px 8px',
                      borderRadius: '6px',
                      backgroundColor: impactColor,
                      color: 'white',
                      fontWeight: 500,
                      fontSize: '12px',
                      textTransform: 'uppercase'
                    }}
                  >
                    {row.impacto}
                  </span>
                </td>
                <td style={{ padding: '12px', borderBottom: '1px solid #e2e8f0' }}>
                  <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                    {renderRiskBadges(row.riesgo_afectacion)}
                  </div>
                </td>
                <td style={{ padding: '12px', borderBottom: '1px solid #e2e8f0', fontStyle: 'italic', color: '#6b7280' }}>
                  {row.mitigacion || 'Pendiente de definir'}
                </td>
                <td style={{ padding: '8px', borderBottom: '1px solid #e2e8f0' }}>
                  <input
                    type="text"
                    placeholder="Asignar responsable..."
                    value={editableResponsables[row.id] || row.responsable || ''}
                    onChange={(e) => handleResponsableChange(row.id, e.target.value)}
                    style={{
                      width: '100%',
                      padding: '6px 8px',
                      border: '1px solid #d1d5db',
                      borderRadius: '4px',
                      fontSize: '13px',
                      backgroundColor: '#ffffff',
                      transition: 'border-color 0.2s'
                    }}
                    onFocus={(e) => {
                      e.target.style.borderColor = '#3b82f6'
                      e.target.style.outline = '2px solid #dbeafe'
                    }}
                    onBlur={(e) => {
                      e.target.style.borderColor = '#d1d5db'
                      e.target.style.outline = 'none'
                    }}
                  />
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>

      {/* Leyenda */}
      <div style={{ marginTop: '20px', padding: '16px', backgroundColor: '#f8fafc', borderRadius: '8px', fontSize: '13px' }}>
        <h4 style={{ margin: '0 0 12px 0', color: '#374151', fontSize: '14px' }}>Leyenda:</h4>

        <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
          <div>
            <strong style={{ color: '#374151' }}>Probabilidad:</strong>
            <div style={{ display: 'flex', gap: '8px', marginTop: '4px', flexWrap: 'wrap' }}>
              {[1, 2, 3, 4, 5].map(num => {
                const info = getProbabilityInfo(num)
                return (
                  <span
                    key={num}
                    style={{
                      padding: '2px 6px',
                      borderRadius: '4px',
                      backgroundColor: info.bgColor,
                      color: info.color,
                      fontWeight: 500,
                      fontSize: '12px'
                    }}
                  >
                    {num} - {info.text}
                  </span>
                )
              })}
            </div>
          </div>

          <div>
            <strong style={{ color: '#374151' }}>Riesgo/Afectación:</strong>
            <div style={{ display: 'flex', gap: '8px', marginTop: '4px' }}>
              <span style={{ padding: '2px 6px', borderRadius: '4px', backgroundColor: '#3b82f6', color: 'white', fontSize: '12px' }}>
                Alcance
              </span>
              <span style={{ padding: '2px 6px', borderRadius: '4px', backgroundColor: '#ef4444', color: 'white', fontSize: '12px' }}>
                Costo
              </span>
              <span style={{ padding: '2px 6px', borderRadius: '4px', backgroundColor: '#f59e0b', color: 'white', fontSize: '12px' }}>
                Tiempo
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default RiskMatrixTable