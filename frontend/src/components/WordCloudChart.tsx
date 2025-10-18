import { useEffect, useRef } from 'react'
import cloud from 'd3-cloud'
import { select } from 'd3-selection'

interface WordCloudData {
  text: string
  value: number
}

interface WordCloudChartProps {
  data?: WordCloudData[]
  width?: number
  height?: number
}

interface CloudWord extends WordCloudData {
  size: number
  x?: number
  y?: number
  rotate?: number
}

const WordCloudChart = ({ data = [], width = 600, height = 400 }: WordCloudChartProps) => {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!data || data.length === 0 || !svgRef.current) return

    // Limpiar el SVG anterior
    select(svgRef.current).selectAll("*").remove()

    // Escalar los valores para el tamaño de fuente
    const maxValue = Math.max(...data.map(d => d.value))
    const minValue = Math.min(...data.map(d => d.value))
    const fontScale = (value: number) => {
      const minFont = 18
      const maxFont = 72
      if (maxValue === minValue) return 24
      return minFont + ((value - minValue) / (maxValue - minValue)) * (maxFont - minFont)
    }

    // Colores para las palabras
    const colors = [
      '#2563eb', '#3b82f6', '#60a5fa', '#93c5fd',
      '#1d4ed8', '#1e40af', '#1e3a8a', '#164084',
      '#6366f1', '#8b5cf6', '#a855f7', '#c084fc'
    ]

    // Preparar datos para d3-cloud
    const words: CloudWord[] = data.map((d) => ({
      text: d.text,
      value: d.value,
      size: fontScale(d.value)
    }))

    // Crear layout de la nube de palabras
    const layout = cloud()
      .size([width, height])
      .words(words)
      .padding(8)
      .rotate(() => Math.random() < 0.5 ? 0 : 90) // Solo horizontal (0°) o vertical (90°)
      .font('Arial, sans-serif')
      .fontSize(d => d.size || 16) // Valor por defecto si size es undefined
      .on('end', drawWords)

    function drawWords(words: CloudWord[]) {
      const svg = select(svgRef.current)

      const g = svg
        .append('g')
        .attr('transform', `translate(${width / 2}, ${height / 2})`)

      const text = g
        .selectAll('text')
        .data(words)
        .enter()
        .append('text')
        .style('font-size', d => `${d.size}px`)
        .style('font-family', 'Arial, sans-serif')
        .style('font-weight', 'bold')
        .style('fill', (_, i) => colors[i % colors.length])
        .style('cursor', 'pointer')
        .attr('text-anchor', 'middle')
        .attr('transform', d => `translate(${d.x || 0}, ${d.y || 0}) rotate(${d.rotate || 0})`)
        .text(d => d.text)

      // Agregar interactividad
      text
        .on('mouseenter', function(_, d) {
          select(this)
            .style('opacity', 0.7)
            .style('transform', `translate(${d.x || 0}px, ${d.y || 0}px) rotate(${d.rotate || 0}deg) scale(1.1)`)
        })
        .on('mouseleave', function(_, d) {
          select(this)
            .style('opacity', 1)
            .style('transform', `translate(${d.x || 0}px, ${d.y || 0}px) rotate(${d.rotate || 0}deg) scale(1)`)
        })
        .append('title')
        .text(d => `${d.text}: ${d.value} ocurrencias`)
    }

    // Iniciar el layout
    layout.start()

  }, [data, width, height])

  if (!data || data.length === 0) {
    return (
      <div
        style={{
          width,
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
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
          <p style={{ margin: 0, fontSize: '14px' }}>
            No hay datos para mostrar en la nube de palabras
          </p>
        </div>
      </div>
    )
  }

  return (
    <div style={{ width, height, overflow: 'hidden' }}>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        style={{
          border: '1px solid #e2e8f0',
          borderRadius: '8px',
          backgroundColor: '#ffffff'
        }}
      />
    </div>
  )
}

export default WordCloudChart