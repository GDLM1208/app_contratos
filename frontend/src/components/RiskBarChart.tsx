import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

type Clause = {
  numero: number;
  contenido: string;
  truncado?: boolean;
  clasificacion?: string;
  confianza?: number;
  nivel_riesgo?: string;
}

type ChartItem = { frase: string; frecuencia: number; color?: string }

const exampleData: ChartItem[] = [
  { frase: 'Daños y perjuicios', frecuencia: 3, color: '#ef4444' },
  { frase: 'Penalidades', frecuencia: 2, color: '#f59e0b' },
  { frase: 'Fallos', frecuencia: 1, color: '#10b981' },
  { frase: 'Responsabilidad', frecuencia: 4, color: '#3b82f6' },
  { frase: 'Garantías', frecuencia: 2, color: '#8b5cf6' },
];

const colorPalette = ['#2563eb', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#3b82f6', '#a3e635']

const CustomTooltip = ({ active, payload, label }: { active?: boolean; payload?: unknown[]; label?: string }) => {
  if (active && payload && payload.length) {
    const first = payload[0]
    let value: number | string | undefined = undefined
    if (typeof first === 'object' && first !== null) {
      const obj = first as Record<string, unknown>
      if ('value' in obj) {
        const v = obj['value']
        if (typeof v === 'number' || typeof v === 'string') value = v
      }
    }
    return (
      <div style={{
        backgroundColor: 'white',
        border: '1px solid #e2e8f0',
        borderRadius: '8px',
        padding: '12px',
        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
        fontSize: '14px',
        fontFamily: 'Inter, sans-serif'
      }}>
        <p style={{ margin: '0 0 8px 0', fontWeight: 600, color: '#1e293b' }}>
          {label}
        </p>
        <p style={{ margin: 0, color: '#64748b' }}>
          Frecuencia: <span style={{ fontWeight: 600, color: '#2563eb' }}>{value ?? ''}</span>
        </p>
      </div>
    );
  }
  return null;
};

const buildChartFromAnalysis = (clauses: Clause[]): ChartItem[] => {
  const counts: Record<string, number> = {}
  clauses.forEach(c => {
    const key = c.clasificacion || c.nivel_riesgo || 'Desconocido'
    counts[key] = (counts[key] || 0) + 1
  })
  const items = Object.entries(counts).map(([frase, frecuencia], idx) => ({ frase, frecuencia, color: colorPalette[idx % colorPalette.length] }))
  // sort descending
  items.sort((a, b) => b.frecuencia - a.frecuencia)
  return items
}

const RiskBarChart = ({ data }: { data?: ChartItem[] }) => {
  const [chartData, setChartData] = useState<ChartItem[]>(data ?? exampleData)

  useEffect(() => {
    if (data && data.length) {
      setChartData(data)
      return
    }

    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail
      if (!detail) return
      const clauses: Clause[] = detail.clausulas_analizadas || []
      const built = buildChartFromAnalysis(clauses)
      if (built.length) setChartData(built)
    }

    window.addEventListener('analysis:completed', handler as EventListener)
    return () => window.removeEventListener('analysis:completed', handler as EventListener)
  }, [data])

  return (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart
        data={chartData}
        margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
        style={{ fontFamily: 'Inter, sans-serif' }}
      >
        <CartesianGrid
          strokeDasharray="3 3"
          stroke="#e2e8f0"
          opacity={0.6}
        />
        <XAxis
          dataKey="frase"
          tick={{ fill: '#64748b', fontSize: 12 }}
          axisLine={{ stroke: '#e2e8f0' }}
          tickLine={{ stroke: '#e2e8f0' }}
          angle={-45}
          textAnchor="end"
          height={80}
        />
        <YAxis
          allowDecimals={false}
          tick={{ fill: '#64748b', fontSize: 12 }}
          axisLine={{ stroke: '#e2e8f0' }}
          tickLine={{ stroke: '#e2e8f0' }}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          wrapperStyle={{
            paddingTop: '20px',
            fontSize: '14px',
            color: '#64748b'
          }}
        />
        <Bar
          dataKey="frecuencia"
          fill="#2563eb"
          radius={[4, 4, 0, 0]}
          name="Frecuencia de Riesgos"
        />
      </BarChart>
    </ResponsiveContainer>
  )
}

export default RiskBarChart;