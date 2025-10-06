import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const exampleData = [
  { frase: 'Daños y perjuicios', frecuencia: 3, color: '#ef4444' },
  { frase: 'Penalidades', frecuencia: 2, color: '#f59e0b' },
  { frase: 'Fallos', frecuencia: 1, color: '#10b981' },
  { frase: 'Responsabilidad', frecuencia: 4, color: '#3b82f6' },
  { frase: 'Garantías', frecuencia: 2, color: '#8b5cf6' },
];

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
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
          Frecuencia: <span style={{ fontWeight: 600, color: '#2563eb' }}>{payload[0].value}</span>
        </p>
      </div>
    );
  }
  return null;
};

const RiskBarChart = ({ data = exampleData }: { data?: { frase: string; frecuencia: number; color?: string }[] }) => (
  <ResponsiveContainer width="100%" height={400}>
    <BarChart 
      data={data} 
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
);

export default RiskBarChart; 