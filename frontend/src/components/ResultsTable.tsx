import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import { Chip } from '@mui/material';

const exampleRows = [
  { id: 1, descripcion: 'Cláusula de daños y perjuicios', impacto: 'Alto', mitigacion: 'Negociar límites', comentarios: 'Revisar con abogado' },
  { id: 2, descripcion: 'Penalidades por retraso', impacto: 'Medio', mitigacion: 'Ajustar cronograma', comentarios: 'Verificar plazos' },
  { id: 3, descripcion: 'Responsabilidad por defectos', impacto: 'Alto', mitigacion: 'Asegurar calidad', comentarios: 'Implementar controles' },
  { id: 4, descripcion: 'Cambios en el alcance', impacto: 'Medio', mitigacion: 'Documentar cambios', comentarios: 'Proceso de cambio formal' },
];

const getImpactColor = (impacto: string) => {
  switch (impacto.toLowerCase()) {
    case 'alto':
      return 'error';
    case 'medio':
      return 'warning';
    case 'bajo':
      return 'success';
    default:
      return 'default';
  }
};

export default function ResultsTable({ rows = exampleRows }:{ rows?:
  { 
    id: number;
    descripcion: string;
    impacto: string;
    mitigacion: string;
    comentarios: string;
  }[] }) {
  return (
    <TableContainer component={Paper} sx={{ 
      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1)',
      borderRadius: '12px',
      overflow: 'hidden',
      border: '1px solid #e2e8f0'
    }}>
      <Table sx={{ minWidth: 650 }}>
        <TableHead>
          <TableRow sx={{ backgroundColor: '#f8fafc' }}>
            <TableCell sx={{ fontWeight: 600, color: '#1e293b' }}>ID</TableCell>
            <TableCell sx={{ fontWeight: 600, color: '#1e293b' }}>Descripción del Riesgo</TableCell>
            <TableCell sx={{ fontWeight: 600, color: '#1e293b' }}>Impacto</TableCell>
            <TableCell sx={{ fontWeight: 600, color: '#1e293b' }}>Estrategias de Mitigación</TableCell>
            <TableCell sx={{ fontWeight: 600, color: '#1e293b' }}>Comentarios</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((row) => (
            <TableRow 
              key={row.id}
              sx={{ 
                '&:nth-of-type(odd)': { backgroundColor: '#fafafa' },
                '&:hover': { backgroundColor: '#f0f9ff' },
                transition: 'background-color 0.2s ease'
              }}
            >
              <TableCell sx={{ fontWeight: 500, color: '#64748b' }}>
                #{row.id}
              </TableCell>
              <TableCell sx={{ color: '#1e293b', fontWeight: 500 }}>
                {row.descripcion}
              </TableCell>
              <TableCell>
                <Chip 
                  label={row.impacto} 
                  color={getImpactColor(row.impacto) as any}
                  size="small"
                  sx={{ fontWeight: 600 }}
                />
              </TableCell>
              <TableCell sx={{ color: '#64748b', maxWidth: 200 }}>
                {row.mitigacion}
              </TableCell>
              <TableCell sx={{ color: '#94a3b8', fontSize: '0.875rem', maxWidth: 200 }}>
                {row.comentarios}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
} 