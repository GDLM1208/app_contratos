import { useState } from 'react';

const UploadForm = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFile(e.target.files?.[0] || null);
    setError(''); // Clear error when new file is selected
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setError('');

    // Disparar evento de inicio de análisis para resetear estado previo
    window.dispatchEvent(new CustomEvent('analysis:started'));

    const formData = new FormData();
    formData.append('pdf_file', file);
    try {
      const res = await fetch(`${API_URL}/analizar-contrato-pdf`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
        if (data.success && data.data) {
          // Disparar un evento con el resultado completo del análisis para que el App lo capture
          console.log('Análisis completo:', data);
          const event = new CustomEvent('analysis:completed', { detail: data });
          window.dispatchEvent(event);
        } else {
          setError(data.error || 'No se pudo procesar el documento.');
        }
    } catch (err) {
      console.log(err);
      setError('Error al subir el archivo. Verifica tu conexión e intenta nuevamente.');
    }
    setLoading(false);
  };

  return (
    <div className="upload-area">
      <h3 style={{ fontSize: '1.5rem', lineHeight: '2rem', fontWeight: 600, color: '#111827', marginBottom: '1rem' }}>
        Selecciona tu documento PDF
      </h3>
      <p style={{ color: '#4b5563', marginBottom: '2rem' }}>
        Sube un archivo PDF para analizar los riesgos en tu contrato de construcción
      </p>

      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div>
          <input
            type="file"
            accept="application/pdf"
            onChange={handleFileChange}
            className="file-input"
          />
        </div>

        <button
          type="submit"
          disabled={loading || !file}
          className="btn-primary"
          style={{
            minWidth: '200px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto'
          }}
        >
          {loading && <div className="loading-spinner"></div>}
          {loading ? 'Procesando...' : 'Analizar Documento'}
        </button>

        {error && <div className="error-message">{error}</div>}
      </form>
    </div>
  );
};

export default UploadForm;