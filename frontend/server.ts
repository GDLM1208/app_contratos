import express from 'express'
import path from 'path'
import { fileURLToPath } from 'url'

const app = express()
const port = process.env.PORT || 3000

// Necesario para __dirname con ESModules
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// Ruta al build generado por Vite
const distPath = path.join(__dirname, 'dist')

// Servir archivos estáticos
app.use(express.static(distPath))

// Manejo del "fallback" para React Router (SPA)
app.get('*', (req, res) => {
  res.sendFile(path.join(distPath, 'index.html'))
})

app.listen(port, () => {
  console.log(`Servidor corriendo en http://localhost:${port}`)
})
