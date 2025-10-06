import fitz  # PyMuPDF
import pytesseract
import io
from PIL import Image
from multiprocessing import Pool, cpu_count

def is_text_page(page):
    """Verifica si la página contiene texto digital."""
    text = page.get_text().strip()
    return bool(text)

def process_page(args):
    """Procesa una sola página del PDF."""
    pdf_path, page_num = args
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    if is_text_page(page):
        text = page.get_text()
    else:
        pix = page.get_pixmap(dpi=150)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang="spa")

    doc.close()
    print(f"✅ Procesada página {page_num}")

    return page_num + 1, text

def extract_pdf_to_txt(pdf_path, output_txt):
    total_pages = fitz.open(pdf_path).page_count
    tasks = [(pdf_path, i) for i in range(total_pages)]
    try:
        pytesseract.get_tesseract_version()  # Verifica que Tesseract esté correctamente instalado
        print("✅ Tesseract instalado correctamente")
    except pytesseract.TesseractNotFoundError:
        print("❌ Tesseract no encontrado. Asegúrate de que esté instalado y en el PATH.")

    with Pool(min(cpu_count(), 3)) as pool:
        results = pool.map(process_page, tasks)

    # Ordenar por número de página
    results.sort(key=lambda x: x[0])

    # Guardar en TXT
    with open(output_txt, "w", encoding="utf-8") as f:
        for page_num, text in results:
            f.write(f"--- Página {page_num} ---\n")
            f.write(text.strip() + "\n\n")

if __name__ == "__main__":
    pdf_path = "data/contrato.pdf"  # tu archivo
    output_txt = "contrato_extraido.txt"
    extract_pdf_to_txt(pdf_path, output_txt)
    print(f"✅ Extracción completada. Guardado en: {output_txt}")
