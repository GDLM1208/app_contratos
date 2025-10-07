import fitz  # PyMuPDF
import pytesseract
import io
from PIL import Image
from multiprocessing import Pool, cpu_count

def is_text_page(page):
    """Verifica si la página contiene texto digital."""
    text = page.get_text().strip()
    return bool(text)

def process_page(page, page_num):
    """Procesa una sola página del PDF."""
    try:
        if is_text_page(page):
            text = page.get_text()
        else:
            pix = page.get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang="spa")

        return page_num + 1, text

    except Exception as e:
        print(f"❌ Error en página {page_num + 1}: {str(e)}")
        return page_num + 1, f"[Error procesando página {page_num + 1}]"

def extract_pdf_to_txt(pdf_path, output_txt):
    total_pages = fitz.open(pdf_path).page_count
    tasks = [(pdf_path, i) for i in range(total_pages)]
    try:
        pytesseract.get_tesseract_version()  # Verifica que Tesseract esté correctamente instalado
        print("✅ Tesseract instalado correctamente")
    except pytesseract.TesseractNotFoundError:
        print("❌ Tesseract no encontrado. Asegúrate de que esté instalado y en el PATH.")
        raise Exception("Tesseract OCR no está instalado")

    """ with Pool(min(cpu_count(), 3)) as pool:
        results = pool.map(process_page, tasks) """

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    results = []
    pages_with_ocr = 0
    pages_with_text = 0

    for page_num in range(total_pages):
        page = doc[page_num]

        # Determinar tipo de página antes de procesar
        has_text = is_text_page(page)
        if has_text:
            pages_with_text += 1
        else:
            pages_with_ocr += 1

        # Procesar
        result = process_page(page, page_num)
        results.append(result)

    doc.close()

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
