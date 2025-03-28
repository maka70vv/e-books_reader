import fitz


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)  # Открываем PDF
    text = ""

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # Загружаем страницу
        text += page.get_text("text")  # Извлекаем текст из страницы

    return text