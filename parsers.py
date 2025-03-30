def extract_text_from_pdf(pdf_path):
    import fitz

    doc = fitz.open(pdf_path)  # Открываем PDF
    text = ""

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # Загружаем страницу
        text += page.get_text("text")  # Извлекаем текст из страницы

    return text


def extract_text_from_epub(epub_path):
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    book = epub.read_epub(epub_path)
    text = ""

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:  # Проверка, что это документ (HTML)
            soup = BeautifulSoup(item.content, "html.parser")

            # Удаляем все теги <title>
            for title_tag in soup.find_all('title'):
                title_tag.decompose()  # Удаляем тег <title> и его содержимое

            # Извлекаем текст
            text += soup.get_text()

    return text
