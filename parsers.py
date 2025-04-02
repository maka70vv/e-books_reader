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


def extract_text_from_fb2(fb2_path):
    from bs4 import BeautifulSoup
    import os

    if not os.path.exists(fb2_path):
        print(f"Ошибка: файл {fb2_path} не существует.")
        return ""

    with open(fb2_path, 'r', encoding='utf-8') as file:
        content = file.read()

    soup = BeautifulSoup(content, 'lxml-xml')
    text = ""
    seen_text = set()  # Множество для отслеживания уникальных строк

    body = soup.find('body')
    if not body:
        print("Ошибка: в FB2-файле не найден тег <body>.")
        return ""

    for p_tag in body.find_all('p', recursive=True):
        paragraph_text = p_tag.get_text().strip()
        if paragraph_text and paragraph_text not in seen_text:  # Добавляем только уникальные строки
            text += paragraph_text + "\n"
            seen_text.add(paragraph_text)

    final_text = text.strip()
    if not final_text:
        print("Итоговый текст пустой. Возможно, в FB2 нет содержимого в <body>.")
    else:
        print(f"Извлечённый текст (первые 50 символов): {final_text[:50]}...")

    return final_text
