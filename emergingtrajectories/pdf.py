"""
This is a very simple set of utility function(s) for loading PDF content. In fact, it might be easier to just use PyPDF directly and avoid this altogether. In the future, we might create specialized functions and classes for doing "fancy" things with PDFs (e.g., OCR, tables, etc.) so have created this module as a way to keep this in mind.
"""

from pypdf import PdfReader
import requests
import io


def get_PDF_content_from_file_by_page(file_path: str) -> list:
    """
    Loads a PDF file and extracts the text into a list of strings, one for each page.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        list: A list of strings, one for each page.
    """
    reader = PdfReader(file_path)
    content = []
    for page in reader.pages:
        content.append(page.extract_text())
    return content


def get_PDF_content_from_url_by_page(url: str) -> list:
    """
    Loads a PDF file from a URL and extracts the text into a list of strings, one for each page.

    Args:
        url (str): The URL to the PDF file.

    Returns:
        list: A list of strings, one for each page.
    """
    response = requests.get(url=url, timeout=120)
    pdf_file = io.BytesIO(response.content)
    reader = PdfReader(pdf_file)
    content = []
    for page in reader.pages:
        content.append(page.extract_text())
    return content


def get_PDF_content_by_page_from_file(file_path: str) -> str:
    """
    Loads a PDF file and extracts the text into one big string.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The text content of the PDF file.
    """
    reader = PdfReader(file_path)
    content = ""
    for page in reader.pages:
        content += page.extract_text() + "\n"
    return content


def get_PDF_content_by_page_from_url(url: str) -> str:
    """
    Loads a PDF file from a URL and extracts the text into one big string.

    Args:
        url (str): The URL to the PDF file.

    Returns:
        str: The text content of the PDF file.
    """
    response = requests.get(url=url, timeout=120)
    pdf_file = io.BytesIO(response.content)
    reader = PdfReader(pdf_file)
    content = ""
    for page in reader.pages:
        content += page.extract_text() + "\n"
    return content
