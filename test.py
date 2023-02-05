import pyminizip
from PyPDF2 import PdfWriter, PdfReader

pdf_file_path = "testdocument.pdf"
pdf_file_path_encrypted = "testdocument.pdf"

pdfWriter = PdfWriter()
pdf = PdfReader(pdf_file_path)

for page_num in range(len(pdf.pages)):
    pdfWriter.add_page(pdf.pages[page_num])

password = "12345"
pdfWriter.encrypt(password)

with open(pdf_file_path_encrypted, "wb") as f:
    pdfWriter.write(f)
    f.close()
