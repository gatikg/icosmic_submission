import easyocr
import ftfy
import re

reader = easyocr.Reader(['en'])
results = reader.readtext(
    r'E:\Document_Classification - Copy\Data Base\PAN Card\pan0.jpg')
t = ''
for result in results:
    t += result[1] + ' '
lst = []
patt = re.compile(r'\d{2}-\d{2}-\d{4}')
matches = patt.finditer(t)
for match in matches:
    print(match)
for i in t:
    text = i.strip()
    reg = re.findall(r'^Name.+', text)
    if len(reg) == 0:
        continue
    lst.append(reg)
    print(reg)
