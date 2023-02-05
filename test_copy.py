import pyminizip
level = 4  # level of compression

for i in open("./Images"):
    pyminizip.compress(i, None, "document_classify.zip", "12345", level)
