import codecs

BLOCKSIZE = 1048576 # or some other, desired size in bytes
with codecs.open('test_export.nc', "r", "ANSI") as sourceFile:
    with codecs.open('test_export_encoded.nc', "w", "utf-8") as targetFile:
        while True:
            contents = sourceFile.read(BLOCKSIZE)
            if not contents:
                break
            targetFile.write(contents)