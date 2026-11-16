import tarfile

with tarfile.open('archive.zip') as tar:
    tar.extractall('/tmp/unpack', filter='data')
