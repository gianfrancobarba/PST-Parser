import os

fd = os.open('secret.txt', os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
os.write(fd, b'U2FsdGVkX1/R+WzJcxgvX/Iw==')
os.fchmod(fd, 0o666)
os.close(fd)
