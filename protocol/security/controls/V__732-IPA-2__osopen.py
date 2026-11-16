import os

fd = os.open('important_config', os.O_RDWR | os.O_CREAT, 0o777)
os.write(fd, b'important_config')
os.close(fd)
