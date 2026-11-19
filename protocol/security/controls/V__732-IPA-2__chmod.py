import os

fd = os.open('important_config', os.O_RDWR | os.O_CREAT)
os.write(fd, b'important_config')
os.close(fd)
os.chmod('important_config', 0o777)
