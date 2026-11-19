import os

fd = os.open('important_config', os.O_CREAT | os.O_WRONLY)
os.fchmod(fd, 0o777)
os.write(fd, b'important_config')
os.close(fd)
