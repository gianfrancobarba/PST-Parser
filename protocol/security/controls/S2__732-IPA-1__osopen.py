import os

fd = os.open('important_config', os.O_CREAT | os.O_WRONLY, 0o600)
os.write(fd, b'important_config')
os.close(fd)
