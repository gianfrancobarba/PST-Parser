import os

MODE = 0o777
with open('important_config', 'w') as handle:
    handle.write('important_config')
os.chmod('important_config', MODE)
