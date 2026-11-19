import os

with open('important_config', 'w') as handle:
    handle.write('important_config')
os.chmod('important_config', mode=0o766)
