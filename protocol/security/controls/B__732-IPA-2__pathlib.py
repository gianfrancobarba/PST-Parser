from pathlib import Path

config = Path('important_config')
config.write_text('important_config')
config.chmod(0o766)
