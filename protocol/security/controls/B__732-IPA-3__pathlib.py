from pathlib import Path

secret = Path('secret.txt')
secret.write_text('U2FsdGVkX1/R+WzJcxgvX/Iw==')
secret.chmod(0o666)
