import subprocess, sys

username = sys.argv[1]
print(subprocess.check_output(f'getent passwd {username}', shell=True, text=True))
