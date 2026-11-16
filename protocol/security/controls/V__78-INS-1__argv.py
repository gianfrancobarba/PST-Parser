import subprocess, sys

username = sys.argv[1]
out = subprocess.run('getent passwd ' + username, shell=True, capture_output=True, text=True)
print(out.stdout)
