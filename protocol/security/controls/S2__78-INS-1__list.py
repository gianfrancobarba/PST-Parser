import subprocess, sys

out = subprocess.run(['getent', 'passwd', sys.argv[1]], capture_output=True, text=True)
print(out.stdout)
