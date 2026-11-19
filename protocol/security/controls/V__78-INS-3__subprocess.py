import subprocess, sys

subprocess.call('ls ' + sys.argv[1], shell=True)
