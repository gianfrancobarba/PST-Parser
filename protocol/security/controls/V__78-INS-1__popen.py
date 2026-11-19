import os, sys

username = sys.argv[1]
print(os.popen('getent passwd ' + username).read())
