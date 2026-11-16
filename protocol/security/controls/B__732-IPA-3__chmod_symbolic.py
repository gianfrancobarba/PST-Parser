import os, stat

with open('secret.txt', 'w') as f:
    f.write('U2FsdGVkX1/R+WzJcxgvX/Iw==')
os.chmod('secret.txt', stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
