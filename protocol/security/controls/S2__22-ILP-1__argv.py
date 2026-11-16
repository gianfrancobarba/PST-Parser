import os, sys

BASE = '/safe/'

def main():
    path = os.path.realpath(os.path.join(BASE, sys.argv[1]))
    if not path.startswith(BASE):
        raise SystemExit('forbidden')
    with open(path) as f:
        print(f.read())

main()
