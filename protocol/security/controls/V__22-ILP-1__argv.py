import os, sys

def main():
    name = sys.argv[1]
    with open(os.path.join('/safe/', name)) as f:
        print(f.read())

main()
