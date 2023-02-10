import sys
import os

PYTHON_PATHS = ['cp37-cp37m', 'cp38-cp38', 'cp39-cp39', 'cp310-cp310', 'cp311-cp11']

COMMAND = """
docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin publish -u {} -p "{}" -i /opt/python/{}/bin/python
"""

def main():
    os.system("docker pull ghcr.io/pyo3/maturin")
    user, password = sys.argv[1:]
    for p in PYTHON_PATHS:
        print(COMMAND.format(user, password, p))
        os.system(COMMAND.format(user, password, p))


if __name__ == "__main__":
    main()
