import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lox.telemetry.run_id import generate_run_id

if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else None
    print(generate_run_id(prefix))
