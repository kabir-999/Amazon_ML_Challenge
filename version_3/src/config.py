import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.abspath(os.path.join(ROOT, "..", "student_resource", "dataset"))
WORK = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
SEED = 42

os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
