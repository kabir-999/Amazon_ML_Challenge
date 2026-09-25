import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.abspath(os.path.join(ROOT, "..", "student_resource", "dataset"))
WORK = os.path.join(ROOT, "data")   # cached subsets / features
OUT = os.path.join(ROOT, "out")     # models, metrics, submission files
N_S1_SUBSET = 500_000                # train S1 entities used for development
SEED = 42
