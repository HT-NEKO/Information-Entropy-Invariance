import re
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
# import zstandard as zstd
import json
from tqdm import tqdm

def print_running_command():
    command = " ".join(sys.argv)
    print(f"Running command: {command}")
