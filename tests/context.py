import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import autom8


class Accumulator(autom8.Observer):
    def __init__(self):
        self.warnings = []
        self.pipelines = []

    def receive_pipeline(self, pipeline, report):
        self.pipelines.append((pipeline, report))

    def warn(self, message):
        self.warnings.append(message)
