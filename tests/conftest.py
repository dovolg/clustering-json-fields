import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# app.py now imports clustering.semantic, which pulls in sentence-transformers/
# torch/gensim -- a one-time ~7-8s import cost that exceeds AppTest's default
# 3s run() timeout if it lands inside whichever test happens to run first.
# Paying it here, once, before any test's timer starts, keeps that cost out
# of individual test budgets regardless of execution order.
import app  # noqa: F401
