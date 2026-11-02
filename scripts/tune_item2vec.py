#!/usr/bin/env python3
"""Prepare, run and evaluate isolated Item2Vec experiments."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

from item2vec.model_selection import main


if __name__=='__main__':
    main()
