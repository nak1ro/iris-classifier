import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from predict import predict, target_names

def test_predict_returns_valid_species():
    sample_features = [5.1, 3.5, 1.4, 0.2]

    species = predict(sample_features)

    # Ensure output is a valid species name
    assert species in target_names
