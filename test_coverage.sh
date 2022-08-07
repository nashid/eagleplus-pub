#! /bin/bash

# unittest run pattern
# coverage run --parallel-mode --source . -m unittest discover . "test*.py"

# pytest run pattern
# CUBLAS_WORKSPACE_CONFIG=:16:8 coverage run --source . --concurrency=multiprocessing -m pytest torchrec -v -s -W ignore::pytest.PytestCollectionWarning --continue-on-collection-errors -k 'not test_sharding_gloo_cw'
# coverage run --source . --concurrency=multiprocessing -m pytest torchrec -v -s -W ignore::pytest.PytestCollectionWarning --continue-on-collection-errors -k 'not test_sharding_gloo_cw'
coverage run -m pytest torchrec -v -s -W ignore::pytest.PytestCollectionWarning --continue-on-collection-errors -k 'not test_sharding_gloo_cw'

coverage combine

# coverage report
coverage html -i --skip-empty --omit "*/tests/test_*.py","*/test_utils/test_*.py","*_tests.py" --include="torchrec/*"
