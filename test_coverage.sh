#! /bin/bash

# unittest run pattern
# coverage run --parallel-mode --source . -m unittest discover . "test*.py"

# pytest run pattern
coverage run --source . -m pytest torchrec -v -s -W ignore::pytest.PytestCollectionWarning --continue-on-collection-errors -k 'not test_sharding_gloo_cw'


# coverage report
coverage html -i --omit "*/test_*.py","*/*_test.py" --include="torchrec/*"
