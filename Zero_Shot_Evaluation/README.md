This script reproduces the zero shot testing performed in the Wanda paper. It tests 6 different datasets using the `lm_eval` library.

In order to resolve dependency conflicts with `lm_eval`, a new environment is needed. After creating your venv, please run: 

`pip install -r requirements_for_zero_shot.txt`

The metric assessed by the Wanda paper is the `acc,none` category of the final dictionary (plain accuracy measure).

It takes a long time to execute (~40 minutes on a RTX 5090) due to the size of the datasets under test.