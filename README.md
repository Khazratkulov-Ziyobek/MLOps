# MLOps

## Personal Loan Prediction

The project on the MLOPs course. In this project, we will consider classification dataset where we need to find whether the customer will opt for personal loan or not from the bank.

## Table of Contents

- [Background](#background)
- [Installation](#installation)
- [Data](#data)
- [CLI](#cli)
- [Mlflow](#mlflow)
- [HW-3](#hw-3)
- [Useful links](#useful-links)
- [Contacts](#contacts)

## Background

Dataset Description:

| Feature              | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| ID                   | Customer   ID                                                               |
| Age                  | Customer's age in   completed years                                         |
| Experience           | #years of professional experience                                           |
| Income               | Annual income of the customer ($000)                                        |
| ZIPCode              | Home Address ZIP code.                                                      |
| Family               | Family size of the customer                                                 |
| CCAvg                | Avg.spending on credit cards per month ($000)                               |
| Education            | Education Level. 1: Undergrad; 2: Graduate; 3: Advanced/Professional        |
| Mortgage             | Value of house mortgage if any. ($000)                                      |
| Personal Loan        | Did this customer accept the personal loan offered in the last campaign?    |
| Securities   Account | Does the customer have a securities account with the bank?                  |
| CD Account           | Does the customer have a certificate of deposit (CD) account with the bank? |
| Online               | Does the customer use internet banking facilities?                          |
| CreditCard           | Does the customer use a credit card issued by Bank?               |

## Installation

To set up the project locally, follow these:
Clone the repository:
```
git clone https://github.com/Khazratkulov-Ziyobek/MLOps.git
cd MLOps
```

## Data

To load dataset run:
```
dvc pull
```

## CLI

To setup all the dependencies and run, follow these steps:
```
python3 -m venv mlops_env
source mlops_env/bin/activate
poetry install
pre-commit run -a
python3 bank_personal_loan_modelling/train.py
python3 bank_personal_loan_modelling/infer.py
```

## Mlflow

To run **second assignment** you should:

1. Specify `tracking_uri` parameter in the [config](https://github.com/Khazratkulov-Ziyobek/MLOps/blob/main/configs/config.yaml)
2. Set `is_mlflow_logging` parameter `True`
3. Run ```python3 run_server.py```. This python script utilizes the `subprocess` module to execute a sequence of command-line operations on the terminal, which includes starting an MLflow server. Following this, it proceeds to run a set of commands related to software setup and model training. Upon complementing these commands, it opens a web browser to access the MLflow server hosted locally.

<!-- ## HW-3

Triton server -->


## Useful links

[Lecture recordings on YouTube, autumn 2023](https://www.youtube.com/playlist?list=PLJR10EXrBaAuJzCa9HKmLRdUpgajnh1g7)

Useful links:
- python packaging: https://realpython.com/python-modules-packages/
- poetry: https://python-poetry.org/docs/
- pyproject.toml
    - [declaring project metadata](https://packaging.python.org/en/latest/specifications/declaring-project-metadata/#declaring-project-metadata)
    - [pyproject at pip site](https://pip.pypa.io/en/stable/reference/build-system/pyproject-toml/)
- fire (tool to create CLI): https://google.github.io/python-fire/
- reference projects
    - template for ml/ds: https://github.com/v-goncharenko/data-science-template
    - regular pip package: https://github.com/v-goncharenko/somepytools
    - ml project: https://github.com/v-goncharenko/mimics/
- [secure jupyter notebook server](https://jupyter-notebook.readthedocs.io/en/5.6.0/public_server.html)

## Contacts

Khazratkulov Ziyobek - [Khazratkulov_Z](https://t.me/Khazratkulov_Z) - khazratkulovziyobek@gmail.com

***Supervisor:*** [Vladislav Goncharenko](https://t.me/white_pepper)
