# MLOps

## Personal Loan Prediction

The first assignment of MLOps course.

For the first task is included dataset of personal loan prediction. The task is to predict customer will accept the personal loan or not.
Columns of the data:
    <ul>
        <li>**ID** - Customer ID</li>
        <li>**Age**  - Customer's age in completed years</li>
        <li>**Experience** - #years of professional experience</li>
        <li>**Income** - Annual income of the customer</li>
        <li>**ZIPCode** - Home Address ZIP code</li>
        <li>**Family** - Family size of the customer</li>
        <li>**CCAvg** - Avg. spending on credit cards per month ($000)</li>
        <li>**Education** - Education Level. 1: Undergrad; 2: Graduate; 3: Advanced/Professional</li>
        <li>**Mortgage** - Value of house mortgage if any. ($000)</li>
        <li>**Personal Loan** - Did this customer accept the personal loan offered in the last campaign?</li>
        <li>**Securities Account** - Does the customer have a securities account with the bank?</li>
        <li>**CD Account** - Does the customer have a certificate of deposit (CD) account with the bank?</li>
        <li>**Online** - Does the customer use internet banking facilities?</li>
        <li>**CreditCard** - Does the customer use a credit card issued by UniversalBank?
</li>
    </ul>

## How to use
```
# Install Poetry
curl -sSL https://install.python-poetry.org | python3
poetry install
poetry run python3 ./bank_personal_loan_modelling/train.py
poetry run python3 ./bank_personal_loan_modelling/infer.py
```
