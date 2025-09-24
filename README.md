# Creditworthiness Prediction

This project was done as a part of the requirements for the course Intelligent Data Analysis & Machine Learning I, offered at the University of Potsdam, 
by the Institut für Informatik.

## Instructions

### Requirements:

1. Python 3.11.1 or later 
2. Git
3. Jupyter Labs or equivalent

## Problem Definition

-  Objective:

A bank aims to develop a predictive model to assess the **creditworthiness of its customers**. 
Using customer records—including financial information, credit history, and demographic attributes—each customer should be classified as 
either **creditworthy or not creditworthy**.

- Learning Problem:

Supervised Learning: given input–output pairs (features + known label).   
Classification problem: target variable is categorical (creditworthy vs. not creditworthy).

- Input Attributes (Features)

1. Status of existing checking account
2. Duration in months
3. Credit history
4. Purpose
5. Credit amount
6. Savings account/bonds
7. Present employment since
8. Installment rate in % of disposable income
9. Personal status and sex
10. Other debtors/guarantors
11. Present residence since
12. Property
13. Age in years
14. Other installment plans
15. Housing
16. Number of existing credits at this bank
17. Job
18. Number of people liable to provide maintenance
19. Telephone
20. Foreign worker

- Target Variable (Label)

Let (y) be the target variable  
Creditworthiness ->  
(y=1) = Applicant is creditworthy         
(y=2) = Applicant is not creditworthy

- Business Impact

Accurately predicting creditworthiness provides multiple benefits:
  
1. Reduce financial risk: Decrease loan defaults by identifying risky applicants.  
2. Improve efficiency: Automate credit decision-making.  
3. Increase profitability: Grant safe loans faster and to more customers.
  

```
cd existing_repo
git remote add origin https://gitup.uni-potsdam.de/ugale/creditworthiness-prediction.git
git branch -M main
git push -uf origin main
```

## To run the project:

1. Clone the repo using SSH or HTTPS from GitUP or the GitHub mirror: 

  GitUP:
  SSH
  `git clone git@gitup.uni-potsdam.de:ugale/creditworthiness-prediction.git`

  HTTPS:
  `git clone https://gitup.uni-potsdam.de/ugale/creditworthiness-prediction.git`
  
2. Navigate to project directory:
  'cd creditworthiness-prediction'

3. Run the jupyter notebook(ipynb file)

## Authors and acknowledgment
Ritika Ugale
ritikaugale2000@gmail.com

## License
Licensed under MIT - license
