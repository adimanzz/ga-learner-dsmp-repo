# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 
data = pd.read_csv(path)
bank = pd.DataFrame(data)


categorical_var = bank.select_dtypes(include = 'object')
print(categorical_var)
numerical_var = bank.select_dtypes(include = 'number')
print(numerical_var)

# code starts here






# code ends here


# --------------
# code starts here


banks = bank.drop('Loan_ID',axis = 1)
print(banks.isnull().sum())

bank_mode = banks.mode()
print(bank_mode)
# for x in banks.columns:
banks['Gender'] = banks['Gender'].fillna(banks['Gender'].mode()[0])
banks['Married'] = banks['Married'].fillna(banks['Married'].mode()[0])
banks['Dependents'] = banks['Dependents'].fillna(banks['Dependents'].mode()[0])
banks['Self_Employed'] = banks['Self_Employed'].fillna(banks['Self_Employed'].mode()[0])
banks['LoanAmount'] = banks['LoanAmount'].fillna(banks['LoanAmount'].mode()[0])
banks['Loan_Amount_Term'] = banks['Loan_Amount_Term'].fillna(banks['Loan_Amount_Term'].mode()[0])
banks['Credit_History'] = banks['Credit_History'].fillna(banks['Credit_History'].mode()[0])


print(banks.isnull().sum())
print(banks)






#code ends here


# --------------
# Code starts here





avg_loan_amount = banks.pivot_table(index = ['Gender','Married','Self_Employed'], values = 'LoanAmount')


# code ends here



# --------------
# code starts here



loan_approved_se = len(banks[(banks['Self_Employed']== 'Yes') & (banks['Loan_Status']=='Y')])

loan_approved_nse = len(banks[(banks['Self_Employed']== 'No') & (banks['Loan_Status']=='Y')])

percentage_se = loan_approved_se/614 *100
percentage_nse = loan_approved_nse/614 *100









# code ends here


# --------------
# code starts here

loan_term = banks['Loan_Amount_Term'].apply(lambda x: x/12 )

big_loan_term = len(loan_term[loan_term >= 25])

print(loan_term)

# code ends here


# --------------
# code starts here

loan_groupby = banks.groupby('Loan_Status')

loan_groupby = loan_groupby['ApplicantIncome','Credit_History']

mean_values =loan_groupby.mean()



# code ends here


