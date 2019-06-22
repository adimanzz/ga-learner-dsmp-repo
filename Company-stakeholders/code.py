# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(path)
loan_status = data['Loan_Status'].value_counts()

loan_status.plot(kind='bar')












#Code starts here


# --------------
#Code starts here



property_and_loan = data.groupby(['Property_Area','Loan_Status']).size().unstack()

property_and_loan.plot(kind='bar',stacked=False)
plt.xlabel("Property Area")
plt.ylabel("Loan Status")
plt.xticks(rotation=45)
plt.show()












# --------------
#Code starts here


education_and_loan = data.groupby(['Education','Loan_Status']).size().unstack()

education_and_loan.plot(kind='bar',stacked=True)

plt.xlabel("Education Status")
plt.ylabel("Loan Status")

plt.xticks(rotation=45)











# --------------
#Code starts here



graduate = data[data['Education']=='Graduate']

not_graduate = data[data['Education']== 'Not Graduate']

graduate['LoanAmount'].plot(kind='density',label='Graduate')

not_graduate['LoanAmount'].plot(kind='density',label='Not Graduate')












#Code ends here

#For automatic legend display
plt.legend()


# --------------
#Code starts here

fig ,(ax_1,ax_2,ax_3) = plt.subplots(3,1)

# res_1 = data.groupby(['ApplicantIncome','LoanAmount'])
data.plot(kind='scatter',x='ApplicantIncome',y='LoanAmount',ax =ax_1,title='Applicant Income')
# data.set_title('Applicant Income')


# res_2 = data.groupby(['CoapplicantIncome','LoanAmount'])
data.plot(kind='scatter',x='CoapplicantIncome',y='LoanAmount',ax =ax_2,title='Coapplicant Income')
# data.set_title('Coapplicant Income')


data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']


# res_3 = data.groupby(['TotalIncome','LoanAmount'])
data.plot(kind='scatter',x='TotalIncome',y='LoanAmount',ax =ax_3,title='Total Income')
# data.set_title('Total Income')









