# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here

df = pd.read_csv(path)
p_a = len(df[df['fico']>700])/len(df)

df1 = df[df['purpose']=='debt_consolidation']


# print(len((df[df['purpose']=='debt_consolidation']) | (df['fico']>700])))

p_b = len(df[df['purpose']=='debt_consolidation'])/len(df)

a_or_b = len(df[(df['purpose']=='debt_consolidation') | (df['fico']>700)])/len(df)
p_a_b = a_or_b/p_a

result = (p_a_b == p_b)

print(result)






# code ends here


# --------------
# code starts here



prob_lp = len(df[df['paid.back.loan']=='Yes'])/len(df)
prob_cs = len(df[df['credit.policy']=='Yes'])/len(df)

new_df = df[df['paid.back.loan']=='Yes']
df1 = df[df['credit.policy']=='Yes']

# a_or_b = len(df[(df['paid.back.loan']=='Yes') | (df['credit.policy']=='Yes')])/len(df)
prob_pd_cs = len(new_df[new_df['credit.policy']=='Yes'])/len(new_df)#a_or_b/prob_cs

bayes = (prob_pd_cs*prob_lp)/prob_cs      #a_or_b/prob_lp
print(bayes)




# code ends here


# --------------
# code starts here


plt.bar(df['purpose'],height=100)

df1 = df[df['paid.back.loan']=='No']

plt.bar(df1['purpose'],height=100)












# code ends here


# --------------
# code starts here

inst_median = df['installment'].median()
inst_mean = df['installment'].mean()

plt.hist(df['installment'],height=100)
plt.hist(df['log.annual.inc'],height=100)


# code ends here


