# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



data = pd.read_csv(path)

data['Gender'].replace('-','Agender',inplace=True)
gender_count = data['Gender'].value_counts()
plt.bar(gender_count,10)


#path of the data file- path

#Code starts here 




# --------------
#Code starts here



alignment = data['Alignment'].value_counts()

plt.pie(alignment)
plt.plot(label='Character Alignment')







# --------------
#Code starts here



sc_df = data[['Strength','Combat']]
sc_covariance = sc_df.cov().iloc[0,1]
sc_strength = sc_df['Strength'].std()
sc_combat = sc_df['Combat'].std()

sc_pearson = sc_covariance/(sc_strength*sc_combat)

ic_df = data[['Intelligence','Combat']]
ic_covariance = ic_df.cov().iloc[0,1]
ic_intelligence = ic_df['Intelligence'].std()
ic_combat = ic_df['Combat'].std()

ic_pearson = ic_covariance/(ic_intelligence*ic_combat)

print(sc_pearson)

print('=======================')
print(ic_pearson)









# --------------
#Code starts here


total_high = data['Total'].quantile(q=0.99)

super_best =  data[data['Total']>total_high]
super_best_names = list(super_best['Name'])
print(super_best_names)
















# --------------
#Code starts here

fig ,(ax_1,ax_2,ax_3) = plt.subplots(1,3,figsize=(20,10))

# print(super_best)
super_best.plot(kind='box',y='Intelligence',ax =ax_1,title='Intelligence')
super_best.plot(kind='box',y='Speed',ax =ax_2,title='Speed')
super_best.plot(kind='box',y='Power',ax =ax_3,title='Power')


