import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, ttest_ind, norm
import json

data = pd.read_csv('mussl.csv')
may_data = data[(data['Month'] == 'May') & data['Site'].isin(['S4', 'S5', 'S6'])]
results = []

for site1 in ['S4', 'S5', 'S6']:
    for site2 in ['S4', 'S5', 'S6']:
        if site1 < site2:
            site1_data = may_data[may_data['Site'] == site1]['Length'].apply(np.log)
            site2_data = may_data[may_data['Site'] == site2]['Length'].apply(np.log)

            t_stat, p_value = ttest_ind(site1_data, site2_data)

            df = len(site1_data) + len(site2_data) - 2
            alpha = 0.05
            t_crit = t.ppf(1 - alpha / 2, df)

            if p_value > alpha:
                results.append({'left': site1, 
                                'right': site2, 
                                'pvalue': p_value, 
                                'statistic': t_stat, 
                                'threshold': t_crit})

with open('ttest.json', 'w') as f:
    json.dump(results, f, indent=2)


from sklearn.linear_model import LinearRegression
from scipy.stats import chisquare


s5_data = data[data['Site'] == 'S5']
X = s5_data['Day'].values.reshape(-1, 1)
y = np.log(s5_data['Length'])
model = LinearRegression().fit(X, y)

residuals = y - model.predict(X)
unit_error_variance = np.mean(residuals ** 2)
std_res = residuals / np.sqrt(unit_error_variance)

hist, bins = np.histogram(std_res, bins=6)
freq = bins.tolist()
statistic, p_value = chisquare(hist)


with open('chi2.json', 'w') as f:
    json.dump({'freq': freq, 
               'pvalue': p_value, 
               'statistic': statistic}, f, indent=2)

expected = norm.pdf((bins[:-1] + bins[1:]) / 2) 
plt.hist(std_res, bins=freq, density=True, alpha=0.5, label='Empirical')
plt.plot((bins[:-1] + bins[1:]) / 2, expected, 'r-', label='Theoretical')
plt.legend()
plt.grid()
plt.savefig('chi2.png') 

