import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from scipy.stats import kendalltau

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df = pd.read_csv(r"C:\Users\ZHA244\Coding\QLD\baffle_creek\dry_season.csv")
for c in [c for c in df.columns if df[c].dtype in numerics]:
    df[c] = df[c].abs()
# df = df.loc[4368:,:]

print(df)

df.drop('TIMESTAMP', axis=1, inplace=True)
print(df.describe())





# scaler = MinMaxScaler()
# scaled_values = scaler.fit_transform(df)
# df.loc[:,:] = scaled_values

print(df)



# Drawing



# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

# fig, ax = plt.subplots(nrows=2, ncols=2)

# ax.spines["top"].set_visible(False)
# ax.spines["bottom"].set_visible(False)
# ax.spines["right"].set_visible(False)
# ax.spines["left"].set_visible(False)
#
# plt.scatter(df['Turbidity_NTU'],df['DO_mg'],c=tableau20[4])


# sns.jointplot(df['Turbidity_NTU'],df['DO_mg'], kind="hex", stat_func=kendalltau, color="#4CB391")

# sns.set(font_scale=1.8)
# # Use JointGrid directly to draw a custom plot
# grid = sns.JointGrid(df['Turbidity_NTU'],df['DO_mg'], space=0, size=6, ratio=50).set_axis_labels(xlabel='Turbidity (NTU)',ylabel='DO (mg/l)')
# grid.plot_joint(plt.scatter, color=tableau20[5])
# grid.plot_marginals(sns.rugplot, height=1, color=tableau20[4])
#
#
# grid = sns.JointGrid(df['pH'],df['DO_mg'], space=0, size=6, ratio=50).set_axis_labels(xlabel='pH',ylabel='DO (mg/l)')
# grid.plot_joint(plt.scatter, color=tableau20[5])
# grid.plot_marginals(sns.rugplot, height=1, color=tableau20[4])


fig, ax = plt.subplots(nrows=3, ncols=2)

plt.subplot(3, 2, 1)
# sns.jointplot(df['Temp_degC'],df['DO_mg'], kind="hex", stat_func=kendalltau, color="#4CB391")
plt.scatter(df['DO_mg'],df['Temp_degC'],c=tableau20[4])

plt.subplot(3, 2, 2)
plt.scatter(df['DO_mg'],df['Chloraphylla_ugL'],c=tableau20[6])

plt.subplot(3, 2, 3)
plt.scatter(df['DO_mg'],df['pH'],c=tableau20[8])

plt.subplot(3, 2, 4)
plt.scatter(df['DO_mg'],df['Turbidity_NTU'],c=tableau20[10])

plt.subplot(3, 2, 5)
plt.scatter(df['DO_mg'],df['EC_uScm'],c=tableau20[10])

plt.show()






# plt.subplot(2, 2, 1)
# true_line, = plt.plot(df['Unnamed: 0'],df['DO_mg'], '-', lw=1, color=tableau20[2],
#                          label='True Value')
# predict_line, = plt.plot(df['Unnamed: 0'],df['Temp_degC'], '--', lw=1, color=tableau20[18],
#                            label='Prediction Value')
#
# plt.legend(handles=[true_line, predict_line], fontsize=12)
#
# plt.subplot(2, 2, 2)
# true_line, = plt.plot(df['Unnamed: 0'],df['DO_mg'], '-', lw=1, color=tableau20[2],
#                          label='True Value')
# predict_line, = plt.plot(df['Unnamed: 0'],df['Chloraphylla_ugL'], '--', lw=1, color=tableau20[18],
#                            label='Prediction Value')
#
# plt.legend(handles=[true_line, predict_line], fontsize=12)
#
# plt.subplot(2, 2, 3)
# true_line, = plt.plot(df['Unnamed: 0'],df['DO_mg'], '-', lw=1, color=tableau20[2],
#                          label='True Value')
# predict_line, = plt.plot(df['Unnamed: 0'],df['pH'], '--', lw=1, color=tableau20[18],
#                            label='Prediction Value')
#
# plt.legend(handles=[true_line, predict_line], fontsize=12)
#
# plt.subplot(2, 2, 4)
# true_line, = plt.plot(df['Unnamed: 0'],df['DO_mg'], '-', lw=1, color=tableau20[2],
#                          label='True Value')
# predict_line, = plt.plot(df['Unnamed: 0'],df['Turbidity_NTU'], '--', lw=1, color=tableau20[18],
#                            label='Prediction Value')
#
# plt.legend(handles=[true_line, predict_line], fontsize=12)
#
# plt.show()








# true_line, = plt.plot_date(axis_data, scaler_do_y.inverse_transform(y_test_do)[0:496], '-', lw=1, color=tableau20[2],
#                          label='True Value')
# predict_line, = plt.plot_date(axis_data, np.array(y_predicted)[0:496], '--', lw=1, color=tableau20[18],
#                            label='Prediction Value')
#
#
# plt.legend(handles=[true_line, predict_line], fontsize=12)
# plt.title('Water Quality Prediction', fontsize=16)
# plt.xlabel('Date', fontsize=14)
# plt.ylabel('DO (mg/l)', fontsize=14)
# plt.savefig(r'C:\Users\ZHA244\Pictures\paper-figure\90min-7days.png', dpi=200)
# plt.show()

