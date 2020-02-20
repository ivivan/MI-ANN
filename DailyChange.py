import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from scipy.stats import kendalltau
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator


df = pd.read_csv(r"C:\Users\ZHA244\Coding\QLD\baffle_creek\baffle-creek-buoy-quality-2013-all-forpca.csv")
df = df.loc[4368:,:]
df.drop('TIMESTAMP', axis=1, inplace=True)

scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(df)
df.loc[:,:] = scaled_values

print(df)

df2 = pd.read_csv(r"C:\Users\ZHA244\Coding\QLD\baffle_creek\baffle-creek-buoy-quality-2013-all-forpca.csv")
df2 = df2.loc[4368:,:]
df2['TIMESTAMP'] = pd.to_datetime(df2['TIMESTAMP'], dayfirst=True)





# scaler = MinMaxScaler()
# scaled_values = scaler.fit_transform(df)
# df.loc[:,:] = scaled_values





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








# Drawing



# ax = plt.gca()
# xfmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
# ax.xaxis.set_major_formatter(xfmt)

# turbidity_line, = plt.plot_date(df2['TIMESTAMP'],df['Turbidity_NTU'] , 'b-', color=tableau20[2],
#                              label='turbidity')
# chloraphylla_line, = plt.plot_date(df2['TIMESTAMP'], df['Chloraphylla_ugL'], 'b-', color=tableau20[4],
#                                 label='chloraphylla')
# pH_line, = plt.plot_date(df2['TIMESTAMP'], df['pH'], 'b-', color=tableau20[6],
#                                 label='pH')
# temp_line, = plt.plot_date(df2['TIMESTAMP'], df['Temp_degC'], 'b-', color=tableau20[8],
#                                 label='temp')
# do_line, = plt.plot_date(df2['TIMESTAMP'], df['DO_mg'], 'b-', color=tableau20[10],
#                                 label='do')
# ec_line, = plt.plot_date(df2['TIMESTAMP'], df['EC_uScm'], 'b-', color=tableau20[12],
#                                 label='ec')
#
# plt.legend(handles=[turbidity_line, chloraphylla_line,pH_line,temp_line,do_line,ec_line])
# plt.gcf().autofmt_xdate()
# plt.show()





fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(nrows=3, ncols=2,sharex=True,sharey=True)
# xfmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
# ax.xaxis.set_major_formatter(xfmt)
# plt.subplot(2, 2, 1)
# sns.jointplot(df['Temp_degC'],df['DO_mg'], kind="hex", stat_func=kendalltau, color="#4CB391")
ax1.plot_date(df2['TIMESTAMP'],df['Turbidity_NTU'] , 'b-', color=tableau20[2],
                             label='turbidity')
ax1.plot_date(df2['TIMESTAMP'], df['DO_mg'], 'b-',ls='dashed', color=tableau20[12],
                                label='temp')

ax1.get_xaxis().set_minor_locator(AutoMinorLocator())
ax1.grid(b=True, which='major', color='w', linewidth=1.5)
ax1.grid(b=True, which='minor', color='w', linewidth=0.75)
plt.setp(ax1.get_xticklabels(), rotation=50, horizontalalignment='right')
ax1.set_title('Turbidity')


# plt.subplot(2, 2, 2)
ax2.plot_date(df2['TIMESTAMP'], df['Chloraphylla_ugL'], 'b-', color=tableau20[4],
                                label='chloraphyll')
ax2.plot_date(df2['TIMESTAMP'], df['DO_mg'], 'b-',ls='dashed', color=tableau20[12],
                                label='temp')
ax2.get_xaxis().set_minor_locator(AutoMinorLocator())
ax2.grid(b=True, which='major', color='w', linewidth=1.5)
ax2.grid(b=True, which='minor', color='w', linewidth=0.75)
plt.setp(ax2.get_xticklabels(), rotation=50, horizontalalignment='right')
ax2.set_title('Chloraphyll')

# plt.subplot(2, 2, 3)
ax3.plot_date(df2['TIMESTAMP'], df['pH'], 'b-', color=tableau20[6],
                                label='pH')
ax3.plot_date(df2['TIMESTAMP'], df['DO_mg'], 'b-',ls='dashed', color=tableau20[12],
                                label='temp')
ax3.get_xaxis().set_minor_locator(AutoMinorLocator())
ax3.grid(b=True, which='major', color='w', linewidth=1.5)
ax3.grid(b=True, which='minor', color='w', linewidth=0.75)
plt.setp(ax3.get_xticklabels(), rotation=50, horizontalalignment='right')
ax3.set_title('pH')

# plt.subplot(2, 2, 4)
ax4.plot_date(df2['TIMESTAMP'], df['Temp_degC'], 'b-', color=tableau20[8],
                                label='temp')
ax4.plot_date(df2['TIMESTAMP'], df['DO_mg'], 'b-',ls='dashed', color=tableau20[12],
                                label='temp')
ax4.get_xaxis().set_minor_locator(AutoMinorLocator())
ax4.grid(b=True, which='major', color='w', linewidth=1.5)
ax4.grid(b=True, which='minor', color='w', linewidth=0.75)
plt.setp(ax4.get_xticklabels(), rotation=50, horizontalalignment='right')
ax4.set_title('Temperature')

# plt.subplot(2, 2, 3)
ax5.plot_date(df2['TIMESTAMP'], df['EC_uScm'], 'b-', color=tableau20[10],
                                label='pH')
ax5.plot_date(df2['TIMESTAMP'], df['DO_mg'], 'b-',ls='dashed', color=tableau20[12],
                                label='temp')
ax5.get_xaxis().set_minor_locator(AutoMinorLocator())
ax5.grid(b=True, which='major', color='w', linewidth=1.5)
ax5.grid(b=True, which='minor', color='w', linewidth=0.75)
plt.setp(ax5.get_xticklabels(), rotation=50, horizontalalignment='right')
ax5.set_title('EC')

# plt.subplot(2, 2, 4)
ax6.plot_date(df2['TIMESTAMP'], df['DO_mg'], 'b-',ls='dashed', color=tableau20[12],
                                label='temp')
ax6.get_xaxis().set_minor_locator(AutoMinorLocator())
ax6.grid(b=True, which='major', color='w', linewidth=1.5)
ax6.grid(b=True, which='minor', color='w', linewidth=0.75)
plt.setp(ax6.get_xticklabels(), rotation=50, horizontalalignment='right')
ax6.set_title('DO')

plt.show()








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


# fig, ax = plt.subplots(nrows=2, ncols=2)
#
# plt.subplot(2, 2, 1)
# # sns.jointplot(df['Temp_degC'],df['DO_mg'], kind="hex", stat_func=kendalltau, color="#4CB391")
# plt.scatter(df['Temp_degC'],df['DO_mg'],c=tableau20[4])
#
# plt.subplot(2, 2, 2)
# plt.scatter(df['Chloraphylla_ugL'],df['DO_mg'],c=tableau20[4])
#
# plt.subplot(2, 2, 3)
# plt.scatter(df['pH'],df['DO_mg'],c=tableau20[4])
#
# plt.subplot(2, 2, 4)
# plt.scatter(df['Turbidity_NTU'],df['DO_mg'],c=tableau20[4])
#
# plt.show()






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

