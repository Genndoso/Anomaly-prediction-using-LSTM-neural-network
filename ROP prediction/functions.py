# импортирование необходимых библиотек
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import seaborn as sns
from scipy.signal import savgol_filter
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras import optimizers
import xgboost as XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import RootMeanSquaredError



def data_preproccesing(data):
    # Очистка пропущенных данных в столбце глубины
    data_1 = data
    # data_1 = data.dropna(axis=0,subset=['Measured Depth m'])

    # очистка данных от выбросов путем разработки модели изоляционного леса
    clf = IsolationForest(n_estimators=200, max_samples='auto', contamination=float(.08), \
                          max_features=5, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
    clf.fit(data_1)
    pred = clf.predict(data_1)
    data_1['anomaly'] = pred
    outliers = data_1.loc[data_1['anomaly'] == -1]
    outlier_index = list(outliers.index)

    # Найти количество найденных аномалий  
    print(data_1['anomaly'].value_counts())
    data_1 = data_1[data_1['anomaly'] == 1]

    # Заполнение пропущенных данных
    data_1['Weight on Bit kkgf'] = data['Weight on Bit kkgf'].fillna(data['Weight on Bit kkgf'].median())
    data_1['Average Hookload kkgf'] = data['Average Hookload kkgf'].fillna(data['Average Hookload kkgf'].median())
    data_1['Average Rotary Speed rpm'] = data['Average Rotary Speed rpm'].fillna(
        data['Average Rotary Speed rpm'].median())
    data_1['Average Standpipe Pressure kPa'] = data['Average Standpipe Pressure kPa'].fillna(
        data['Average Standpipe Pressure kPa'].median())
    data_1['Average Surface Torque kN.m'] = data['Average Surface Torque kN.m'].fillna(
        data['Average Surface Torque kN.m'].median())
    data_1['Mud Density In g/cm3'] = data['Mud Density In g/cm3'].fillna(data['Mud Density In g/cm3'].median())
    data_1['Rate of Penetration m/h'] = data['Rate of Penetration m/h'].fillna(data['Rate of Penetration m/h'].median())
    data_1['Mud Flow In L/min'] = data['Mud Flow In L/min'].fillna(data['Mud Flow In L/min'].median())
    data_1['USROP Gamma gAPI'] = data['USROP Gamma gAPI'].fillna(data['USROP Gamma gAPI'].median())

    return data_1


def data_distribution_plot(data):
    # Распределение данных

    plt.style.use('seaborn')
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(18,20))
    fig.suptitle("Распределение параметров режима бурения", fontsize=22)

    sns.distplot(ax=ax[0,0],a =data['Weight on Bit kkgf'], hist=True,kde=False,color = 'blue')
    ax[0,0].set_xlabel('WOB, kkgf',size=22)
    ax[0,0].set_ylabel('Distribution',size=22)

    sns.distplot(ax=ax[0,1],a =data['Average Hookload kkgf'], hist=True,kde=False,color = 'blue')

    ax[0,1].set_xlabel('Hookload, kkgf',size=22)
    ax[0,1].set_ylabel('Distribution',size=22)


    sns.distplot(ax=ax[0,2],a =data['Average Surface Torque kN.m'], hist=True,kde=False,color = 'blue')
    # Add labels
    #ax[0].title('Depth')
    ax[0,2].set_xlabel('Torque, kN.m',size=22)

    sns.distplot(ax=ax[1,0],a =data['Average Rotary Speed rpm'], hist=True,kde=False,color = 'blue')

    ax[1,0].set_xlabel('Rotatary speed, rpm',size=22)
    ax[1,0].set_ylabel('Distribution',size=22)

    sns.distplot(ax=ax[1,1],a =data['Rate of Penetration m/h'], hist=True,kde=False,color = 'blue')
    ax[1,1].set_xlabel('ROP, m/h',size=22)

    sns.distplot(ax=ax[1,2],a =data['Average Standpipe Pressure kPa'], hist=True,kde=False,color = 'blue')
    ax[1,2].set_xlabel('Standpipe pressure, kPa',size=22)


    sns.distplot(ax=ax[2,0],a =data['Mud Density In g/cm3'], hist=True,kde=False,color = 'blue')
    ax[2,0].set_xlabel('Mud Density In g/cm3',size=22)
    ax[2,0].set_ylabel('Distribution',size=22)


    sns.distplot(ax=ax[2,1],a =data['Mud Flow In L/min'], hist=True,kde=False,color = 'blue')
    ax[2,1].set_xlabel('Mud Flow In L/min',size=22)


    sns.distplot(ax=ax[2,2],a =data['USROP Gamma gAPI'], hist=True,kde=False,color = 'blue')
    ax[2,2].set_xlabel('GR',size=22)

def log_plot(data_1):
        fig, ax = plt.subplots(nrows=1, ncols=7, figsize=(18, 20), sharey=True)
        fig.suptitle("Каротажные диаграммы", fontsize=22)
        # fig.subplots_adjust(top=0.75,wspace=0.1)
        # General setting for all axis
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=15)
        for axes in ax:
            # axes.set_ylim (490, 1300)
            axes.invert_yaxis()
            axes.yaxis.grid(True)
            axes.get_xaxis().set_visible(False)

        ax01 = ax[0].twiny()
        ax01.spines['top'].set_position(('outward', 0))
        ax01.set_xlabel("WOB kkgf]")
        # ax01.set_ylim(0,20)
        ax01.plot(data_1['Weight on Bit kkgf'], data_1['Measured Depth m'], label='WOB kkgf', color='blue')
        ax01.set_xlabel('WOB kkgf', color='blue', fontsize=15)
        ax01.tick_params(axis='x', colors='blue', labelsize=15)
        ax01.grid(True)

        ax02 = ax[0].twiny()
        ax02.plot(data_1['Average Hookload kkgf'], data_1['Measured Depth m'], label='Hookload kkgf', color='green')
        ax02.spines['top'].set_position(('outward', 40))
        ax02.set_xlabel('Hookload kkgf', color='green', fontsize=15)
        ax02.tick_params(axis='x', colors='green', labelsize=15)

        ax03 = ax[1].twiny()

        ax03.plot(data_1['Mud Flow In L/min'], data_1['Measured Depth m'], label='Mud Flow In L/min', color='red')
        ax03.spines['top'].set_position(('outward', 0))
        ax03.set_xlabel('Mud Flow In L/min', color='red', fontsize=15)
        ax03.tick_params(axis='x', colors='red', labelsize=15)

        ax04 = ax[1].twiny()
        ax04.grid(True)
        ax04.spines['top'].set_position(('outward', 40))
        ax04.set_xlabel('Mud Density In g/cm3', color='black', fontsize=15)
        ax04.scatter(data_1['Mud Density In g/cm3'], data_1['Measured Depth m'], label='Mud Density In g/cm',
                     color='black')
        ax04.tick_params(axis='x', colors='black', labelsize=15)

        ax04 = ax[2].twiny()
        ax04.grid(True)
        ax04.spines['top'].set_position(('outward', 40))
        ax04.set_xlabel('Standpipe pressure,kpa', color='black', fontsize=12)
        ax04.plot(data_1['Average Standpipe Pressure kPa'], data_1['Measured Depth m'],
                  label='Average Standpipe Pressure kPa', color='black')
        ax04.tick_params(axis='x', colors='black', labelsize=15)

        ax04 = ax[3].twiny()
        ax04.grid(True)
        ax04.spines['top'].set_position(('outward', 0))
        ax04.set_xlabel('Rate pf penetration, m/h', color='red', fontsize=15)
        ax04.plot(data_1['Rate of Penetration m/h'], data_1['Measured Depth m'], label='Rate pf penetration, m/h',
                  color='red')
        ax04.tick_params(axis='x', colors='red', labelsize=15)

        ax04 = ax[4].twiny()
        ax04.grid(True)
        ax04.spines['top'].set_position(('outward', 0))
        ax04.set_xlabel('Torque, kN.m', color='purple', fontsize=15)
        ax04.plot(data_1['Average Surface Torque kN.m'], data_1['Measured Depth m'],
                  label='Average Surface Torque kN.m',
                  color='purple')
        ax04.tick_params(axis='x', colors='purple', labelsize=15)

        ax05 = ax[5].twiny()
        ax05.grid(True)
        ax05.spines['top'].set_position(('outward', 0))
        ax05.set_xlabel('Rotary speed rpm', color='orange', fontsize=15)
        ax05.plot(data_1['Average Rotary Speed rpm'], data_1['Measured Depth m'], label='Rotary speed rpm',
                  color='orange')
        ax05.tick_params(axis='x', colors='orange', labelsize=15)

        ax06 = ax[5].twiny()
        ax06.grid(True)
        ax06.spines['top'].set_position(('outward', 0))
        ax06.set_xlabel('Rotary speed rpm', color='orange', fontsize=15)
        ax06.plot(data_1['Average Rotary Speed rpm'], data_1['Measured Depth m'], label='Rotary speed rpm',
                  color='orange')
        ax06.tick_params(axis='x', colors='orange', labelsize=15)

        ax07 = ax[6].twiny()
        ax07.grid(True)
        ax07.spines['top'].set_position(('outward', 0))
        ax07.set_xlabel('GR', color='blue', fontsize=15)
        ax07.plot(data_1['USROP Gamma gAPI'], data_1['Measured Depth m'], label='Rotary speed rpm', color='blue')
        ax07.tick_params(axis='x', colors='blue', labelsize=15)

    # Корреляционная диаграмма
def CorrMatrix(dataset):
        corr = dataset.corr()
        x = plt.figure(figsize=(20, 12))
        ax = sns.heatmap(
            corr,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200, ),
            square=True,
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        );
def data_encoding(x_tr,y_tr):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_scaled = scaler_x.fit_transform(x_tr)
    y_scaled = scaler_y.fit_transform(y_tr.reshape(-1,1))
    return x_scaled,y_scaled,scaler_x,scaler_y

# Cглаживание тестовых данных
def data_smoothing(data): 
    data['Rate of Penetration m/h'] = savgol_filter(data['Rate of Penetration m/h'].values,89,3)
    data['Average Hookload kkgf'] = savgol_filter(data['Average Hookload kkgf'].values,101,3)
    data['Weight on Bit kkgf'] = savgol_filter(data['Weight on Bit kkgf'].values,101,3)
    data['Average Rotary Speed rpm'] = savgol_filter(data['Average Rotary Speed rpm'].values,101,3)
    data['USROP Gamma gAPI'] = savgol_filter(data['USROP Gamma gAPI'].values,101,3)
    data['Average Standpipe Pressure kPa'] = savgol_filter(data['Average Standpipe Pressure kPa'].values,101,3)
    data['Average Surface Torque kN.m'] = savgol_filter(data['Average Surface Torque kN.m'].values,101,3)
    return data