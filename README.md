# logistic-
python 3.6 下 logistic回归实现鸢尾花分类
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
if __name__ == "__main__":
    path = 'C:\\Users\\Administrator\\PycharmProjects\\python program\\4.Advertising.csv'
    data = pd.read_csv(path)
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data[['Sales']]
    # print(x)
    # print(y)
    # plt.plot(data['TV'], y, 'ro', label = 'TV')
    # plt.plot(data['Radio'], y, 'g^', label = 'Radio')
    # plt.plot(data['Newspaper'], y, 'b*',label = 'Newspaper')
    # plt.legend(loc = 'lower right')
    # plt.grid()
    # plt.show()
    # plt.figure(figsize= (9,12))
    # plt.subplot(311)
    # plt.plot(data['TV'], y, 'ro')
    # plt.title('TV')
    # plt.grid()
    # plt.subplot(312)
    # plt.plot(data['Radio'], y, 'ro')
    # plt.title('Radio')
    # plt.grid()
    # plt.subplot(313)
    # plt.plot(data['Newspaper'], y, 'ro')
    # plt.title('Newspaper')
    # plt.grid()
    # plt.tight_layout()
    # plt.show()
    x_train, x_test , y_train , y_test = train_test_split(x , y)
    #print(x_train,y_train)
    linreg = LinearRegression()
    model = linreg.fit(x_train,y_train)
    print(model)
    print(linreg.coef_)#系数
    print(linreg.intercept_)#截距
    y_hat = linreg.predict(x_test)
    mse = np.average((y_hat - y_test)** 2) #Mean Squared Error均方误差
    rmse = np.sqrt(mse)#Root Mean Squared Error 均方根误差
    print(mse,rmse)
    t = np.arange(len(x_test))
    plt.plot(t,y_test,'r-',linewidth = 2,label = 'Test')
    plt.plot(t,y_hat,'g-',linewidth = 2,label = 'Predict')
    plt.legend(loc = 'upper right')
    plt.grid()
    plt.show()
