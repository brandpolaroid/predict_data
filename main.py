# ------------------------------------------------------
# ---------------------- main.py -----------------------
# ------------------------------------------------------
from PyQt5.QtCore import QFileInfo
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi

from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
import numpy as np
import math
from pandas import read_csv
import random

h = 10


def POCID_TENDENCY(real, pred):
  dt = 0
  real_n = np.array([ real[i] - real[i-1] for i in range(1, len(real)) ])
  pred_n = np.array([ pred[i] - pred[i-1] for i in range(1, len(real)) ])
  prod = real_n * pred_n
  dt = len([x for x in prod if x > 0]) # кол-во положительных значений
  dt_minus = len([x for x in prod if x < 0]) # кол-во отрицательных значений
  pocid = 100 * dt / len(real)
  tendency = dt_minus / len(real)
  return round(pocid, 2), round(tendency, 2)


def SLG(real, pred):
  lt = 0
  real_n = np.array([ real[i] - real[i-1] for i in range(1, len(real)) ])
  pred_n = np.array([ pred[i] - pred[i-1] for i in range(1, len(real)) ])
  product = real_n * pred_n
  for i in range(len(real_n)):
    if(product[i] > 0):
      lt += abs(real_n[i])
    else:
      lt -= abs(real_n[i])
    slg = lt / len(real)
  return round(slg, 2)

def AFF(real, pred): # k = 1
  sum = 0
  n = 0 # количество точек

  for i in range(len(real)):
    if (abs(real[i]) >= 0.001): # если Ti >= 10^(-3)
        sum += abs((pred[i] - real[i]) / real[i])
        n += 1
        #print(f"{pred[i]} и {real[i]} = {abs((pred[i] - real[i]) / real[i])}")
  aff = 100 * sum / n
  return round(aff, 2)

def PDA(real, pred):
    dt = 0
    gt = 0
    remax = 0.002
    for i in range(1, len(real)):
        prod = (real[i] - real[i - 1]) * (pred[i] - pred[i - 1])
        reT = abs((pred[i] - real[i]) / real[i])

        if (prod > 0):
            dt = 1
        else:
            dt = 0

        if (dt == 1):
            if (reT < remax):
                gt += 1 - reT / remax
            else:
                gt += 0
        elif (dt == 0):
            if (reT < remax):
                gt += -1 + reT / remax
            else:
                gt += -1
    pda = gt / len(real)
    return round(pda, 2)

def ARV(real, pred):
  et = np.array([ (pred[i] - real[i])**2 for i in range(len(real)) ])
  et2sum = et.sum()
  realarray = np.array(real)
  avgreal = realarray.sum() / len(realarray)
  avg = np.array([ (pred[i] - avgreal)**2 for i in range(len(real)) ])
  avgsum = avg.sum()
  arv =  et2sum / avgsum
  return round(arv, 2)

def GetMeasures(real, pred):
    pocid, tendency = POCID_TENDENCY(real, pred)
    slg = SLG(real, pred)
    aff = AFF(real, pred)
    pda = PDA(real, pred)
    arv = ARV(real, pred)
    return pocid, tendency, slg, aff, pda, arv

# метод, принимающий массив x-ов и возвращающий массив значений первой встроенной функции
def y1(t):
    y = 10 * np.sin(2 * np.pi * t / 100)
    return y

# метод, принимающий массив x-ов и возвращающий массив значений второй встроенной функции
def y5(t):
    y = np.sin(np.pi * t / 20) * np.exp(0.02 * t) + 3
    return y

# расчёт H
def H(n, h, q):
    s = [np.power(i - q, n) for i in range(h + 1)]
    return sum(s)

# расчёт F1
def F1_calc(i, i0, ii, a0, a1, q, f):
    f1 = [f[j] * np.power(j - i0 - q, 2) for j in range (i0, ii + 1)]
    f1i_res = sum(f1) - a0[i] * H(2, h, q) - a1[i] * H(3, h, q)
    return f1i_res

# расчёт F2
def F2_calc(i, i0, ii, a0, a1, q, f):
    f2 = [f[j] * np.power(j - i0 - q, 3) for j in range (i0, ii + 1)]
    f2i_res = sum(f2) - a0[i] * H(3, h, q) - a1[i] * H(4, h, q)
    return f2i_res

# основной метод, дающий прогноз
def RSS(h, ro, n, f, y):
    i0 = 0
    i = 1
    m = h + 1
    q = 0
    k = n - (i0 + h) + 1
    koef_ii = [ i0 + h + i for i in range(k)] # коэффициенты ii
    d_ii = m + k
    a_ii = k + 1
    a0 = np.zeros(a_ii) # слздание массивов с 0
    a1 = np.zeros(a_ii)
    a2 = np.zeros(a_ii)
    a3 = np.zeros(a_ii)
    F1 = np.zeros(a_ii)
    F2 = np.zeros(a_ii)
    S = np.zeros(d_ii)
    d = np.zeros(d_ii)
    a0[0] = y[0]
    A = 6 * (1 - ro) * (h**3) * (h - 2 * q) + ro * H(5, h, q) # h^4
    B = 4 * (1 - ro) * (h**3) + ro * H(4, h, q)               # h^3
    C = 12 * (1 - ro) * (h**3) * ((h**2) - 3*h*q + 3*(q**2)) + ro * H(6, h, q) # h^5
    for ii in koef_ii:
        a0[i] = a0[i-1] + a1[i-1] * 1 + a2[i-1] * (1**2) + a3[i-1] * (1**3)
        a1[i] = a1[i-1] + 2 * a2[i-1] * 1 + 3 * a3[i-1] * (1**2)
        F1[i] = F1_calc(i, i0, ii, a0, a1, q, f)
        F2[i] = F2_calc(i, i0, ii, a0, a1, q, f)
        a2[i] = (ro * (F1[i] * C - F2[i] * A)) / (C * B - A**2)
        a3[i] = (ro * (F2[i] * B - F1[i] * A)) / (C * B - A**2)
        S[m+i0] = a0[i] + a1[i] * m + a2[i] * (m**2) + a3[i] * (m**3)
        d[m+i0] = S[m+i0] - y[m+i0]
        i = i + 1
        i0 = i0 + 1
    temp = 100 / (max(y) - min(y))
    d_s = [d[i]**2 for i in range(h, n-h+m+1)]
    d_sum = sum(d_s)
    sig = math.sqrt(float(d_sum / (n - 2 * h))) * temp # RMSPE
    return S, d, sig


class MatplotlibWidget(QMainWindow):

    file_file = ""

    def __init__(self):
        QMainWindow.__init__(self)

        loadUi("qt_designer2.ui", self)

        self.setWindowTitle("Прогнозирование в РРВ")

        # привязка функций к кнопкам
        self.pushButton.clicked.connect(self.load_2graphs)
        self.pushButton_2.clicked.connect(self.change_graphs)
        self.pushButton_openDialog.clicked.connect(self.load_csv)
        # сделать кнопку неактивной
        self.pushButton_2.setEnabled(False)

        self.addToolBar(NavigationToolbar(self.MplWidget.canvas, self))

        # настройка высоты строк
        self.tableWidget.setRowHeight(0, 45)
        self.tableWidget.setRowHeight(1, 45)
        self.tableWidget.setRowHeight(2, 45)
        self.tableWidget_2.setRowHeight(0, 45)
        self.tableWidget_2.setRowHeight(1, 45)
        self.tableWidget_2.setRowHeight(2, 45)


    def load_2graphs(self):

        self.MplWidget.canvas.axes1.set_visible(True)
        self.MplWidget.canvas.axes2.set_visible(True)
        self.MplWidget.canvas.axes3.set_visible(False)

        n = 100
        x_plot = list(range(0, n + 2))

        y11 = [y1(i) for i in x_plot]
        s = 0.1 * max(y11)
        f1 = [y11[i] + (-1) ** i * random.uniform(0, s) for i in x_plot]
        h = 10

        ro1 = 0.2
        rss1, ost1, osh1 = RSS(h, ro1, n, f1, y11)

        ro2 = 0.5
        rss2, ost2, osh2 = RSS(h, ro2, n, f1, y11)

        ro3 = 0.9
        rss3, ost3, osh3 = RSS(h, ro3, n, f1, y11)

        y55 = [y5(i) for i in x_plot]
        s = 0.1 * max(y55)
        f5 = [y55[i] + (-1) ** i * random.uniform(0, s) for i in x_plot]

        ro1 = 0.2
        rss51, ost51, osh51 = RSS(h, ro1, n, f5, y55)

        ro2 = 0.5
        rss52, ost52, osh52 = RSS(h, ro2, n, f5, y55)

        ro3 = 0.9
        rss53, ost53, osh53 = RSS(h, ro3, n, f5, y55)


        titleTable1 = "10 * sin(2*п*t/100)"
        self.MplWidget.canvas.axes1.clear()
        self.MplWidget.canvas.axes1.set_title(titleTable1) #10 * np.sin(2 * np.pi * t / 100)
        self.MplWidget.canvas.axes1.plot(x_plot[11:], y11[11:], label="y")
        self.MplWidget.canvas.axes1.plot(x_plot[11:], rss1[11:], label=f"RSS, ro = {ro1}, RMSPE = {round(osh1, 2)}")
        self.MplWidget.canvas.axes1.plot(x_plot[11:], rss2[11:], label=f"RSS, ro = {ro2}, RMSPE = {round(osh2, 2)}")
        self.MplWidget.canvas.axes1.plot(x_plot[11:], rss3[11:], label=f"RSS, ro = {ro3}, RMSPE = {round(osh3, 2)}")
        self.MplWidget.canvas.axes1.plot(x_plot[11:], f1[11:], label="f")
        self.MplWidget.canvas.axes1.legend(loc='best')

        real = y11[14:]
        rmspe_array = np.array([osh1, osh2, osh3])
        min_osh = rmspe_array.min()
        # выбор ro с минимальной RMSPE-ошибкой
        if(min_osh == osh1):
            pred = rss1[14:]
            titleTable1Ro = str(ro1)
        elif(min_osh == osh2):
            pred = rss2[14:]
            titleTable1Ro = str(ro2)
        else:
            pred = rss3[14:]
            titleTable1Ro = str(ro3)

        # подсчёт показателей тенденций
        pocid, tendency, slg, aff, pda, arv = GetMeasures(real, pred)
        # вывод показателей тенденций в таблицу
        self.labelTable1.setText(f"{titleTable1}, ro = {titleTable1Ro}")
        self.tableWidget.setItem(1, 0, QTableWidgetItem(str(pocid)))
        self.tableWidget.setItem(1, 1, QTableWidgetItem(str(tendency)))
        self.tableWidget.setItem(1, 2, QTableWidgetItem(str(slg)))
        self.tableWidget.setItem(1, 3, QTableWidgetItem(str(aff)))
        self.tableWidget.setItem(1, 4, QTableWidgetItem(str(pda)))
        self.tableWidget.setItem(1, 5, QTableWidgetItem(str(arv)))

        titleTable2 = "sin(п*t/20) * exp(0.02*t) + 3"
        self.MplWidget.canvas.axes2.clear()
        self.MplWidget.canvas.axes2.set_title(titleTable2) # np.sin(np.pi * t / 20) * np.exp(0.02 * t) + 3
        self.MplWidget.canvas.axes2.plot(x_plot[11:], y55[11:], label="y")
        self.MplWidget.canvas.axes2.plot(x_plot[11:], rss51[11:], label=f"RSS, ro = {ro1}, RMSPE = {round(osh51, 2)}")
        self.MplWidget.canvas.axes2.plot(x_plot[11:], rss52[11:], label=f"RSS, ro = {ro2}, RMSPE = {round(osh52, 2)}")
        self.MplWidget.canvas.axes2.plot(x_plot[11:], rss53[11:], label=f"RSS, ro = {ro3}, RMSPE = {round(osh53, 2)}")
        self.MplWidget.canvas.axes2.plot(x_plot[11:], f5[11:], label="f")
        self.MplWidget.canvas.axes2.legend(loc='best')

        real = y55[14:]
        rmspe_array = np.array([osh51, osh52, osh53])
        min_osh = rmspe_array.min()

        if(min_osh == osh51):
            pred = rss51[14:]
            titleTable1Ro = str(ro1)
        elif(min_osh == osh52):
            pred = rss52[14:]
            titleTable1Ro = str(ro2)
        else:
            pred = rss53[14:]
            titleTable2Ro = str(ro3)

        # подсчёт показателей тенденций
        pocid, tendency, slg, aff, pda, arv = GetMeasures(real, pred)
        # вывод показателей тенденций в таблицу
        self.labelTable2.setText(f"{titleTable2}, ro = {titleTable2Ro}")
        self.tableWidget_2.setItem(1, 0, QTableWidgetItem(str(pocid)))
        self.tableWidget_2.setItem(1, 1, QTableWidgetItem(str(tendency)))
        self.tableWidget_2.setItem(1, 2, QTableWidgetItem(str(slg)))
        self.tableWidget_2.setItem(1, 3, QTableWidgetItem(str(aff)))
        self.tableWidget_2.setItem(1, 4, QTableWidgetItem(str(pda)))
        self.tableWidget_2.setItem(1, 5, QTableWidgetItem(str(arv)))

        self.MplWidget.canvas.figure.tight_layout()
        self.MplWidget.canvas.draw()


    def change_graphs(self):

        self.MplWidget.canvas.axes1.set_visible(False)
        self.MplWidget.canvas.axes2.set_visible(False)
        self.MplWidget.canvas.axes3.set_visible(True)
        self.MplWidget.canvas.draw()
        self.show_my_graph()

    def load_csv(self):
        # открываем проводник, выбираем .csv файл
        filepath = QFileDialog.getOpenFileName(self, 'Open file', '/ТПУ/УИРС 3 курс/6 СЕМЕСТР', "Text (*.csv)")[0]
        self.file_file = filepath
        self.show_my_graph()
        self.pushButton_2.setEnabled(True)


    def show_my_graph(self):
        filepath = self.file_file
        filename = QFileInfo(filepath).baseName()

        series = read_csv(filepath, header = 0, index_col = 0)
        series_values2 = series.values
        series_values = series_values2.flatten().tolist()

        h = 10
        n = len(series_values) - 2
        x_pl = list(range(len(series_values)))

        ro1 = 0.2
        rss_51, ost_51, osh_51 = RSS(h, ro1, n, series_values, series_values)

        ro2 = 0.5
        rss_52, ost_52, osh_52 = RSS(h, ro2, n, series_values, series_values)

        ro3 = 0.9
        rss_53, ost_53, osh_53 = RSS(h, ro3, n, series_values, series_values)

        real = series_values[14:]
        rmspe_array = np.array([osh_51, osh_52, osh_53])
        min_osh = rmspe_array.min()

        if(min_osh == osh_51):
            pred = rss_51[14:]
            titleTable1Ro = str(ro1)
        elif(min_osh == osh_52):
            pred = rss_52[14:]
            titleTable1Ro = str(ro3)
        else:
            pred = rss_53[14:]
            titleTable1Ro = str(ro3)

        # подсчёт показателей тенденций
        pocid, tendency, slg, aff, pda, arv = GetMeasures(real, pred)
        # вывод показателей тенденций в таблицу
        self.labelTable1.setText(f"{filename}, ro = {titleTable1Ro}")
        self.tableWidget.setItem(1, 0, QTableWidgetItem(str(pocid)))
        self.tableWidget.setItem(1, 1, QTableWidgetItem(str(tendency)))
        self.tableWidget.setItem(1, 2, QTableWidgetItem(str(slg)))
        self.tableWidget.setItem(1, 3, QTableWidgetItem(str(aff)))
        self.tableWidget.setItem(1, 4, QTableWidgetItem(str(pda)))
        self.tableWidget.setItem(1, 5, QTableWidgetItem(str(arv)))

        self.labelTable2.setText("")
        self.tableWidget_2.setItem(1, 0, QTableWidgetItem(""))
        self.tableWidget_2.setItem(1, 1, QTableWidgetItem(""))
        self.tableWidget_2.setItem(1, 2, QTableWidgetItem(""))
        self.tableWidget_2.setItem(1, 3, QTableWidgetItem(""))
        self.tableWidget_2.setItem(1, 4, QTableWidgetItem(""))
        self.tableWidget_2.setItem(1, 5, QTableWidgetItem(""))

        self.MplWidget.canvas.axes1.set_visible(False)
        self.MplWidget.canvas.axes2.set_visible(False)
        self.MplWidget.canvas.axes3.set_visible(True)

        self.MplWidget.canvas.axes3.clear()

        self.MplWidget.canvas.axes3.set_title(f"{filename}")
        self.MplWidget.canvas.axes3.plot(x_pl[11:], series_values[11:], label="y")
        self.MplWidget.canvas.axes3.plot(x_pl[11:], rss_51[11:], label=f"RSS, ro = {ro1}, RMSPE = {round(osh_51, 2)}")
        self.MplWidget.canvas.axes3.plot(x_pl[11:], rss_52[11:], label=f"RSS, ro = {ro2}, RMSPE = {round(osh_52, 2)}")
        self.MplWidget.canvas.axes3.plot(x_pl[11:], rss_53[11:], label=f"RSS, ro = {ro3}, RMSPE = {round(osh_53, 2)}")
        self.MplWidget.canvas.axes3.legend(loc='best')

        self.MplWidget.canvas.draw()


app = QApplication([])
window = MatplotlibWidget()
window.show()
app.exec_()



