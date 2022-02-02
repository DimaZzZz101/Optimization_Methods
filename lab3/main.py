# Copyright 2021 DimaZzZz101 zabotin.d@list.ru


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

a_ = 1
b_ = 4

T_MAX_ = 10000
T_MIN_ = 0.1

coefficient_ = 0.95


def unimodal_func(x):
    return -np.sqrt(x) * np.sin(x) + 2


def multimodal_func(x):
    return unimodal_func(x) * np.sin(5 * x)


x_array_ = np.arange(a_, b_, 0.01)

y1 = unimodal_func(x_array_)
y2 = multimodal_func(x_array_)


def show_graphic(name, x=None, y=None):
    fig = plt.figure(figsize=(7, 7))

    plt.plot(x, y)

    plt.title(name, fontsize=15)  # заголовок.
    plt.xlabel("x", fontsize=14)  # ось абсцисс.
    plt.ylabel("y", fontsize=14)  # ось ординат.

    plt.grid(True)  # включение отображение сетки.
    plt.show()

    save = "Graphics/" + name + ".png"
    fig.savefig(save)


# Функция формирования таблицы.
def create_table(t_max_array, x_array, y_array, p_array, accept_array, x_best_array, y_best_array):
    pd.set_option('display.max_rows', None)
    table = pd.DataFrame({
        'N': [i for i in range(1, len(x_array) + 1)],
        'T': t_max_array,
        'x': x_array,
        'f(x)': y_array,
        'P': p_array,
        'accept': accept_array,
        'x best': x_best_array,
        'f(x) best': y_best_array
    })

    table.set_index('N', inplace=True)

    print(tabulate(table, headers='keys', tablefmt='psql'), end='\n\n')


# Метод имитации отжига
def annealing(a, b, t_max, t_min, coeff, function):
    # Массивы для хранения данных для будущей таблицы.
    t_max_array = []
    x_array = []
    y_array = []
    p_array = []
    accept_array = []
    x_best_array = []
    y_best_array = []

    # Первоначальный выбор случайной точки на заданном отрезке и вычисление знечения функции в ней.
    x_min = np.random.uniform(a, b)
    f_min = function(x_min)

    while t_max > t_min:
        # Вычисляем текущие x и f(x).
        x_i = np.random.uniform(a, b)
        f_i = function(x_i)

        # Находим разность между текущим и предыдущим значением функций.
        delta = f_i - f_min

        # Если текущее значение функции меньше либо равно предыдущему, то...
        if delta <= 0:
            # Вычисляем новые x и f(x).
            x_min = x_i
            f_min = function(x_min)
            # Переход осуществляется безусловно.
            accept_array.append('Y')
            p_array.append(1)
        else:
            # Если текущее значение больше предыдущего...
            # Вычисляется вероятность перехода
            p = np.exp(-delta / t_max) * 100
            # Переход выполняется с вероятностью, которая убывает с ростом delta и уменьшением температуры.
            if p >= np.random.uniform(0, 100):
                x_min = x_i
                f_min = function(x_min)
                accept_array.append('Y')
            else:
                # В противном случае переход не осуществляется.
                accept_array.append('-')
            p_array.append(p / 100)

        # Запоминаем вычисленные на текущей итерации значения.
        t_max_array.append(t_max)
        x_array.append(x_i)
        y_array.append(f_i)
        x_best_array.append(x_min)
        y_best_array.append(f_min)

        # Понижаем температуру.
        t_max *= coeff
    print(f"Результат: x_min = {x_min:.3f}, y_min = {f_min:.3f}")
    create_table(t_max_array, x_array, y_array, p_array, accept_array, x_best_array, y_best_array)


annealing(a_, b_, T_MAX_, T_MIN_, coefficient_, unimodal_func)
annealing(a_, b_, T_MAX_, T_MIN_, coefficient_, multimodal_func)

"""
show_graphic("Унимодальная функция", x_array_, y1)
print("Для унимодальной функции:")
annealing(a_, b_, T_MAX_, T_MIN_, coefficient_, unimodal_func)

show_graphic("Мультимодальная функция", x_array_, y2)
print("Для мультимодальной функции:")
annealing(a_, b_, T_MAX_, T_MIN_, coefficient_, multimodal_func)
"""
