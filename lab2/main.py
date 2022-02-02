# Copyright 2021 DimaZzZz101 zabotin.d@list.ru


from tabulate import tabulate
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Начальные параметры:
_a = 1
_b = 4

_p = [0.9 + i / 100 for i in range(0, 10)]
_q = [0.005 * i for i in range(1, 21)]


# Унимодальная функция.
def unimodal_func(x):
    return -np.sqrt(x) * np.sin(x) + 2


# Мультимодальная функция.
def multimodal_func(x):
    return unimodal_func(x) * np.sin(5 * x)


# Функция подсчета количества итераций.
def number_of_iterations(p, q):
    return np.ceil(np.log(1 - p) / np.log(1 - q)).astype('int')


# Функция формирования таблицы.
def create_table(data):
    pd.set_option('display.max_rows', None)
    table = pd.DataFrame(data=data)
    table.set_index('q\P', inplace=True)
    print(tabulate(table, headers='keys', tablefmt='psql'), end='\n\n')


# Функция случайного поиска.
def random_search(function, a, b, P, Q):
    n_list = []
    y_min_list = []

    for p in P:
        for q in Q:
            n = number_of_iterations(p, q)
            y_min = None
            for i in range(0, n):
                x = np.random.uniform(a, b)
                if y_min is None or function(x) < y_min:
                    y_min = function(x)
            n_list.append(n)
            y_min_list.append(y_min)

    data = {'q\P': Q}

    # Таблица с количеством итераций.
    data.update({P[i]: n_list[i * 20:i * 20 + 20] for i in range(10)})
    create_table(data)

    # Таблица с результатами поиска минимума.
    data.update({P[i]: y_min_list[i * 20:i * 20 + 20] for i in range(10)})
    create_table(data)


# Функция вывода графика.
def show_graphic(name="", x=None, y=None):
    if y is None:
        y = []

    if x is None:
        x = []

    fig = plt.figure(figsize=(7, 7))
    plt.plot(x, y)

    plt.title(name, fontsize=15)  # Заголовок.
    plt.xlabel("x", fontsize=14)  # Ось абсцисс.
    plt.ylabel("y", fontsize=14)  # Ось ординат.
    plt.grid(True)  # Включение отображение сетки.
    plt.show()
    save = "Graphics/" + name + ".png"
    fig.savefig(save)


print("Унимодальная функция")
random_search(unimodal_func, _a, _b, _p, _q)

print("Мультимодальная функция")
random_search(multimodal_func, _a, _b, _p, _q)

X = np.arange(_a, _b, 0.01)

Y1 = unimodal_func(X)
Y2 = multimodal_func(X)

show_graphic("Унимодальная функция", X, Y1)
show_graphic("Мультимодальная функция", X, Y2)
