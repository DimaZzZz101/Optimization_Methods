# Copyright 2021 DimaZzZz101 zabotin.d@list.ru


from tabulate import tabulate
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def to_fixed(num_obj, digits=0):
    return f"{num_obj:.{digits}f}"


# Заданные константы.
_epsilon = 0.1  # Интервал неопределенности.
_begin = 1  # Левая граница.
_end = 4  # Правая граница.

# Список co со значениями длин при пассивном поиске.
list_of_length_passive = [(_end - _begin) / (n + 1) for n in range(1, 60)]

# Список со значениями длин при методе дихотомии.
list_of_length_dichotomy = []


# Функция из варианта.
def func_from_task(x):
    return -np.sqrt(x) * np.sin(x) - 1.5


# Сравнение погрешностей методов пассивного поиска и дихотомии.
def compare_inaccuracy(name="", length_passive=None, length_dichotomy=None):
    if length_passive is None:
        length_passive = []

    if length_dichotomy is None:
        length_dichotomy = []

    x1 = [n for n in range(1, len(length_passive) + 1)]
    x2 = [n for n in range(1, len(length_dichotomy) + 1)]

    y1 = length_passive
    y2 = length_dichotomy

    fig = plt.figure(figsize=(7, 7))
    plt.plot(x1, y1, label="Пассивный поиск")
    plt.plot(x2, y2, label="Метод дихотомии")
    plt.title(name, fontsize=15)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()
    save = "Graphics/" + name + ".png"
    fig.savefig(save)


# Функция подсчета количества итераций.
def count_of_iterations(begin, end, epsilon):
    return 2 * (end - begin) / epsilon - 1


# Функция подсчета точности.
def count_delta(epsilon):
    return epsilon / 2


# Функция вывода таблицы для метода пассивного поиска на экран.
def create_table_passive_search(x_array, y_array, n):
    pd.set_option('display.max_rows', n)

    table = pd.DataFrame({
        'N': list(range(1, n + 1)),
        'x': x_array,
        'f(x)': y_array,
    })

    table.set_index('N', inplace=True)

    print(tabulate(table, headers='keys', tablefmt='psql'), end='\n\n')


# Метод оптимального пассивного поиска.
def optimal_passive_search(begin, end, epsilon):
    delta = count_delta(epsilon)

    list_of_x = np.arange(begin + delta, end, delta)
    list_of_y = [func_from_task(x) for x in list_of_x]

    n = int(count_of_iterations(begin, end, epsilon))

    # Задание начального состояния переменных для поиска минимума.
    y_min = 5
    x_min = 0

    # Поиск y_min и соответствующего x_min.
    for x in list_of_x:
        y = func_from_task(x)
        if y < y_min:
            x_min = x
            y_min = y

    # Вывод таблицы со всеми x и y.
    print("Метод оптимального пассивного поиска:")
    create_table_passive_search(list_of_x, list_of_y, n)

    # Вывод результата - искомых x_min и y_min.
    print(f"Результат: x_min = {to_fixed(x_min, 3)} ± {delta}, y_min = {to_fixed(y_min, 3)}", end='\n\n')


# Функция печати таблицы для метода дихотомии.
def create_table_dichotomy(list_a, list_b, list_x1, list_x2, list_f1, list_f2, list_yes_no, list_a_, list_b_, length):
    pd.set_option('display.max_rows', None)

    table = pd.DataFrame({
        'a': list_a,
        'b': list_b,
        'x1': list_x1,
        'x2': list_x2,
        'f(x1)': list_f1,
        'f(x2)': list_f2,
        'f(x1) > f(x2) - ?': list_yes_no,
        'a_k+1': list_a_,
        'b_k+1': list_b_,
        'length': length
    })
    table.set_index('a', inplace=True)

    print(tabulate(table, headers='keys', tablefmt='psql'), end='\n\n')


# Поиск методом дихотомии.
def dichotomy_search(begin, end, epsilon):
    new_a = begin
    new_b = end

    delta = 0.01

    # Списки для сохранения данных будущей таблицы
    list_a = []
    list_b = []

    list_x1 = []
    list_x2 = []

    list_f1 = []
    list_f2 = []

    list_yes_no = []

    list_a_ = []
    list_b_ = []

    while True:
        # Сохраняем текущие начало и конец отрезка
        list_a.append(new_a)
        list_b.append(new_b)

        # Вычисляем расстояние между концами отрезка.
        length = new_b - new_a

        # Вычисление точности.
        precision = length / 2

        # Запоминаем текущую дляину отрезка
        list_of_length_dichotomy.append(length)

        # Вычисляем, затем сохраняем полученные значения точек в списки.
        x1 = (new_a + new_b) / 2 - delta
        x2 = (new_a + new_b) / 2 + delta

        list_x1.append(x1)
        list_x2.append(x2)

        # К текущим значениям точек находим соответсвующие значения функций.
        f_x1 = func_from_task(x1)
        f_x2 = func_from_task(x2)

        # Сохраняем их.
        list_f1.append(f_x1)
        list_f2.append(f_x2)

        if f_x1 > f_x2:
            new_a = x1
            list_yes_no.append('yes')
        else:
            new_b = x2
            list_yes_no.append('no')

        # Сохраняем новые начало и конец отрезка.
        list_a_.append(new_a)
        list_b_.append(new_b)

        # Условие выхода из цикла.
        if length < epsilon:
            x_part1 = (new_a + new_b) / 2  # Запоминаем полученный минимальный x.

            y_min = func_from_task(x_part1)  # Ему в соответсвие вычисляем значение функции в нем.

            # Вывод таблицы для метода дихотомии.
            print("Метод дихотомии:")
            create_table_dichotomy(list_a, list_b, list_x1,
                                   list_x2, list_f1, list_f2,
                                   list_yes_no, list_a_, list_b_,
                                   list_of_length_dichotomy)

            # Вывод результата.
            print(f'Результат: x_min = {to_fixed(x_part1, 4)} ± {to_fixed(precision, 4)}, y_min = {to_fixed(y_min, 4)}')
            break


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


# Вывод пояснительной надписи.
print('Лабораторная работа №1', 'Вариант 7', 'Функция: -sqrt(x) * sin(x) - 1.5', sep='\n', end='\n\n')

# Метод пассивного поиска.
optimal_passive_search(_begin, _end, _epsilon)

# Метод дихотомии.
dichotomy_search(_begin, _end, _epsilon)

# Построение графика функции.
x_array = np.arange(_begin, _end, 0.01)
y_array = func_from_task(x_array)
show_graphic("График функции f(x)", x_array, y_array)

# Построение графика сравнения работы метода пассивного поиска и метода дихотомии, в зависимости от N.
compare_inaccuracy("График зависимости погрешности от количества итераций", list_of_length_passive,
                   list_of_length_dichotomy)
