import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import functions as fn
import numpy as np


# Функция формирования таблицы.
def create_table_final(_data):
    pd.set_option('display.width', None)
    table = pd.DataFrame(data=_data)
    print(tabulate(table, tablefmt='psql'), end='\n\n')


# Функция построения графиков функций.
def plot_results(x_source, y_source, y_noisy, y_filtered, title, filename):
    plt.title(title)
    plt.plot(x_source, y_source, color='blue')
    plt.plot(x_source, y_noisy, color='orange')
    plt.plot(x_source, y_filtered, color='green')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend(['Normal', 'Noisy', 'Filtered'])
    plt.savefig("Results/Graphics/" + filename)
    plt.clf()


# Графическое отображение найденных приближений к оптимальным критериям в системе координат (noise, difference).
def plot_criteria(noise, difference, filename):
    plt.xlabel('Noise criteria')
    plt.xscale('log')
    plt.ylabel('Difference criteria')
    colors = np.random.rand(len(noise))
    plt.scatter(noise, difference, c=colors)
    plt.grid()
    plt.savefig("Results/Graphics/" + filename)
    plt.clf()


if __name__ == '__main__':
    # Начальная функция и зашумленная функция.
    function = lambda x: np.sin(x) + 0.5
    noised_function = fn.uniform_noise(function, amplitude=0.5)

    # Первоначальные данные:
    # l = 0, ..., L, L = 10
    arrangement = (0., 1., 11)

    # Количество попыток.
    tries = fn.tries_count(x_min=0., x_max=np.pi, eps=0.01, probability=0.95)

    # Генерация интервала.
    x_source = fn.gen_x()

    # Значения функций.
    y_source = function(x_source)
    y_noised = noised_function(x_source)

    # Основная часть:
    # Для r = 3.
    index, results = fn.find_weights(y_noised, 3, tries, arrangement)
    results.to_csv('Results/results_3.csv')
    weights = results.loc[index]['Alpha']
    y_filtered = fn.mean_harmonic_window(y_noised, weights)

    # Строим графики для окна r = 3.
    plot_results(x_source, y_source, y_noised, y_filtered, 'f(x) = sin(x) + 0.5, r = 3', 'length_3/results_3.png')
    plot_criteria(results['W'].values, results['D'].values, 'length_3/criteria_3.png')

    # Итоговый результат для окна размером 3.
    print('\nРезультат для для окна размером 3:', end='\n\n')

    create_table_final(results.loc[index])

    print('+=========+==========+==========+==========+==========+==========+============+')
    print('+=========+==========+==========+==========+==========+==========+============+', end='\n\n')

    # Для r = 5.
    index, results = fn.find_weights(y_noised, 5, tries, arrangement)
    results.to_csv('Results/results_5.csv')
    weights = results.loc[index]['Alpha']
    y_filtered = fn.mean_harmonic_window(y_noised, weights)

    # Строим графики для окна r = 5.
    plot_results(x_source, y_source, y_noised, y_filtered, 'f(x) = sin(x) + 0.5, r = 5', 'length_5/results_5.png')
    plot_criteria(results['W'].values, results['D'].values, 'length_5/criteria_5.png')

    # Итоговый результат для окна размером 5.
    print('Результат для для окна размером 5:', end='\n\n')
    create_table_final(results.loc[index])
