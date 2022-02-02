# Copyright 2021 DimaZzZz101 zabotin.d@list.ru

import functions as fn
import numpy as np


# Задание 3.
if __name__ == '__main__':
    print('Взвешивание и объединение критериев:', end='\n\n')

    alternatives = {
        'A': "Анатолий",
        'B': "Александр",
        'C': "Владимир",
        'D': "Сергей",
    }

    names = np.array(list('ABCD'))

    # Матрица оценок альтернатив.
    criteria_matrix = np.array([[3, 9, 1, 8],
                                [5, 6, 3, 9],
                                [7, 4, 4, 4],
                                [9, 1, 1, 1]])

    comparisons = [1, 0, 0, 0.5, 0.5, 0.5]

    # Нормализация матрицы.
    sums = np.sum(np.transpose(criteria_matrix), axis=1)
    criteria_matrix = np.divide(criteria_matrix, sums)

    # Создание массива весов из попарных сравнений.
    weights = fn.create_weights_matrix(4, comparisons, lambda x: 1 - x)

    # Вектор весов критериев.
    weights = np.sum(weights, axis=1)

    # Нормализация ветора весов критериев.
    weights = weights / np.sum(weights)

    # Умножение нормализованной матрицы на нормализованный вектор весов критериев.
    # В результате получим значение объединенного критерия для всех альтернатив.
    result = np.dot(criteria_matrix, weights)

    # Вывод результатов.
    print('Нормализированный вектор весов: ', weights, end='\n\n')

    print('Нормализированная матрица критериев:')
    for i, row in zip(names, criteria_matrix):
        print(i, *[format(e, '.3f') for e in row], sep=' | ')
    print()
    print('Объединенные критерии:')
    for i, r in zip(names, result):
        print(i, format(r, '.3f'), sep=' | ')
    print()
    print('Альтернатива: \n\t', names[result.argmax()], ' - ', alternatives[names[result.argmax()]])
