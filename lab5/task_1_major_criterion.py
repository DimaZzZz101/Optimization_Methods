# Copyright 2021 DimaZzZz101 zabotin.d@list.ru

import functions as fn
import numpy as np


# Задание 1.
if __name__ == '__main__':
    print('Метод главного критерия:', end='\n\n')

    alternatives = {
        'A': "Анатолий",
        'B': "Александр",
        'C': "Владимир",
        'D': "Сергей",
    }

    names = np.array(list('ABCD'))

    # Вектор весов критериев.
    criteria_weights = np.array([8, 6, 2, 4])

    # Нормальзуем вектор весов критериев.
    normalized_criteria_weights = np.divide(criteria_weights, np.sum(criteria_weights))

    # Матрица оценок альтернатив.
    rating_matrix = np.array([[3., 9., 1., 8.],
                              [5., 6., 3., 9.],
                              [7., 4., 4., 4.],
                              [9., 1., 1., 1.]])

    normed_matrix = fn.ForMajorCriterion(rating_matrix, 1).matrix_normalization()

    # Минимально допутсимые уровни для критериев.
    lower_thresholds = np.array([0.4, 1, 0.6, 0.6])

    # Заменяем критерии на ограничения для приска подходящего решения.
    indices = fn.criteria_to_restrictions(normed_matrix, lower_thresholds)

    # Вывод результатов.
    print('Вектор весов критериев:', criteria_weights, end='\n\n')
    print('Нормализованный вектор весов критериев: ', np.round(normalized_criteria_weights, 2), end='\n\n')
    print('Ограничения на веса: ', lower_thresholds, end='\n\n')

    print('Матрица оценок альтернатив:')
    for i, row in zip(names, rating_matrix):
        print(str(i), *row, sep=' | ')
    print()

    print('Нормированная матрица:')
    for i, row in zip(names, normed_matrix):
        print(i, *[format(e, '.3f') for e in row], sep=' | ')
    print()

    print('Выбранная альтернатива:')
    for i, row in zip(names[indices], rating_matrix[indices]):
        print(f'{alternatives[i]}:')
        print('\t', str(i), *row, sep=' | ')
