# Copyright 2021 DimaZzZz101 zabotin.d@list.ru

import functions as fn
import numpy as np


def print_matrix(matrix, columns=None, form='.3f', sep=' | '):
    if columns is not None:
        print(*columns, sep=sep)
    for row in matrix:
        print(*[format(e, form) for e in row], sep=sep)


# Задание 4.
if __name__ == '__main__':
    print('Метод анализа иерархий:', end='\n\n')

    alternatives = {
        'A': "Анатолий",
        'B': "Александр",
        'C': "Владимир",
        'D': "Сергей",
    }

    # Показатель случайной согласованности.
    # Random consistency score.
    rcs = 0.90

    names = np.array(list('ABCD'))
    # GM - geometry mean - среднее геометрическое.
    # NPV - normalized priority vector - нормализованный вектор приоритетов.
    columns = ['A    ', 'B    ', 'C    ', 'D    ', 'GM   ', 'NPV  ']
    print('Матрица для критерия "Образование": ')
    criteria = [1, 1 / 5, 1 / 7, 1 / 3, 1 / 7, 1 / 3]
    criteria_weights = fn.create_weights_matrix(4, criteria, lambda x: 1 / x)
    means, npv_1 = fn.normalized_priority_vector(criteria_weights)
    print_matrix(np.hstack([criteria_weights, means[:, np.newaxis], npv_1[:, np.newaxis]]), columns)
    accordance = fn.count_accordance(criteria_weights, npv_1, rcs)
    print('Отношение согласованности: {}'.format(round(accordance, 3)), '< 0.1', end='\n\n')

    print('Матрица для критерия "Физическая подготовка": ')
    criteria = [3, 7, 7, 5, 5, 3]
    criteria_weights = fn.create_weights_matrix(4, criteria, lambda x: 1 / x)
    means, npv_2 = fn.normalized_priority_vector(criteria_weights)
    print_matrix(np.hstack([criteria_weights, means[:, np.newaxis], npv_2[:, np.newaxis]]), columns)
    accordance = fn.count_accordance(criteria_weights, npv_2, rcs)
    print('Отношение согласованности: {}'.format(round(accordance, 3)), '< 0.1', end='\n\n')

    print('Матрица для критерия "Внешность": ')
    criteria = [5, 1, 5, 1 / 7, 1, 5]
    criteria_weights = fn.create_weights_matrix(4, criteria, lambda x: 1 / x)
    means, npv_3 = fn.normalized_priority_vector(criteria_weights)
    print_matrix(np.hstack([criteria_weights, means[:, np.newaxis], npv_3[:, np.newaxis]]), columns)
    accordance = fn.count_accordance(criteria_weights, npv_3, rcs)
    print('Отношение согласованности: {}'.format(round(accordance, 3)), '< 0.1', end='\n\n')

    print('Матрица для критерия "Характер": ')
    criteria = [1, 5, 5, 5, 5, 1]
    criteria_weights = fn.create_weights_matrix(4, criteria, lambda x: 1 / x)
    means, npv_4 = fn.normalized_priority_vector(criteria_weights)
    print_matrix(np.hstack([criteria_weights, means[:, np.newaxis], npv_4[:, np.newaxis]]), columns)
    accordance = fn.count_accordance(criteria_weights, npv_4, rcs)
    print('Отношение согласованности: {}'.format(round(accordance, 3)), '< 0.1', end='\n\n')

    print('Оценка приоритетов критериев: ')
    criteria = [5, 3, 7, 1 / 3, 5, 3]
    criteria_weights = fn.create_weights_matrix(4, criteria, lambda x: 1 / x)
    means, npv = fn.normalized_priority_vector(criteria_weights)
    print_matrix(np.hstack([criteria_weights, means[:, np.newaxis], npv[:, np.newaxis]]), columns)
    accordance = fn.count_accordance(criteria_weights, npv, rcs)
    print('Отношение согласованности: {}'.format(round(accordance, 3)), 'незначительно больше 0.1', end='\n\n')

    print('Нормальная матрица приоритетов:')
    npv_matrix = np.transpose([npv_1, npv_2, npv_3, npv_4])
    print_matrix(npv_matrix)
    print()
    print('Результирующий вектор')
    for i, j in zip(names, np.dot(npv_matrix, npv)):
        print(i, format(j, '.3f'), sep='|')
    print()
    print('Альтернатива:')
    alt = names[np.dot(npv_matrix, npv).argmax()]
    print(f'\t{alternatives[alt]}: {alt}')
