# Copyright 2021 DimaZzZz101 zabotin.d@list.ru

import functions as fn
import numpy as np


# Задание 2.
if __name__ == '__main__':
    print('Формирование множества Парето:', end='\n\n')

    alternatives = {
        'A': "Анатолий",
        'B': "Александр",
        'C': "Владимир",
        'D': "Сергей",
    }

    names = np.array(list('ABCD'))

    # Матрица оценок альтернатив.
    rating_matrix = np.array([[3, 9, 1, 8],
                              [5, 6, 3, 9],
                              [7, 4, 4, 4],
                              [9, 1, 1, 1]])

    # Получаем массив расстояний до от каждой точки с координатами из двух критериев до точки утопии.
    dist = fn.create_pareto_set(rating_matrix, [0, 2], (10, 10), ['Образование', 'Внешность'], names)

    # Получаем индекс решения с минимальным расстоянием до точки утопии.
    index = np.argmin(dist)

    # Вывод результатов.
    print('Матрица критериев:')
    for i, row in zip(names, rating_matrix):
        print(i, *row, sep=' | ')
    print()
    print('Евклидово расстояние:')
    for i, d in zip(names, dist):
        print(i, d, sep=' | ')
    print()
    print('Альтернатива: ', alternatives[names[index]], names[index], *rating_matrix[index], sep=' | ')
