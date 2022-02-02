# Copyright 2021 DimaZzZz101 zabotin.d@list.ru

import matplotlib.pyplot as plt
import numpy as np
import math as m


# По варианту:
# Задача: Выбор спутника жизни;
# Поиск расстояния: Евклидова метрика;

# Альтернативы:
# А. Анатолий;
# В. Александр;
# С. Владимир;
# D. Сергей;

# Критерии:
# 1. Образование;
# 2. Физическая подготовка;
# 3. Внешность;
# 4. Характер;

# Метод замены критериев ограничениями.
class ForMajorCriterion:
    def __init__(self, matrix, index):
        self.matrix = matrix.copy()
        self.index = index

    def matrix_normalization(self):
        min_ = np.min(self.matrix, axis=0)
        max_ = np.max(self.matrix, axis=0)
        for i in range(4):
            if i != self.index:
                for j in range(4):
                    self.matrix[j][i] = (self.matrix[j][i] - min_[i]) / (max_[i] - min_[i])
        return self.matrix


# Возврат индексов жизнеспособных альтернатив по порядку на основе главного критерия.
def criteria_to_restrictions(weights, restrictions):
    assert 1. in restrictions, "Отсутствует главный критерий."

    # Запоминаем столбец с главным критерием.
    main = np.where(restrictions == 1.)
    assert len(main), "Главный критерий может быть только один."

    # Транспонируем матрицу и находим максимальные элементы в строках этой матрицы.
    maxes = np.max(np.transpose(weights), axis=1)

    # Индикаторы перехода выше допустимого ограничения.
    indicators = np.array([weight >= restrictions * maxes for weight in weights.astype(float)])

    # Чтобы ориентироваться по главному критерию переводим все критерии до него в true.
    indicators[:, main] = True

    # Построчно берем индексы тех строк, которые равны true.
    idx_list = np.flatnonzero(np.all(indicators, axis=1))

    return idx_list


# Расчет Евклидовой метрики.
def euclidean_metric(s_point, f_point):
    return m.sqrt((s_point[0] - f_point[0]) ** 2 + (s_point[1] - f_point[1]) ** 2)


# Построение множества Парето, вычисление расстояния и возврат индекса наилучшего решения.
def create_pareto_set(weights, criteria, utopia, criteria_names, names):
    assert len(criteria) == 2, "Необходимо 2 критерия."

    # Из критериев(столбцов матрицы оценок) берем координаты точек.
    points = weights[:, criteria]

    # Для каждой точки вычисляем евклидово расстояние до точки утопии -> массив расстояний.
    dist = [euclidean_metric(point, utopia) for point in points]

    # Строим график для демонстрации.
    x, y = np.transpose(points)
    colors = np.random.rand(len(x))
    plt.title('Множество Парето:')
    plt.scatter(x, y, c=colors)
    for i, label in enumerate(names):
        plt.annotate(label, (x[i], y[i]))
    plt.scatter(*utopia, c='red')
    plt.annotate('Точка утопии', utopia)
    plt.xticks(np.arange(0, 11))
    plt.yticks(np.arange(0, 11))
    plt.xlabel(criteria_names[0])
    plt.ylabel(criteria_names[1])
    plt.grid()
    plt.plot(x, y)
    plt.savefig("Graphics/pareto.png")

    return dist


# Создание матрицы весов.
def create_weights_matrix(amount, pair_comparisons, inv_function):
    result = [[1. for _ in range(amount)] for _ in range(amount)]
    offset = 0

    for i in range(amount - 1):
        j = amount - 1 - i
        base = pair_comparisons[offset:offset + j]
        result[i][i + 1:] = base
        mirrored = [inv_function(elem) for elem in base[::-1]]
        result[j][:j] = mirrored
        offset += j

    return result


# Нормализация матрицы.
def normalized_priority_vector(matrix):
    # Вычисление среднего геометрического в каждой строке матрицы.
    # Произведение элементов строки матрицы в степени 1 / (размер матрицы).
    geometric_means = np.array([np.prod(row) ** (1. / len(matrix)) for row in matrix])
    geometric_means_sum = np.sum(geometric_means)

    return geometric_means, geometric_means / geometric_means_sum


# Вычисление коэффициента согласованности.
def count_accordance(matrix, npv, rci):
    column_sums = np.sum(np.transpose(matrix), axis=1)
    own_value = sum(np.multiply(column_sums, npv))
    cons_i = (own_value - len(npv)) / (len(npv) - 1)

    return cons_i / rci
