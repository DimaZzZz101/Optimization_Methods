# Copyright 2021 DimaZzZz101 zabotin.d@list.ru


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tabulate import tabulate

# Параметры популяции:
ITERATIONS = 10  # Количество итераций.
SPECIES = 4  # Первоначальное количество особей.
P_MUTATION = 0.25  # Вероятность мутаций.
FIXED = 4  # Количество знаков после запятой.
DELTA = 0.095

X1 = -2  # Левая граница для х.
X2 = 2  # Правая граница для х.

Y1 = -2  # Левая граница для х.
Y2 = 2  # Правая граница для х.


# Фитнес функция.
def fit_function(x_, y_):
    return np.exp(-(x_ ** 2) - (y_ ** 2))


# Сохранение трехмерного графика.
def show_figure(x1, x2, y1, y2, show=False):
    x, y = np.meshgrid(np.arange(x1, x2, 0.01), np.arange(y1, y2, 0.01))
    z = fit_function(x, y)

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_surface(x, y, z, cmap=plt.get_cmap('jet'))

    if show:
        plt.show()

    plt.title("График популяции", fontsize=15)
    save = "Graphics/gen_algorithm.png"
    fig.savefig(save)


# Информация для таблицы.
class InitPopulation:
    def __init__(self, population=None, max_fit=0, avg_fit=0):
        if population is None:
            population = []

        self.x_array = [x for x, y in population]
        self.y_array = [y for x, y in population]
        self.fit_array = [fit_function(x, y) for x, y in population]

        self.max = max_fit
        self.avg = avg_fit


class GeneticAlgorithm:
    # Конструктор класса.
    def __init__(self):
        self.history = {}
        self.population_number = 0
        self.population_size = SPECIES
        self.mutation_probability = P_MUTATION
        self.mutation_delta = DELTA

    # Главная функция.
    def __call__(self, function, interval_x, interval_y):
        self._interval_x = interval_x
        self._interval_y = interval_y
        self._function = function
        self._iterations = ITERATIONS

        # Создаем изначальную популяцию.
        population = self.source_population()

        # Selection -> crossover -> mutation
        for i in range(self._iterations + 1):
            population = self.mutation(self.crossover(self.selection(population)))

            population_data = InitPopulation(population=population,
                                             max_fit=self._max_fit(population),
                                             avg_fit=self.average_fit(population))

            print(f"Поколение: {self.population_number}")

            pd.set_option('display.max_rows', None)
            table = pd.DataFrame({
                'X': population_data.x_array,
                'Y': population_data.y_array,
                'FIT': population_data.fit_array,
                'MAX': [population_data.max for _ in range(len(population_data.x_array))],
                'AVERAGE': [population_data.avg for _ in range(len(population_data.x_array))]
            })

            table.set_index('X', inplace=True)

            print(tabulate(table, headers='keys', tablefmt='psql'), end='\n\n')

            self.population_number += 1

    # Генерация исходной популяции.
    def source_population(self):
        population = []

        for i in range(0, self.population_size):
            pair = (
                np.random.uniform(self._interval_x[0], self._interval_x[1]),
                np.random.uniform(self._interval_y[0], self._interval_y[1]))

            population.append(pair)

        return population

    # Функция отбора.
    def selection(self, population=None):
        if population is None:
            population = []

        selected_individs = []

        # Массив вероятностей (рулеточный алгоритм отбора).
        choice = self.make_choice(population)

        for i in range(0, 3):
            index = self._individual_selection(choice)
            selected_individs.append(population[index])
            selected_individs.sort(key=lambda individ: self._function(individ[0], individ[1]), reverse=True)

        return selected_individs

    # Функция кроссовера.
    def crossover(self, selected_population=None):
        if selected_population is None:
            selected_population = []

        # Массив для особой после кроссовера
        crossover_population = []

        # Выполнение кроссовера:
        # (x0, y0) + (x1, y1) -> (x0, y1) и (x1, y0)
        # (x0, y0) + (x2, y2) -> (x0, y2) и (x2, y0)
        first_pair = self._crossover_parents(selected_population[0], selected_population[1])
        second_pair = self._crossover_parents(selected_population[0], selected_population[2])

        crossover_population.extend(first_pair)
        crossover_population.extend(second_pair)

        return crossover_population

    # Функция мутации.
    def mutation(self, crossover_population):
        if crossover_population is None:
            crossover_population = []

        # Массив для новой популяции.
        mutation_population = []

        # Этап мутации.
        for (x, y) in crossover_population:
            mut_x, mut_y = x, y
            if np.random.uniform() <= self.mutation_probability:
                mut_x = self.mutate_gen_x(x)
            if np.random.uniform() <= self.mutation_probability:
                mut_y = self.mutate_gen_y(y)
            mutation_population.append((mut_x, mut_y))

        return mutation_population

    # Вычисление среднего значения FIT-функции.
    def average_fit(self, population=None):
        if population is None:
            population = []

        return self._get_sum(population) / len(population)

    # Алгоритм рулеточного отбора.
    def make_choice(self, population=None):
        if population is None:
            population = []

        # Массив вероятностей, полученных алгоритмом.
        choice = []

        if not population:
            return choice

        bound = 0

        for (x, y) in population:
            bound += self._function(x, y) / self._get_sum(population)
            choice.append(bound)

        return choice

    # Мутация.
    def mutate_gen_x(self, gen=0):
        if np.random.uniform() < self.mutation_probability:
            if gen - self.mutation_delta > self._interval_x[0]:
                gen -= self.mutation_delta
        else:
            if gen + self.mutation_delta < self._interval_x[1]:
                gen += self.mutation_delta

        return gen

    def mutate_gen_y(self, gen=0):
        if np.random.uniform() < self.mutation_probability:
            if gen - self.mutation_delta > self._interval_y[0]:
                gen -= self.mutation_delta
        else:
            if gen + self.mutation_delta < self._interval_y[1]:
                gen += self.mutation_delta

        return gen

    # Статические методы класса.
    @staticmethod
    def _get_sum(population=None):
        if population is None:
            population = []

        fit_population = [fit_function(x, y) for (x, y) in population]

        return sum(fit_population)

    # Вычисление максимума фитнес-функции.
    @staticmethod
    def _max_fit(population=None):
        if population is None:
            population = []

        fit_population = [fit_function(x, y) for (x, y) in population]

        return max(fit_population)

    # Индивидуальный отбор.
    @staticmethod
    def _individual_selection(choice=None):
        if choice is None:
            choice = []

        rnd = np.random.uniform()

        index = 0

        for probability in choice:
            if rnd <= probability:
                index = choice.index(probability)
                choice[index] = 0
                break

        return index

    # Кроссовер для "родителей".
    @staticmethod
    def _crossover_parents(first_parent=(), second_parent=()):
        first_child = (first_parent[0], second_parent[1])
        second_child = (second_parent[0], first_parent[1])

        return [first_child, second_child]


if __name__ == "__main__":
    gen_alg = GeneticAlgorithm()
    gen_alg(function=fit_function, interval_x=(X1, X2), interval_y=(Y1, Y2))

    show_figure(X1, X2, Y1, Y2)
