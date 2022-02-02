import numpy as np
import pandas as pd
import random

SEED = 41


# Количество попыток.
def tries_count(x_min, x_max, eps, probability):
    return int(np.log(1 - probability) / np.log(1 - eps / (x_max - x_min)))


# Вычисление евклидова расстояния.
def euclid_dist(f_array, s_array):
    result = np.subtract(f_array, s_array)
    return np.sqrt(np.sum(result ** 2))


# Критерий зашумленности.
def euclid_noise_criterion(array):
    return euclid_dist(array[1:], array[:-1])


# Критерий отличия.
def euclid_diff_criterion(f_array, s_array):
    assert len(f_array) == len(s_array)
    result = euclid_dist(f_array, s_array)
    return result / len(f_array)


# Генерация интервала.
def gen_x(x_max=0, x_min=np.pi, K=101):
    return [(x_min + k * (x_max - x_min) / K) for k in range(0, K)]


# Равномерное зашумливание.
def uniform_noise(func, amplitude, seed=SEED):
    assert amplitude > 0
    # Задаем амплитуду равномерного шума.
    # 2a = 0.5 -> a = 0.5 / 2
    amplitude /= 2
    rnd_gen = np.random.default_rng(seed=seed)
    return lambda x: np.add(func(x), rnd_gen.uniform(-amplitude, amplitude, len(x)))


# Генератор окна скольжения.
def gen_window(window_size=3, seed=SEED):
    assert window_size > 1 and window_size % 2 == 1
    # Инициализация рандомайзера.
    random.seed(seed)
    while True:
        # Центральный вес.
        mid = random.uniform(0, 1)
        # Массив весов.
        w = []
        for _ in range(0, (window_size - 1) // 2 - 1):
            w.append(0.5 * random.uniform(0, 1 - mid - 2 * sum(w)))

        w.append(0.5 * (1 - mid - 2 * sum(w)))

        yield [*reversed(w), mid, *w]


# Скользящее среднее (среднее гармоническое).
def mean_harmonic_window(y_noisy, weights):
    m = (len(weights) - 1) // 2
    y_padded = np.pad(y_noisy, (m, m), 'constant', constant_values=1)
    return [np.sum([(w / y_padded[i + j]) for j, w in enumerate(weights)]) ** -1 for i, _ in enumerate(y_noisy)]


# Функция случайного поиска.
def random_search(y_noisy, window, tries, lam):
    weights = [next(window) for _ in range(tries)]

    y_filtered = [mean_harmonic_window(y_noisy, w) for w in weights]

    noise_criterion = [euclid_noise_criterion(y_a) for y_a in y_filtered]

    diff_criterion = [euclid_diff_criterion(y_a, y_noisy) for y_a in y_filtered]

    # n_c - omega, d_c - delta.
    # omega - критерий зашумленности.
    # delta - критерий отличия.
    # lin_convolution - линейная свертка.
    lin_convolution = [lam * n_c + (1. - lam) * d_c for n_c, d_c in zip(noise_criterion, diff_criterion)]

    index_min = np.argmin(lin_convolution, axis=0)

    return [np.round(weights[index_min], 6), np.round(noise_criterion[index_min], 6),
            np.round(diff_criterion[index_min], 6), np.round(lin_convolution[index_min], 6)]


def find_weights(y_noisy, window_size, tries, arrangement, seed=SEED):
    result = []

    weights = gen_window(window_size, seed)
    for lam in np.linspace(*arrangement):
        result.append([np.round(lam, 1), *random_search(y_noisy, weights, tries, lam)])
    result = pd.DataFrame(data=result, columns=['Lambda', 'Alpha', 'W', 'D', 'J'])

    # Считаем расстояние до идеальной точки.
    distance = np.sqrt(result['W'] ** 2 + result['D'] ** 2)

    result['Distance'] = np.round(distance, 6)
    return np.argmin(np.round(distance, 6)), result
