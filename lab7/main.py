# Copyright 2021 DimaZzZz101 zabotin.d@list.ru

import pandas as pd

TIME = {
    'a': 3,
    'b': 5,
    'c': 2,
    'd': 4,
    'e': 3,
    'f': 1,
    'g': 4,
    'h': 3,
    'i': 3,
    'j': 2,
    'k': 5
}


# Функция формирования таблицы.
def create_table(data_, id_):
    with pd.option_context('display.width', None):
        table = pd.DataFrame(data_, index=id_)
        print(table)


# Парсер файла с исходными данными.
def parse_file(file_):
    counter_ = -1
    tasks_ = {}
    for line in file_:
        parsed_line = line.split(',')
        counter_ += 1
        for i in range(len(parsed_line)):
            tasks_['task' + str(parsed_line[0])] = dict()
            tasks_['task' + str(parsed_line[0])]['id'] = parsed_line[0]
            tasks_['task' + str(parsed_line[0])]['name'] = parsed_line[0]
            tasks_['task' + str(parsed_line[0])]['duration'] = parsed_line[1]

            if parsed_line[2] != "\n":
                tasks_['task' + str(parsed_line[0])]['dependencies'] = parsed_line[2].strip().split(';')
            else:
                tasks_['task' + str(parsed_line[0])]['dependencies'] = ['-1']

            tasks_['task' + str(parsed_line[0])]['e_s'] = 0
            tasks_['task' + str(parsed_line[0])]['e_f'] = 0
            tasks_['task' + str(parsed_line[0])]['l_s'] = 0
            tasks_['task' + str(parsed_line[0])]['l_f'] = 0
            tasks_['task' + str(parsed_line[0])]['f'] = 0
            tasks_['task' + str(parsed_line[0])]['isCritical'] = False

    return tasks_


def forward(__data):
    for forward_bypass in __data:
        # Если это первая задача.
        if '-1' in __data[forward_bypass]['dependencies']:
            __data[forward_bypass]['e_s'] = 0

            __data[forward_bypass]['e_f'] = (__data[forward_bypass]['duration'])
        else:
            for key in __data.keys():
                for dependence in __data[key]['dependencies']:
                    # Если у задачи есть одна предшествующая задача.
                    if dependence != '-1' and len(__data[key]['dependencies']) == 1:
                        __data[key]['e_s'] = int(__data['task' + dependence]['e_f'])

                        __data[key]['e_f'] = int(__data[key]['e_s']) + int(__data[key]['duration'])
                    # Если у задачи есть более одной предшествующей задачи.
                    elif dependence != '-1':
                        if int(__data['task' + dependence]['e_f']) > int(__data[key]['e_s']):
                            __data[key]['e_s'] = int(__data['task' + dependence]['e_f'])

                            __data[key]['e_f'] = int(__data[key]['e_s']) + int(__data[key]['duration'])
    return __data


def backward(data__, tasks):
    for backward_bypass in data__:
        # Если задача последняя.
        if data__.index(backward_bypass) == 0:
            tasks[backward_bypass]['l_f'] = tasks[backward_bypass]['e_f']

            tasks[backward_bypass]['l_s'] = tasks[backward_bypass]['e_s']

        for dependence in tasks[backward_bypass]['dependencies']:
            # Если задача не последняя.
            if dependence != '-1':
                if tasks['task' + dependence]['l_f'] == 0:
                    tasks['task' + dependence]['l_f'] = int(tasks[backward_bypass]['l_s'])

                    tasks['task' + dependence]['l_s'] = \
                        int(tasks['task' + dependence]['l_f']) - int(tasks['task' + dependence]['duration'])

                    tasks['task' + dependence]['f'] = \
                        int(tasks['task' + dependence]['l_f']) - int(tasks['task' + dependence]['e_f'])

                if int(tasks['task' + dependence]['l_f']) > int(tasks[backward_bypass]['l_s']):
                    tasks['task' + dependence]['l_f'] = int(tasks[backward_bypass]['l_s'])

                    tasks['task' + dependence]['l_s'] = \
                        int(tasks['task' + dependence]['l_f']) - int(tasks['task' + dependence]['duration'])

                    tasks['task' + dependence]['f'] = \
                        int(tasks['task' + dependence]['l_f']) - int(tasks['task' + dependence]['e_f'])

    return tasks


if __name__ == '__main__':
    file = open('input.txt')

    tasks = parse_file(file)

    # Прямой проход.
    tasks = forward(tasks)

    # Копируем и переворачиваем tasks.
    array1 = []
    for task in tasks.keys():
        array1.append(task)
    array2 = array1[:]
    array2.reverse()

    # Обратный проход.
    tasks = backward(array2, tasks)

    ID = []

    data = {
        'Длительность задачи': [],
        'Раннее начало': [],
        'Раннее окончание': [],
        'Позднее начало': [],
        'Позднее окончание': [],
        'Полный резерв': [],
        'Критический путь': []
    }

    for task in tasks:
        ID.append(tasks[task]['id'])
        data['Длительность задачи'].append(tasks[task]['duration'])
        data['Раннее начало'].append(tasks[task]['e_s'])
        data['Раннее окончание'].append(tasks[task]['e_f'])
        data['Позднее начало'].append((tasks[task]['l_s']))
        data['Позднее окончание'].append((tasks[task]['l_f']))
        data['Полный резерв'].append((tasks[task]['f']))
        data['Критический путь'].append(tasks[task]['f'] == 0)

    create_table(data, ID)

    print('\nДлина критического пути:', tasks['taskstop']['l_s'])
