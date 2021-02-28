import json


def save_data(arch, folder_name, n_epochs=30, n_clients=10, n_experiments=10,
              learning_rate=1e-3,
              batch_size=128,
              epoch_step=1):
    data = dict()
    data['arch'] = str(type(arch))
    data['n_epochs'] = n_epochs
    data['n_clients'] = n_clients
    data['n_experiment'] = n_experiments
    data['learning_rate'] = learning_rate
    data['batch_size'] = batch_size
    data['epoch_step'] = epoch_step
    with open(folder_name + '/data.json', 'w') as fp:
        json.dump(data, fp)