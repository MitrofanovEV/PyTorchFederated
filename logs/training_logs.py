
def log(string, folder):
    try:
        file = open(folder + '/experiment_log.txt', 'x')
        file.write(string + '\n')
        file.close()
    except FileExistsError:
        with open(folder + '/experiment_log.txt', 'a') as file:
            file.write(string + '\n')
    print(string)