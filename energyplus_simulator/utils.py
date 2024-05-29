import os


def save_path_handler(name, prefix, save_format: str, results_path: str, reversion=True):

    def auto_versioning(path_filename: str, save_format: str):
        version = 0
        path = path_filename
        while True:
            extension = str(version) + save_format
            if not os.path.isfile(path + extension):
                return path + extension
            version += 1

    path = results_path + "/data/"

    if name is None:
        reversion = True
        path = path
    else:
        path = path + name

    path = path if prefix is None else path + prefix

    if reversion is False:
        return path + '0' + save_format
    return auto_versioning(path, save_format)
