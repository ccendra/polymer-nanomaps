import os
import time


class ProgressBar():
    def __init__(self, total_tasks, interval=5):
        self.total_tasks = total_tasks
        self.milestone = 0
        self.gap = interval - 1
        self.tic = time.time()
        self.current_task = 0
        self.finished = False
        print('0%')

    def update(self, current_task, verbose=False, clear_when_finished=False):
        if current_task == '+1':
            self.current_task += 1
        else:
            self.current_task = current_task
        progress = int(self.current_task / self.total_tasks * 100)
        if progress - self.gap > self.milestone or (verbose and progress > 0):
            self.milestone = progress

            # Estimate time remaining
            self.toc = time.time()
            average_rate = (self.toc - self.tic) / progress
            time_remaining = (100 - progress) * average_rate
            m, s = divmod(time_remaining, 60)
            time_string = "%d:%02d" % (m, s)
            print(progress, '%', ' ' * 10, time_string, ' remaining')
            if progress > 99:
                total_time = self.toc - self.tic
                m, s = divmod(total_time, 60)
                print('Total Time ', "%d:%02d" % (m, s))
        if self.current_task == self.total_tasks - 1:
            total_time = self.toc - self.tic
            m, s = divmod(total_time, 60)
            time_string = "%d:%02d" % (m, s)
            print('Finished in ', time_string)

    def get_total_time(self):
        return self.toc - self.tic


def choose_file(path, file_number, file_type, extension=False):
    if extension:
        path = path + extension
    try:
        files = [f for f in os.listdir(path) if file_type in f]
        files.sort()
        all_filenames = {str(i): f for i, f in enumerate(files)}
        print('File Numbers for this directory:')
        for key in all_filenames:
            print(key, ": ", all_filenames[key])

        filename = all_filenames[str(file_number)]
    except FileNotFoundError:
        print('Faulty Directory Name - no file selected')
        raise
    except KeyError:
        print('\n File number out of range - no file selected')
        raise

    print('\nData Selected: ', path + '/' +  filename)

    return filename


def get_filenames(path, file_type, extension=False):
    if extension:
        path = path + extension
    try:
        files = [f for f in os.listdir(path) if file_type in f]
    except FileNotFoundError:
        print('Faulty Directory Name - no file selected')
    except KeyError:
        print('\n File number out of range - no file selected')
    return files


def format_full_path(path, filename, suffix, *args, extension=False, **kwargs):
    """Checks that a directory exists, makes a directory if necessary, and returns
    a formatted string for the full path of the file.  Args and Kwargs are formatted in a readable way."""
    # Later work on making it format strings for the OS

    if suffix == False:
        suffix = ''

    if extension:
        path = path + '/' + extension

    # Make directory if needed
    if not os.path.exists(path):
        os.makedirs(path)

    # remove suffix from filename if present
    if '.' in filename:
        split = filename.split('.')
        if len(split) == 2:
            filename = split[0]
        else:
            filename = '.'.join(split[:-1])

    # Join pieces
    for arg in args:
        filename = filename + ' ' + str(arg)
    for key in kwargs:
        filename = filename + ' ' + key + '=' + str(kwargs[key])
        filename = filename + '.' + suffix
    full_path = path + '/' + filename + suffix

    return full_path


