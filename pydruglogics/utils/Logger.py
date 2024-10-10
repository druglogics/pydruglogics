class Logger:
    def __init__(self, verbosity=3):
        self.verbosity = verbosity

    def log(self, message, level):
        if self.verbosity >= level:
            print(message)
