class DataModule:
    def __init__(self, data_path):
        self.data_path = data_path
    def setup(self):
        print(f"Setting up data from {self.data_path}")