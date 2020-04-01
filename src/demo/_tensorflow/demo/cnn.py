import os

class Cifar10:


    def run(self):
        data_path = '/apps/data/ai_nlp_testing/raw/tensorflow/cifar-10-batches-py'
        file_batch_1 = 'data_batch_1'
        path_batch_1 = os.path.join(data_path, file_batch_1)
        batch_1 = self.unpickle(path_batch_1)
        print(batch_1)

