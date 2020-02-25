# ------------------------------------------------------------------------------
# Divide one dataset into an array of datasets representing different tasks, for
# a continual learning setup.
# ------------------------------------------------------------------------------

import random
from src.data.dataset_obj import Dataset, Instance

class TaskSplitter():
    """Divides the dataset into a list of datasets.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.nr_tasks = None

    def get_datasets(self):
        pass

class ClassSplitter(TaskSplitter):
    """Divides the dataset into different tasks according to the class.
    """
    def __init__(self, dataset, nr_tasks):
        super().__init__(dataset=dataset)
        self.nr_tasks = nr_tasks

    def get_datasets(self, randomize_class_order=False):
        classes = self.dataset.classes
        nr_classes = len(classes)
        datasets = []
        assert float(nr_classes)/self.nr_tasks == float(nr_classes)//self.nr_tasks, 'The number of tasks must be a divider of the number of classes'
        new_nr_classes = int(nr_classes/nr_tasks)
        if randomize_class_order:
            random.shuffle(classes)
        # Has the form task_index -> list(int), where the order of the list 
        # is the remapped class e.g. {0: ['0', '1'], 1: ['2', '3'], 2: ['4', '5'], 3: ['6', '7'], 4: ['8', '9']}            
        remapping = {task_ix: classes[task_ix*new_nr_classes:(task_ix+1)*new_nr_classes] for task_ix in range(nr_tasks)}
        print(remapping)
        for task_ix, class_names in remapping.items():
            instances = 
            new_dataset = Dataset(name=self.dataset.name, 
                img_shape=self.dataset.img_shape, 
                file_type=self.dataset.file_type, 
                nr_channels=self.dataset.nr_channels, instances=instances, 
                hold_out_test_ixs=[])
            # TODO: is leaving hold_out_test_ixs empty correct?
            datasets.append(new_dataset)
        return datasets

