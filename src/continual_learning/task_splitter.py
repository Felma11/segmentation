# ------------------------------------------------------------------------------
# Divide one dataset into an array of datasets representing different tasks, for
# a continual learning setup.
# ------------------------------------------------------------------------------

import random
from src.data.dataset_obj import Dataset, Instance

class TaskSplitter():
    """Divides the dataset into a list of datasets.
    """
    def __init__(self, dataset, save_path):
        self.dataset = dataset
        self.nr_tasks = None
        self.save_path = save_path

    def split_indexes(self, index_dictinary):
        """Receives a dictionary with an index list for each split and returns
        self.nr_tasks dictionaries with index lists.
        """
        pass

class ClassTaskSplitter(TaskSplitter):
    """Divides the indexes into different tasks according to the class.
    """
    def __init__(self, dataset, save_path, nr_tasks, class_remapping=True):
        """
        :param class_remapping: if true, the class is remapped to the index of
            the class in the task. E.g. '4' -> 0, '5' -> 1
        """
        super().__init__(dataset=dataset, save_path=save_path)
        self.nr_tasks = nr_tasks
        self.task_class_mapping, self.class_task_mapping = self.set_task_mapping()

    def set_task_mapping(self, randomize_class_order=False):
        classes = tuple(sorted(list(self.dataset.classes)))
        nr_classes = len(classes)
        assert float(nr_classes)/self.nr_tasks == float(nr_classes)//self.nr_tasks, 'The number of tasks must be a divider of the number of classes'
        new_nr_classes = int(nr_classes/self.nr_tasks)
        if randomize_class_order:
            random.shuffle(classes)
        # Has the form task_index -> list(int), where the order of the list 
        # is the remapped class e.g. {0: ['0', '1'], 1: ['2', '3'], 2: ['4', '5'], 3: ['6', '7'], 4: ['8', '9']}            
        task_class_mapping = {task_ix: classes[task_ix*new_nr_classes:(task_ix+1)*new_nr_classes] for task_ix in range(self.nr_tasks)}
        class_task_mapping = {class_name: task_ix for task_ix in range(self.nr_tasks) for class_name in task_class_mapping[task_ix] }
        return task_class_mapping, class_task_mapping
    
    def get_task_ixs(self, exp_ixs, task_ix):
        return [ix for ix in exp_ixs if self.get_index_task(ix)==task_ix]
        
    def get_index_task(self, example_ix):
        return self.class_task_mapping[self.dataset.instances[example_ix].y]