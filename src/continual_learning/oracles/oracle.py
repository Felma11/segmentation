import torch
import os

from src.utils.load_restore import pkl_load, join_path, pkl_dump
from src.eval.visualization.confusion_matrix import ConfusionMatrix

class Oracle():
    def __init__(self, pytorch_datasets, save_path, batch_size=1, lowest_score=False, name=None):
        """
        :param batch_size: batch size of dataloaders, usually 1
        :param name: name of the oracle (name of subclass)
        :param lowest_score: select the task id for which the score is lowest?
            Otherwise that with highest score is selected.
        """
        '''
        experiment_path: a path to some folder where intermediate values can be stored
        '''
        self.name = name
        self.lowest_score = lowest_score
        self.nr_tasks = len(pytorch_datasets)
        self.save_path = save_path
        self.splits = ['train', 'val', 'test']
        self.datasets = pytorch_datasets
        
        # Initialize scores. The first dimension if the split, the second the
        # dataloader index and the last the model index.
        self.scores = {split: [[[] for model_task in range(self.nr_tasks)] 
            for dl_task in range(self.nr_tasks)] for split in self.splits}
        self.load_scores() # Load scores which have been set

        # Build data loaders with the specified batch size
        self.dataloaders = [{split: torch.utils.data.DataLoader(
            self.datasets[task_ix][split], batch_size=batch_size, shuffle=False) 
            for split in self.splits} 
            for task_ix in range(self.nr_tasks)]

    def accuracy(self, prediction, target):
        assert len(prediction) == len(target)
        correct = sum([1 if prediction[i]==target[i] else 0 for i in range(len(prediction))])
        return float(correct)/len(prediction)

    def class_tp_tn_fp_fn(self, prediction, target, label=1):
        targets_label = [1 if x==label else 0 for x in target]
        predictions_label = [1 if x==label else 0 for x in prediction]
        tp, tn, fp, fn = 0, 0, 0, 0
        for i in range(len(targets_label)):
            if targets_label[i] == 1:
                if predictions_label[i] == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if predictions_label[i] == 1:
                    fp += 1
                else:
                    tn += 1
        return tp, tn, fp, fn

    def select_model(self, dl_task, split='val'):
        if len(self.scores[split][dl_task][0]) == 0:
            self.set_scores(dl_task=dl_task, split=split)
            self.save_scores()
        dataset = self.datasets[dl_task][split]
        selected_models = []
        for example_ix in range(len(dataset)):
            models_score = [self.scores[split][dl_task][model_task][example_ix] for model_task in range(self.nr_tasks)]
            if self.lowest_score:
                selected_model = models_score.index(min(models_score))
            else:
                selected_model = models_score.index(max(models_score))
            selected_models.append(selected_model)
        return selected_models

    def set_scores(self, dl_task, split='val'):
        """Template method for setting scores."""
        pass

    def save_scores(self):
        full_path = join_path([self.save_path, 'oracles', 'obj'])
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        pkl_dump(self.scores, 'scores_'+self.name, path=full_path)

    def load_scores(self):
        try:
            full_path = join_path([self.save_path, 'oracles', 'obj'])
            scores = pkl_load('scores_'+self.name, path=full_path)
            if scores is not None:
                self.scores = scores
        except:
            pass

    def get_domain_confusion(self, split):
        cm = ConfusionMatrix(self.nr_tasks)
        for dl_task in range(self.nr_tasks):
            selected_model = self.select_model(dl_task=dl_task, split=split)
            for model in selected_model:
                cm.add(predicted=model, actual=dl_task)
        return cm



        
