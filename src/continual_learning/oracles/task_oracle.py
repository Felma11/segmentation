from src.continual_learning.oracles.oracle import Oracle

class TaskOracle(Oracle):
    def __init__(self, pytorch_datasets, save_path):
        super().__init__(pytorch_datasets=pytorch_datasets, save_path=save_path, batch_size=1, lowest_score=False, name='TaskOracle')

    def set_scores(self, dl_task, split='val'):
        print('Setting scores for task {}, split {}'.format(dl_task, split))
        dataloader = self.dataloaders[dl_task][split]
        for model_task_ix in range(self.nr_tasks):
            score = 1.0 if model_task_ix==dl_task else 0.0
            for x, y in dataloader:
                self.scores[split][dl_task][model_task_ix].append(score)