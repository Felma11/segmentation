import argparse

def get_args(argv):
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--agent_class_path', type=str, default='NormalNN', help="The class path of the agent.")
    parser.add_argument('--model_class_path', type=str, default='NormalNN', help="The class path of the model.")

    # Experiment
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--experiment_notes', type=str)
    parser.add_argument('--experiment_run_class_path', type=str, help="Optionally write a custum ExperimentRun subclass to write additional information in the review.json file.")

    # Dataset
    parser.add_argument('--restore_dataset', dest='restore_dataset', default=False, action='store_true', help='Reload dataset object.')
    parser.add_argument('--nr_runs', type=int, default=1)
    parser.add_argument('--cross_validation', dest='cross_validation', default=False, action='store_true')
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of validation data from train+validation data.')
    parser.add_argument('--test_ratio', type=float, default=0.2, help="Only relevant if 'cross_validation' if false.")

    # Training
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)    
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--optimizer', type=str, default='Adam', help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--schedule', nargs="+", type=int, default=[2],
                        help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number is the end epoch")

    # Continual learning
    parser.add_argument('--nr_tasks', type=int)
    parser.add_argument('--joined_training', dest='joined_training', default=False, action='store_true')
    parser.add_argument('--randomize_class_order', dest='randomize_class_order', default=False, action='store_true')

    args = parser.parse_args(argv)
    return args