import argparse

# TODO: adapt this class to the arguments we need

def get_args(argv):
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batch_size', type=int)

    # Additional
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--joined_training', dest='joined_training', default=False, action='store_true')

    # Dataset
    parser.add_argument('--ds_val_ratio', type=float, default=0.3)
    parser.add_argument('--ds_stratesfied', dest='ds_stratesfied', default=False, action='store_true')
    parser.add_argument('--splits_name', type=None, help='Pickle file where task splits are stored.')
    parser.add_argument('--randomize_class_order', dest='randomize_class_order', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=0) # REMOVE
    parser.add_argument('--model_weights', type=str, default=None)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--model_type', type=str, default='mlp', help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--model_name', type=str, default='MLP', help="The name of actual model for the backbone")
    parser.add_argument('--agent_type', type=str, default='default', help="The type (filename) of agent")
    parser.add_argument('--agent_name', type=str, default='NormalNN', help="The class name of agent")
    parser.add_argument('--optimizer', type=str, default='Adam', help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")

    parser.add_argument('--schedule', nargs="+", type=int, default=[2],
                        help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number is the end epoch")
    parser.add_argument('--retraining_schedule', nargs="+", type=int, default=[2])

    parser.add_argument('--problem', type=str, help="The problem formulation.")
    parser.add_argument('--normalize', type=int, help="Normalization.")
    parser.add_argument('--search_forward', type=int, help="search_forward.")
    parser.add_argument('--add_forward', type=int, help="add_forward.")
    
    parser.add_argument('--calculate_importance', type=str, help="calculate_importance.")
    parser.add_argument('--noise_type', type=str, help="noise_type.")
    parser.add_argument('--norm_clamp_btm', type=int)
    parser.add_argument('--norm_clamp_top', type=int)
    parser.add_argument('--norm_start', type=float)
    parser.add_argument('--ratio_remaining', type=float)
    args = parser.parse_args(argv)
    return args