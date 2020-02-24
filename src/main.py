ON_JUPYTER = False
try:
    from IPython import get_ipython
    # Autoreload imported modules for Jupyter
    get_ipython().magic('load_ext autoreload') 
    get_ipython().magic('autoreload 2')
    ON_JUPYTER = True
except AttributeError:
    pass

import torch
assert torch.cuda.is_available()
from cl.telegram_bot import TelegramBot

def set_gpu(gpu_nr):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_nr)

def define_configs():
    configs = []

    return configs

from src.eval.results import ExperimentResult

def run(experiment, config):
    # TODO
    results = ExperimentResult(config['measures'])

    #TODO


    # 1. create dataset

    # If not loaded, create repetitions


    # Define configuration. The configuration includes the number of repetitions
    # or cross-validation folds. This starts an array of experiments, each for 
    # one repetition\fold

    
    experiment.finish(results=results)

if __name__ == '__main__':
    bot = TelegramBot()
    if not ON_JUPYTER:
        if len(sys.argv[1:]) > 0:
            print('Console arguments')
            args = get_args(sys.argv[1:])
            run(Experiment(args, label=''))
        else:
            print('Config arguments')
            configs = define_configs()
            
            for ix, config in enumerate(configs):
                print('\nNew experiment')
                args = args_standard(config)
                print(args)
                notes = ''
                splits = 
                experiment = Experiment(args, notes=notes)
                try:
                    set_gpu(args.gpu)
                    run(experiment, config)
                except Exception as e: 
                    state = 'ERROR'
                    experiment.finish(exception=e)
                bot.send_msg('Exp. {} of {} PC1 using gpu {} is finished with {}'.format(ix+1, len(configs), args.gpu, state))