# segmentation

Install requirements with: pip install -r requirements.txt
Execute tests from the root directory with 'pytest'
Install PyTorch over conda (get command from https://pytorch.org/)

src: contains all the code
    agents: define the training and evaluation of models
    data: selection of training, validation and testing examples, and data loading operations
    eval: includes 
        measures: functions to calculate measures, e.g. dice
        results: internal representation of results
        visualization: creation of charts
    models: definition of models
    utils: helper functions
        arguments: definition of arguments and passing to dictionary
        experiment: deifnition of experiment for repeated runs

