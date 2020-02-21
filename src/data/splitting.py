# ------------------------------------------------------------------------------
# A dataset is usually split into train, validation and test sets. This
# splitting can be performed at random, for instance to perform n repetitions,
# or through cross-validation. Each repetition or fold is used for a different
# experiment.
# In addition, a hold-out test dataset may be kept which is always the same and
# initialized together with a Dataset instance.
# ------------------------------------------------------------------------------

class Split:
    def __init__(self, name, train_ixs, val_ixs, test_ixs):
    self.name = name
    self.train_ixs = train_ixs
    self.val_ixs = val_ixs
    self.test_ixs = test_ixs

def get_repetition(dataset, name, val_ratio=0.2, test_ratio=0.2):
    """
    Divides the instances of a Dataset at random, given as parameters the ratio
    of the validation and test data.

    :returns: A Split object
    """
    pass

def get_repetitions(dataset, name, k=1, val_ratio=0.2, test_ratio=0.2):
    """
    Creates k repetitions.

    :returns: A list of Split objects
    """
    return [get_repetition(dataset, name=name+'_'+str(i), val_ratio=val_ratio, 
        test_ratio=test_ratio) for i in range(k)]

def get_cross_validation_folds(dataset, name, k=5, val_ratio=0.2):
    """
    Divides the instances of a Dataset into k folds. For each Split, the test
    data is made out of a different fold. The validation data is defined by the 
    val_ratio parameter and taken from the remaining folds.

    :returns: A list of Split objects
    """
    pass
