

# Experiment
# string experiment_name
# string experiment_notes
# string experiment_class_path optionally write a custum ExperimentRun subclass to write additional information in the review.json file

# Train, validation and test splits
# boolean cross_validation
# boolean nr_runs
# float val_ratio. The validation ratio is of the train+validation (excluding test) indexes
# float test_ratio
