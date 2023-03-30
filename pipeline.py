import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE',
                        sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 1.0
exported_pipeline = GradientBoostingClassifier(
    learning_rate=1.0, max_depth=6, max_features=0.1, min_samples_leaf=9, min_samples_split=16, n_estimators=100, subsample=0.55)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
