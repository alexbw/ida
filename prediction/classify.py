import whetlab
import pandas as pd
import numpy as np
from scipy import ndimage
from sklearn import cross_validation
from sklearn import decomposition as decomp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import click
from blurd.util import *
from scipy import stats

np.set_printoptions(suppress=True, precision=3)


def whiten(x):
    mu, L = np.mean(x,axis=0), np.linalg.cholesky(np.cov(x,rowvar=0,bias=1))
    return np.linalg.solve(L, (x-mu).T).T, L, mu


def unwhiten(x, L, mu):
    return L.dot(x) + mu

# param_to_predict = "%Hyper"
# filename = "/Users/Alex/Code/blurd/data.pandas"
# n_runs = 100
# n_folds = 5
# seed = 0


@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("--n-runs", default=100, type=int)
@click.option("--n-folds", default=5, type=int)
@click.option("--seed", default=0, type=int)
def predict(filename, n_runs, n_folds, seed):
    #####################################
    # Data Preparation
    #####################################

    # Load it up
    sdf = pd.read_pickle(filename)

    # Create a unique tag for each patient
    sdf['patient'] = sdf.sample+"-"+sdf.date

    # Select only good vials
    sdf = sdf[sdf.good == True]

    # Fill in the micro/hypo calls based on these criteria (AND THEN MOVE TO THE EXTRACTION NOTEBOOK)
    # ALL THESE INDICES ARE TO CALL THE MICRO/HYPOCHROMIC ANEMIC POPULATION
    hypo_idx = sdf['%Hypo'] >= 3.9
    anemic_woman_idx = (sdf['Age'] > 15) & (sdf['Sex'] == "F") & (sdf.HGB < 12.0) & hypo_idx
    anemic_man_idx = (sdf['Age'] > 15) & (sdf['Sex'] == "M") & (sdf.HGB < 13.0) & hypo_idx
    anemic_infant_idx = (sdf['Age'] < 5) & (sdf.HGB < 11.0) & hypo_idx
    anemic_child_idx = (sdf['Age'] >= 5) & (sdf['Age'] < 15) & (sdf.HGB < 11.5) & hypo_idx
    anemic_population_idx = anemic_woman_idx | anemic_man_idx | anemic_infant_idx | anemic_child_idx
    sdf.blood_type.values[np.argwhere(anemic_population_idx)[()]] = 'anemic'
    sdf.blood_type.values[np.argwhere(anemic_population_idx == False)[()]] = 'normal'

    # Gather the blood-type for prediction
    int_y = sdf['blood_type'].values.copy()
    int_y[int_y == 'anemic'] = 1
    int_y[int_y == 'normal'] = 0
    int_y = int_y.astype('int32')
    sdf['int_blood_type'] = int_y

    # Get the patients
    patients = sdf.patient.unique()

    # For each patient, get the blood-type
    patient_blood_type = []
    for patient in patients:
        patient_blood_type.append(sdf[sdf.patient == patient].blood_type.values[0])
    patient_blood_type = np.array(patient_blood_type)

    # Hold out a final test set
    np.random.seed(0)
    train_patient_idx, test_patient_idx = list(cross_validation.StratifiedShuffleSplit(patient_blood_type, n_iter=1, test_size=1.0/n_folds))[0]

    train_patients = patients[train_patient_idx]
    train_patient_blood_type = patient_blood_type[train_patient_idx]
    # test_patients = patients[test_patient_idx]
    # test_patient_blood_type = patient_blood_type[test_patient_idx]

    # Divide the training patients into the remaining folds for K-fold cross-validated training.
    # This will help us guard a little against overfitting.
    trainvalidation_folds = list(cross_validation.StratifiedKFold(train_patient_blood_type, n_folds=n_folds-1))

    # Stack all data and labels for indexing in a moment
    all_data = np.vstack(sdf.data.values)
    blood_param = sdf['int_blood_type'].values

    all_train_idx = np.array([p in train_patients for p in sdf.patient])
    # all_test_idx = np.array([p in test_patients for p in sdf.patient])

    #####################################
    # Set up experiment
    #####################################

    # Define parameters to optimize
    parameters = dict(nphase=dict(type='integer', min=2, max=3),
                        centrifugation_time = dict(type='integer', min=1, max=5),
                        n_dimensions=dict(type='int', min=2, max=184),
                        softwindow_mean = dict(type='float', min=0, max=184),
                        softwindow_variance = dict(type='float', min=1, max=1000),
                        reduc_method=dict(type='int', min=0,max=3),
                        C=dict(type='float', min=1e-2, max=1e2),
                        penalty=dict(type='enum', options=['l1','l2']))

    name = 'Classifying IDA (patient-based folds)'
    description = ''
    outcome = dict(name="AUC")
    scientist = whetlab.Experiment(name=name, access_token=None,
                                   description=description,
                                   parameters=parameters,
                                   outcome=outcome)

    #####################################
    # Optimize
    #####################################

    for i in range(n_runs):
        pendings = scientist.pending()
        if len(pendings) > 0:
            job = pendings[0]
        else:
            job = scientist.suggest()
        print job

        # Prepare the blood parameter index
        blood_param_idx = ((sdf.run_time == 2*int(job['centrifugation_time'])) & (sdf.ida == "IDA%d" % job['nphase'])).values

        # Apply a soft window to the data (better than cropping)
        window = stats.norm.pdf(np.arange(all_data.shape[1]), loc=job['softwindow_mean'], scale=job['softwindow_variance'])
        window /= window.max()
        cropped_data = all_data*window[None,:]

        # NOTE: Doing the unsupervised step on all the training data.
        # Other options include:
        # - all data (it's unsupervised so should be fine)
        # - all training data, irrespective of blood parameters (e.g. include all centrifugation times together)
        all_train_data = cropped_data[all_train_idx & blood_param_idx].copy()
        reduc_method = ['raw','pca','nmf','kernelpca'][job['reduc_method']]
        if reduc_method == "raw":
            processed_data = ndimage.zoom(cropped_data,(1,job['n_dimensions']/cropped_data.shape[1]))
        elif reduc_method == "nmf":
            preprocessor = decomp.NMF(job['n_dimensions']).fit(all_train_data)
            processed_data = preprocessor.transform(cropped_data)
        elif reduc_method == "pca":
            preprocessor = decomp.PCA(job['n_dimensions']).fit(all_train_data)
            processed_data = preprocessor.transform(cropped_data)
        elif reduc_method == "kernelpca":
            preprocessor = decomp.KernelPCA(job['n_dimensions'],'rbf',gamma=1.0).fit(all_train_data)
            processed_data = preprocessor.transform(cropped_data)

        if np.isnan(processed_data).any():
            print "Invalid parameter value. Nans in the processed data"
            scientist.update(job, np.nan)
            continue

        auc_across_folds = []
        for train, validation in trainvalidation_folds:

            # Prepare the indices for this fold (only using the training patients)
            train_idx = blood_param_idx & np.array([p in train_patients[train] for p in sdf.patient])
            validation_idx = blood_param_idx & np.array([p in train_patients[validation] for p in sdf.patient])

            # Get the data and labels
            train_data = processed_data[train_idx]
            validation_data = processed_data[validation_idx]
            train_y = blood_param[train_idx]
            validation_y = blood_param[validation_idx]

            assert len(train_data) == len(train_y), "Unequal data and label sizes"
            assert len(validation_data) == len(validation_y), "Unequal data and label sizes"

            # Train a model
            learner = LogisticRegression(penalty=job['penalty'], C=job['C'], fit_intercept=True).fit(train_data,train_y)
            this_auc = roc_auc_score(validation_y, learner.decision_function(validation_data).ravel())
            auc_across_folds.append(this_auc)

        auc = np.mean(auc_across_folds)

        print auc

        # Tell Whetlab about the result
        scientist.update(job, auc)

if __name__ == "__main__":
    predict()
