import whetlab
import pandas as pd
import numpy as np
from scipy import ndimage
from sklearn import cross_validation
from sklearn import decomposition as decomp
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from blurd.util import *
import click
from scipy import stats
np.set_printoptions(suppress=True, precision=3)

raise ValueError("This file no longer works, because it depended on whetlab, a company that is now out of business")

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
@click.argument("param-to-predict", type=str, required=True)
@click.option("--n-runs", default=100, type=int)
@click.option("--n-folds", default=5, type=int)
@click.option("--seed", default=0, type=int)
def predict(filename, param_to_predict, n_runs, n_folds, seed):
    #####################################
    # Data Preparation
    #####################################

    # Load it up
    sdf = pd.read_pickle(filename)

    # Create a unique tag for each patient
    sdf['patient'] = sdf.sample+"-"+sdf.date

    # Select only good vials
    sdf = sdf[sdf.good == True]

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
    blood_param = sdf[param_to_predict].values.astype('float')

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
                        gamma=dict(type='float', min=1e-2, max=1e2),
                        epsilon=dict(type='float', min=1e-2, max=1e2))
                        # whiten_input=dict(type='integer', min=0, max=1),
                        # whiten_output=dict(type='integer', min=0, max=1))

    name = 'Predicting %s (patient-based folds)' % (param_to_predict)
    description = 'Predicting %s from blood test' % (param_to_predict)
    outcome = dict(name="R2")
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

        # Test for reasonable parameters, first

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

        # Possible further statistical preprocessing of the data
        # if job['whiten_input']:
        #     processed_data, L, mu = whiten(processed_data)
        #     if np.isnan(processed_data).any():
        #         scientist.update(job, np.nan)
        #         continue
        # if job['whiten_output']:
        #     processed_blood_param,L,mu = whiten(blood_param)
        # else:
        processed_blood_param = blood_param.copy()

        r2_across_folds = []
        for train, validation in trainvalidation_folds:

            # Prepare the indices for this fold (only using the training patients)
            train_idx = blood_param_idx & np.array([p in train_patients[train] for p in sdf.patient])
            validation_idx = blood_param_idx & np.array([p in train_patients[validation] for p in sdf.patient])

            # Get the data and labels
            train_data = processed_data[train_idx]
            validation_data = processed_data[validation_idx]
            train_y = processed_blood_param[train_idx]
            validation_y = processed_blood_param[validation_idx]

            assert len(train_data) == len(train_y), "Unequal data and label sizes"
            assert len(validation_data) == len(validation_y), "Unequal data and label sizes"

            # Train a model
            learner = SVR('rbf', C=job['C'], gamma=job['gamma'], epsilon=job['epsilon']).fit(train_data, train_y)
            pred_y = learner.predict(validation_data).ravel()
            this_r2 = r2_score(validation_y, pred_y)
            r2_across_folds.append(this_r2)

        r2 = np.mean(r2_across_folds)

        print r2

        # Tell Whetlab about the result
        scientist.update(job, r2)

if __name__ == "__main__":
    predict()
