# Paroxysmal-Atrial-Fibrillation-Events-Detection
Paroxysmal Atrial Fibrillation Events Detection from Dynamic ECG Recordings challenge for ICBEB2021

The data format must be formed into WFDB and read via relative package.Cancel changes

## How to run this code
Configure the matlab and install WFDB package from https://github.com/ikarosilva/wfdb-app-toolbox

Change the working dictory to current dictory and run:

    predict_endpoints=Result(sample_name,sample_path,save_path)

where 'sample_name' is the path to store the sample name and the sample name (RECORDS), 'sample_path' is the relative path where the recording is stored and refer to wfdb toolbox, 'save_path' is the path where the results are stored.

## Where to reference
https://github.com/CPSC-Committee/cpsc2021-matlab-entry
