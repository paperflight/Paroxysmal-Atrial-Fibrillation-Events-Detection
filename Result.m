
%%
% Written by: Caiyun Ma, Chengyu Liu
%             School of Instrument Science and Engineering
%             Southeast University, China
%             chengyu@seu.edu.cn
%%
function predict_endpoints=Result(sample_name,sample_path,save_path)
% Save the results generated by each sample.
%
% inputs
%   sample_name: The path to store the sample name and the sample name (RECORDS)   
%   sample_path: The relative path where the recording is stored and refer to wfdb toolbox
%     save_path: The path where the results are stored
% outputs
%   predict_endpoints: Predicted atrial fibrillation endpoints 

Records=importdata(sample_name);
len=length(Records);
for i=1:len
    Rrcord=char(Records(i)); 
    sample_path_all=[sample_path,Rrcord]
%   [ann,anntype,subtype,chan,num,comments]=rdann('\data_0001', 'atr');
    predict_endpoints = challenge(sample_path_all);
    filename = strcat(save_path,Rrcord, '.mat');
    save (filename,'predict_endpoints');
end
