% endp = Result('name.csv', './', './')


function predict_endpoints=Result(sample_name,sample_path,save_path)
% Save the results generated by each sample.
%
% inputs
%   sample_name: The path to store the sample name and the sample name (RECORDS)   
%   sample_path: The relative path where the recording is stored and refer to wfdb toolbox
%     save_path: The path where the results are stored
% outputs
%   predict_endpoints: Predicted atrial fibrillation endpoints 
% net = load('Net_1_1.mat')
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
end

function predict_endpoints = challenge(sample_path)
load('Net_2_1.mat');
[signal,Fs,tm]=rdsamp(sample_path);
sig=signal(:,1);
% sig = signal{1,1}(1,1);
y_seq=zeros(length(sig),1);
end_points=[];
fs = 200;
L = 10;
D = 1;

Seg = floor(length(sig)/fs/D);

for i = 1:Seg-10
    region_head = (i-1) * fs * D + 1;
    region_end = (i-1) * fs * D + fs * L;
    predict_res = classify(net, sig(region_head:region_end));
    if double(predict_res) == 2
        y_seq(region_head:region_end) = 1;
    end
end
predict_res = classify(net, sig(length(sig)-fs*L + 1:length(sig)));
if double(predict_res) == 2
    y_seq(length(sig)-fs*L + 1:length(sig)) = 1;
end

g1=0;
g2=0;
clear start_points end_points
for z=1:length(y_seq)
    if z == 1 && y_seq(z)==1
        g1=g1+1;
        start_points(g1,:)=z;
    elseif z==length(y_seq) && y_seq(z)==1
        g2=g2+1;
        end_points(g2,:)=z;
    elseif z==length(y_seq)
        break
    elseif y_seq(z)==0 && y_seq(z + 1)==1
       g1=g1+1;
       start_points(g1,:)=z+1;
    elseif y_seq(z)==1 && y_seq(z + 1)==0
       g2=g2+1;
       end_points(g2,:)=z;
    end
end

if exist('start_points')
    predict_endpoints=[start_points,end_points];
else
    predict_endpoints=[];
end

end


 
