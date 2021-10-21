clear
clc

for i = 0:104
    for j = 1:30
       Data(i+1,j) = compose("data_%d_%d",i,j);
       exist (Data(i+1,j)+'.dat');
       if ans == 2
           ReadHead = textread(Data(i+1,j)+'.hea','%s',30);
           L = ReadHead{24,1};
           L = convertCharsToStrings(L);
           if L == 'non'
                Label(i+1,j) = 1;
           elseif L == 'persistent'
                Label(i+1,j) = 2;
           elseif L == 'paroxysmal'
                Label(i+1,j) = 3;
           end
           clear ReadHead L
           DataName = convertStringsToChars(Data(i+1,j));
           [signal,Fs,time]=rdsamp(DataName);
           Signal{i+1,j} = signal;
       end
    end
end
clear ans i j time signal DataName

size(Label);
I = ans(1);
J = ans(2);
Segments = [];
Labels = [];
Count = 0;

for i = 1:I
    for j = 1:J
        SignalData = Signal{i,j};
        SignalLabel = Label(i,j);
        S = size(SignalData);
        if S(1) ~= 0
            Count = Count+1;
            for k = 1:2
                [Segment, Seg] = SegFunction(SignalData(:,k),5);
                Segments = [Segments; Segment];
                Labels = [Labels; SignalLabel*ones(Seg,1)];
            end
        end
    end
end

for i = 0:104
    for j = 1:30
       Data(i+1,j) = compose("data_%d_%d",i,j);
       exist (Data(i+1,j)+'.dat');
       if ans == 2
           ReadHead = textread(Data(i+1,j)+'.hea','%s',30);
           L = ReadHead{24,1};
           L = convertCharsToStrings(L);
           if L == 'paroxysmal'
               DataName = convertStringsToChars(Data(i+1,j));
               [signal,Fs,time]=rdsamp(DataName);
               
               [ATR_T,~,~,~,~,ATR_L] = rdann(DataName,'atr');
               [idxstart,idxend] = ATRCOV(ATR_L);
               Onset = ATR_T(idxstart);
               End = ATR_T(idxend);
               if idxstart(1) == 1
                   Onset(1) = 1;
               end
               EndCheck = max(End);
               if EndCheck > length(signal)
                   End(end) = length(signal);
               end
               Signal{i+1,j} = signal;
               OnsetandEnd{i+1,j} = sort([Onset; End]);
               label = ones(length(signal),1);
               for k = 1:length(idxstart)
                   label(Onset(k):End(k)) = 2; 
               end
               Label{i+1,j} = label;
           end
           clear ReadHead L
       end
    end
end

% clear ans i j time signal DataName Data Output ATR_L ATR_T idxend idxstart k label Onset End

size(Signal);
I = ans(1);
J = ans(2);
Count = 0;
Segments = [];
Labels = [];
for i = 1:I
    for j = 1:J
        SignalData = Signal{i,j};
        SignalLabel = Label{i,j};
        S = size(SignalData);
        if S(1) ~= 0
            Count = Count+1;
            for k = 1:2
                [Segment, Seg] = SegFunction(SignalData(:,k),5,1);
                Segments = [Segments; Segment];
            end
                [labels, ~] = SegFunction(SignalLabel,5,1);
                labels = mean(labels');
                labels(labels>=1.3) = 2;
                labels(labels<1.3) = 1;
%                 labels = imbinarize(labels-1);
%                 labels = double(labels)+1;
                Labels = [Labels; labels';labels'];
%                 Labels = [Labels; labels'];
        end
    end
end

Idx = [];

for i = 1:length(Labels)
    if Labels(i)==3
        Idx = [Idx; i];  
    end
end

Segments(Idx,:) = [];
Labels(Idx,:) = [];

for i = 1:length(Labels)
    Data{i} = Segments(i,:)';
end

for j = length(Labels)+1:length(Labels)+length(Labels3)
   Data{j} = Segments3(j-length(Labels),:)'; 
end

Data = Data';
Labels = [Labels; Labels3];
Labels = categorical(Labels);
clear i Idx Labels3 Segments Segments3

n =length(Labels);
hpartition = cvpartition(n,'Holdout',0.3);
idTrain = training(hpartition);
Train = Data(idTrain,:);
TrainLabels = Labels(idTrain,:);
idValidation = test(hpartition);
Validation = Data(idValidation,:);
ValidationLabels = Labels(idValidation,:);

layers = [ ...
    sequenceInputLayer(1000)
    bilstmLayer(50,"OutputMode","sequence")
    bilstmLayer(50,"OutputMode","last")
    fullyConnectedLayer(numel(unique(TrainLabels)))
    softmaxLayer
    classificationLayer];

options = trainingOptions("sgdm", ...
    'InitialLearnRate',0.001, ... 
    "Shuffle","every-epoch", ...
    'LearnRateDropFactor',0.2, ...
    'MiniBatchSize',256, ...
    'MaxEpochs',10, ...
    "ValidationData",{Validation,ValidationLabels}, ...
    "Plots","training-progress", ...
    "Shuffle","every-epoch", ...
    "Verbose",false);

net = trainNetwork(Train,TrainLabels,layers,options);


function [idxstart,idxend] = ATRCOV(Input)
tf = cellfun('isempty',Input);
Input(tf) = {0};
Output = string(Input);
idx1 = find(Output == '(AFIB');
idx2 = find(Output == '(AFL');
idxend = find(Output == '(N');
idxstart = [idx1,idx2];
IES = isempty(idxstart);
IEN = isempty(idxend);
    if IES == 1 || length(idxend) > length(idxstart)
        idxstart = [1;idxstart];
    end
    if IEN == 1 || length(idxend) < length(idxstart)
        idxend = [idxend;length(Input)];
    end
end

% Moving Window Segmentation
function [Segments, Seg] = SegFunction(Data,L,D)

fs = 200;

Seg = floor((length(Data)-L*fs)/D/fs);

for i = 1:Seg
Segments(i,:) = Data((i-1)*fs*D+1:(i-1)*fs*D+fs*L);
% Segments(i,:) = normalize(Segments(i,:));
end

end
