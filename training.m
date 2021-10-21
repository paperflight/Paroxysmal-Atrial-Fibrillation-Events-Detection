%% TrainingBLSTM
clear all
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
save('Signal','Signal');
save('Label','Label');
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

Data = Data';
Labels = categorical(Labels);
clear i Idx

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

% No_Moving Window Segmentation
function [Segments, Seg] = SegFunction(Data,L)

fs = 200;

Seg = floor(length(Data)/fs/L);

for i = 1:Seg
Segments(i,:) = Data((i-1)*fs*L+1:i*fs*L);
end

end

%% TrainingFE_FC
load('Label.mat')
load('Signal.mat')

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
               Labels{i+1,j} = label;
           end
           clear ReadHead L
       end
    end
end

size(Label);
I = ans(1);
J = ans(2);

for i = 1:I
    for j = 1:J
        SignalData = Signal{i,j};
        S = size(SignalData);
        if S(1) ~= 0
            sig = SignalData(:,1);
            Count = Count+1;
        end
        Data(i,j) = compose("data_%d_%d",i-1,j);
        exist (Data(i,j)+'.dat');
         if ans == 2
           ReadHead = textread(Data(i,j)+'.hea','%s',30);
           L = ReadHead{24,1};
           L = convertCharsToStrings(L);
           if L == 'non'
               Lab = ones(length(sig),1);
           end
           if L == 'persistent'
               Lab = 2*ones(length(sig),1);
               Onset = 1;
               End = length(sig);
               OnsetandEnd{i,j} = sort([Onset; End]);
           end
           Labels{i,j} = Lab;           
     end
    end
end

for i = 1:I
    for j = 1:J
        SignalData = Signal{i,j};
        L = Labels{i,j};
        S = size(SignalData);
        if S(1) ~= 0
            sig = SignalData(:,1);
            Count = Count+1;
            [QRS,~,~] = qrs_detect(sig',0.25,0.6,Fs);
            if length(QRS) <= 10;
                [~, QRS] = MYPantompkins(sig, Fs);
            end
            if length(QRS) <= 10;
                SS = filloutliers(sig,'linear');
                SS = MYBandPass(SS,200,1,25,3);
                [~, QRS] = MYPantompkins(SS, Fs);
                clear SS
            end
            RR = diff(QRS)/Fs;
            RR_N = length(RR);
            is_af = [];
            a = 1;
            n = RR_N/5;
            q = rem(RR_N,5);
            for f=1:n
                b=a+4;
                [cosEn,sentropy,mRR,minRR,stdRR,medFreq,meanFreq] = FeaEx(RR(a:b));
                feature = [cosEn,sentropy,mRR,minRR,stdRR,medFreq,meanFreq];
                Feature(f,:) = feature;
                LL = mean(L(QRS(a):QRS(b)));
                LL = round(LL);
                Lab(f,:) = LL;
                a=a+5;
            end
            if q~=0
                RR_end = RR(RR_N-4:RR_N);
                [cosEn,sentropy,mRR,minRR,stdRR,medFreq,meanFreq] = FeaEx(RR_end);
                feature = [cosEn,sentropy,mRR,minRR,stdRR,medFreq,meanFreq];
                Feature(f+1,:) = feature;
                LL = mean(L(QRS(b):end));
                LL = round(LL);
                Lab(f+1,:) = LL;
            end
            FE{i,j} = Feature;
            LB{i,j} = Lab;
            Feature = [];
            Lab = [];
        end
    end
end

size(FE);
I = ans(1);
J = ans(2);
FEs = [];
LBs = [];
for i = 1:I
    for j = 1:J
        fe = FE{i,j};
        lb = LB{i,j};
        S = size(fe);
        if S(1) ~= 0
            FEs = [FEs; fe];
            LBs = [LBs; lb];
        end
    end
end

Labels = categorical(LBs);

for i = 1:length(Labels)
    Data{i} = FEs(i,:)';
end
Data = Data';

n =length(Labels);
hpartition = cvpartition(n,'Holdout',0.3);
idTrain = training(hpartition);
Train = Data(idTrain,:);
TrainLabels = Labels(idTrain,:);
idValidation = test(hpartition);
Validation = Data(idValidation,:);
ValidationLabels = Labels(idValidation,:);

layers = [ ...
    featureInputLayer(7)
    fullyConnectedLayer(64)
    fullyConnectedLayer(64)
    leakyReluLayer
    fullyConnectedLayer(2)
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
