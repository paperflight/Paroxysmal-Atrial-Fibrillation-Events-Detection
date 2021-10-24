function predict_endpoints = challenge(sample_path)
load('Net_2_4.mat');
[signal,Fs,tm]=rdsamp(sample_path);
sig=signal(:,1);
% sig = signal{1,1}(1,1);
y_seq=zeros(length(sig),1);
end_points=[];
fs = 200;
L = 5;
D = 1;

Seg = floor((length(sig)-L*fs)/D/fs);

for i = 1:Seg
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

for i=1:length(y_seq) - 1100
    x = y_seq(i:i+1100);
    if any(x) && x(1) == 0 && x(length(x)) == 0
        x(x==1) = 0;
        y_seq(i:i+1100) = x;
    end
end

if ~any(y_seq) || sum(y_seq) / length(sig) < 0.05
    predict_endpoints = [];
    return
end

y_seq = y_seq * 0;

window_size = 6 - 1;
[r_peak, signal_r_peak] = RdetectFEextract(sig);
% r_peak: 1 * num r_peak
% signal_r_peak: (num r_peak - 4) * 7

load('Net_3_3.mat'); % net_3_1
res = classify(net_3_3, signal_r_peak);
res = double(res) - 1;

if all(res)
    predict_endpoints = [1, length(signal)];
    return
end

for i=1:length(res)
    if res(i)
        y_seq(r_peak(i):r_peak(i+window_size)) = y_seq(r_peak(i):r_peak(i+window_size)) + 1;
    end
end

y_seq = uint8(y_seq / 5);

if length(r_peak) > 7
    if y_seq(r_peak(6)) == 1
        y_seq(1:r_peak(6)) = 1;
    end
    if y_seq(r_peak(length(r_peak) - 6)) == 1
        y_seq(r_peak(length(r_peak) - 6) : length(y_seq)) = 1;
    end
end
for i=1:length(y_seq) - 800
    x = y_seq(i:i+800);
    if any(x) && x(1) == 1 && x(length(x)) == 1
        x(x==0) = 1;
        y_seq(i:i+800) = x;
    end
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

if sum(y_seq) / length(sig) > 0.5
    predict_endpoints = [1, length(signal)];
    return
end

if ~any(y_seq)
    predict_endpoints = [];
else
    predict_endpoints=[start_points,end_points];
end

end


function [QRS, Feature] = RdetectFEextract(sig)
    Fs = 200;
    [QRS,~,~] = qrs_detect(sig',0.25,0.6,Fs);
    if length(QRS) <= 10
        [~, QRS] = MYPantompkins(sig, Fs);
    end
    if length(QRS) <= 10
        SS = filloutliers(sig,'linear');
        SS = MYBandPass(SS,200,1,25,3);
        [~, QRS] = MYPantompkins(SS, Fs);
        clear SS
    end
    RR = diff(QRS)/Fs;
    RR_N = length(RR);
    n = RR_N - 4;
%     q = rem(RR_N,5);
    for f=1:n
        [cosEn,sentropy,mRR,minRR,stdRR,medFreq,meanFreq] = FeaEx(RR(f:f+4));
        feature = [cosEn,sentropy,mRR,minRR,stdRR,medFreq,meanFreq];
        Feature(f,:) = feature;
    end
%     if q~=0
%         RR_end = RR(RR_N-4:RR_N);
%         [cosEn,sentropy,mRR,minRR,stdRR,medFreq,meanFreq] = FeaEx(RR_end);
%         feature = [cosEn,sentropy,mRR,minRR,stdRR,medFreq,meanFreq];
%         Feature(f+1,:) = feature;
%     end
end



function [qrs_pos,sign,en_thres] = qrs_detect(ecg,varargin)
WIN_SAMP_SZ = 7;
REF_PERIOD = 0.250; 
THRES = 0.6; 
fs = 1000; 
fid_vec = [];
SIGN_FORCE = [];
debug = 0;

switch nargin
    case 1
    case 2
        REF_PERIOD=varargin{1};
    case 3
        REF_PERIOD=varargin{1}; 
        THRES=varargin{2};
    case 4
        REF_PERIOD=varargin{1}; 
        THRES=varargin{2};
        fs=varargin{3};  
    case 5
        REF_PERIOD=varargin{1}; 
        THRES=varargin{2}; 
        fs=varargin{3}; 
        fid_vec=varargin{4};
    case 6
        REF_PERIOD=varargin{1}; 
        THRES=varargin{2}; 
        fs=varargin{3};
        fid_vec=varargin{4}; 
        SIGN_FORCE=varargin{5};
    case 7
        REF_PERIOD=varargin{1}; 
        THRES=varargin{2}; 
        fs=varargin{3};
        fid_vec=varargin{4};
        SIGN_FORCE=varargin{5};          
        debug=varargin{6};
    case 8
        REF_PERIOD=varargin{1}; 
        THRES=varargin{2}; 
        fs=varargin{3};
        fid_vec=varargin{4};
        SIGN_FORCE=varargin{5};          
        debug=varargin{6};
        WIN_SAMP_SZ = varargin{7};
    otherwise
        error('qrs_detect: wrong number of input arguments \n');
end


[a b] = size(ecg);
if(a>b); NB_SAMP=a; elseif(b>a); NB_SAMP=b; ecg=ecg'; end;
tm = 1/fs:1/fs:ceil(NB_SAMP/fs);
MED_SMOOTH_NB_COEFF = round(fs/100);
INT_NB_COEFF = round(WIN_SAMP_SZ*fs/256);
SEARCH_BACK = 1; 
MAX_FORCE = []; 
MIN_AMP = 0.1; 
NB_SAMP = length(ecg);
try
    b1 = [-7.757327341237223e-05  -2.357742589814283e-04 -6.689305101192819e-04 -0.001770119249103 ...
         -0.004364327211358 -0.010013251577232 -0.021344241245400 -0.042182820580118 -0.077080889653194...
         -0.129740392318591 -0.200064921294891 -0.280328573340852 -0.352139052257134 -0.386867664739069 ...
         -0.351974030208595 -0.223363323458050 0 0.286427448595213 0.574058766243311 ...
         0.788100265785590 0.867325070584078 0.788100265785590 0.574058766243311 0.286427448595213 0 ...
         -0.223363323458050 -0.351974030208595 -0.386867664739069 -0.352139052257134...
         -0.280328573340852 -0.200064921294891 -0.129740392318591 -0.077080889653194 -0.042182820580118 ...
         -0.021344241245400 -0.010013251577232 -0.004364327211358 -0.001770119249103 -6.689305101192819e-04...
         -2.357742589814283e-04 -7.757327341237223e-05];

    b1 = resample(b1,fs,250);
    bpfecg = filtfilt(b1,1,ecg)';
    
    if (sum(abs(ecg-median(ecg))>MIN_AMP)/NB_SAMP)>0.05
        dffecg = diff(bpfecg');  
        sqrecg = dffecg.*dffecg; 
        intecg = filter(ones(1,INT_NB_COEFF),1,sqrecg); 
        mdfint = medfilt1(intecg,MED_SMOOTH_NB_COEFF);  
        delay  = ceil(INT_NB_COEFF/2); 
        mdfint = circshift(mdfint,-delay);
        if isempty(fid_vec); mdfintFidel = mdfint; else mdfintFidel(fid_vec>2) = 0; end;
        if NB_SAMP/fs>90; xs=sort(mdfintFidel(fs:fs*90)); else xs = sort(mdfintFidel(fs:end)); end;

        if isempty(MAX_FORCE)
           if NB_SAMP/fs>10
                ind_xs = ceil(98/100*length(xs)); 
                en_thres = xs(ind_xs);
            else
                ind_xs = ceil(99/100*length(xs)); 
                en_thres = xs(ind_xs);
            end 
        else
           en_thres = MAX_FORCE;
        end
        poss_reg = mdfint>(THRES*en_thres); 
        if isempty(poss_reg); poss_reg(10) = 1; end;
        if SEARCH_BACK
            indAboveThreshold = find(poss_reg); 
            RRv = diff(tm(indAboveThreshold));  
            medRRv = median(RRv(RRv>0.01));
            indMissedBeat = find(RRv>1.5*medRRv); 
            indStart = indAboveThreshold(indMissedBeat);
            indEnd = indAboveThreshold(indMissedBeat+1);

            for i=1:length(indStart)
                poss_reg(indStart(i):indEnd(i)) = mdfint(indStart(i):indEnd(i))>(0.5*THRES*en_thres);
            end
        end

        left  = find(diff([0 poss_reg'])==1);  
        right = find(diff([poss_reg' 0])==-1); 

        if SIGN_FORCE
            sign = SIGN_FORCE;
        else
            nb_s = length(left<30*fs);
            loc  = zeros(1,nb_s);
            for j=1:nb_s
                [~,loc(j)] = max(abs(bpfecg(left(j):right(j))));
                loc(j) = loc(j)-1+left(j);
            end
            sign = mean(ecg(loc)); 
        end
        compt=1;
        NB_PEAKS = length(left);
        maxval = zeros(1,NB_PEAKS);
        maxloc = zeros(1,NB_PEAKS);
        for i=1:NB_PEAKS
            if sign>0
                [maxval(compt) maxloc(compt)] = max(ecg(left(i):right(i)));
            else
                [maxval(compt) maxloc(compt)] = min(ecg(left(i):right(i)));
            end
            maxloc(compt) = maxloc(compt)-1+left(i); 

            if compt>1
                if maxloc(compt)-maxloc(compt-1)<fs*REF_PERIOD && abs(maxval(compt))<abs(maxval(compt-1))
                    maxloc(compt)=[]; maxval(compt)=[];
                elseif maxloc(compt)-maxloc(compt-1)<fs*REF_PERIOD && abs(maxval(compt))>=abs(maxval(compt-1))
                    maxloc(compt-1)=[]; maxval(compt-1)=[];
                else
                    compt=compt+1;
                end
            else
                compt=compt+1;
            end
        end

        qrs_pos = maxloc; 
        R_t = tm(maxloc); 
        R_amp = maxval; 
        hrv = 60./diff(R_t); 
    else
        qrs_pos = [];
        R_t = [];
        R_amp = [];
        hrv = [];
        sign = [];
        en_thres = [];
    end
catch ME
    rethrow(ME);
    for enb=1:length(ME.stack); disp(ME.stack(enb)); end;
    qrs_pos = [1 10 20]; sign = 1; en_thres = 0.5; 
end

if debug
    figure;
    FONTSIZE = 20;
    ax(1) = subplot(4,1,1); plot(tm,ecg); hold on;plot(tm,bpfecg,'r')
        title('raw ECG (blue) and zero-pahse FIR filtered ECG (red)'); ylabel('ECG');
        xlim([0 tm(end)]);  hold off;
    ax(2) = subplot(4,1,2); plot(tm(1:length(mdfint)),mdfint);hold on;
        plot(tm,max(mdfint)*bpfecg/(2*max(bpfecg)),'r',tm(left),mdfint(left),'og',tm(right),mdfint(right),'om'); 
        title('Integrated ecg with scan boundaries over scaled ECG');
        ylabel('Int ECG'); xlim([0 tm(end)]); hold off;
    ax(3) = subplot(4,1,3); plot(tm,bpfecg,'r');hold on;
        plot(R_t,R_amp,'+k');
        title('ECG with R-peaks (black) and S-points (green) over ECG')
        ylabel('ECG+R+S'); xlim([0 tm(end)]); hold off;
    ax(4) = subplot(4,1,4); plot(R_t(1:length(hrv)),hrv,'r+')
        hold on, title('HR')
        ylabel('RR (s)'); xlim([0 tm(end)]);
    
    %linkaxes(ax,'x');
    set(gca,'FontSize',FONTSIZE);
    allAxesInFigure = findall(gcf,'type','axes');
    set(allAxesInFigure,'fontSize',FONTSIZE);
end
end

function [cosEn,sentropy,mRR,minRR,stdRR,medFreq,meanFreq] = FeaEx(segment)
r=0.03;      
M=2;     

mNc=5;   
dr= 0.001;    
A=-1000*ones(M,1);  

while A(M,1)< mNc
  [e,A,B]=sampen(segment,M,r);
  r=r+dr;
end

if A(M,1)~=-1000
    mRR=mean(segment);
    stdRR=std(segment);
    cosEn= e(M,1)+log(2*(r-dr))-log(mRR);
else
    cosEn=-1000;
end

sentropy=e(M,1);
minRR=min(segment);
medFreq=median(1./segment);
meanFreq=mean(1./segment);

end

function [y,x] = MYPantompkins(Raw_signal,fs)

Filter = designfilt('bandpassiir', 'FilterOrder', 6, 'PassbandFrequency1', 5, 'PassbandFrequency2', 15, 'PassbandRipple', 1, 'SampleRate', fs);
Filtered_signal = filtfilt(Filter,Raw_signal);

int_c = (5-1)/(fs*1/40);
b = interp1(1:5,[1 2 0 -2 -1].*(1/8)*fs,1:int_c:5);

D_Filtered_signal = filtfilt(b,1,Filtered_signal);
D_Filtered_signal = D_Filtered_signal/max(D_Filtered_signal);

D_Filtered_signal(find(D_Filtered_signal<=0)) = 0;
D_Filtered_signal = D_Filtered_signal.^2;

[y_test,x_test] = findpeaks(D_Filtered_signal,'MINPEAKDISTANCE',round(0.2*fs));

Sort_max = sort(unique(y_test), 'descend');
if 0.5*Sort_max(1)>Sort_max(2)
    Max_data = Sort_max(2);
else
    Max_data = Sort_max(1);
end

[y,x] = findpeaks(D_Filtered_signal,'MINPEAKDISTANCE',round(0.2*fs),'MinPeakHeight',0.3*Max_data);

x=LocalMax(Raw_signal,x);

end

function Max = LocalMax(y,x)
i=1;
if y(x)>y(x-1)
    while y(x+i)-y(x+i+1) <= 0
        i = i+1;
    end
        Max = x+i;
else
    while y(x-i-1)-y(x-i) >= 0
        i = i+1;
    end
        Max = x-i;
end

end

function Filtered = MYBandPass(Raw,fs,low,high,Order)

[b,a] = butter(Order,low/(fs/2),'high');
Filtered = filtfilt(b,a,Raw);

[b,a] = butter(Order,high/(fs/2),'low');
Filtered = filtfilt(b,a,Filtered);

end

function [e,A,B]=sampen(y,M,r);
%function [e,A,B]=sampenc(y,M,r);
%
%Input
%
%y input data
%M maximum template length
%r matching tolerance
%
%Output
%
%e sample entropy estimates for m=0,1,...,M-1
%A number of matches for m=1,...,M
%B number of matches for m=1,...,M excluding last point

n=length(y);
lastrun=zeros(1,n);
run=zeros(1,n);
A=zeros(M,1);
B=zeros(M,1);
p=zeros(M,1);
e=zeros(M,1);
for i=1:(n-1)
   nj=n-i;
   y1=y(i);
   for jj=1:nj
      j=jj+i;      
      if abs(y(j)-y1)<r
         run(jj)=lastrun(jj)+1;
         M1=min(M,run(jj));
         for m=1:M1           
            A(m)=A(m)+1;
            if j<n
               B(m)=B(m)+1;
            end            
         end
      else
         run(jj)=0;
      end      
   end
   for j=1:nj
      lastrun(j)=run(j);
   end
end
N=n*(n-1)/2;
p(1)=A(1)/N;
e(1)=-log(p(1));
for m=2:M
   p(m)=A(m)/B(m-1);
   e(m)=-log(p(m));
end
end
