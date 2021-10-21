function predict_endpoints = challenge(sample_path)
load('Net_2_2.mat');
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

if ~all(y_seq)
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
        y_seq(r_peak(i):r_peak(i+window_size)) = 1;
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

predict_endpoints=[start_points,end_points];

end
