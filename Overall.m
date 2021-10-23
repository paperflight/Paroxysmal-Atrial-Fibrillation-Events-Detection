function Overall()
names = dir('./');
for x = 1:length(names)
    na = names(x).name;
    if startsWith(na, 'data') && endsWith(na, '.dat')
        predict_endpoints = challenge(na);
        na = na(1:length(na)-4);
        filename = strcat('./',na, '.mat');
        save (filename,'predict_endpoints');
    end
end
end
