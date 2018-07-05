file = 'ProbMap_class_1_CT-11.mat'; % Input file name
loaded = load(file);
scan = loaded.vol;
X = scan;

% Removes all values less than .95, sets all values geq .95 to 1
X(scan<.95) = 0;
X(scan>.95) = 1;

CC = bwconncomp(X);

% Gets largest connect component
S = regionprops(CC, 'Area');
seg = bwareaopen(X,max([S.Area]));

save('Skull-11.mat', 'seg'); % Output file name
