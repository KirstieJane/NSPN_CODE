subs = dir('*')
addpath(genpath('/home/kw401/Downloads/'))
addpath(genpath('/home/kw401/MATLAB_TOOLBOXES/'))
for i = 3:length(subs)
dirname = fullfile(subs(i).name, 'DTI/MRI0/CONNECTIVITY')
cd (dirname)
A = load('Msym.txt');
NetworkMeasuresDTI(A, subs(i).name);
cd ../../../..
end