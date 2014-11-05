%% Load Grolier encyclopedia dataset
% Processed version of data provided at http://cs.nyu.edu/~roweis/data.html
load('data/grolier15276.mat');
% Use small dataset for demo (May want to change depending on your computer)
grolier = grolier(:,1:1000); 
words = words(1:1000);

%% Train single PMRF model (i.e. only 1 topic)
fprintf('Training single PMRF (i.e. only 1 topic) on subset of Grolier encyclopedia data\n');
ops = [];
ops.saveVerbosity = 2; % Also, save files that can be opened in the graph visualization software Gephi http://gephi.org
ops.baseFilename = 'single-pmrf';
numTopics = 1;
[Wt, thetaNodeArray, thetaEdgesArray] = apm( grolier, numTopics, words, ops );

%% Train APM with 3 topics
fprintf('Training APM with 3 topics on subset of Grolier encyclopedia data\n');
ops = [];
ops.baseFilename = 'apm-3topics';
ops.numWorkers = 12; % Parallel execution with 12 workers (NOTE: Must have Parallel Computing Toolbox)
numTopics = 3;
[Wt, thetaNodeArray, thetaEdgesArray] = apm( grolier, numTopics, words, ops );
