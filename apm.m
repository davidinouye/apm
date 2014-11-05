function [Wt, thetaNodeArray, thetaEdgesArray, apmNums] = apm( Xt, numTopics, words, ops )
%APM - Trains an Admixture of Poisson MRFs (APM) model on an input document word matrix Xt
% Author: David I. Inouye  Email: dinouye@cs.utexas.edu  Website: cs.utexas.edu/~dinouye
% Please cite: 
%   Capturing Semantically Meaningful Word Dependencies with an Admixture of Poisson MRFs
%   D. Inouye, P. Ravikumar, I. Dhillon
%   Neural Information Processing Systems (NIPS) 27, 2014.
%
% Input:
% Xt        N x P document word matrix where P is the number of dimensions 
%           (e.g. number of words) and N is the number of instances
%           (NOTE: Data matrix will be augmented by column of ones)
% numTopics Number of topics to train (Default: 3)
% words     P x 1 cell array of words
% ops       Options structure for APM.  Defaults will be used for all
%           fields unless specified in this structure.  Most common options
%           are below but more options can be found in utils/APMNums.m
%           file.
%           ops.lambda - Regularization parameter (e.g. 1, 0.1, 0.001). 
%             If 'trace' is given, then the algorithm will trace through 
%             different values of lambda and train multiple models. 
%             (Default: 'trace')
%           ops.heldOutPercent - Used in combination with ops.lambda =
%             'trace'.  This specifies the percentage of data to use as a
%             held out dataset to automatically determine a reasonable
%             lambda. (Default: 0.1)
%           ops.numWorkers - The number of parallel workers to attempt to
%             launch for parallel computation. (Default: 8)
%           ops.isMixture - If set to true, fit a mixture of Poisson MRFs
%             instead of an admixture (i.e. admixture weights will be
%             indicator vectors). (Default: false)
%           ops.maxTraceIter - Number of different lambda values to try
%             when using ops.lambda = 'trace'. (Default: 15)
%
% Output:
% Wt        Fitted admixture weights
% thetaNodeArray K x 1 cell array of the node parameter vector for each topic
% thetaEdgesArray K x 1 cell array of the edge parameter matrix for each topic
% apmNums   APM options and auxiliary numbers
%
% WARNING: If ops.lambda = 'trace', this will NOT return the model
% parameters for all lambda values that were attempted.  If
% ops.heldOutPercent > 0, then this will return the final model trained
% after determining lambda by using the held-out dataset. Otherwise, if
% ops.lambda = 'trace', the models should be saved to mat files with the
% ops.saveVerbosity >= 1 option.
%
% [Wt, thetaNodeArray, thetaEdgesArray, apmNums] = apm( Xt, numTopics, words, ops )
if(nargin < 2); numTopics = 3; fprintf('NOTE: Setting number of topics to 3 since not specified.\n'); end;
if(nargin < 3); words = cell(size(Xt,2),1); fprintf('NOTE: Creating empty word list since not given.'); end;
if(nargin < 4); ops = []; end;

%% Initialization
addpath(fullfile(fileparts(mfilename('fullpath')), 'utils'));
% Rename and inflate by column of ones
Zt = Xt; clear Xt;
Zt = setmatrixtype([ones(size(Zt,1),1), Zt]); % Add column of ones for computing node parameters (i.e. "intercept" term in Poisson regressions)
apmNums = configureAPM(size(Zt), numTopics, ops); clear ops; % Setup apmNums object

% Attempt to create output directories
dirArray = {'', 'gexf', 'mat'};
for i = 1:length(dirArray)
    dirPath = fullfile(apmNums.baseDir, dirArray{i});
    try if(apmNums.saveVerbosity >= 1 && exist(dirPath, 'dir') ~= 7); mkdir(dirPath); end
    catch err; fprintf('Error creating dirPath = %s, error\n%s', dirPath, err.message); end
end

% Setup parallel pool of MATLAB workers if parallel toolbox is installed
initParallel(apmNums.numWorkers);

% Display options and metadata for this run
disp(apmNums); disp(apmNums.metadata);

if(apmNums.trace && apmNums.heldOutPercent <= 0)
    if(apmNums.saveVerbosity == 0)
        fprintf('WARNING: %s\n%s\n%s\n%s\n%s\n','Trained models will be lost because you are tracing through lambda values',...
            'but the function will not return the parameters',...
            '(because storing the parameters for all lambda values would take up too much memor).',...
            'Please specify heldOutPercent to let the model automatically determine a reasonable lambda.',...
            'Or, specify a particular lambda rather than ''trace''.');
    else
        fprintf('NOTE: All models will be stored in files rather than being returned by the function');
    end
end

%% Initialize W and PMRF parameters
[Wt thetaNodeArray thetaEdgesArray] = initVars(apmNums);
% Initialize held out set if using to automatically determine a good lambda
if(apmNums.heldOutPercent > 0)
    rngSettings = rng; rng(1);
    permutation = randperm(apmNums.n);
    splitIdx = round(apmNums.heldOutPercent*apmNums.n);
    rng(rngSettings);

    % Create ZtTune and reset Zt
    ZtHeldOut = Zt(permutation(1:splitIdx),:);
    Zt = Zt(permutation((splitIdx+1):end),:);

    % Create WtTune and reset Wt
    WtHeldOut = Wt(permutation(1:splitIdx),:);
    Wt = Wt(permutation((splitIdx+1):end),:);
    % Fix the current number of instances
    apmNums.n = size(Zt,1);
    
    % Initialize empty best model
    heldOutObjective = NaN(apmNums.maxTraceIter, 1);
    bestModel.thetaNodeArray = thetaNodeArray;
    bestModel.thetaEdgesArray = thetaEdgesArray;
    bestModel.Wt = Wt;
    bestModel.WtHeldOut = WtHeldOut;
    bestModel.apmNums = APMNums.loadobj(APMNums.getstruct(apmNums));
    bestModel.apmNums.innerAfter = Inf;
end

%% Trace through different lambda values
% Setup loop variables
apmNums.outerAfter = apmNums.n; % If all parameters start at 0, this is a good objective value estimate
pmrfTime = 0; admixWeightsTime = 0;
traceIter = 1; heldOutFinalIter = false;
fprintf('Starting iterations\n');
while traceIter <= apmNums.maxTraceIter
    % Update lambda for tracing
    if(apmNums.trace && traceIter > 1)
        % Starting value is from the maximum gradient (thus if lambda is
        % greater than maxGradient the model will stay independent)
        apmNums.lambda = apmNums.maxGradient*0.5^(traceIter-1); % Decrease exponentially 
        apmNums.independent = false;
    elseif(traceIter > 1)
        break; % Skip other iterations if non-trace
    end
    
    %% Alternate between optimizing PMRF parameters and admixture weights
    isConverged = false;
    for iter = 1:apmNums.maxAlternateIter
        apmNums.outerBefore = apmNums.outerAfter; % Set objective to previous value
        
        % Remove independent flag to compute minLam from gradient if last
        % independent iteration (because when independent is true, pmrfs 
        % does not compute full gradient)
        if(apmNums.trace && iter == apmNums.maxAlternateIter); apmNums.independent = false; end;
        
        %% Compute k PMRF components with fixed W
        tstart = tic;
        [thetaNodeArray, thetaEdgesArray, ~, tempMaxGradient] = pmrfs( Zt, Wt, thetaNodeArray, thetaEdgesArray, apmNums);
        
        %% PMRFs post processing
        apmTimeOneIter = toc(tstart);
        pmrfTime = pmrfTime + apmTimeOneIter;
        apmNums.outerAfter = evalObj(Zt, Wt, thetaNodeArray, thetaEdgesArray, apmNums); % Needed to implicitly compute regularization/Lasso term for objective value later
        % Set maxGradient during first trace iteration (i.e. during independent phase using gradient)
        if(apmNums.trace && traceIter == 1)
            apmNums.maxGradient = tempMaxGradient;
        end
        % Check objective for debugging
        if(apmNums.debug)
            checkobjective(apmNums.outerBefore, apmNums.outerAfter, 'Not decreasing from pmrfs.outerBefore to pmrfs.outerAfter', 'decreasing');
            checkobjective(apmNums.outerAfter, apmNums.innerAfter, 'Not equal from pmrfs.outerAfter to pmrfs.innerAfter');
        else
            apmNums.outerAfter = apmNums.innerAfter;
        end

        %% Compute W with k PMRF components fixed
        apmNums.outerBefore = apmNums.outerAfter;
        if(apmNums.k > 1)
            Zt = Zt'; Wt = Wt'; % Transpose inplace for computational reasons
            tstart = tic;
            [Wt, ~] = admixweights(Zt(2:end,:), Wt, thetaNodeArray, thetaEdgesArray, apmNums);
            admixWeightsTimeOneIter = toc(tstart);
            Zt = Zt'; Wt = Wt'; % Reverse transpose inplace
            apmNums.outerAfter = evalObj(Zt, Wt, thetaNodeArray, thetaEdgesArray, apmNums);
        else
            admixWeightsTimeOneIter = 0; % Don't do anything if only 1 topic
        end
        
        %% Admix weights post-processing
        admixWeightsTime = admixWeightsTime + admixWeightsTimeOneIter;
        % Check objective for debugging
        if(apmNums.debug)
            checkobjective(apmNums.outerBefore, apmNums.outerAfter, 'Not decreasing from admixweights.outerBefore to admixweights.outerAfter','decreasing');
            checkobjective(apmNums.outerAfter, apmNums.innerAfter, 'Not equal from admixweights.outerAfter to admixweights.innerAfter');
            checkobjective(apmNums.prevOuterAfter, apmNums.outerAfter, 'Not decreasing from prevOuterAfter to outerAfter', 'decreasing');
        else
            apmNums.outerAfter = apmNums.innerAfter; % Use inner objective function value
        end
        
        %% Check convergence condition based on relative difference in objective
        %  Also mark convergence if only 1 topic (i.e. single PMRF iteration is 
        %   sufficient since admixture weights not changing)
        relativeDiffObjective = abs((apmNums.prevOuterAfter - apmNums.outerAfter)/apmNums.prevOuterAfter);
        if(relativeDiffObjective < apmNums.convergeThreshOuter || apmNums.k == 1)
            % If it is independent and has converged, do one more iteration so that
            %  the full gradient is calculated and hence a good starting position for
            %  tracing through lambda is obtained
            if(apmNums.trace && apmNums.independent); apmNums.independent = false; 
            else isConverged = true; end;
        end
        
        %% Print out information for current iteration
        % Compute +nnz
        nnzPosVec = zeros(1, apmNums.k);
        nnzNegVec = zeros(1, apmNums.k);
        for j = 1:apmNums.k; nnzPosVec(j) = nnz(thetaEdgesArray{j} > 0); end
        for j = 1:apmNums.k; nnzNegVec(j) = nnz(thetaEdgesArray{j} < 0); end
        % Print header line
        if(iter == 1 && traceIter == 1)
            fprintf('cols, traceIter, iter, relativeDiff, objective, pmrfsOneIter, pmrfsTotal, weightsOneIter, weightsTotal, lambda, nnzPosVec, nnzNegVec, avgNNZ\n')
        end
        fprintf('iter, %d, %d, %.2e, %.6e, %.2f, %.2f, %.2f, %.2f, %.2g, %s, %s, %d\n', ...
            traceIter, iter, relativeDiffObjective, apmNums.outerAfter, ...
            apmTimeOneIter, pmrfTime, admixWeightsTimeOneIter, admixWeightsTime, ...
            apmNums.lambda, ['[' num2str(nnzPosVec, '%d ') ']'], ['[' num2str(nnzPosVec, '%d ') ']'], ...
            round(mean(nnzPosVec+nnzNegVec)));

        %% Save results of main iteration
        % Save original filename the first trace iteration
        if(apmNums.trace)
            if(traceIter == 1 && iter == 1); origFilename = apmNums.baseFilename; end;
            if(heldOutFinalIter)
                apmNums.baseFilename = sprintf('%s_finalModel', origFilename);
            else
                apmNums.baseFilename = sprintf('%s_traceIter-%02d', origFilename, traceIter);
            end
        end
        % Save profile information if profiling on
        if(apmNums.profileAlg); profileInfo = profile('info'); end;
        
        % Only save when converged and only the last iteration if a
        %  held-out dataset is being used
        if((isConverged || iter == apmNums.maxAlternateIter) && (apmNums.heldOutPercent == 0 || heldOutFinalIter))
            %% Save all variables every iteration or every last iteration depending on saveVerbosity
            if(apmNums.saveVerbosity >= 1)
                matFilename = [apmNums.baseDir '/mat/' apmNums.baseFilename '.mat'];
                fprintf('Saving model (i.e. Wt, thetaNodeArray, thetaEdgesArray, apmNums) to file: %s\n', matFilename);
                if(apmNums.debug)
                    save(matFilename);
                else
                    save(matFilename, 'Wt', 'thetaNodeArray', 'thetaEdgesArray', 'apmNums');
                end
            end

            %% Save topic graphs
            if(apmNums.saveVerbosity >= 2)
                gexfFilename = [apmNums.baseDir '/gexf/' apmNums.baseFilename];
                fprintf('Saving model as GEXF file to be opened in Gephi <gephi.org>: %s\n', [gexfFilename '.gexf']);
                savegexf(gexfFilename, thetaNodeArray, thetaEdgesArray, words);
            end
        end

        %% Display summary output and break if converged or last iteration
        if(isConverged || iter == apmNums.maxAlternateIter)
            %% Display summary
            topN = 10;
            summary = cell(apmNums.k*(topN+1)+1, 6);
            summary(1, :) = {'NodeWgt', 'TopNodes', 'EdgeWgt', 'Top+Edges', 'EdgeWgt', 'Top-Edges'};
            for j = 1:apmNums.k
                offset = (j-1)*(topN+1)+1;
                summary(offset+1, :) = {sprintf('Topic %d',j), '', '', '', '', ''};
                [sortedNode, sortedNodeI] = sort(thetaNodeArray{j}, 'descend');
                
                edges = (thetaEdgesArray{j} + thetaEdgesArray{j}')/2;
                edges = edges - triu(edges); % Remove duplicate edges
                [sortedValues, sortedValuesI]= sort(edges(:));
                minValues = sortedValues;
                maxValues = flipud(sortedValues);
                [min1, min2] = ind2sub(size(edges), sortedValuesI);
                [max1, max2] = ind2sub(size(edges), flipud(sortedValuesI));

                offset = offset + 1;
                for topI = 1:topN
                    % Weight, node, weight, +edge, weight -edge
                    summary{offset+topI, 1} = sortedNode(topI);
                    summary{offset+topI, 2} = words{sortedNodeI(topI)};
                    summary{offset+topI, 3} = maxValues(topI);
                    summary{offset+topI, 4} = sprintf('%s+%s', words{max1(topI)}, words{max2(topI)});
                    summary{offset+topI, 5} = minValues(topI);
                    summary{offset+topI, 6} = sprintf('%s-%s', words{min1(topI)}, words{min2(topI)});
                end
            end
            disp(summary);
            
            %% Compute held out loss and update best model parameters
            if(apmNums.heldOutPercent > 0)
                % Copy APMnums for fitting held out set
                apmNumsHeldOut = APMNums.loadobj(APMNums.getstruct(apmNums));
                apmNumsHeldOut.n = size(ZtHeldOut,1);
                apmNumsHeldOut.verbosity = 1;
                apmNumsHeldOut.lassoTerm = 0; % Remove regularization term
                % Fit WtHeldOut
                ZtHeldOut = ZtHeldOut'; WtHeldOut = WtHeldOut'; % Transpose inplace for computational reasons
                [WtHeldOut, fWHeldOut] = admixweights(ZtHeldOut(2:end,:), WtHeldOut, thetaNodeArray, thetaEdgesArray, apmNumsHeldOut);
                ZtHeldOut = ZtHeldOut'; WtHeldOut = WtHeldOut'; % Reverse transpose inplace

                heldOutObjective(traceIter) = apmNumsHeldOut.innerAfter;
                % Check if this has better held out error/objective and save model
                fprintf('traceIter = %d, Held-out objective: %g\n', traceIter, heldOutObjective(traceIter));
                if(apmNumsHeldOut.innerAfter < bestModel.apmNums.innerAfter)
                    bestModel.thetaNodeArray = thetaNodeArray;
                    bestModel.thetaEdgesArray = thetaEdgesArray;
                    bestModel.Wt = Wt;
                    bestModel.WtHeldOut = WtHeldOut;
                    bestModel.apmNums = apmNumsHeldOut;
                end
                
                % On final trace iteration or when held-out likelihood is 50% greater than best,
                % reset to be bestModel parameters
                if(traceIter == apmNums.maxTraceIter || apmNumsHeldOut.innerAfter > 1.5*bestModel.apmNums.innerAfter)
                    fprintf('Best lambda based on held-out documents was lambda = %g.\nThis lambda being set for final training with all data.\n',bestModel.apmNums.lambda);
                    
                    % Reset data and parameters for final model fitting
                    Zt = [Zt; ZtHeldOut];
                    Wt = [bestModel.Wt; bestModel.WtHeldOut];
                    apmNums.n = size(Zt,1);
                    n = apmNums.n;
                    thetaNodeArray = bestModel.thetaNodeArray;
                    thetaEdgesArray = bestModel.thetaEdgesArray;

                    % Reset lambda to best model
                    apmNums.lambda = bestModel.apmNums.lambda;
                    apmNums.independent = bestModel.apmNums.independent;
                    apmNums.heldOutPercent = 0; % Stop testing held out
                    
                    % Reset objective value (except for outerAfter in order to provide reasonable estimate for first iteration)
                    apmNums.lassoTerm = NaN; apmNums.outerBefore = NaN; apmNums.innerBefore = NaN; apmNums.innerAfter = NaN;
                    
                    % Do one more trace iteration to train final model
                    traceIter = apmNums.maxTraceIter;
                    apmNums.maxTraceIter = apmNums.maxTraceIter + 1;
                    heldOutFinalIter = true;
                end
            end
            
            %% Post-process model if function provided
            if(isa(apmNums.postProcessFunc, 'function_handle'))
                try apmNums.postProcessFunc(Zt, Wt, thetaNodeArray, thetaEdgeArray, apmNums);
                catch err; fprintf('Error running postProcess function: %s\n', err.message); end
            end

            break; % Break out of inner trace iteration loop
        end
        
        apmNums.prevOuterAfter = apmNums.outerAfter; % Save old value
    end
    traceIter = traceIter + 1;
end

%% Save profiling information if option set
if(apmNums.saveVerbosity >= 1 && apmNums.profileAlg)
    profileInfo = profile('info');
    profsave(profileInfo, [apmNums.baseDir '/mat/' apmNums.baseFilename '_PROFILE.html']);
    save([apmNums.baseDir '/mat/' apmNums.baseFilename '_PROFILE.mat'], 'profileInfo');
end

%% Clear variables if tracing over different lambda but not using heldOut dataset to automatically determine lambda
if(apmNums.trace && apmNums.heldOutPercent <= 0)
    Wt = [];
    thetaNodeArray = [];
    thetaEdgesArray = [];
end

end

function apmNums = configureAPM(sizeZt, numTopics, ops)
    apmNums = APMNums();
    
    % Loop through fields in ops structure
    numsFields = fieldnames(apmNums);
    for field = fieldnames(ops)'
        if(sum(strcmp(field{1}, numsFields)) > 0)
            apmNums.(field{1}) = ops.(field{1});
        end
    end
    
    % Change heldOutPercent = 0 if lambda is not 'trace'
    if(apmNums.heldOutPercent ~= 0 && ~strcmp(apmNums.lambda, 'trace'))
        fprintf('NOTE: Changing heldOutPercent = 0 because lambda was not set to ''trace''.\n')
        apmNums.heldOutPercent = 0;
    end
    
    % If ops was not set, then set saveVerbosity based on filename
    if(~isfield(ops,'saveVerbosity'))
        if(strcmp(apmNums.baseFilename,''))
            fprintf('No baseFilename given in ops so not saving any files\n');
            apmNums.saveVerbosity = 0;
        else
            apmNums.saveVerbosity = 1;
        end
    end
    
    % Set # of words, instances and topics
    apmNums.p = sizeZt(2)-1;
    apmNums.n = sizeZt(1);
    apmNums.k = numTopics;

    % Setup lambda parameter(s)
    if(strcmp(apmNums.lambda,'trace') || apmNums.lambda == Inf)
        fprintf('NOTE: ops.lambda = ''trace'' so starting with very large lambda\nand tracing through different lambda values.\n');
        lamTemp = 1e100;
        apmNums.independent = true;
        if(strcmp(apmNums.lambda,'trace')); apmNums.trace = true; end;
    else
        lamTemp = apmNums.lambda;
    end

    % Set lambda and starting max gradient
    apmNums.lambda = lamTemp;
    apmNums.maxGradient = 1e100;

    % Turn on profiler if requested
    if(apmNums.profileAlg)
        profile off; profile clear; profile on -timer real; fprintf('Turned on profiling...\n');
    end
end

function obj = evalObj(Zt, Wt, thetaNodeArray, thetaEdgesArray, apmNums)
    n = apmNums.n; k = apmNums.k;
    % Compute lasso term for innerBefore and innerAfter
    apmNums.lassoTerm = 0;
    for j = 1:k
        apmNums.lassoTerm = apmNums.lassoTerm + apmNums.lambda*sum(abs(thetaEdgesArray{j}(:)));
    end
    
    % Only compute full objective if debugging
    % NOTE: VERY slow objective computation especially for large n but 
    % good to double check objective for debugging.
    if(apmNums.debug)
        obj = 0;
        for i = 1:n
            gamVec = zeros(size(thetaNodeArray{1}));
            for j = 1:k
                gamVec = gamVec + Wt(i,j)*(thetaNodeArray{j} + (Zt(i,2:end)*thetaEdgesArray{j})');
            end
            linearTerm = (Zt(i,2:end)*gamVec)';
            expTerm = sum(exp(gamVec));
            obj = obj + (1/n)*(- linearTerm + expTerm);
        end
        obj = obj + apmNums.lassoTerm;
    else
        obj = NaN;
    end
end

function [Wt, thetaNodeArray, thetaEdgesArray] = initVars(apmNums)
    % Initialize W randomly
    rng(1); % Set random seed
    if(apmNums.k > 1)
        if(apmNums.isMixture)
            % Initialize indicator matrix
            fprintf('Initializing W as random indicator matrix\n');
            [~, temp] = sort(rand(apmNums.k, apmNums.n), 1);
            W = double(temp == apmNums.k);
        else
            fprintf('Initializing W randomly with Dirichlet random vectors\n');
            W = randdirichlet(apmNums.k, apmNums.n, 1); % Uniform dirichlet (i.e. uniformly from simplex)
        end
    else
        W = ones(apmNums.k, apmNums.n);
    end
    Wt = W';

    % Initialize thetaNodeArray, thetaEdgesArray
    fprintf('Initializing theta to 0\n');
    thetaNodeArray = cell(apmNums.k,1);
    thetaEdgesArray = cell(apmNums.k,1);
    for j = 1:apmNums.k
        thetaNodeArray{j} = zeros(apmNums.p,1);
        thetaEdgesArray{j} = sparse(apmNums.p,apmNums.p);
    end
end
