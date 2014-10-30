function [W, fw] = admixweights( X, W, thetaNodeArray, thetaEdgesArray, apmNums )
%ADMIXWEIGHTS Estimate admixture weights for APM using Newton-like method
% Author: David I. Inouye  Email: dinouye@cs.utexas.edu  Website: cs.utexas.edu/~dinouye
% Please cite: 
%   Capturing Semantically Meaningful Word Dependencies with an Admixture of Poisson MRFs
%   D. Inouye, P. Ravikumar, I. Dhillon
%   Neural Information Processing Systems (NIPS) 27, 2014.
%
% Wrapper function to estimate all n admixture parameters
% (See ADMIXSINGLE subfunction in code)
%
% Input:
% X         P x N sparse matrix of count data with ones column
% W         K x N matrix of current admixture weight estimates
% thetaNodeArray Current estimate of thetaNode parameters
% thetaEdgesArray Current estimate of thetaEdges parameters
% apmNums   APM options and auxiliary numbers
%
% Output:
% W         New estimate of admixture weights parameters
% fW        Vector of function values for each subproblem
%
% [W, fw] = admixweights( X, W, thetaNodeArray, thetaEdgesArray, apmNums )
addpath(fullfile(fileparts(mfilename('fullpath')), 'utils'));
n = apmNums.n; p = apmNums.p; k = apmNums.k;

%% Iterate over batches of size 5000
maxBatchSize = apmNums.admixweightsBatchSize;
PiTensor = zeros(p,k,maxBatchSize);
fw = zeros(n,1); fw0 = zeros(n,1);
if(apmNums.verbosity >= 2); fprintf('Training admixture weights for each data instance\n'); end
for offset = 0:maxBatchSize:n
    % Calculate actual batch size which may be smaller than maxBatchSize
    batchSize = min(n-offset, maxBatchSize);
    
    %% Preprocessing
    for j=1:k
        PiTensor(:,j,1:batchSize) = full(bsxfun(@plus, thetaNodeArray{j}, thetaEdgesArray{j}'*X(:,(offset+1):(offset+batchSize)) ));
    end

    %% Main parallel loop
    parfor (i = 1:batchSize, apmNums.numWorkers)
        [W(:,offset+i), fw(offset+i), fw0(offset+i)] = admixsingle(W(:,offset+i), X(:,offset+i), PiTensor(:,:,i), offset+i, apmNums);
    end
end

%% Check objective
apmNums.innerBefore = sum(fw0) + apmNums.lassoTerm;
apmNums.innerAfter = sum(fw) + apmNums.lassoTerm;

if(apmNums.debug)
    checkobjective(apmNums.innerBefore, apmNums.innerAfter, 'Inner admixweights not decreasing', 'decreasing');
    checkobjective(apmNums.innerBefore, apmNums.outerBefore, 'Inner admixweights not equal to outerBefore');
end

end

function [w, fw, fw0] = admixsingle(w, x, PiT, i, apmNums)
%ADMIXSINGLE Estimate admixture weights for a single instance using Newton-like method
% Input:
% w         K x 1 vector of current admixture weights estimate
% x         P x 1 document/instance vector
% PiT       Variable constructed from thetaNodeArray and thetaEdgesArray
%           (helps simplify syntax of function)
% i         Current instance index
% Phi       The initial estimate of the parameters for this variable
% apmNums   APM options and auxiliary numbers
%
% Output:
% w         New K x 1 parameter estimate of admixture weights
% fw        Function value at new w
% fw0       Initial function value at old w
n = apmNums.n; p = apmNums.p; k = apmNums.k;

%% Preprocessing
Pi = PiT.';
psi = Pi*x; % x is the last part of z
gamVec = computegamvec(PiT, w);
fw0 = evalobj(w, psi, gamVec, apmNums);
fw = fw0;

%% Handling mixtures is much simpler
if(apmNums.isMixture)
    % Test all different components and find minimum negative pseudo likelihood
    fw = Inf;
    for j = 1:k
        wTest = (1:k == j).'; % Indicator vector for w
        gamVec = computegamvec(PiT,wTest);
        fwTest = evalobj(wTest, psi, gamVec, apmNums);
        if(fwTest < fw)
            fw = fwTest;
            w = wTest;
        end
    end
    return; % Don't do anything more for mixtures
end

outerIter = 1; outerMaxIter = 500;
relativeDiff = 1;
while outerIter <= outerMaxIter
    %% Find gradient and Hessian in closed-form
    gradW = -psi + Pi*gamVec;
    HessianW = Pi*bsxfun(@times, PiT, gamVec);
    
    %% Solve for Newton direction using dual coordinate descent
    % Precomputation
    aPrecomp = bsxfun(@plus, diag(HessianW), bsxfun(@plus, diag(HessianW)', (-2)*HessianW));
    bPrecomp = repmat(gradW,1,size(gradW,1)) - repmat(gradW',size(gradW,1),1);
    
    % Loop through coordinate pairs
    maxInnerIter = 8;
    d = zeros(k,1); r = zeros(k,1);
    for innerIter = 1:maxInnerIter
        dOld = d;
        % Loop through all coordinate pairs
        for minJ = 1:k
            for maxI = (minJ+1):k % Loop in column order as much as possible
                % Compute the solution to the 1D problem
                % Original: a = HessianW(maxI,maxI) - 2*HessianW(maxI,minJ) + HessianW(minJ, minJ);
                a = aPrecomp(maxI,minJ);
                % Original: b = gradW(maxI) - gradW(minJ) + r(maxI) - r(minJ);
                b = bPrecomp(maxI,minJ) + r(maxI) - r(minJ);
                mu = min( max(-b/a, -(w(maxI) + d(maxI))), w(minJ) + d(minJ) );

                % Update direction
                d(maxI) = d(maxI) + mu;
                d(minJ) = d(minJ) - mu;
                
                % Update auxillary variables
                r = r + mu*(HessianW(:,maxI) - HessianW(:,minJ));
            end
        end
        
        % Output debugging information for coordinate descent iteration
        if(apmNums.verbosity >= 4)
            innerRelative = norm(dOld-d)/norm(d);
            fprintf('      coordDescentIter = %d, newtonIter = %d, i = %d, relativeDiff = %g\n', innerIter, outerIter, i, innerRelative);
        end
    end
    
    %% Find step size
    maxStepExp = 50;
    for stepExp = 0:maxStepExp
        % Evaluate possible new w
        stepSize = apmNums.stepParam1^stepExp;
        wNew = w + stepSize*d;
        gamVec = computegamvec(PiT, wNew); % Overwrite old gamma vector since no longer needed
        fwNew = evalobj(wNew, psi, gamVec, apmNums);
        % Armijo rule
        if( fwNew <= fw + stepSize*apmNums.stepParam2*(gradW'*d) )
            break;
        end
    end
    
    % Compute relative difference and only update if decreased
    relativeDiff = abs(fw-fwNew)/abs(fw);
    if(fwNew <= fw)
        fw = fwNew;
        w = wNew;
    elseif(stepExp == maxStepExp)
        relativeDiff = 0; % i.e. no update
    else
        fprintf('  Warning: Armijo rule is not correct since fwNew < fw but stepExp ~= maxStepExp\n');
        relativeDiff = 1e-100;
    end
    
    if(apmNums.verbosity >= 3)
        fprintf('    newtonIter = %d, i = %d, stepSize = %d, obj = %g, rel. diff = %g\n', outerIter, i, stepSize, fwNew, relativeDiff);
    end
    
    % Check convergence condition (either converged to within threshold or function value is miniscule in comparison to complete objective value)
    %   (Second condition is needed for words that are very rare)
    if(relativeDiff < apmNums.convergeThreshInner || abs(fw/apmNums.outerBefore) < apmNums.convergeThreshInner/n)
        break;
    end
    outerIter = outerIter + 1;
end

%% Post-processing and debug output
checkobjective(fw0, fw, 'Not decreasing after admixsingle', 'decreasing');
outputStr = sprintf('i = %d, newtonIter = %d, stepSize = %g, obj = %g, rel. diff = %g', i, outerIter, stepSize, fw, relativeDiff);
if(outerIter >= outerMaxIter && apmNums.verbosity >= 2)
    fprintf('  Warning newtonIterMax reached for i = %d\n', i);
    fprintf('  %s\n',outputStr);
elseif(apmNums.verbosity >= 3)
    fprintf('  %s\n',outputStr);
end

end

function f = evalobj(w, psi, gamVec, apmNums)
    f = (1/apmNums.n)*(-w'*psi + sum(gamVec));
end

function gamVec = computegamvec(PiT,w)
    gamVec = exp(PiT*w);

    % Cap values to be under Inf
    gamVec(gamVec > 1e300) = 1e300;
end
