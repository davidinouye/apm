function [thetaNodeArray, thetaEdgesArray, fPhi, maxGradient] = pmrfs( Zt, Wt, thetaNodeArray, thetaEdgesArray, apmNums )
%PMRFS Estimate k PMRFs using proximal Newton-like method
% Author: David I. Inouye  Email: dinouye@cs.utexas.edu  Website: cs.utexas.edu/~dinouye
% Please cite: 
%   Capturing Semantically Meaningful Word Dependencies with an Admixture of Poisson MRFs
%   D. Inouye, P. Ravikumar, I. Dhillon
%   Neural Information Processing Systems (NIPS) 27, 2014.
%
% Wrapper function to estimate all p Phi parameters
% (See PMRFSINGLE subfunction in code)
%
% Input:
% Zt        N x (P+1) sparse matrix of count data with ones column
% Wt        N x K matrix of admixture weights
% thetaNodeArray Current estimate of thetaNode parameters
% thetaEdgesArray Current estimate of thetaEdges parameters
% apmNums   APM options and auxiliary numbers
%
% Output:
% thetaNodeAray New estimate of thetaNode parameters
% thetaEdgesArray New estimate of thetaEdges parameters
% fPhi      Vector of function values for each subproblem
% maxGradient Maximum gradient (used for determining first lambda when
%           tracing through different lambdas
%
% [thetaNodeArray, thetaEdgesArray, fPhi, maxGradient] = pmrfs( Zt, Wt, thetaNodeArray, thetaEdgesArray, apmNums )
addpath(fullfile(fileparts(mfilename('fullpath')), 'utils'));

%% Initialization
n = apmNums.n; p = apmNums.p;
fPhi = zeros(p,1); fPhi0 = zeros(p,1);
maxGradientVec = zeros(p,1);
PhiTensor = theta2phi(thetaNodeArray, thetaEdgesArray, apmNums);

%% Main loop
if(apmNums.verbosity >= 2); fprintf('Training component Poisson MRFs\n'); end
if(apmNums.numWorkers > 1)
    parfor (s = 1:p, apmNums.numWorkers)
        [PhiTensor(:,:,s), fPhi(s), maxGradientVec(s), fPhi0(s)] = pmrfsingle(Zt, Wt, s, PhiTensor(:,:,s), apmNums);
    end
else
    for s = 1:p
        [PhiTensor(:,:,s), fPhi(s), maxGradientVec(s), fPhi0(s)] = pmrfsingle(Zt, Wt, s, PhiTensor(:,:,s), apmNums);
    end
end

%% Post processing
% Extract new PhiTensor into arrays for output
[thetaNodeArray, thetaEdgesArray] = phi2theta(PhiTensor, thetaNodeArray, thetaEdgesArray, apmNums);

% Find maximum of the minimum lambdas to ensure that the solution is independent
maxGradient = max(maxGradientVec);

% Check objective for debugging purposes
apmNums.innerBefore = sum(fPhi0);
apmNums.innerAfter = sum(fPhi);

if(apmNums.debug)
    checkobjective(apmNums.innerBefore, apmNums.innerAfter, 'Inner pmrfs not decreasing', 'decreasing');
    checkobjective(apmNums.innerBefore, apmNums.outerBefore, 'Inner pmrfs not equal to outerBefore');
end

end

function [Phi, fPhi, maxGradient, fPhi0] = pmrfsingle(Zt, Wt, s, Phi, apmNums)
%PMRFSINGLE Estimate PMRF node conditional of variable s with proximal Newton method
% Input:
% Zt        N x (P+1) sparse matrix of count data with ones column
% Wt        N x K matrix of admixture weights
% s         Current variable index
% Phi       The initial estimate of the parameters for this variable
% apmNums   APM options and auxiliary numbers
%
% Output:
% Phi       Parameter estimate for this variable
% fPhi      Function value at this Phi
% maxGradient Maximum gradient (used for determining first lambda when
%           tracing through different lambdas
% fPhi0     Initial function value
tic;

%% Compute initial Zsum, gamVec, and objective
Zsum = full(bsxfun(@times, Zt, Zt(:,s+1))'*Wt);
Phi = setmatrixtype(Phi); % Set to sparse or dense matrix based on sparsity
gamVec = computegamvec(Zt, Phi, Wt, apmNums);
fPhi0 = evalobj(Phi, gamVec, Zsum, apmNums);
if(fPhi0 > 1e10)
    % Reset Phi if objective is abnormally high, this could be 
    % caused by the instability added when held-out instances are
    % added to the dataset after finding appropriate lambda
    % NOTE: This should not change the solution since the problem is convex
    Phi(Phi ~= 0) = 0; % Reset to 0
    fPhi0 = evalobj(Phi, gamVec, Zsum, apmNums); % Recalculate initial objective
end
fPhi = fPhi0;

%% Initialize loop variables
n = apmNums.n; p = apmNums.p; k = apmNums.k;
GradG = zeros(size(Phi));
outerIter = 1; outerMaxIter = 500;
debugOutput = cell(outerMaxIter, 1);
relativeDiff = 1;

%% Main outer Newton loop
while outerIter <= outerMaxIter
    %% Find gradient
    GradG = gradient(GradG, Zt, Wt, gamVec, Zsum, s, apmNums);
    
    % Store the max gradient over edges if not independent (i.e. full
    %  gradient has been computed).  Needed for determing first lambda when
    % using 'trace'
    if(apmNums.independent); maxGradient = apmNums.maxGradient;
    else maxGradient = max(max(abs(GradG(2:end,:)))); end
    
    %% Determine free set
    if(apmNums.independent)
        freeSetBool = false(size(GradG)); % Only optimize nodes
    else
        freeSetBool = ((abs(GradG) >= apmNums.lambda) | Phi ~= 0);
    end
    % Always update node thetas, never update self-edge (i.e. edge from s to s b/c must = 0)
    freeSetBool(1,:) = true; freeSetBool(s+1,:) = false;
    [r, c] = find(freeSetBool);
    freeSet = [r, c];
    
    %% Find Newton direction
    % Precompute some values that will not change
    aPrecomp = zeros(1,size(freeSet,1));
    bPrecomp = zeros(n,size(freeSet,1));
    WtZt = zeros(size(bPrecomp));
    for idx = 1:size(freeSet,1)
        t = freeSet(idx,1); j = freeSet(idx,2);
        WtZt(:,idx) = Wt(:,j).*Zt(:,t);
    end
    
    WtZt = setmatrixtype(WtZt); % Sparsify if appropriate for computational reasons
    for idx = 1:size(freeSet,1)
        bPrecomp(:,idx) = (1/n)*gamVec.*WtZt(:,idx);
        aPrecomp(idx) = sum(bPrecomp(:,idx).*WtZt(:,idx)); % Squared term
    end
    bPrecomp = setmatrixtype(bPrecomp); % Sparsify if appropriate for computational reasons
    
    % Initialize some loop variables
    D = sparse(p+1, k);
    r = zeros(n,1);
    innerIter = 1;
    innerMaxIter = 1 + outerIter/3;  % QUIC setting for stopping condition
    % Vectorize sparse matrices
    Dv = full(D(sub2ind(size(D), freeSet(:,1), freeSet(:,2))));
    Phiv = full(Phi(sub2ind(size(Phi), freeSet(:,1), freeSet(:,2))));
    while innerIter <= innerMaxIter
        DvOld = Dv;
        % Coordinate descent
        for idx = 1:size(freeSet,1)
            t = freeSet(idx,1); j = freeSet(idx,2);
            % Precomputed version of: a = gamVec'*((Wt(:,j).*Zt(:,t)).^2);
            a = aPrecomp(idx);
            % Only step if curvature is nonzero
            if(a == 0)
                a = 1; % If a = 0, then linear function so just do not scale the gradient
            end
            % Using precomputed values for: b = GradG(t,j) + sum(gamVec.*Wt(:,j).*Zt(:,t).*r);
            b = GradG(t,j) + r'*bPrecomp(:,idx);
            if(t==1)
                % Update without thresholding if node parameter (i.e. t=1)
                mu = -b/a;
            else
                % Update edge parameter with thresholding closed-form solution
                c = Phiv(idx) + Dv(idx);
                z = c - b/a;
                mu = -c + sign(z)*max(abs(z) - apmNums.lambda/a, 0);
            end
            % Update direction and r vector
            Dv(idx) = Dv(idx) + mu;
            r = r + mu*WtZt(:,idx);
        end
        
        % Output debugging information for coordinate descent iteration
        if(apmNums.verbosity >= 4)
            innerRelative = norm(DvOld-Dv)/norm(Dv);
            fprintf('      coordDescentIter = %d, newtonIter = %d, s = %d, relativeDiff = %g\n', innerIter, outerIter, s, innerRelative);
        end
        innerIter = innerIter + 1;
    end
    % Reconstruct sparse D after calculations
    D = sparse(freeSet(:,1), freeSet(:,2), Dv, size(D,1), size(D,2));
    
    %% Find step size
    maxStepExp = 50;
    for stepExp = 0:maxStepExp;
        % Evaluate possible new Phi
        stepSize = apmNums.stepParam1^stepExp;
        PhiNew = Phi + stepSize*D;
        gamVec = computegamvec(Zt, PhiNew, Wt, apmNums); % Overwrite old gamma vector since no longer needed
        fPhiNew = evalobj(PhiNew, gamVec, Zsum, apmNums);
        
        % Armijo step rule
        if(fPhiNew <= fPhi + stepSize*apmNums.stepParam2*(...
                sum(GradG(freeSetBool).*D(freeSetBool)) ...
                + sum(sum(abs(Phi(2:end,:)+D(2:end,:)))) ...
                - sum(sum(abs(Phi(2:end,:)))) ...
                ) )
            break;
        end
        
        if(stepExp == maxStepExp && apmNums.verbosity >= 1)
            fprintf('  Warning: maxBetaExp reached, s = %d, outerIter = %d, +nnz = %d, -nnz = %d, objective = %g, relative diff = %g\n', s, outerIter, nnz(Phi>0), nnz(Phi<0), fPhi, relativeDiff);
        end
    end
    
    %% Post processing of new Phi
    % Compute relative difference based on previous full objective value (i.e. apmNums.outerBefore)
    relativeDiff = abs(fPhi-fPhiNew)/abs(fPhi);
    
    % Only update if decreased
    if(fPhiNew <= fPhi)
        fPhi = fPhiNew;
        Phi = PhiNew;
    else
        relativeDiff = 0; % i.e. no update
    end
    
    % Save some debug information
    iterationOutput = sprintf('    newtonIter = %d, stepSize = %g, s = %d, obj = %g, rel. diff = %g, +nnz = %d, -nnz = %d', outerIter, stepSize, s, fPhi, relativeDiff, full(nnz(Phi(2:end,:)>0)), full(nnz(Phi(2:end,:)<0)));
    if(apmNums.verbosity >= 3)
        fprintf('%s\n',iterationOutput);
    else
        debugOutput{outerIter} = iterationOutput;
    end
    
    % Check convergence condition (either converged to within threshold or function value is miniscule in comparison to complete objective value)
    %   (Second condition is needed for words that are very rare)
    if(relativeDiff < apmNums.convergeThreshInner || abs(fPhi/apmNums.outerBefore) < apmNums.convergeThreshInner/p)
        break;
    end
    
    outerIter = outerIter + 1;
end

singleVariableTime = toc;

%% Post-processing and debug output
% Output debug information
checkobjective(fPhi0, fPhi, 'Not decreasing after pmrfsingle', 'decreasing');
outputStr = sprintf('s = %d, newtonIter = %d, stepSize = %g, obj = %g, rel. diff = %g, +nnz = %d, -nnz = %d, time = %g s', s, outerIter, stepSize, fPhi, relativeDiff, full(nnz(Phi(2:end,:)>0)), full(nnz(Phi(2:end,:)<0)), singleVariableTime);
if(outerIter >= outerMaxIter && apmNums.verbosity >= 1)
    fprintf('  Warning newtonIterMax reached for s = %d\n', s);
    if(apmNums.verbosity >= 3 || apmNums.debug)
        fprintf('Debug loop output:\n');
        for ii = 1:size(debugOutput,1); fprintf('%s\n',debugOutput{ii}); end
    end
    fprintf('  %s\n',outputStr);
elseif(apmNums.verbosity >= 2)
    fprintf('  %s\n',outputStr);
end

% Post processing of final output
Phi = full(Phi); % Make sure return type of Phi is full

end

function f = evalobj(Phi, gamVec, Zsum, apmNums)
    if(apmNums.independent)
        f = (1/apmNums.n)*(-sum(Phi(1,:).*Zsum(1,:)) + sum(gamVec));
    else
        singleLassoTerm = apmNums.lambda*sum(sum(abs(Phi(2:end,:))));
        f = (1/apmNums.n)*(-(Phi(:)'*Zsum(:)) + sum(gamVec)) + singleLassoTerm;
    end
    f = full(f); % Make non-sparse
end

function gamVec = computegamvec(Zt, Phi, Wt, apmNums)
    if(apmNums.independent)
        % Only use 1s column of Zt and top row of Phi since all 0s
        % Thus transformation is simple
        gamVec = exp(Wt*Phi(1,:)');
    else
        gamVec = exp(sum((Zt*Phi).*Wt,2));
    end
    
    % Cap infinities by realmax = 1e300 (Don't know if this is actually a
    %  problem anymore but don't want to remove it unless I know)
    gamVec(gamVec > 1e300) = 1e300;
end

function PhiTensor = theta2phi(thetaNodeArray, thetaEdgesArray, apmNums)
    PhiTensor = zeros(apmNums.p+1,apmNums.k,apmNums.p);
    % Extract values from thetaNodeArray and thetaEdgeArray
    for s = 1:apmNums.p
        for j = 1:apmNums.k
            PhiTensor(:,j,s) = [thetaNodeArray{j}(s); thetaEdgesArray{j}(:,s)];
        end
    end
end

function [thetaNodeArray, thetaEdgesArray] = phi2theta(PhiTensor, thetaNodeArray, thetaEdgesArray, apmNums)
    for s = 1:apmNums.p
        for j = 1:apmNums.k
            thetaNodeArray{j}(s) = full(PhiTensor(1,j,s));
            thetaEdgesArray{j}(:,s) = sparse(PhiTensor(2:end,j,s));
        end
    end
end

function GradG = gradient(GradG, Zt, Wt, gamVec, Zsum, s, apmNums)
    % If independent, only using top row of Z and gamVec since we only need to
    % compute gradient for top row of Phi
    n = apmNums.n; p = apmNums.p;
    if(apmNums.independent)
        GradG(1,:) = (1/n)*(-Zsum(1,:) + gamVec'*Wt);
    else
        % Trying to split it up so that this operation uses less memory
        threshN = 5000;
        if(n > threshN)
            pBlockSize = round(p/(n/threshN))+1; % Split p into blocks based on how large the dataset is
            for pBlockStart = 1:pBlockSize:p
                pBlock = pBlockStart:min(pBlockStart + (pBlockSize-1), p);
                GradG(pBlock,:) = (1/n)*(-Zsum(pBlock,:) + bsxfun(@times, Zt(:, pBlock), gamVec)'*Wt);
            end
        else
            GradG = (1/n)*(-Zsum + bsxfun(@times, Zt, gamVec)'*Wt);
        end
        GradG(s+1,:) = 0; % Don't update s+1 coordinate, also needed for minimum lambda that stays independent
    end
end
