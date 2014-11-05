classdef APMNums < handle
%APMNUMS Object to pass APM parameters and auxiliary information between functions
% This is used to pass all the APM options to the functions.  This can make
%  adding new features in the future easier.  In addition, some auxiliary
%  information is passed between functions for debugging and other
%  purposes.

    properties
        %% APM Standard Options
        lambda = 'trace'; % Regularization parameter. Either a number or 'trace' which will train multiple models with different values of lambda. (e.g. 1, 0.01)
        heldOutPercent = 0.1; % Percentage of data for held-out dataset so that lambda can be chosen automatically (e.g. 0.1 means 10% held out)
        numWorkers = 8; % Number of workers for parallel execution (NOTE: Must have MATLAB Parallel Computing toolbox.)
        isMixture = false; % If true, fits a mixture instead (i.e. admixture weights will be indicator vectors)
        maxTraceIter = 15; % Number of different lambda to try when using lam = 'trace'
        
        %% APM Other Options
        verbosity = 2; % Amount of output to show.
        % 0 - Only show main iteration and errors
        % 1 - Also show warnings
        % 2 - Also show some output for each PMRF subproblem
        % 3 - Also show inner Newton iterations and each instance (admix weights)
        
        baseFilename = ''; % Base filename (extensions will be added. For example, if 'apm-model' is given then the output files may be like 'apm-model_traceIter-10.mat')
        baseDir = 'data'; % Defaults to 'data' directory
        saveVerbosity = 1; % Determines which files to save to disk
        % 0 - Don't save any files
        % 1 - Only save the last iteration models for each lambda value
        % 2 - Save GEXF files for each lambda value (NOTE: This can be slow if there are many edges)
        
        maxAlternateIter = 50; % Max number of alternating iterations (# of times to optimize PMRF parameters and then admixture weights)
        
        %% Developer/Experimental features
        debug = false; % Turn on debugging (checking to make sure objective is decreasing, etc.) NOTE: Could be much slower.
        profileAlg = false; % Turn on MATLAB profiling
        
        % Post process function plugin function be run after solution is found for each lambda
        %   signature of post process function: 
        %   @(Zt, Wt, thetaNodeArray, thetaEdgeArray, apmNums){};
        postProcessFunc = [];
        metadata = []; % Can be a structure that contains extra metadata that might be used in the postProcessFunc
        
        %% Algorithm Tuning Parameters
        stepParam1 = 0.5; % Step size parameter (i.e. will try 1, 0.5, 0.25, 0.125, etc.)
        stepParam2 = 1e-10; % Step size parameter for Armijo rule which only needs to be > 0 (commonly the "alpha" parameter)
        convergeThreshOuter = 1e-4; % Convergence threshold for outer alternating iterations
        convergeThreshInner = 1e-5; % Convergence threshold for individual Poisson node regressions and individual admixture weights
        admixweightsBatchSize = 5000; % Batch size for fitting admixture weights in order to use less memory for large datasets
        
        %% Useful auxiliary variables to pass around
        % WARNING: These variables are set during the execution of the
        % program and should not be modified directly.
        n; % Number of documents/instances
        p; % Number of words/dimensions
        k; % Number of topics/factors
        
        % Used to keep track of the maximum gradient for edges.
        % Needed for determining first lambda when using trace.
        maxGradient = 1e100;
        independent = false; % Set during first trace iteration so that full gradient is not usually computed
        trace = false; % Set if lam = 'trace'
        
        lassoTerm = NaN; % Used to save regularization term for computing objective
        % Auxilary variables for debugging purposes
        outerBefore = NaN;
        outerAfter = NaN;
        innerBefore = NaN;
        innerAfter = NaN;
        prevOuterAfter = Inf;
    end
    
    methods
        % Simple constructor
        function apmNums = APMNums()
        end
        
        % Save all properties of object into structure so that it can save
        % easily
        function struct = saveobj(obj)
            fields = fieldnames(obj);
            for fI = 1:size(fields,1);
                struct.(fields{fI}) = obj.(fields{fI});
            end
        end
        
    end
    
    methods (Static)
        % Load properties from saved structure
        % Allows for objects to be loaded from Matlab file
        function obj = loadobj(struct)
            obj = APMNums();
            structFields = fieldnames(struct);
            fields = fieldnames(obj);
            for fI = 1:size(fields,1);
                if(sum(strcmp(fields{fI}, structFields)) > 0)
                    obj.(fields{fI}) = struct.(fields{fI});
                else
                    warning('APMNums:LoadError', 'Could not find %s field in saved structure\n', fields{fI});
                end
            end
        end
        
        % Return struct from object properties
        function struct = getstruct(obj)
            fields = fieldnames(obj);
            for fI = 1:size(fields,1);
                struct.(fields{fI}) = obj.(fields{fI});
            end
        end
    end
end
