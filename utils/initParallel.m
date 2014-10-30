function initParallel( nWorkers )
%INITPARALLEL Attempt to start a parallel pool of size nWorkers.
%
% initParallel(nWorkers)
if(nWorkers > 1)
    fprintf('Starting parallel pool...\n');
    
    % Try to create temp directory for storing the job files
    tempDir = tempname();
    mkdir(tempDir);
    
    % If before R2014a (i.e. 8.3.0), then cap at 12 workers because of restrictions
    if(verLessThan('matlab', '8.3.0'))
        targetNumWorkers = min(nWorkers, 12);
    else
        targetNumWorkers = nWorkers;
    end

    try
        c = parcluster();
        c.JobStorageLocation = tempDir;
        c.NumWorkers = targetNumWorkers;
        
        % Check current or empty parallel pool and create or reinitialize pool
        curPool = gcp('nocreate');
        if(isempty(curPool))
            curPool = parpool(c, targetNumWorkers); % Initialize with as many workers as possible up to requested number of workers
        elseif(curPool.NumWorkers < targetNumWorkers) 
            delete(curPool); % Delete old pool if still around
            curPool = parpool(c, targetNumWorkers);
        end
    catch err
        try
            % NOTE: Try deprecated parallel pool scheduler for older versions of MATLAB
            sched = findResource('scheduler', 'type', 'local');
            sched.DataLocation = tempDir;
            sched.ClusterSize = nWorkers;

            % Check current or empty parallel pool and create or reinitiate pool
            if(matlabpool('size') <= 0)
                matlabpool(sched, targetNumWorkers); % Initialize with as many workers as possible up to requested number of workers
            elseif(matlabpool('size') < targetNumWorkers) 
                matlabpool('close'); % Attempt to close pool if too small
                matlabpool(sched, targetNumWorkers); % Start new pool with enough workers
            end
        catch err2
            try
                matlabpool('open');
                fprintf('Successfully created a parallel pool using simple matlabpool(open) with %d workers\n', matlabpool('size'));
            catch err3
                fprintf('Error starting parallel pool of workers. Using only 1 process.  The errors for parpool, matlabpool and barebones matlabpool were the following:\n%s\n\n%s\n\n%s\n', err.message, err2.message, err3.message);
            end
        end
    end
end

end

