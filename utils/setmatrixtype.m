function X = setmatrixtype(X)
%SETMATRIXTYPE - Set the type of the matrix to be sparse or full depending
% on amount of sparsity
%
% X = setmatrixtype(X)

densityX = nnz(X)/(size(X,1)*size(X,2));
threshold = 0.05;
if(densityX > threshold)
    X = full(X);
else
    X = sparse(X);
end

end