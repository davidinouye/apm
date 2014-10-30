function checkobjective( obj1, obj2, msg, type )
%CHECKOBJECTIVE Check to see if objective is decreasing or staying the same
%
% checkobjective(obj1, obj2, msg, type)

if(nargin < 4); type = 'equal'; end;

if(strncmp(type,'dec',3))
    % Assert objective has decreased
    increaseAmount = obj2 - obj1;
    if(increaseAmount > 0)
        fprintf(['ObjectiveError: ' msg ', %g <= %g, increaseAmount = %g\n'], obj1, obj2, increaseAmount);
    end
else
    % Assert nearly equal (i.e. within 5 epsilon)
    relativeDiff = abs((obj1 - obj2)/mean([obj1,obj2]));
    nearlyEqual = relativeDiff < 100*eps;
    if(~nearlyEqual)
        fprintf(['ObjectiveError: ' msg ', %g ~= %g, relativeDiff = %g\n'], obj1, obj2, relativeDiff);
    end
end

end