function [P, iterations] = randgamma(N, M, b, c)
% Released under BSD license:
% Copyright (c) 2011, Matthew Roughan
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
% 
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution
%     * Neither the name of the University of Adelaide nor the names
%       of its contributors may be used to endorse or promote products derived
%       from this software without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
%
% original file:      	rand_gamma.m, (c) Matthew Roughan, Mon Apr  4 2011
% created: 	Mon Apr  4 2011 
% author:  	Matthew Roughan 
% email:   	matthew.roughan@adelaide.edu.au
%
% Generate a Gamma random variable
%    "Statistical Distributions", Evans, Hastings, Peacock, 2nd Edition,
%    Wiley, 1993, p.75-81
%
% INPUTS: 
%       (N,M) = size of array of random variables to generate
%       b = scale parameter > 0
%       c = shape parameter > 0
%
% probability density function (pdf)
%    p(x) = (x/b)^(c-1) * exp(-x/b)  / (b * gamma(c))
%
% where gamma(c) is the gamma function (http://en.wikipedia.org/wiki/Gamma_function)
%
% Basic stats of the gamma distribution
%       mean = b c
%   variance = b^2 c
%
% generation method comes from 
%    http://en.wikipedia.org/wiki/Gamma_distribution#Generating_gamma-distributed_random_variables
%          notation:   theta = b 
%                          k = c
% the algorithm exploits several properties of the gamma
%   (1) Gamma(b,1) ~ Exp(b)   (an exponential random variable)
%   (2) sum_{i=1}^{n} Gamma(b,c_i) ~ Gamma(b,c)   where c=sum_{i=1}^{n} c_i
% plus acceptance-rejectance sampling
%

% basic tests on the inputs
if (nargin < 4)
  error('must enter all inputs');
end
if (~isfinite(b) | b<=0 )
  error('b > 0');
end
if (~isfinite(c) | c<=0 )
  error('c > 0');
end


% first break the problem into N-integral bits, which can be done as a sum of exponentials,
% and then a fractional part.

n = floor(c);
delta = c - n;


% initialize
v0 = exp(1) / (exp(1) + delta);

if (delta > 0)
  for i=1:N
    for j=1:M
      m = 1;
      not_end = 1;
      
      % acceptance rejection method
      while (not_end & m<1000)
	v = rand(1,3);
	if (v(1) <= v0)
	  xi_m = v(2)^(1/delta);
	  eta_m = v(3) * xi_m^(delta-1);
	else
	  xi_m = 1 - log(v(2));
	  eta_m = v(3) * exp(-xi_m);
	end
	m = m + 1;
	if (eta_m <= xi_m^(delta-1) * exp(-xi_m) )
	  not_end = 0;
	end
      end
      xi(i,j) = xi_m;
      iterations(i,j) = m;
    end
  end
else
  xi = zeros(N,M);
end

U = rand(N,M,n);
P = b*( xi - sum(log(U), 3) ); 




