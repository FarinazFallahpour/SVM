% Farinaz Fallahpour
% Date: 2012 
% https://github.com/FarinazFallahpour

function [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = one_RbfSVC(Samples,Gamma, nu)
% USAGES: 
%    [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = one_RbfSVC(Samples)
%    [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = one_RbfSVC(Samples, Gamma)
%    [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = one_RbfSVC(Samples, Gamma, nu)
%
% DESCRIPTION: 
%   Construct a non-linear SVM classifier with a radial based kernel, or Guassian kernel, 
%     from the training Samples and Labels
%
% INPUTS:
%   Samples: all the training patterns. (a row of column vectors)
%   Lables: the corresponding class labels for the training patterns in Samples, (a row vector)
%   Gamma: parameters of the radial based kernel, which has the form
%            of (exp(-Gamma*|X(:,i)-X(:,j)|^2)). (default 1)
%   nu: the nu parameter of the one_svm (default 0.5) 
%
% OUTPUTS:
%    AlphaY    - Alpha * Y, where Alpha is the non-zero Lagrange Coefficients, and
%                    Y is the corresponding Labels, (L-1) x sum(nSV);
%                All the AlphaYs are organized as follows: (pretty fuzzy !)
%      				classifier between class i and j: coefficients with
%			  	         i are in AlphaY(j-1, start_Pos_of_i:(start_Pos_of_i+1)-1),
%				         j are in AlphaY(i, start_Pos_of_j:(start_Pos_of_j+1)-1)
%    SVs       - Support Vectors. (Sample corresponding the non-zero Alpha), M x sum(nSV),
%                All the SVs are stored in the format as follows:
%                 [SVs from Class 1, SVs from Class 2, ... SVs from Class L];
%    Bias      - Bias of all the 2-class classifier(s), 1 x L*(L-1)/2;
%    Parameters -  Output parameters used in training;
%    nSV       -  numbers of SVs in each class, 1xL;
%    nLabel    -  Labels of each class, 1xL.
%
% By Junshui Ma, and Yi Zhao (02/15/2002)
%

if (nargin < 1) & (nargin > 3)
   disp(' Incorrect number of input variables.\n');
   help RbfSVC;
   return;
else
   Labels = ones(1,size(Samples,2)); 
   if (nargin == 1)
       Parameters = [2 1 1 1 1 45 0.001 2];
       [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = SVMTrain(Samples, Labels, Parameters);
   elseif  (nargin == 2)
       Parameters = [2 1 Gamma 1 1 45 0.001 2];
       [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = SVMTrain(Samples, Labels, Parameters);
   elseif  (nargin == 3)
       Parameters = [2 1 Gamma 1 1 45 0.001 2 nu];
       [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = SVMTrain(Samples, Labels, Parameters);
   end
   nLabel = [-1 1];
   nSV = [0 length(AlphaY)];
end
