function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(featureNum, imageNum, poolRow, poolCol)
%     

numImages = size(convolvedFeatures, 2);
numFeatures = size(convolvedFeatures, 1);
convolvedDim = size(convolvedFeatures, 3);
numPools = floor(convolvedDim / poolDim);
pooledFeatures = zeros(numFeatures, numImages, numPools, numPools);

% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   numFeatures x numImages x (convolvedDim/poolDim) x (convolvedDim/poolDim) 
%   matrix pooledFeatures, such that
%   pooledFeatures(featureNum, imageNum, poolRow, poolCol) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region 
%   (see http://ufldl/wiki/index.php/Pooling )
%   
%   Use mean pooling here.
% -------------------- YOUR CODE HERE --------------------
for featureNum = 1:numFeatures
	for imageNum = 1:numImages
		pooledFeature = zeros(floor(convolvedDim / poolDim), floor(convolvedDim / poolDim));
		x = 1;
		for i=1:numPools
			y = 1;
			for j=1:numPools
				pooledFeature(i, j) = mean(mean(convolvedFeatures(featureNum, imageNum, x:x + poolDim - 1, y:y + poolDim - 1)));
				y = y + poolDim;
			end
			x = x + poolDim;
		end
		pooledFeatures(featureNum, imageNum, :, :) = pooledFeature;
	end
end

end

