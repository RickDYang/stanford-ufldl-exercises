function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for i = 1:numel(stack)
    stackgrad{i}.w = zeros(size(stack{i}.w));
    stackgrad{i}.b = zeros(size(stack{i}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

% forwardpropagation
layers = cell(numel(stack) + 1, 1);
layers{1}.a =  data;
for i = 1:numel(stack)
	layers{i+1}.a = sigmoid(bsxfun(@plus, stack{i}.w *layers{i}.a, stack{i}.b));
end
% end forwardprogation


% softmax
i = numel(stack) + 1;

fx = softmaxTheta * layers{i}.a;
fx = bsxfun(@minus,fx , max(fx ,[],1));

fx = exp(fx);
fx = bsxfun(@rdivide, fx, sum(fx));

softmaxThetaGrad =  - (groundTruth - fx) * layers{i}.a' / M + lambda * softmaxTheta;

i = numel(stack) + 1;
layers{i}.d =  - softmaxTheta' * (groundTruth - fx).* (layers{i}.a.*(1 - layers{i}.a));
% backpropagation
for i = numel(stack):-1:2
	layers{i}.d = (stack{i}.w' * layers{i+1}.d).* (layers{i}.a.*(1 - layers{i}.a));
end

for i = numel(stack):-1:1
	stackgrad{i}.w =  layers{i+1}.d *  layers{i}.a' / M  + stack{i}.w * lambda;
	stackgrad{i}.b = sum(layers{i+1}.d, 2) / M;
end
%

% cost
cost = - sum(sum(log(fx).*groundTruth)) / M + lambda * sum(sum(softmaxTheta .^2)) / 2;

for i = 1:numel(stack)
	cost = cost + lambda * sum(sum(stack{i}.w.^2)) / 2;
end





% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
