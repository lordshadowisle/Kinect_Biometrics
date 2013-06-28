%% Classify using Nearest Neighbor Distances (with Optimizer v.3)
% For each test sample, compare the distance to the nearest neighbor, and 
% the distance of the nearest neighbor to its nearest neighbor.
% This algorithm has a prototype optimizer; the optimizer determines a
% scale factor by minimizing the error on training set.
%
% 28/06: First implemented.

function [confusionTable, oocDetectionRate] = ClassifyNNDopt3_ooc(data, labels, trainSplit)
    % Perform CV-split evaluation
    confusionTable = zeros(max(labels), max(labels));
    oocDetectionRate = zeros(4,max(labels));        %tp, fp, tn, fn
    
    %determine the OOC label
    for oocLabel = 1 : max(labels)
        tempOOCclass = labels ~= oocLabel;
        for k = 1 : size(trainSplit,2)
            % Learning Task
            tempTrainSplit = trainSplit(:,k);
            tempTrainSplit = (tempTrainSplit & tempOOCclass);
            trainData = data(tempTrainSplit,:);
            trainLabels = labels(tempTrainSplit);
            [firstNeighborDistance, secondNeighborDistance] = NearestNeighborDistances2(trainData, trainLabels);
            scaleFactor = OptimizeScaleFactor(trainData, trainLabels, firstNeighborDistance, secondNeighborDistance);

            % Evaluation Task
            tempTrainSplit = trainSplit(:,k);
            actualLabel = labels(~tempTrainSplit);
            [nearestNeighbor, nearestDistance] = knnsearch(trainData, data(~tempTrainSplit,:));
            %reject OOC samples
            OOCreject = firstNeighborDistance(nearestNeighbor) < scaleFactor * nearestDistance;
            
            predictedLabel = trainLabels(nearestNeighbor);
            %[predictedLabel == actualLabel, oocReject, neighborDistances(nearestNeighbor), nearestDistance, neighborDistances(nearestNeighbor)./ nearestDistance ]
            % actually need some predictive mapping to select a better weight.
            predictedLabel(OOCreject) = oocLabel;
                     
            % compute confusion tables
            for i = 1 : max(labels)
                for j = 1 : max(labels)
                    confusionTable(i,j)=confusionTable(i,j) + sum((predictedLabel == j) .* (actualLabel==i));
                end
            end
            
            % compute outlier detection statistics
            % true ooc positive (when both predicted and actual labels are ooc)
            oocDetectionRate(1,oocLabel) = oocDetectionRate(1, oocLabel) + sum((predictedLabel == oocLabel) .* (actualLabel == oocLabel));
            % false ooc positive (when predicted is ooC but actual is not)
            oocDetectionRate(2,oocLabel) = oocDetectionRate(2, oocLabel) + sum((predictedLabel == oocLabel) .* (actualLabel ~= oocLabel)); 
            % true ooc negative (when predicted and actual are not ooc)
            oocDetectionRate(3,oocLabel) = oocDetectionRate(3, oocLabel) + sum((predictedLabel ~= oocLabel) .* (actualLabel ~= oocLabel));
            % false ooc negative (when predicted is not ooc but acutal is)
            oocDetectionRate(4,oocLabel) = oocDetectionRate(4, oocLabel) + sum((predictedLabel ~= oocLabel) .* (actualLabel == oocLabel));
        end
    end
end

%% Compute Nearest Neighbor and second nearest neighbor distances for training data
function [neighborDistances, secondNeighborDistances] = NearestNeighborDistances2(data, labels)
    neighborDistances = [];
    secondNeighborDistances = [];
    % for each class, compute the nearest and 2nd nearest neighbors 
    for i = 1 : max(labels)
        if sum(labels == i) > 0
            inData = data(labels == i, :);
            [~, distances] = knnsearch(inData, inData, 'K', 3);
            neighborDistances = [neighborDistances; distances(:,2)];
            secondNeighborDistances = [secondNeighborDistances; distances(:,3)];
        end
    end
end

% Optimize Scale Factor
% Select a scalefactor that reduces the cost of misclassification on the
% test data.
% For each data point, first determine if it is correctly classified or
% misclassified. Compute the multipliers for both correct and incorrect 
% samples, then find the threshold level which minimizes error.
function [scaleFactor] = OptimizeScaleFactor(data, labels, n1, n2)
   costFactor = 2;  % penalize wrong as costFactor right entries
   
   % Perform classification on dataset to determine nearest neighbor identity
   [idx, distances] = knnsearch(data, data,'K',2);
   idx = idx(:,2);   %first is itself, 2nd is nearest neighbor
   isRight = labels(idx) == labels;
   % for right cases, determine 2nd neighbor distances
   rightDist = [distances(isRight, 2), n2(idx(isRight))];
   % for wrong cases, determine distances
   [idx, wrongDist] = knnsearch(data, data(find(~isRight),:), 'K',2);
   wrongDist = [wrongDist(:,2), n1(idx(:,2))];
   rightMultiplier = rightDist(:,2)./rightDist(:,1);
   wrongMultiplier = wrongDist(:,2)./wrongDist(:,1);
   
   %% Search wrongMultiplier for error minimizing scaleFactor
   scaleFactorCost = zeros(length(wrongMultiplier),1);
   wrongMultiplier = sort(wrongMultiplier, 'descend');
   for i = 1 :  length(wrongMultiplier)
       scaleFactorCost(i) = sum(rightMultiplier < wrongMultiplier(i)) + costFactor * i;
   end
   [~,scaleFactorCost] = min(scaleFactorCost);
   scaleFactor = wrongMultiplier(scaleFactorCost);
end
