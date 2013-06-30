%% Classify using Large Margin Nearest Neighbor Distances (with Optimizer v.3)
% The data space is reprojected using large margins. For each test sample,
% compare the distance to the nearest neighbor, and the distance of the 
% nearest neighbor to its nearest neighbor.
%
% Note: Uses a slightly different coding structure for efficiency.
% Note: Needs a scale factor chooser.
% 0706: Added output for ooc detection statistics 
% 2706: Applied Large Margin projection

function [confusionTable, oocDetectionRate] = ClassifyLMNNDopt3_ooc(data, labels, trainSplit)
    % Perform CV-split evaluation
    confusionTable = zeros(max(labels), max(labels));
    oocDetectionRate = zeros(4,max(labels));        %tp, fp, tn, f
    
    %determine the OOC label
    for oocLabel = 1 : max(labels)
        tempOOCclass = labels ~= oocLabel;
        for k = 1 : size(trainSplit,2)
            [oocLabel, k]
            % Learning Task
            tempTrainSplit = trainSplit(:,k);
            tempTrainSplit = (tempTrainSplit & tempOOCclass);
            trainData = data(tempTrainSplit,:);
            trainLabels = labels(tempTrainSplit);
            % Obtain Large Margin projection
            [M,L] = lmnn2(data(tempTrainSplit,:)', labels(tempTrainSplit)', 'quiet', 1); M = M';  
            
            [firstNeighborDistance, secondNeighborDistance] = NearestNeighborDistances2(trainData*M, trainLabels);
            scaleFactor = OptimizeScaleFactor(trainData*M, trainLabels, firstNeighborDistance, secondNeighborDistance);

            % Evaluation Task
            tempTrainSplit = trainSplit(:,k);
            actualLabel = labels(~tempTrainSplit);
            [nearestNeighbor, nearestDistance] = knnsearch(trainData*M, data(~tempTrainSplit,:)*M);
            %reject OOC samples
            OOCreject = firstNeighborDistance(nearestNeighbor) < scaleFactor * nearestDistance;
            
            predictedLabel = trainLabels(nearestNeighbor);
            %[predictedLabel == actualLabel, oocReject, neighborDistances(nearestNeighbor), nearestDistance, neighborDistances(nearestNeighbor)./ nearestDistance ]
            % actually need some predictive mapping to select a better
            % weight.
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
   if length(wrongMultiplier) > 0
       scaleFactorCost = zeros(length(wrongMultiplier),1);
       wrongMultiplier = sort(wrongMultiplier, 'descend');
       for i = 1 :  length(wrongMultiplier)
           scaleFactorCost(i) = sum(rightMultiplier < wrongMultiplier(i)) + costFactor * i;
       end
       [~,scaleFactorCost] = min(scaleFactorCost);
       scaleFactor = wrongMultiplier(scaleFactorCost);
   else
       %for cases where projected space has no misclassified items
       scaleFactor = min(rightMultiplier);
   end
end