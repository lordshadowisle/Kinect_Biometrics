%% Classify using Absolute Large Margin Nearest Neighbor Distances
% The data space is reprojected using large margins. For each test sample,
% compare the distance to the nearest neighbor, and the distance of the 
% nearest neighbor to its nearest neighbor.
%
% Note: Uses a slightly different coding structure for efficiency.
% Note: Needs a distance factor chooser!!
% 0706: Added output for ooc detection statistics 
% 2706: Applied Large Margin projection
% 0107: Modified to use absolute distances

function [confusionTable, oocDetectionRate] = ClassifyLMNNDabs_ooc(data, labels, trainSplit)
    scaleFactor = .3; 
    % Perform CV-split evaluation
    confusionTable = zeros(max(labels), max(labels));
    oocDetectionRate = zeros(4,max(labels));        %tp, fp, tn, fn
    
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
            [M,L] = lmnn2(data(tempTrainSplit,:)', labels(tempTrainSplit)', 'quiet', 1, 'maxiter', 1000); M = M';
            neighborDistances = NearestNeighborDistances(trainData*M, trainLabels);
            distanceFactor = ComputeDistanceFactor(trainData*M, trainLabels, neighborDistances);

            % Evaluation Task
            tempTrainSplit = trainSplit(:,k);
            actualLabel = labels(~tempTrainSplit);
            [nearestNeighbor, nearestDistance] = knnsearch(trainData*M, data(~tempTrainSplit,:)*M);
            %reject OOC samples
            OOCreject = nearestDistance - neighborDistances(nearestNeighbor) > distanceFactor;
            
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


% Compute Nearest Neighbor for Training Data
% For each class, compute the distance of each member to the nearest
% neighbor.
function [neighborDistances] = NearestNeighborDistances(data, labels)
    neighborDistances = [];
    % for each class
    for i = 1 : max(labels)
        if sum(labels == i) > 0
            subData = data(labels == i, :);
            distanceMatrix = squareform(pdist(subData));
            distanceMatrix = distanceMatrix + (max(max(distanceMatrix))) * eye(size(distanceMatrix,1));
            neighborDistances = [neighborDistances, min(distanceMatrix)];
        end
    end
    
    neighborDistances = neighborDistances';
end

% Compute Distance Factor for Outlier Rejection
% compute nearest outlier distance, and obtain a safety margin
function [distanceFactor] = ComputeDistanceFactor(data, labels, neighborDistances)
   % Perform classification on dataset(?)
   oocDistances = [];
   for i = 1 : max(labels)
       if sum(labels == i) > 0
           inData = data(labels == i,:);
           outData = data(labels ~=i,:);
           [~,distance] = knnsearch(outData, inData);
           oocDistances = [oocDistances; distance];
       end
   end
   distanceFactor = 0.5*(median(oocDistances - neighborDistances));
   %pause
end
