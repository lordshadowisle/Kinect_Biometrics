%% Classify using k Nearest Neighbor Distances
% For each test sample, compare the distance to the nearest neighbor, and 
% the distance of the nearest neighbor to its nearest neighbor.
% Note: Uses a slightly different coding structure for efficiency.
% Note: Extended to K-NN.
% Note: Needs a scale factor chooser.
% 09/06: Added output for ooc detection statistics

function [confusionTable, oocDetectionRate] = ClassifykNND_ooc(data, labels, trainSplit, K)
    scaleFactor = .5;
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
            neighborDistances = NearestNeighborDistances(trainData, trainLabels, K);            

            % Evaluation Task
            tempTrainSplit = trainSplit(:,k);
            actualLabel = labels(~tempTrainSplit);
            % a more complex classification procedure
            [predictedLabel, OOCreject] = classifyData(trainData, trainLabels, data(~tempTrainSplit,:), neighborDistances, scaleFactor, K);
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


% Compute K Nearest Neighbors for Training Data
% For each class, compute the average distance of each member to the K 
% nearest neighbors.
function [neighborDistances] = NearestNeighborDistances(data, labels, K)
    neighborDistances = [];
    % for each class
    for i = 1 : max(labels)
        if sum(labels == i) > 0
            subData = data(labels == i, :);
            distanceMatrix = squareform(pdist(subData));
            distanceMatrix = distanceMatrix + (max(max(distanceMatrix))) * eye(size(distanceMatrix,1));
            distanceMatrix = sort(distanceMatrix,1,'ascend');
            neighborDistances = [neighborDistances, mean(distanceMatrix(1:K,:),1)];
        end
    end
    
    neighborDistances = neighborDistances';
end

%  Assign Labels and Reject OOC 
function [predictedLabel, OOCreject] = classifyData(trainData, trainLabels, testData, neighborDistances, scaleFactor, K)
    quotients = zeros(size(testData,1),max(trainLabels));   % ratio of neighbor distance to this distance
    toLabelDistance = zeros(size(testData,1), max(trainLabels));
    for i = 1 : max(trainLabels)
        if sum(trainLabels == i) > 0
           subData = trainData(trainLabels == i, :);
           subND = neighborDistances(trainLabels ==i);
           [idx, nearestDistance] = knnsearch(subData, testData, 'K', K);
           quotients(:,i) = mean(subND(idx),2) ./ mean(nearestDistance,2);     % neighbordistance / this distance
           %quotients(:,i) = subND(idx) ./ nearestDistance;toLabelDistance(:,i) = nearestDistance;
           toLabelDistance(:,i) = mean(nearestDistance,2);
        else
           toLabelDistance(:,i) = inf;
        end
    end
    
    [~,predictedLabel] = min(toLabelDistance,[],2);
    OOCreject= zeros(size(predictedLabel));
    
    for j = 1 : length(predictedLabel)
       OOCreject(j) = quotients(j, predictedLabel(j));
    end
    OOCreject = OOCreject < scaleFactor;
end