%% Classify using Nearest Neighbor Distances (with Optimizer)
% For each test sample, compare the distance to the nearest neighbor, and 
% the distance of the nearest neighbor to its nearest neighbor.
% This algorithm has a prototype optimizer.
%
% Note: Uses a slightly different coding structure for efficiency.
% Note: Needs an extension to K-NN (ought to be simple)
% Note: Needs a scale factor chooser.

function [confusionTable] = ClassifyNNDopt_ooc(data, labels, trainSplit)
    scaleFactor = .5;
    % Perform CV-split evaluation
    confusionTable = zeros(max(labels), max(labels));
    %determine the OOC label
    for oocLabel = 1 : max(labels)
        tempOOCclass = labels ~= oocLabel;
        for k = 1 : size(trainSplit,2)
            % Learning Task
            tempTrainSplit = trainSplit(:,k);
            tempTrainSplit = (tempTrainSplit & tempOOCclass);
            trainData = data(tempTrainSplit,:);
            trainLabels = labels(tempTrainSplit);
            neighborDistances = NearestNeighborDistances(trainData, trainLabels);
            scaleFactor = OptimizeScaleFactor(trainData, trainLabels, neighborDistances);

            % Evaluation Task
            tempTrainSplit = trainSplit(:,k);
            actualLabel = labels(~tempTrainSplit);
            [nearestNeighbor, nearestDistance] = knnsearch(trainData, data(~tempTrainSplit,:));
            %reject OOC samples
            OOCreject = neighborDistances(nearestNeighbor) < scaleFactor * nearestDistance;
            
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
            inData = data(labels == i, :);
            distanceMatrix = squareform(pdist(inData));
            distanceMatrix = distanceMatrix + (max(max(distanceMatrix))) * eye(size(distanceMatrix,1));
            neighborDistances = [neighborDistances, min(distanceMatrix)];
        end
    end
    
    neighborDistances = neighborDistances';
end

% Optimize Scale Factor
% Select a scalefactor that reduces the cost of misclassification on the
% test data.
% Compute the median minimum distance between two non-class objects.
function [scaleFactor] = OptimizeScaleFactor(data, labels, neighborDistances)
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
   scaleFactor = median(neighborDistances ./ oocDistances);
end
