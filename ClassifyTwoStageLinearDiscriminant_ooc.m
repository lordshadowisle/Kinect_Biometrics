%% Classify using Two-stage OOC Rejection using LDA
% The two-stage classification and OOC rejection system uses a LMNNd system
% to detect and reject outliers. The remaining samples are classified with
% traditional classifier, in this case LDA.
%
% 0307: First implemented

function [confusionTable, oocDetectionRate] = ClassifyTwoStageLinearDiscriminant_ooc(data, labels, trainSplit)
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
            learner = ClassificationDiscriminant.fit(data(tempTrainSplit,:), labels(tempTrainSplit));
            [neighborDistances, secondNeighborDistances] = NearestNeighborDistances(trainData*M, trainLabels);
            distanceFactor = ComputeDistanceFactor(trainLabels, neighborDistances, secondNeighborDistances);

            % Evaluation Task
            tempTrainSplit = trainSplit(:,k);
            actualLabel = labels(~tempTrainSplit);
            [nearestNeighbor, nearestDistance] = knnsearch(trainData*M, data(~tempTrainSplit,:)*M);
            %reject OOC samples
            OOCreject = nearestDistance - neighborDistances(nearestNeighbor) > distanceFactor(trainLabels(nearestNeighbor));     %-> solved error

            [predictedLabel] = predict(learner, data(~tempTrainSplit,:));
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
%% Compute Nearest Neighbor and second nearest neighbor distances for training data
function [neighborDistances, secondNeighborDistances] = NearestNeighborDistances(data, labels)
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

% Compute Label-specific Distance Factor for Outlier Rejection
% For each label, compute the difference between the second and first
% nearest neighbor distances. The distance factor is the maximum difference
% for the label.
function [distanceFactor] = ComputeDistanceFactor(labels, neighborDistances, secondNeighborDistances)
   distanceFactor = zeros(1,max(labels));
   for i = 1 : max(labels)
       if sum(labels == i) > 0
           localFactor = secondNeighborDistances(labels == i) - neighborDistances(labels == i);
           distanceFactor(i) = max(localFactor);
       else
           distanceFactor(i) = 0;
       end
   end
   distanceFactor = distanceFactor';
end
