%% Classify using Two-stage OOC Rejection using Support Vector Machines (out-of-class)
% The two-stage classification and OOC rejection system uses a LMNNd system
% to detect and reject outliers. The remaining samples are classified with
% traditional classifier, in this case SVM.

function [confusionTable, oocDetectionRate] = ClassifyTwoStageSVM_ooc(data, labels, trainSplit)
    % Perform CV-split evaluation
    confusionTable = zeros(max(labels), max(labels));
    oocDetectionRate = zeros(4,max(labels));        %tp, fp, tn, fn
    
    addpath('SVMLight\');
    
    oocDetectionRate = zeros(4,max(labels));        %tp, fp, tn, fn
    
    % Construct data structure compatible with SVMLight
    dataStruct=  struct;
    dataStruct.X = data';
    dataStruct.y = labels';
    
    % Perform CV-split evaluation
    confusionTable = zeros(max(labels), max(labels));
    %determine the OOC label
    for oocLabel = 1 : max(labels)
        tempOOCclass = labels ~= oocLabel;
        for k = 1 : size(trainSplit,2)
            [oocLabel, k]
            tempTrainSplit = trainSplit(:,k);
            tempTrainSplit = (tempTrainSplit & tempOOCclass);

            % Generate training data and model 
            trainData = dataStruct;
            trainData.X = dataStruct.X(:,tempTrainSplit);
            trainData.y = dataStruct.y(tempTrainSplit);
			[neighborDistances, secondNeighborDistances] = NearestNeighborDistances(trainData.X', trainData.y);
			distanceFactor = ComputeDistanceFactor(trainData.y, neighborDistances, secondNeighborDistances);
            svmlight_train(trainData);

            % Generate test data and evaluate
            tempTrainSplit = trainSplit(:,k);
            testData = dataStruct;
            testData.X = dataStruct.X(:,~tempTrainSplit);
            testData.y = dataStruct.y(~tempTrainSplit);
            [predictedLabel] = svmlight_test(testData);
            actualLabel = labels(~tempTrainSplit);
            
            [nearestNeighbor, nearestDistance] = knnsearch(trainData.X', testData.X');
            %reject OOC samples
            OOCreject = nearestDistance - neighborDistances(nearestNeighbor) > distanceFactor(trainData.y(nearestNeighbor));     %-> solved error
            predictedLabel(OOCreject) = oocLabel;

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
        
    % Remove temporary files
    ClearTemp();
    rmpath('SVMLight\');
end

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