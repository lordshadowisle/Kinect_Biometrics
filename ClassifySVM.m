%% Classify using Support Vector Machines
% Use SVMLight implementation to asssign the classes to labels.

function [confusionTable] = ClassifySVM(data, labels, trainSplit)
    addpath('SVMLight\');
    
    % Construct data structure compatible with SVMLight
    dataStruct=  struct;
    dataStruct.X = data';
    dataStruct.y = labels';
    
    % Perform CV-split evaluation
    confusionTable = zeros(max(labels), max(labels));
    for k = 1 : size(trainSplit,2)
        tempTrainSplit = trainSplit(:,k);
        
        % Generate training data and model 
        trainData = dataStruct;
        trainData.X = dataStruct.X(:,tempTrainSplit);
        trainData.y = dataStruct.y(tempTrainSplit);
        svmlight_train(trainData);
        
        % Generate test data and evaluate
        testData = dataStruct;
        testData.X = dataStruct.X(:,~tempTrainSplit);
        testData.y = dataStruct.y(~tempTrainSplit);
        predictedLabel = svmlight_test(testData);
        actualLabel = labels(~tempTrainSplit);

        for i = 1 : max(labels)
            for j = 1 : max(labels)
                confusionTable(i,j)=confusionTable(i,j) + sum((predictedLabel == j) .* (actualLabel==i));
            end
        end
    end
    
    % Remove temporary files
    ClearTemp();
    rmpath('SVMLight\');
end