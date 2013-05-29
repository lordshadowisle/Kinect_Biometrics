%% Evaluation Framework for Palm Biometrics
% 
% This shell program evaluates the algorithms for palm biometric
% recognition.
%
% There are two evaluation tasks for the palm biometrics
% i) Simple user verification
% ii) Novel user verification

function [evaluationResult, caseData, caseLabels] = EvaluationFramework(switchCode)
    switchCode;
    learningMethod = 2;
    testSplit = 5;
    
    %% Preprocessing
    % Load data and labels from memory
    [caseData, caseLabels] = LoadData(1);
    % Generate training and test proportions (using CV)
    [trainTestMembership] = GenerateCVTrainTestMembership(caseLabels, testSplit);
    
    %% Learning and Evaluation
    switch learningMethod
        case 1
            %Multi-class random forest
            confusionTable = ClassifyRandomForest(caseData, caseLabels(:,3), trainTestMembership);
        case 2
            %Placeholder method
            [trainTestMembership] = GenerateCVTrainTestMembership(caseLabels, 5);
            confusionTable = ClassifyRandomForest(caseData, caseLabels(:,3), trainTestMembership);
    end
    evaluationResult = confusionTable;
    
end

%% Load Data
% loadSetting determines the setting at which the data is loaded.
function [caseData, caseLabels] = LoadData(loadSetting)
    caseData = [];
    caseLabels = [];
    for switchCode = [1,2,3]
        switch switchCode
            % The following are the new cases collected by the special method
            case 1
                fileName = 'DCI_twl2';
            case 2
                fileName = 'DCI_tcg';
            case 3
                fileName = 'DCI_txm';        
            case 4
                fileName = 'DCI_wly'; %--> as expected, mostly unusable data
        end
        load(['..\HandSegmentation\Descriptors\' fileName '_descriptors'], 'datasetDescriptors');
        tempCaseData = datasetDescriptors.Descriptors;
        tempCaseLabels= datasetDescriptors.Left_Right;
        isUsable = (datasetDescriptors.isUsable) > 0;
        caseData = [caseData;tempCaseData(isUsable,:)];
        tempCaseLabels = tempCaseLabels(isUsable);
        tempCaseLabels(:,2) = switchCode;
        tempCaseLabels(:,3) = 2*(tempCaseLabels(:,2)-1)+ tempCaseLabels(:,1);
        caseLabels =[caseLabels; tempCaseLabels];
    end
    
    if 1
        adjustedDimensions = [2,11:18]; 
        adjustedDimensions = [5*adjustedDimensions, 5*adjustedDimensions-1, 5*adjustedDimensions-2, 5*adjustedDimensions-3, 5*adjustedDimensions-4];
        adjustedDimensions = sort(adjustedDimensions);
        caseData = caseData(:, adjustedDimensions);
    end
    
    % standardize dimensions
    caseData = zscore(caseData);
end

%% Generate TrainTest Memberships
% testSplit determines the relative split between the training and test
% classes.
% trainTestMembership is a vector, 1 represents training, 0 test.
function [trainTestMembership] = GenerateTrainTestMembership(caseLabels, testSplit)
    trainTestMembership = ones(size(caseLabels, 1),1);
    % Generate and assign splits for each class
    for i = 1 : caseLabels(end,3)
        isClass = find(caseLabels(:,3) == i);
        randOrder = randperm(length(isClass));
        trainTestMembership(isClass(randOrder(1:ceil(length(isClass)/testSplit)))) = 0;
    end
    
    trainTestMembership = trainTestMembership > 0;
end

%% Generate CV TrainTest Memberships
% testSplit determines the number of CV folds. 
% trainTestMembership is a n*split vector, 1 represents train, 0 test.
function [trainTestMembership] = GenerateCVTrainTestMembership(caseLabels, testSplit)
    trainTestMembership = ones(size(caseLabels, 1),testSplit);

    % Generate and assign splits for each class
    for i = 1 : caseLabels(end,3)            
        isClass = find(caseLabels(:,3) == i);
        randOrder = randperm(length(isClass));
        itemsPerSplit = ceil(length(isClass) / testSplit);
        for j = 1 : testSplit
            %rotate initial split.
            trainTestMembership(isClass(randOrder(1:itemsPerSplit)),j) = 0;
            randOrder = circshift(randOrder,[1,itemsPerSplit]);
        end
    end
    trainTestMembership = trainTestMembership > 0;
end