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
    testSplit = 0.2;
    
    %% Preprocessing
    % Load data and labels from memory
    [caseData, caseLabels] = LoadData(1);
    % Generate training and test proportions
    [trainTestMembership] = GenerateTrainTestMembership(caseLabels, testSplit);
    
    evaluationResult = 0;
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
        trainTestMembership(isClass(randOrder(1:ceil(length(isClass)/5)))) = 0;
    end
end

%% Generate CV TrainTest Memberships
function [trainTestMembershipCV] = GenerateCVTrainTestMembership(caseLabels, testSplit)
    trainTestMembershipCV = 0;
end