%% Clustering Verification
%
% Verify number of clusters seen in data.
function [allData, allLabels, idx, confusionTable] = ClusteringVerification(numClusters)
    clc
    close all

    allData = [];
    allLabels = [];
    
    %% Loading routines-> some processing also
    for switchCode = [1,2, 4]
        switch switchCode
            % The following are the new cases collected by the special method
            case 1
                fileName = 'DCI_twl2';
            case 2
                fileName = 'DCI_tcg';
            case 3
                fileName = 'DCI_wly'; %--> as expected, mostly unusable data
            case 4
                fileName = 'DCI_txm';
        end
        [caseData, caseLabels] = LoadData(fileName, switchCode);
        allData = [allData; caseData];
        allLabels = [allLabels; caseLabels];
    end
    
    allData_std = zscore(allData);
    %% Analysis of descriptors wrt class
    [idx] = kmeans(allData_std, numClusters, 'Distance', 'correlation', 'replicates', 5);
    idx = [idx, (2*allLabels(:,2)-2 + allLabels(:,1))];
    % Generate confusion table
    confusionTable = zeros(8, numClusters);
    for i = 1 : 8
        isClass = idx(idx(:,2) == i);
        for j = 1 : numClusters
            confusionTable(i,j) = sum(isClass == j);
        end
    end
    
    % clear the WLY samples if prompted to
    if numClusters == 6
        confusionTable(5:6,:) = [];
    end
end

%% Function for loading data and parsing valid cases
function [caseData, caseLabels] = LoadData(fileName, personIndex)
    load(['Descriptors\' fileName '_descriptors'], 'datasetDescriptors');
    caseData = datasetDescriptors.Descriptors;
    caseLabels= datasetDescriptors.Left_Right;
    isUsable = (datasetDescriptors.isUsable) > 0;
    caseData = caseData(isUsable,:);
    caseLabels = caseLabels(isUsable);
    caseLabels(:,2) = personIndex;
    
    if 1
        adjustedDimensions = [2,11:18]; 
        adjustedDimensions = [5*adjustedDimensions, 5*adjustedDimensions-1, 5*adjustedDimensions-2, 5*adjustedDimensions-3, 5*adjustedDimensions-4];
        adjustedDimensions = sort(adjustedDimensions);
        caseData = caseData(:, adjustedDimensions);
    end
end