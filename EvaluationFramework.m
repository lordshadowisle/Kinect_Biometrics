%% Evaluation Framework for Palm Biometrics
% 
% This shell program evaluates the algorithms for palm biometric
% recognition. An option is included to repeat the cross-validation using
% different random seeds.
% 04/06: Added options for PCA and PCA dimensionality reduction preprocessing
% 05/06: Added options for different data settings
% 25/06: Generalized loadData for different datasets
% 26/06: LOOCV functionality
%
% There are two evaluation tasks for the palm biometrics
% i) Simple user verification
% ii) Novel user verification
%
%% INPUTS AND OUTPUTS:
% switchCode: [1-5], selects a learning method.
% numTrials: [integer], number of trials to run; if set to 0, uses LOOCV.
% varargin(3): [0 or 1], to use PCA transformation
% varargin(4): [0~1, float], to choose the percentage variation to retain using PCA
%%%%
% evaluationResult: A confusion table, the third dimension is the number of trials.
% evaluationMetrics: A 1*3 vector, recording the accuracy, micro-averaged,
% and macro-averaged F-measure.
% caseData: A placeholder for future output variables.
% caseLabels: A placeholder for future output variables.

function [evaluationResult, evaluationMetrics, caseData, caseLabels] = EvaluationFramework(switchCode, numTrials, varargin)
    % Important Settings
    learningMethod = switchCode;% Sets the learning method to be evaluated.
    testSplit = 10;             % Number of CV splits used. Needs to be manually set here.
    usePCA = 0;                 % PCA Switch; will be loaded via varagin.
    loadSetting = 0;            % READ LoadData function header for info. Needs to be manually set here.
    useLOOCV = 0;
    
    % Determine if LOOCV is used by checking numTrials
    if numTrials < 1
        numTrials = 1;
        useLOOCV = 1;
    end
    % Argument check to enable PCA preprocessing
    % If third argument of function exists, it is usePCA switch.
    % If fourth argument exists, it is the % of PCA variance to keep.
    if ~isempty(varargin)
        usePCA = 1;
        if nargin >= 4
            dimReducePCA = varargin{2};
            if ((dimReducePCA < 0) || (dimReducePCA > 1))
                dimReducePCA = 0.95;
            end
        end
    end
    
    %% Preprocessing
    % Load data and labels from memory
    [caseData, caseLabels] = LoadData(loadSetting);
    evaluationResult = zeros(caseLabels(end,3), caseLabels(end,3),numTrials);
    processedData = caseData;
    processedLabels = caseLabels;
    
    %% PCA Processing
    % Apply PCA dimensionality reduction
    if usePCA == 1
        %[~, processedData, latent] = princomp(caseData);
        [processedData, latent] = pca(caseData);
        % If dimensionality reduction is on, filter the least descriptive components
        if exist('dimReducePCA')
            dimReducePCAIndex = find((cumsum(latent) / sum(latent)) > dimReducePCA, 1 );
            processedData = processedData(:, 1:dimReducePCAIndex);
        end
    end
    
    %% Main Training and Testing Loop
    for trialIdx = 1 : numTrials 
        % Generate training and test proportions
        if ~useLOOCV
            [trainTestMembership] = GenerateCVTrainTestMembership(caseLabels, testSplit, trialIdx);
        else
            [trainTestMembership] = GenerateLOOCVTrainTestMembership(caseLabels);
        end

        %% Learning and Evaluation
        switch learningMethod
            case 1
                %Multi-class random forest
                confusionTable = ClassifyRandomForest(processedData, processedLabels(:,3), trainTestMembership);
            case 2
                %K-Nearest Neighbor 
                confusionTable = ClassifyKNN(processedData, processedLabels(:,3), trainTestMembership,3);
            case 3
                %Bayesian classifier
                confusionTable = ClassifyBayes(processedData, processedLabels(:,3), trainTestMembership);
            case 4
                %Linear Discriminant
                confusionTable = ClassifyLinearDiscriminant(processedData, processedLabels(:,3), trainTestMembership);
            case 5
                %Decision Tree
                confusionTable = ClassifyDecisionTree(processedData, processedLabels(:,3), trainTestMembership);
            case 6
                %Multi-class SVM (using SVMLight)
                confusionTable = ClassifySVM(processedData, processedLabels(:,3), trainTestMembership);
            case 6.1
                %Multi-class SMV (using libsvm)
                confusionTable = ClassifySVM2(processedData, processedLabels(:,3), trainTestMembership);
            case 7
                %Random subspace method (26/6)
                confusionTable = ClassifyRandomSubspace(processedData, processedLabels(:,3), trainTestMembership);
            case 8
                % Large Margin nearest neighbor
                confusionTable = ClassifyLargeMarginNN(processedData, processedLabels(:,3), trainTestMembership, 3);
        end
        evaluationResult(:,:,trialIdx) = confusionTable;
            

    end
        evaluationMetrics = ComputeEvaluationMetrics(evaluationResult);
end

%% Load Data
% loadSetting determines the setting at which the data is loaded.
% 0 - Full dataset, no discarded dimensions
% 1 - Non-adjusted dimensions discarded
% 2 - Also discard thumb and little finger
% 25/06: Modified to compact data and class matrices
function [caseData, caseLabels] = LoadData(loadSetting)
    caseData = [];
    caseLabels = [];
    labelIdx = 1;
    %for switchCode = [1,2,3];
    for switchCode = [5,6,7]
        switch switchCode
            % The following are the new cases collected by the special method
            case 1
                fileName = 'DCI_twl';
            case 2
                fileName = 'DCI_tcg';
            case 3
                fileName = 'DCI_txm';        
            case 4
                fileName = 'DCI_wly'; %--> as expected, mostly unusable data
            case 5
                fileName = 'DCI_twl2';
            case 6
                fileName = 'DCI_tcg2';
            case 7
                fileName = 'DCI_txm2';
        end
        load(['..\HandSegmentation\Descriptors\' fileName '_descriptors'], 'datasetDescriptors');
        tempCaseData = datasetDescriptors.Descriptors;
        tempCaseLabels= datasetDescriptors.Left_Right;
        isUsable = (datasetDescriptors.isUsable) > 0;
        caseData = [caseData;tempCaseData(isUsable,:)];
        tempCaseLabels = tempCaseLabels(isUsable);
        tempCaseLabels(:,2) = switchCode;
        %tempCaseLabels(:,3) = 2*(tempCaseLabels(:,2)-1)+ tempCaseLabels(:,1);  % old code, depreciated.
        tempCaseLabels(:,3) = (tempCaseLabels(:,1) > mean(tempCaseLabels(:,1))) + labelIdx;
        labelIdx = labelIdx + 2;
        caseLabels =[caseLabels; tempCaseLabels];
    end
    
    if loadSetting >= 1
        adjustedDimensions = [2,11:18]; 
        adjustedDimensions = [5*adjustedDimensions, 5*adjustedDimensions-1, 5*adjustedDimensions-2, 5*adjustedDimensions-3, 5*adjustedDimensions-4];
        adjustedDimensions = sort(adjustedDimensions);
        caseData = caseData(:, adjustedDimensions);
        %caseData(:,adjustedDimensions) = [];
        % Use only 3 middle fingers
        if loadSetting == 2
            adjustedDimensions = [2:5:size(caseData,2), 3:5:size(caseData,2), 4:5:size(caseData,2)];
            adjustedDimensions = sort(adjustedDimensions);
            caseData = caseData(:, adjustedDimensions);
        end
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
% randSeed controls the rng seed, allows for repeatable trials.
function [trainTestMembership] = GenerateCVTrainTestMembership(caseLabels, testSplit, randSeed)
    s = RandStream('mcg16807', 'Seed', randSeed);
    s = RandStream.setGlobalStream(s);
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

%% Generate LOOCV TrainTestMemberships
% trainTestMembership is a n*split vector, 1 represents train, 0 test.
% In actuality, LOOCV can be computed more directly, but this structure
% here is to maintain compatibility with old code.
function [trainTestMembership] = GenerateLOOCVTrainTestMembership(caseLabels)
    trainTestMembership = ~eye(size(caseLabels,1));
end

%% Compute Evaluation Metrics
% Outputs: Accuracy, Micro-averaged F-measure, Macro-averaged F-measure
function [evaluationMetrics] = ComputeEvaluationMetrics(confusionTable)
    evaluationMetrics = zeros(1,3);
    compactTable = sum(confusionTable,3);
    
    % Compute micro-averaged F-measure
    pi_n = 0;
    pi_d = 0;
    rho_n = 0;
    rho_d = 0;

    for i = 1 : size(compactTable,1)
        pi_n = pi_n + compactTable(i,i);
        pi_d = pi_d + sum(compactTable(i,i) + sum(compactTable(:,i)) - compactTable(i,i));
        rho_n = rho_n + compactTable(i,i);
        rho_d = rho_d + sum(compactTable(i,i) + sum(compactTable(i,:)) - compactTable(i,i));
    end
    pi_ = pi_n / pi_d;
    rho_ = rho_n / rho_d;
    Fmicro = 2 * pi_ * rho_ / (pi_ + rho_);

    % Compute macro-averaged F-measure
    F = 0;
    for i = 1 : size(compactTable,1)
        pi_i = compactTable(i,i) / sum(compactTable(:,i));
        rho_i = compactTable(i,i) / sum(compactTable(i,:));
        % check for edge cases
        if sum(compactTable(:,i)) == 0
            pi_i = 1;
        end    
        F = F + 2 * pi_i * rho_i / (pi_i + rho_i);
    end
    Fmacro = F / size(compactTable,1);
    
    % Export Fmicro and Fmacro
    evaluationMetrics(1) = sum(diag(compactTable)) / sum(sum(compactTable));
    evaluationMetrics(2) = Fmicro;
    evaluationMetrics(3) = Fmacro;
end