%% Evaluation Framework for Palm Biometrics
% 
% This shell program evaluates the algorithms for palm biometric
% recognition. An option is included to repeat the cross-validation using
% different random seeds.
% 04/06: Added options for PCA and PCA dimensionality reduction preprocessing
%
% There are two evaluation tasks for the palm biometrics
% i) Simple user verification
% ii) Novel user verification

function [evaluationResult, evaluationMetrics, caseData, caseLabels] = EvaluationFramework(switchCode, numTrials, varargin)
    switchCode;
    learningMethod = switchCode;
    testSplit = 10;
    usePCA = 0;
    
    % Argument check to enable PCA preprocessing
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
    [caseData, caseLabels] = LoadData(1);
    evaluationResult = zeros(caseLabels(end,3), caseLabels(end,3),numTrials);
    processedData = caseData;
    processedLabels = caseLabels;
    
    %% PCA Processing
    % Apply PCA dimensionality reduction
    if usePCA == 1
        [~, processedData, latent] = princomp(caseData);
        % If dimensionality reduction is on, filter the least descriptive components
        if exist('dimReducePCA')
            dimReducePCAIndex = min(find((cumsum(latent) / sum(latent)) > dimReducePCA));
            processedData = processedData(:, 1:dimReducePCAIndex);
        end
    end
    
    %% Main Training and Testing Loop
    for trialIdx = 1 : numTrials
        % Generate training and test proportions (using CV)
        [trainTestMembership] = GenerateCVTrainTestMembership(caseLabels, testSplit, trialIdx);

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
        end
        evaluationResult(:,:,trialIdx) = confusionTable;
            

    end
        evaluationMetrics = ComputeEvaluationMetrics(evaluationResult);
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