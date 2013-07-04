%% Evaluation Framework for Palm Biometrics (out-of-class)
% 
% This shell program evaluates the algorithms for palm biometric
% recognition. An option is included to repeat the cross-validation using
% different random seeds.
% This framework requires out-of-class detection.
% 04/06: Added options for PCA and PCA dimensionality reduction preprocessing
% 25/06: Generalized loadData for different datasets
% 25/06: LOOCV functionality
%
% There are two evaluation tasks for the palm biometrics
% i) Simple user verification
% ii) Novel user verification
% This performs the novel user verification task.
%
%% INPUTS AND OUTPUTS:
% switchCode: [1-7], selects a learning method.
% numTrials: [integer], number of trials to run
% varargin(3): [0 or 1], to use PCA transformation
% varargin(4): [0~1, float], to choose the percentage variation to retain using PCA
%%%%
% evaluationResult: A confusion table, the third dimension is the number of trials.
% evaluationMetrics: A 2*4 vector, recording the accuracy, micro-averaged,
% and macro-averaged F-measure, and the TPR, TNR, FNR, and P detection
% rate.
% caseData: A placeholder for future output variables.
% caseLabels: A placeholder for future output variables.

function [evaluationResult, evaluationMetrics, caseData, caseLabels] = EvaluationFramework_ooc(switchCode, numTrials, varargin)
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
    oocResult = zeros(4, caseLabels(end,3),numTrials);      % evaluation results for ooc detection
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
                [confusionTable, oocDetectionRate] = ClassifyRandomForest_ooc(processedData, processedLabels(:,3), trainTestMembership);
            case 2
                %K-Nearest Neighbor
                [confusionTable, oocDetectionRate] = ClassifyKNN_ooc(processedData, processedLabels(:,3), trainTestMembership,3);
            case 2.5
                %Large Margin K-Nearest Neighor
                [confusionTable, oocDetectionRate] = ClassifyLargeMarginKNN_ooc(processedData, processedLabels(:,3), trainTestMembership,3);
            case 3
                %Bayesian classifier
                [confusionTable, oocDetectionRate] = ClassifyBayes_ooc(processedData, processedLabels(:,3), trainTestMembership);
            case 4
                %Linear Discriminant 
                [confusionTable, oocDetectionRate] = ClassifyLinearDiscriminant_ooc(processedData, processedLabels(:,3), trainTestMembership);
            case 5
                %Decision Tree
                [confusionTable, oocDetectionRate] = ClassifyDecisionTree_ooc(processedData, processedLabels(:,3), trainTestMembership);
            case 6
                %NND
                [confusionTable, oocDetectionRate] = ClassifyNND_ooc(processedData, processedLabels(:,3), trainTestMembership);
            case 6.1
                %NND with Optimizier v.1 (Compare nearest in and out-of-class samples to decide scaleFctor.
                [confusionTable, oocDetectionRate] = ClassifyNNDopt_ooc(processedData, processedLabels(:,3), trainTestMembership);
            case 6.2
                %NND with Optimizer v.2 (still prototyping -> Generalizes neighbor scales to each class)
                [confusionTable, oocDetectionRate] = ClassifyNNDopt2_ooc(processedData, processedLabels(:,3), trainTestMembership);
            case 6.3
                %NND with Optimizer v.3 (Minimizes classification error to choose scaleFactor)
                [confusionTable, oocDetectionRate] = ClassifyNNDopt3_ooc(processedData, processedLabels(:,3), trainTestMembership);
            case 6.4
                % NND with absolute distances
                [confusionTable, oocDetectionRate] = ClassifyNNDabs_ooc(processedData, processedLabels(:,3), trainTestMembership);
            case 6.5
                % NND using abolute distance with cluster-specific distance radius selection
                [confusionTable, oocDetectionRate] = ClassifyNNDabs_ClusterDistanceRadius_ooc(processedData, processedLabels(:,3), trainTestMembership);  
            case 7
                %k-NND
                [confusionTable, oocDetectionRate] = ClassifykNND_ooc(processedData, processedLabels(:,3), trainTestMembership,3);
            case 8
                % Large margin NND
                [confusionTable, oocDetectionRate] = ClassifyLMNND_ooc(processedData, processedLabels(:,3), trainTestMembership);
            case 8.1
                % Large margin NND with Optimizer v.1 (Compare nearest in and out-of-class samples to decide scaleFctor.
                [confusionTable, oocDetectionRate] = ClassifyLMNNDopt_ooc(processedData, processedLabels(:,3), trainTestMembership);
            case 8.2
                % Large margin NND with Optimizer v.2 (still prototyping -> Generalizes neighbor scales to each class)
                [confusionTable, oocDetectionRate] = ClassifyLMNNDopt2_ooc(processedData, processedLabels(:,3), trainTestMembership);
            case 8.3
                % Large margin NND with Optimizer v.3 (Minimizes classification error to choose scaleFactor)
                [confusionTable, oocDetectionRate] = ClassifyLMNNDopt3_ooc(processedData, processedLabels(:,3), trainTestMembership);
            case 8.4
                % Large margin NND using absolute distance 
                [confusionTable, oocDetectionRate] = ClassifyLMNNDabs_ooc(processedData, processedLabels(:,3), trainTestMembership);
            case 8.5 
                % Large margin NND using abolute distance with cluster-specific distance radius selection
                [confusionTable, oocDetectionRate] = ClassifyLLNNDabs_ClusterDistanceRadius_ooc(processedData, processedLabels(:,3), trainTestMembership);
            case 9.4
                % Two stage model with linear discriminant classifier
                [confusionTable, oocDetectionRate] = ClassifyTwoStageLinearDiscriminant_ooc(processedData, processedLabels(:,3), trainTestMembership);
            case 9.6
                % Two stage model with SVM
                [confusionTable, oocDetectionRate] = ClassifyTwoStageSVM_ooc(processedData, processedLabels(:,3), trainTestMembership);
            otherwise
                %% EXPERIMENTAL
                %[confusionTable, oocDetectionRate] = ClassifyEXPERIMENTAL2(processedData, processedLabels(:,3), trainTestMembership);
                [confusionTable, oocDetectionRate] = ClassifySVM_ooc(processedData, processedLabels(:,3), trainTestMembership);
        end
        evaluationResult(:,:,trialIdx) = confusionTable;
        oocResult(:,:,trialIdx) = oocDetectionRate;
            

    end
        evaluationMetrics = ComputeEvaluationMetrics(evaluationResult, oocResult);
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
    %for switchCode = [1,2,3]
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
% Remains the same for a novel class problem.
% Added new row for ooc evaluation -> reverts to original if empty
% Evaluation Metrics: 
% (1,1:3) = accuracy, micro-averaged F-measure, macro-averaged F-measure
% (2,1:4) = TPR, TNR, FNR, P detection rate
function [evaluationMetrics] = ComputeEvaluationMetrics(confusionTable, oocDetectionRate)
    evaluationMetrics = zeros(2,3);
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
        if sum(compactTable(i,:)) == 0
            rho_i = 1;
        end
        F = F + 2 * pi_i * rho_i / (pi_i + rho_i);
    end
    Fmacro = F / size(compactTable,1);
    
    % Export Fmicro and Fmacro
    evaluationMetrics(1,1) = sum(diag(compactTable)) / sum(sum(compactTable));
    evaluationMetrics(1,2) = Fmicro;
    evaluationMetrics(1,3) = Fmacro;
    
    % Compute Metrics for oocDetection
    if any(oocDetectionRate)
        mergeOOC = sum(sum(oocDetectionRate, 3),2);     %TP, FP, TN, FN
        evaluationMetrics(2,1) = mergeOOC(1) / (mergeOOC(1) + mergeOOC(2)); %TPR
        evaluationMetrics(2,2) = mergeOOC(3) / (mergeOOC(3) + mergeOOC(4)); %TNR
        evaluationMetrics(2,3) = mergeOOC(4) / (mergeOOC(3) + mergeOOC(4)); %FNR
        evaluationMetrics(2,4) = mergeOOC(1) / (mergeOOC(1) + mergeOOC(4)); %P detect
    end
end