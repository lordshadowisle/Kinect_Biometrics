%% Classify using Support Vector Machines
% Use libsvm implementation to asssign the classes to labels.

function [confusionTable] = ClassifySVM2(data, labels, trainSplit)
    addpath('libsvm\');

    % Perform CV-split evaluation
    confusionTable = zeros(max(labels), max(labels));
    for k = 1 : size(trainSplit,2)
        tempTrainSplit = trainSplit(:,k);
        learner = svmtrain_lb(labels(tempTrainSplit), data(tempTrainSplit,:), '-q');
        [predictedLabel]= svmpredict(labels(~tempTrainSplit), data(~tempTrainSplit,:), learner, '-q');
        actualLabel = labels(~tempTrainSplit);

        for i = 1 : max(labels)
            for j = 1 : max(labels)
                confusionTable(i,j)=confusionTable(i,j) + sum((predictedLabel == j) .* (actualLabel==i));
            end
        end
    end

    rmpath('libsvm\');
end