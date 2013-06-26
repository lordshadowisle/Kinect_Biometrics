%% Classify using Random Subspace Method
% This implements a dumb variant of the random subspace method using only
% fixed, predefined parameters.
% -> Todo: Implement an adaptive learning scheme for random subspace.
% Added functionality to perform evaluation using CV splits
% Simplified form

function [confusionTable] = ClassifyRandomSubspace(data, labels, trainSplit)
    % Perform CV-split evaluation
    confusionTable = zeros(max(labels), max(labels));
    dimRandomSubspace = ceil(size(data,2) * 0.4);
    for k = 1 : size(trainSplit,2)
        tempTrainSplit = trainSplit(:,k);
        ens = fitensemble(data(tempTrainSplit,:), labels(tempTrainSplit), 'subspace', 100, 'KNN', 'NPredToSample',dimRandomSubspace);
        predictedLabel = predict(ens, data(~tempTrainSplit,:));
        actualLabel = labels(~tempTrainSplit);

        for i = 1 : max(labels)
            for j = 1 : max(labels)
                confusionTable(i,j)=confusionTable(i,j) + sum((predictedLabel == j) .* (actualLabel==i));
            end
        end
    end
end