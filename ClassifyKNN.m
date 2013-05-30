%% Classify using K-Nearest Neighbors
% Assign to the class, using K neigbors

function [confusionTable] = ClassifyKNN(data, labels, trainSplit, K)
    % Perform CV-split evaluation
    confusionTable = zeros(max(labels), max(labels));
    for k = 1 : size(trainSplit,2)
        tempTrainSplit = trainSplit(:,k);
        %ens = fitensemble(data(tempTrainSplit,:), labels(tempTrainSplit), 'AdaBoostM2', 100, 'Tree');
        learner = ClassificationKNN.fit(data(tempTrainSplit,:), labels(tempTrainSplit), 'NumNeighbors', K);
        predictedLabel = predict(learner, data(~tempTrainSplit,:));
        actualLabel = labels(~tempTrainSplit);

        for i = 1 : max(labels)
            for j = 1 : max(labels)
                confusionTable(i,j)=confusionTable(i,j) + sum((predictedLabel == j) .* (actualLabel==i));
            end
        end
    end
end