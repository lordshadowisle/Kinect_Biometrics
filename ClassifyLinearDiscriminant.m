%% Classify using Discriminant Analysis

function [confusionTable] = ClassifyLinearDiscriminant(data, labels, trainSplit)
    % Perform CV-split evaluation
    confusionTable = zeros(max(labels), max(labels));
    for k = 1 : size(trainSplit,2)
        tempTrainSplit = trainSplit(:,k);
        learner = ClassificationDiscriminant.fit(data(tempTrainSplit,:), labels(tempTrainSplit));
        predictedLabel = predict(learner, data(~tempTrainSplit,:));
        actualLabel = labels(~tempTrainSplit);

        for i = 1 : max(labels)
            for j = 1 : max(labels)
                confusionTable(i,j)=confusionTable(i,j) + sum((predictedLabel == j) .* (actualLabel==i));
            end
        end
    end
end