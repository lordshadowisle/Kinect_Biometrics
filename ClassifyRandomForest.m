%% Verson 2 of Classify using Random Forest
% Added functionality to perform evaluation using CV splits

function [confusionTable] = ClassifyRandomForest(data, labels, trainSplit)
    if size(trainSplit,2) == 1
        ens = fitensemble(data(trainSplit,:), labels(trainSplit), 'AdaBoostM2', 100, 'Tree');
        %ens = fitensemble(data(trainSplit,:), labels(trainSplit), 'Subspace', 100, 'KNN');
        predictedLabel = predict(ens, data(~trainSplit,:));
        actualLabel = labels(~trainSplit);

        confusionTable = zeros(max(labels), max(labels));
        for i = 1 : max(labels)
            for j = 1 : max(labels)
                confusionTable(i,j)=sum((predictedLabel == j) .* (actualLabel==i));
            end
        end
    else
        % Perform CV-split evaluation
        confusionTable = zeros(max(labels), max(labels));
        for k = 1 : size(trainSplit,2)
            tempTrainSplit = trainSplit(:,k);
            ens = fitensemble(data(tempTrainSplit,:), labels(tempTrainSplit), 'AdaBoostM2', 100, 'Tree');
            %ens = fitensemble(data(tempTrainSplit,:), labels(tempTrainSplit), 'Subspace', 100, 'KNN');
            predictedLabel = predict(ens, data(~tempTrainSplit,:));
            actualLabel = labels(~tempTrainSplit);

            for i = 1 : max(labels)
                for j = 1 : max(labels)
                    confusionTable(i,j)=confusionTable(i,j) + sum((predictedLabel == j) .* (actualLabel==i));
                end
            end
        end
    end
end