%% Classify using Large-Margin Nearest Neighbors
% Assign to the class, using Large-Margin K neighbors
% We can choose from 2 different lmnn implementations.
% i) Matlab toolbox for dimensionality reduction
% ii) Kilian Weinberger 2009 (faster, default)

function [confusionTable] = ClassifyLargeMarginNN(data, labels, trainSplit, K)
    % Perform CV-split evaluation
    confusionTable = zeros(max(labels), max(labels));
    for k = 1 : size(trainSplit,2)
        %k
        tempTrainSplit = trainSplit(:,k);
        % apply lmnn to obtain projection 
        %[M, L, Y, C] = lmnn(data(tempTrainSplit,:), labels(tempTrainSplit));
        [M,L] = lmnn2(data(tempTrainSplit,:)', labels(tempTrainSplit)', 'quiet', 1); M = M';
        
        learner = ClassificationKNN.fit(data(tempTrainSplit,:)*M, labels(tempTrainSplit), 'NumNeighbors', K);
        predictedLabel = predict(learner, data(~tempTrainSplit,:)*M);
        actualLabel = labels(~tempTrainSplit);

        for i = 1 : max(labels)
            for j = 1 : max(labels)
                confusionTable(i,j)=confusionTable(i,j) + sum((predictedLabel == j) .* (actualLabel==i));
            end
        end
    end
end