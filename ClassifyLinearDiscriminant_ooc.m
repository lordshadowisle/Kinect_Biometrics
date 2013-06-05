%% Classify using Discriminant Analysis (out-of-class)
% A variant of the classifier, comparing the relative scores.
% The evaluation is performed once per label, treating that label as an OOC
% label.

function [confusionTable] = ClassifyLinearDiscriminant_ooc(data, labels, trainSplit)
    % Perform CV-split evaluation
    confusionTable = zeros(max(labels), max(labels));
    %determine the OOC label
    for oocLabel = 1 : max(labels)
        for k = 1 : size(trainSplit,2)
            tempTrainSplit = trainSplit(:,k);
            learner = ClassificationDiscriminant.fit(data(tempTrainSplit,:), labels(tempTrainSplit));
            [oocThreshold] = SetThreshold(learner, data(tempTrainSplit,:), labels(tempTrainSplit));
            [predictedLabel, score] = predict(learner, data(~tempTrainSplit,:));
            actualLabel = labels(~tempTrainSplit);

            % reject OOC samples
            sortedScores = sort(score,2, 'descend');
            multiplier = sortedScores(:,1) ./ (sortedScores(:,2)+.01);
            OOCreject = multiplier < oocThreshold;
            predictedLabel(OOCreject) = oocLabel;
            % compute confusion tables
            for i = 1 : max(labels)
                for j = 1 : max(labels)
                    confusionTable(i,j)=confusionTable(i,j) + sum((predictedLabel == j) .* (actualLabel==i));
                end
            end
        end
    end
end

%% Set Threshold for OOC detection
% Find a multiplier threshold that maximizes the chance of correct
% classification
function [oocThreshold, correctMultiplier, wrongMultiplier] = SetThreshold(learner, data, labels)
    costFactor = 0.49;
    [predictedLabel, score] = predict(learner, data);
    isCorrect = predictedLabel == labels;
    % compute general multipliers
    sortedScores = sort(score,2, 'descend');
    multiplier = sortedScores(:,1) ./ (sortedScores(:,2)+.01);
    % compute multiplier for correct cases
    correctMultiplier = multiplier(isCorrect);
    % compute multiplier for wrong cases
    wrongMultiplier = sort(multiplier(~isCorrect),'descend');
    
    % Compute cost at each multiplier level
    cost = zeros(1,length(wrongMultiplier));
    for i = 1 : length(cost)
        cost(i) = sum(correctMultiplier < wrongMultiplier(i)) + costFactor*i;
    end
    [~,oocThreshold] = min(cost);
    oocThreshold = wrongMultiplier(oocThreshold);
end