%% Classify using Naive Bayes Classification (out-of-class)


function [confusionTable] = ClassifyBayes_ooc(data, labels, trainSplit)
    % Perform CV-split evaluation
    confusionTable = zeros(max(labels), max(labels));
    
    for oocLabel = 1 : max(labels)
        tempOOCclass = labels ~= oocLabel;
        for k = 1 : size(trainSplit,2)
            % Learning Task
            tempTrainSplit = trainSplit(:,k);
            tempTrainSplit = (tempTrainSplit & tempOOCclass);
            learner = NaiveBayes.fit(data(tempTrainSplit,:), labels(tempTrainSplit), 'Distribution', 'kernel');
            [oocThreshold] = SetThreshold(learner, data(tempTrainSplit,:), labels(tempTrainSplit));

            % Evaluation Task
            tempTrainSplit = trainSplit(:,k);
            [score, predictedLabel] = posterior(learner, data(~tempTrainSplit,:));
            actualLabel = labels(~tempTrainSplit);
            % reject OOC samples
            sortedScores = sort(score,2, 'descend');
            multiplier = sortedScores(:,1) ./ (sortedScores(:,2)+.01);
            OOCreject = multiplier < oocThreshold;
            predictedLabel(OOCreject) = oocLabel;
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
    costFactor = 1;
    [score, predictedLabel] = posterior(learner, data);
    isCorrect = predictedLabel == labels;
    % compute general multipliers
    sortedScores = sort(score,2, 'descend');
    multiplier = sortedScores(:,1) ./ (sortedScores(:,2)+.01);
    % compute multiplier for correct cases
    correctMultiplier = multiplier(isCorrect);
    % compute multiplier for wrong cases
    wrongMultiplier = sort(multiplier(~isCorrect),'descend');
    
    % catch edge cases
    if isempty(wrongMultiplier)
        oocThreshold = min(correctMultiplier);
    else
        % Compute cost at each multiplier level
        cost = zeros(1,length(wrongMultiplier));
        for i = 1 : length(cost)
            cost(i) = sum(correctMultiplier < wrongMultiplier(i)) + costFactor*i;
        end
        [~,oocThreshold] = min(cost);
        oocThreshold = wrongMultiplier(oocThreshold);
    end
end