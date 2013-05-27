%%  Shell Experiment Processing
%
% This shell processes all the data for the data captured during the formal
% experiment data.
% Modified to function on L/R palms and the revised RectCoords.
% Modified for consistency of code

function Shell_Experiment_Processing()

    clear
    clc
    close all

    isDisplay = 1;
    switch 1
        % The following are the new cases collected by the special method
        case 1
            fileName = 'DCI_twl2';
        case 2
            fileName = 'DCI_tcg';
        case 3
            fileName = 'DCI_wly';
        case 4
            fileName = 'DCI_txm';
    end
    load(['..\Biometrics\Data\' fileName]);

    if isDisplay
        figHandle = figure(1);
    end
                
    % Do leftOrRight = 1;        %1 == Right, 2 == Left
    for leftOrRight = 1:2
    %% Extract based on ROI
    for j = 1 : 3
        for i = 1 : 3
            if leftOrRight == 1
                pic = dataR(:,:,j,i);
                rectCoords = depthsR(i,:);
                handROI = ExtractROI(pic, rectCoords);
            else
                pic = dataL(:,:,j,i);
                rectCoords = depthsL(i,:);
                handROI = ExtractROI(pic, rectCoords);
            end
            if isDisplay
                subplot(3,3,i);
                imagesc(pic)
                r = rectangle('Position', rectCoords);
                set(r,'edgecolor','g');    
            % processed input
                subplot(3,3,i+3);
                imagesc(handROI);
                % segmented input
                subplot(3,3,i+6);
            end
            [a b, peaks] = SegmentROI(handROI);
            if isDisplay
                plot(a, b); hold on;
                plot([peaks(1) peaks(1)], [0, max(b)],'r');
                plot([peaks(2) peaks(2)], [0, max(b)],'g');
                plot([peaks(3) peaks(3)], [0, max(b)],'k');
                %plot([peaks(4) peaks(4)], [0, max(b)],'k');
                hold off;
                if leftOrRight == 1
                    set(figHandle, 'name', ['Right, Sample ' num2str(j)]);
                else
                    set(figHandle, 'name', ['Left, Sample ' num2str(j)]);
                end
                imagesc(handROI .* uint16((handROI < peaks(3))));        
            end
        end
        
        if isDisplay
            pause(0.5)
        end
    end
    end
end

%% Compute an extended ROI to extract the hand region.
function [handROI] = ExtractROI(pic, ROI)
    handROI = pic(ROI(2):(ROI(2)+ROI(4)), ROI(1):(ROI(1)+ROI(3)));
    % determine most likely distributions
end

function [fi, xi, mu] = SegmentROI(handROI)
   temp = double(handROI(:));
   [xi, fi] = ksdensity(temp, linspace(400, 2500, 200));
   temp(temp > 1800) = [];              % prevent foreground confusion
   poorSeed = (temp < mean(temp))+1;
   obj = gmdistribution.fit(temp,2, 'Start', poorSeed);
   mu = obj.mu;
   sigma = squeeze(obj.Sigma);
   probComp = obj.PComponents;
   % Perform calculation to determine best intersection point between peaks
   a = -sigma(2) + sigma(1);
   b = 2*sigma(2)*mu(1) - 2*sigma(1)*mu(2);
   c = 2*sigma(1)*sigma(2)*log( ( probComp(2)*sqrt(sigma(2)) ) / ( probComp(1)*sqrt(sigma(1)) ) ) - sigma(2)*mu(1)*mu(1) + sigma(1)*mu(2)*mu(2);
   %mu(3) = sigma(2)*(mu(1) - (sigma(1)*mu(2)/sigma(2)))/(sigma(2)-sigma(1)) - log; 
   mu(3) = (-b + sqrt(b^2 - 4 * a * c)) / (2*a);
   %mu(4) = (-b - sqrt(b^2 - 4 * a * c)) / (2*a);
end