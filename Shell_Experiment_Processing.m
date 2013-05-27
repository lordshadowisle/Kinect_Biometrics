%%  Shell Experiment Processing
%
% This shell processes all the data for the data captured during the formal
% experiment data.

function Shell_Experiment_Processing()

    clear
    clc
    close all

    %% Determine settings and load datasets
    isDisplay = 0;
    switch 5
        case 1
            fileName = 'palm_right'; %2, 3 invalid
        case 2
            fileName = 'palm_left'; %4,13 invalid
        case 3
            fileName = 'palm_right2';
        case 4
            fileName = 'palm_left2';
        case 5
            fileName = 'DCI_twl';
    end
    load(['..\Biometrics\Data\' fileName]);
    try
        load(['Labels\' fileName '_ROI']);
    catch
        disp('No Label File.');
        bw = 1;
    end
    
    %% Extract based on ROI
    for j = 1 : 15
        for i = 1 : 3
            % raw input
            subplot(3,3,i);
            pic = dataR(:,:,j,i);
            imagesc(pic);
            r = rectangle('Position', depths(i,:));
            set(r,'edgecolor','g');    
            % processed input
            handROI = ExtractROI(pic, depths(i,:));
            subplot(3,3,i+3);
            imagesc(handROI);
            % segmented input
            subplot(3,3,i+6);
            [a b, peaks] = SegmentROI(handROI);
            plot(a, b); hold on;
            plot([peaks(1) peaks(1)], [0, max(b)],'r');
            plot([peaks(2) peaks(2)], [0, max(b)],'g');
            plot([peaks(3) peaks(3)], [0, max(b)],'k');
            %plot([peaks(4) peaks(4)], [0, max(b)],'k');
            hold off;
            %text(2000,1e-3, [num2str(peaks)]);
            imagesc(handROI .* uint16((handROI < peaks(3))));
        end
        waitforbuttonpress;
    end
end

%% Compute an extended ROI to extract the hand region.
function [handROI] = ExtractROI(pic, ROI)
    handROI = pic(ROI(1):(ROI(1)+ROI(3)), ROI(2):(ROI(2)+ROI(4)));
    % determine most likely distributions
end

function [fi, xi, mu] = SegmentROI(handROI)
   temp = double(handROI(:));
   [xi, fi] = ksdensity(temp, linspace(400, 2500, 200));
   temp(temp > 2500) = [];
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