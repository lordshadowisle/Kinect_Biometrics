%% Shell Process and Describe
%
% This shell program performs segmentation and descriptor computation for a
% given experiment dataset. It combines the codes from
% Shell_DescribeDataset and Shell_Experiment_Processing.

function [datasetStruct] = Shell_Process_Describe(switchCode, isDisplay)

    clc
    close all
    if ~exist('switchCode')
        isDisplay = 1;
        switchCode =1;
    end
    %% Determine settings and load datasets
    switch switchCode
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
    bw = 1;
    if isDisplay
        figHandle = figure(1);
    end
   
    % Initialize output variables--> refer to notes for meanings
    datasetDescriptors = [];
    datasetLeftOrRight = [];
    datasetHandpoly = [];
    datasetDepth = [];
    datasetReference =[];
   
    % Do leftOrRight = 1;        %1 == Right, 2 == Left
    for leftOrRight = 1:2
        %% Extract based on ROI
        for j = 1 : 51               %sample number
            for i = 1 : 3           %depth level
                if leftOrRight == 1
                    disp(['Processing Right, Sample ' num2str(j) ', Depth ' num2str(i)]);
                    imageD = dataR(:,:,j,i);
                    imageD = double(imageD);
                    rectCoords = depthsR(i,:);
                    handROI = ExtractROI(imageD, rectCoords);
                else
                    disp(['Processing Left, Sample ' num2str(j) ', Depth ' num2str(i) ]);
                    imageD = dataL(:,:,j,i);
                    imageD = double(imageD);
                    rectCoords = depthsL(i,:);
                    handROI = ExtractROI(imageD, rectCoords);
                end
                if isDisplay
                    subplot(3,3,i);
                    imagesc(imageD)
                    r = rectangle('Position', rectCoords);
                    set(r,'edgecolor','g');    
                % processed input
                    subplot(3,3,i+3);
                    imagesc(handROI);
                    % segmented input
                end
                [a, b, peaks] = SegmentROI(handROI);
                if isDisplay
%                     subplot(3,3,i+3);
%                     plot(a, b); hold on;
%                     plot([peaks(1) peaks(1)], [0, max(b)],'r');
%                     plot([peaks(2) peaks(2)], [0, max(b)],'g');
%                     plot([peaks(3) peaks(3)], [0, max(b)],'k');
%                     hold off;
                    if leftOrRight == 1
                        set(figHandle, 'name', ['Right, Sample ' num2str(j)]);
                    else
                        set(figHandle, 'name', ['Left, Sample ' num2str(j)]);
                    end
                    subplot(3,3,i+6);
                    imagesc(handROI .* (handROI < peaks(3)));        
                end
                palmD = zeros(480,640);
                %palmD(rectCoords(2):(rectCoords(2)+rectCoords(4)), rectCoords(1):(rectCoords(1)+rectCoords(3))) = handROI .* (handROI < peaks(3));            %revisions
                %GetPalmDescriptors(palmD, isDisplay);
                % - > actually, no reason not to use relative addressing.
                [palmDescriptors, handpoly] = GetPalmDescriptors(handROI .* handROI .* (handROI < peaks(3)), isDisplay);
                
                % Do some L/R reversal
                if ~isempty(palmDescriptors)
                    datasetDescriptors = [datasetDescriptors; reshape(palmDescriptors,1,90)];
                    datasetLeftOrRight = [datasetLeftOrRight; leftOrRight];
                    datasetHandpoly = [datasetHandpoly; reshape(handpoly, 1, 10);];
                    datasetDepth = [datasetDepth; i];
                    datasetReference = [datasetReference; leftOrRight, i, j];       % technically contains all data... but well.
                end
            end

            if isDisplay
                %pause(0.5)
                waitforbuttonpress
            end
        end
    end
    
    datasetStruct = struct('Descriptors', datasetDescriptors, 'Left_Right', datasetLeftOrRight, 'Handpoly', datasetHandpoly, 'Depth', datasetDepth, 'Reference', datasetReference);
end

%% Compute an extended ROI to extract the hand region.
function [handROI] = ExtractROI(pic, ROI)
    handROI = pic(ROI(2):(ROI(2)+ROI(4)), ROI(1):(ROI(1)+ROI(3)));
    % determine most likely distributions
end

function [fi, xi, mu] = SegmentROI(handROI)
   temp = double(handROI(:));
   [xi, fi] = ksdensity(temp, linspace(400, 2500, 200));        %->actually quite poor, may collapse into single smoothed kernel
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

%% Function to obtain the descriptors for a single palm, if valid.
%% Forked from Shell_DescribeDataset
function [palmDescriptors, handpoly] = GetPalmDescriptors(palmD, isDisplay)
    isNonRecoverable = 0;

    [valleyLoc, peakLoc, palmCent, roiD] = ValleyPeak_Robust3_function(palmD, 0);
    if length(valleyLoc) < 4 || length(peakLoc) < 5
        %disp('Potential non-recoverable situation.');
        isNonRecoverable = 1;
    end
    
    if isDisplay
        hold off;
        imagesc(palmD); hold on;
        plot(peakLoc(:,2), peakLoc(:,1),'gx');
        %plot(peakLoc(:,2), peakLoc(:,1),'go');
        plot(valleyLoc(:,2), valleyLoc(:,1), 'go');
        plot(valleyLoc(:,2), valleyLoc(:,1), 'g+');
        plot(palmCent(:,2), palmCent(:,1), 'go');
    end
        
    if ~isNonRecoverable
        %% Compute and display ROI polygon for each of the 3 middle fingers
        virtualPeakLoc = (peakLoc(1:end-1,:) + peakLoc(2:end, :)) / 2;
        for i = 1  : 3
            handpoly = [];
            handpoly = [valleyLoc(i,:); valleyLoc(i+1,:); virtualPeakLoc(i+1,:); peakLoc(i+1,:); virtualPeakLoc(i,:); valleyLoc(i,:)];
            %plot(handpoly(:,2), handpoly(:,1), 'g');
        end
        
        %% Encapsulate Everything In a Consistent Package
        sfactor = 2;
        tempValleyLoc = [valleyLoc(1,:) + sfactor*(valleyLoc(1,:) - valleyLoc(2,:));valleyLoc; valleyLoc(end,:) + sfactor * (valleyLoc(end,:)- valleyLoc(end-1,:))];
        tempPeakLoc = [2*peakLoc(1,:) - peakLoc(2,:); peakLoc; 2*peakLoc(end,:)-peakLoc(end-1,:)];
        virtualPeakLoc = (tempPeakLoc(1:end-1,:) + tempPeakLoc(2:end, :)) / 2;
        handpoly = [];
        for i = 1  : 5
            handpoly = [handpoly; tempValleyLoc(i,:); tempValleyLoc(i+1,:); virtualPeakLoc(i+1,:); tempPeakLoc(i+1,:); virtualPeakLoc(i,:); tempValleyLoc(i,:)];
        end
        
        %% Perform non-recoverable checks
        % Apply the Y coordinate rule for the thumb and little finger. +may
        % need an X coordinate check as well
        if length(tempValleyLoc) == length(tempPeakLoc) - 1
            digitLengthDifference = min(min(tempValleyLoc(:,1) - virtualPeakLoc(:,1)), min(tempValleyLoc(:,1) - virtualPeakLoc(:,1)));
        else
            %disp('There is a length mismatch');
            digitLengthDifference = -1;
        end
        % Apply the polygon area rule --> need to check the pixels as well.
        fingerPolyAreas = zeros(1,5);
        for i = 0 : 4
            fingerPolyAreas(i+1) = polyarea(handpoly((i*5+1):(i*5+5),1), handpoly((i*5+1):(i*5+5),2));
        end
        if (((digitLengthDifference > 0) && (0.2<(min(fingerPolyAreas) / max(fingerPolyAreas)))))
            isNonRecoverable = 0;
        else
            isNonRecoverable = 1;
            %disp('Potential non-recoverable situation.');
        end
        
        if isDisplay
            if ~isNonRecoverable
                plot(handpoly(:,2), handpoly(:,1), 'g');
            else
                plot(handpoly(:,2), handpoly(:,1), 'r');
            end
        end
    end
    
    if ~isNonRecoverable
        %%Extract each finger segment and display
        palmDescriptors = zeros(5,18);
        for i = 1 : 5
            handpoly = [];
            handpoly = [tempValleyLoc(i,:); tempValleyLoc(i+1,:); virtualPeakLoc(i+1,:); tempPeakLoc(i+1,:); virtualPeakLoc(i,:)];
            BW = poly2mask(handpoly(:,2), handpoly(:,1), size(palmD,1), size(palmD,2));
            finger = BW .* palmD;
            a = sum(finger,1) == 0;
            b = sum(finger,2) == 0;
            finger(b,:) = [];
            finger(:,a) = [];
            if isDisplay && 0          
                subplot(2,3,i);
                imagesc(finger); hold on;           
                improveFinger(finger);
            end

            %findCentralAxis(handpoly, a, b);
            [palmDescriptors(i,:)] = DescribeFinger(finger, handpoly, palmCent,0);
        end
    else
        palmDescriptors = [];
        handpoly = [];
    end
    
    if max(max(isnan(palmDescriptors))) == 1
        palmDescriptors = [];
        handpoly = [];
    end
end