%% Data collection interface
%
% This program provides an interface for the collection of hand biometric
% data for verification experiments.

% Current interface elements
% --------------------------
% Single screen format
% This version works on the webcam stream

function [data, depths] = DataCollectionInterface_beta2(safeGuard)

% SAFEGUARD is to ensure function is not calledby F5;
safeGuard;

%% Instructions:
uiwait(msgbox(['This experiment consists of three rounds of data collection.' char(10) 'Each set is broken into two steps. In the first step, ' ...
'place your palm such that it lies entirely in the red box. In the second step, spread and close your fingers slowly, while keeping your palm in the box.'...
char(10) 'Click ok to begin the experiment.']));
%data = 0 ;return;

%% Initialization for kinect
[a b] = getimagedataN(1);

data = zeros(480, 640, 51,3); 

%% Routine for Repositioning
for depthLevel = 1 : 3
    rectCoords = PositionBox(depthLevel);
    for i = 5 : -0.2 : 0
        [imageD imageRGB] = GetSnapshot();
        imageD(imageD == 0) = 4000;
        imageD = imageD(:,end:-1:1);
        imagesc(imageD);
        %r = rectangle('Position', [200, 200, 500, 500]);
        r = rectangle('Position', rectCoords);
        set(r,'edgecolor','g');
        set(gca, 'xdir', 'reverse');
        %text(0, -25, ['Please reposition hand in box. You may need to move your hand forward or backward. Time until capture begins: ' num2str(i) 's...']);
        text(640, -15, ['Please reposition hand in box. You may need to move your hand forward or backward. Time until capture begins: ' num2str(i) 's...']);
        pause(0.2);
    end

    %% Main Frame Capture Routine
    j = 0;
    % Capture data for 10 seconds
    rectCoords = PositionBox(depthLevel);
    for i = 10 : -.2 : 0
        j = j + 1;
        [imageD imageRGB] = GetSnapshot();
        imageD(imageD == 0) = 4000;
        imageD = imageD(:,end:-1:1);
        imagesc(imageD);
        r = rectangle('Position', rectCoords);
        set(r,'edgecolor','g');    
        set(gca, 'xdir', 'reverse');
        %text(0, -25, ['Time until end of capture: ' num2str(i) 's...']);
        text(640, -15, ['Time until end of capture: ' num2str(i) 's...']);
        data(:,:,j, depthLevel) = imageD;
        pause(0.2);
    end
    
    uiwait(msgbox(['Capture complete for set ' num2str(depthLevel) ' of 3. Click ok when ready.']));
end

uiwait(msgbox(['Thank you for participating in this experiment!']));
data = uint16(data);
depths = [PositionBox(1);PositionBox(2);PositionBox(3)];
end

%% Generate a bounding rectangle
% There are 3 default depth levels -> to be calibrated
% Depths are relatively well caliibrated
% need to add setting for L/R hands
% Re-align offset to maintain(largely) the same angle.
function [rectCoords] = PositionBox(depth)
    switch depth
        case 1
            rectCoords = [150, 150, 200, 200];
        case 2
            rectCoords = [150, 150, 150, 150];
        case 3
            rectCoords = [150, 150, 100, 100];
        otherwise
            rectCoords = [150, 150, 250, 250];
    end
end