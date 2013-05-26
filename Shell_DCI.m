clc
clear
close all

uiwait(msgbox('Right Hand.'));
[dataR, depthsR] = DataCollectionInterface_beta2(1);

uiwait(msgbox('Left Hand.'));
[dataL, depthsL] = DataCollectionInterface_beta2(2);

