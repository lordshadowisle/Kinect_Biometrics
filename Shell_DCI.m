clc
clear
close all

uiwait(msgbox('Right Hand.'));
[dataR, depths] = DataCollectionInterface_beta2(1);

uiwait(msgbox('Left Hand.'));
[dataL, depths] = DataCollectionInterface_beta2(1);

