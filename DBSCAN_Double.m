function [outputArg1,outputArg2] = DBSCAN_Double(img,pathsave_Detected,k)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
im = im2double (img); 
idm = im < 0.12; 
im(idm) = NaN; 
indices = find(isnan(im) == 1);
[I,J] = ind2sub(size(im),indices);
id (:,1) = I;
id (:,2)=J;  
eps = 1 ;
minpt = 5;  
idx = dbscan(id, eps,minpt); 
%
vis = [id,idx];
indices_vis = find(vis(:,3) <0);
vis  (indices_vis,:) = [];
% Logical
BW = zeros (size(im));
for i = 1:length (vis)
    BW(vis(i,1),vis(i,2))=1;
end    
BW_log = logical(BW);
% Centroids
stats = regionprops(BW_log,'centroid');
centroids = cat(1,stats.Centroid);


%% New- 2nd  DBSCAN
BW_2 = BW;
[II,JJ] = ind2sub(size(BW_2),find(BW_2==1));
idd (:,1) = II; idd (:,2)=JJ;
db_2 = dbscan(idd,1,5);
viss = [idd,db_2];
indices_viss = find(viss(:,3) <0);
viss  (indices_viss,:) = [];
BW_2_new = zeros (size(im));

for j = 1:length (viss)
    BW_2_new(viss(j,1),viss(j,2))=1;
end
% Centroids 
BW_2_log = logical(BW_2_new);
statss = regionprops(BW_2_log,'centroid');
centroidss = cat(1,statss.Centroid);

%% Bounding Boxes
xlen = 250 ; ylen =  250;

for f= 1: length (centroidss)
    clearvars xmin ymin F
    xmin = centroidss(f,1) - 125 ; ymin = centroidss(f,2) - 125;
    F = imcrop(img, [xmin ymin xlen ylen]);
    %K=k; F=f; 
    %'KF%d%d.pnd', 
    %text1 = (k+'_%d.png');
    baseFileName = sprintf('%d%d.png',k,f);
    fullFileName = fullfile(pathsave_Detected, baseFileName);
    imwrite(F, fullFileName);
end 



end

