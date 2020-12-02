%% mslMainCompleteUNIQUEPipeline.m
%  This program extracts image patches from any available online database, learns 
%  sparse representation from them, processes images from LIVE, and MULTI 
%  databases and compares them against subjective scores.

%%
clear all
clc

%%
% %Extract patches from ImageNet database
% 
% %Path where ImageNet database is stored
% path = '.\Database\ILSVRC2013_DET_test\';
% %Size of patches to be extracted
% Scale = 8;
% 
% %Call function to extract patches and store in the variable mslPatches
% [mslPatches,numPatches] = mslImageNetPatchExtract(path,Scale);

%%
%Calculate weights and bias using a linear decoder
patchesStruct = load('ImageNet_patches.mat');
mslPatches = patchesStruct.patches;
Scale = 8;
numPatches = length(mslPatches);
[W,b,~] = linearDecoderExercise(mslPatches,Scale,numPatches);

%% Parameter Initialization for score calculation in individual databases

%AEStruct = load('ImageNet_Weights_YGCr.mat');
AEStruct = load('ImageNet_Weights_YGCr_EDF.mat');
W = AEStruct.W;
b = AEStruct.b;

tic
blockSize=[10 10];
paramIndArray=[1];
metInd=[1];
poolInd=[1];
nMet = 1;

%LIVE
%Calculate scores for images in LIVE database
for ii=1:nMet
mslMainPoolingLIVE(poolInd, metInd(ii), paramIndArray, blockSize)
end

%MULTI
%Calculate scores for images in MULTI database
for ii=1:nMet
mslMainPoolingMULTI(poolInd, metInd(ii), paramIndArray, blockSize)
end

%% Compare against subjective scores
for jj=1:nMet

fileDir=['Multi_corrAfter','*metInd_',num2str(metInd(jj)),'_poolInd_',num2str(poolInd)];
files = dir([fileDir,'*.mat']);
load([files(1).name]);
multi_results(jj,:,:)=squeeze(corrMat);


fileDir=['corrAfter','*metInd_',num2str(metInd(jj)),'_poolInd_',num2str(poolInd)];
files = dir([fileDir,'*.mat']);
load([cd,filesep,files(1).name]);
live_results(jj,:,:)=squeeze(corrMat);

end

% Extract and separate three different correlation methods
pool_table_ps=zeros(19,nMet);
pool_table_sm=zeros(19,nMet);
pool_table_kd=zeros(19,nMet);
live_ps=squeeze(live_results(:,1,:));
live_sm=squeeze(live_results(:,2,:));
live_kd=squeeze(live_results(:,3,:));
multi_ps=squeeze(multi_results(:,1,:));
multi_sm=squeeze(multi_results(:,2,:));
multi_kd=squeeze(multi_results(:,3,:));

live_ind=[1,2,5,8,10,17];
live_ind_int=[1,2,3,5,4,6];
multi_ind=[4,6,12,13,18];
multi_ind_int=[1,2,1,2,3];

for jj=1:nMet

    for ii=1:size(live_ind,2)
        if nMet==1 
            pool_table_ps(live_ind(ii))=live_ps(live_ind_int(ii));
            pool_table_sm(live_ind(ii))=live_sm(live_ind_int(ii));
            pool_table_kd(live_ind(ii))=live_kd(live_ind_int(ii));
        else
            pool_table_ps(live_ind(ii),jj)=live_ps(jj,live_ind_int(ii));
            pool_table_sm(live_ind(ii),jj)=live_sm(jj,live_ind_int(ii));
            pool_table_kd(live_ind(ii),jj)=live_kd(jj,live_ind_int(ii));
        end
    end

    for ii=1:size(multi_ind,2)
        if nMet==1 
            pool_table_ps(multi_ind(ii))=multi_ps(multi_ind_int(ii));
            pool_table_sm(multi_ind(ii))=multi_sm(multi_ind_int(ii));
            pool_table_kd(multi_ind(ii))=multi_kd(multi_ind_int(ii));
        else
            pool_table_ps(multi_ind(ii),jj)=multi_ps(jj,multi_ind_int(ii));
            pool_table_sm(multi_ind(ii),jj)=multi_sm(jj,multi_ind_int(ii));
            pool_table_kd(multi_ind(ii),jj)=multi_kd(jj,multi_ind_int(ii));
        end
    end

end

pool_table_ps=abs(pool_table_ps);
pool_table_sm=abs(pool_table_sm);
pool_table_kd=abs(pool_table_kd);

temp=load('ssim.mat');
temp=temp.pool_table;

% LIVE RESULTS
ii=1; 
dist_class{ii}=['Jp2k:    ','  SSIM:  ',num2str(temp(ii)),'    Pearson:  ',num2str(pool_table_ps(ii)), '  Spearman:  ',num2str(pool_table_sm(ii)),'  Kendall:  ',num2str(pool_table_kd(ii))];
ii=2;
dist_class{ii}=['Jpeg:    ','  SSIM:  ',num2str(temp(ii)),'    Pearson:  ',num2str(pool_table_ps(ii)), '  Spearman:  ',num2str(pool_table_sm(ii)),'  Kendall:  ',num2str(pool_table_kd(ii))];
ii=4;
dist_class{ii}=['Blur-Jpeg:    ','  SSIM:  ',num2str(temp(ii)),'    Pearson:  ',num2str(pool_table_ps(ii)), '  Spearman:  ',num2str(pool_table_sm(ii)),'  Kendall:  ',num2str(pool_table_kd(ii))];
ii=5;
dist_class{ii}=['Wn:    ','  SSIM:  ',num2str(temp(ii)),'    Pearson:  ',num2str(pool_table_ps(ii)), '  Spearman:  ',num2str(pool_table_sm(ii)),'  Kendall:  ',num2str(pool_table_kd(ii))];
ii=6;
dist_class{ii}=['Blur-Noise:    ','  SSIM:  ',num2str(temp(ii)),'    Pearson:  ',num2str(pool_table_ps(ii)), '  Spearman:  ',num2str(pool_table_sm(ii)),'  Kendall:  ',num2str(pool_table_kd(ii))];
ii=8;
dist_class{ii}=['FF:    ','  SSIM:  ',num2str(temp(ii)),'    Pearson:  ',num2str(pool_table_ps(ii)), '  Spearman:  ',num2str(pool_table_sm(ii)),'  Kendall:  ',num2str(pool_table_kd(ii))];
ii=10;
dist_class{ii}=['Gblur:    ','  SSIM:  ',num2str(temp(ii)),'    Pearson:  ',num2str(pool_table_ps(ii)), '  Spearman:  ',num2str(pool_table_sm(ii)),'  Kendall:  ',num2str(pool_table_kd(ii))];
ii=17;
jj=11;
dist_class{jj}=['All [LIVE]:    ','  SSIM:  ',num2str(temp(ii)),'    Pearson:  ',num2str(pool_table_ps(ii)), '  Spearman:  ',num2str(pool_table_sm(ii)),'  Kendall:  ',num2str(pool_table_kd(ii))];

% MULTI RESULTS
ii=12;
dist_class{ii}=['Blur-Jpeg:    ','  SSIM:  ',num2str(temp(ii)),'    Pearson:  ',num2str(pool_table_ps(ii)), '  Spearman:  ',num2str(pool_table_sm(ii)),'  Kendall:  ',num2str(pool_table_kd(ii))];
ii=13;
dist_class{ii}=['Blur-Noise:    ','  SSIM:  ',num2str(temp(ii)),'    Pearson:  ',num2str(pool_table_ps(ii)), '  Spearman:  ',num2str(pool_table_sm(ii)),'  Kendall:  ',num2str(pool_table_kd(ii))];
ii=18;
jj=14;
dist_class{jj}=['All [MULTI]:    ','  SSIM:  ',num2str(temp(ii)),'    Pearson:  ',num2str(pool_table_ps(ii)), '  Spearman:  ',num2str(pool_table_sm(ii)),'  Kendall:  ',num2str(pool_table_kd(ii))];

toc

for ii=1:14
   disp(dist_class{ii}) 
end
