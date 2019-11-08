%% In this script the network is being trained with optimized parameters w.r.t. our data. 
%% Hyperparameters are a compromise to avoid overfitting and to get the best posssible accuracy.
%% The loaded data should be separated by class with almost equal in number and size separated by folders.    
%% For detailed information please refer to training network principles by deep learning in MATALAB Documentations.  
%% Setareh Medghalchi (medghalchi@imm.rwth-aachen.de)- IMM - RWTH - November 2019

%% Load data

imds = imageDatastore('Data', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 

%% Automatic split of the data
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);

net = inceptionv3;

analyzeNetwork(net)

net.Layers(1)

inputSize = net.Layers(1).InputSize;

if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer] 

numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% To check that the new layers are connected correctly, plot the new layer graph and zoom in on the last layers of the network.
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])

%% Freeze Initial Layers
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);
miniBatchSize = 10; 
valFrequency = floor(numel(imdsTrain.Files)/miniBatchSize);

%% Network Propeties. 
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',90, ...    
    'InitialLearnRate',3e-4, ... 
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Train the network
tic
net = trainNetwork(imdsTrain,lgraph,options);
toc

%%  Classify Validation Images
[YPred,probs] = classify(net,imdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)

%% Display some validation images:
cd  'I:\Daten\Damage_Matlab_IncpV3_Cat2_CorrectedData_New\Data';
mkdir ('Results_1\');
cd 'I:\Daten\Damage_Matlab_IncpV3_Cat2_CorrectedData_New\Data\Results_1'

%idx = randperm(numel(imdsValidation.Files),numel(imdsValidation.Files));
idx = randperm(numel(imdsValidation.Files),100);

% store the shown validation images
for i = 1: length(idx)
    clearvars I label title
    I = readimage(imdsValidation,idx(1,i));
    %imshow(I)
    label = YPred(idx(i));
    title = title(char(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
    imwrite (I, sprintf('%s.png',get(get(gca,'title'),'string')))
    
end


%% Test new images 

imds_Test = imageDatastore('Resized_IncpV3', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

[YPred_test,probs_test] = classify(net,imds_Test);
accuracy = mean(YPred == imds_Test.Labels)



% Display some results:
cd  'Your Directory';
mkdir ('Results_1\');
cd 'Results_1 folder in your directory'
idx = randperm(numel(imds_Test.Files),numel(imds_Test.Files));

for i = 1: length(idx)-1
    clearvars I label title
    I = readimage(imds_Test,idx(1,i));
    label = YPred_test(idx(i));
    title = title(char(label) + ", " + num2str(100*max(probs_test(idx(i),:)),3) + "%");
    imwrite (I, sprintf('%d_%s.png',i,get(get(gca,'title'),'string')))
   i 
end
     
%% At the end: save the workspace in the current directory 
save ('data.mat')
