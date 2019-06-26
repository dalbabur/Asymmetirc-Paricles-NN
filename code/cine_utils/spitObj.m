%% read data and set library
addpath lib_asym
importfile('mat/LinLUT.mat');


[fileName, pathName, ~] = uigetfile( ...
    {'*.cine','CINE-files (*.cine)'; ...
    '*.*',  'All Files (*.*)'}, ...
    'Select a movie file', ...
    'MultiSelect', 'on');

info=cineInfo(pathName,fileName); % changed a little bit
height = info.Height;
width = info.Width;
numFrames = info.NumFrames;

%Make a data directory
pN = uigetdir();
ImgDirName=[pN, '\',fileName(1:(end-5))];
mkdir(ImgDirName);
cd(ImgDirName);

%% set variables
% for display
minVal =  1000;
maxVal = 3000;
marginLen = 10;
% for constructing background at least 20 frames
% ask user to define the range of backgroud images
prompt  = {'Starting frame # for BG (at least 20 frame clearance'};
def     = {'-80000'};
dlgtitle   = 'Frame range for the BG determination';
lines   = 1;
% options.Resize='on';
% options.WindowStyle='modal';
% options.Interpreter='tex';
answer  = inputdlg(prompt,dlgtitle,lines,def);

bgStartFrame = eval(answer{1})-info.startFrame+1;
bgEndFrame= bgStartFrame+20;

% bgStartFrame = 95; %80th frame (Current frame-initial frame)
% bgEndFrame = 115; %100th frame (Current frame-initial frame)
numTrainFrames = bgEndFrame-bgStartFrame+1;
% for finding frames containing objects
isObjThre = 500;
filterSize = 3;
% for removing small objects
dilationSize = 7;
smallObjSize = 50;
% for detection

% ask user to define the range of backgroud images
prompt  = {' numSamples','numIteration'};
def     = {'2000', '20'};
dlgtitle   = 'Please define interation parameters (increment of 1000 &10)';
lines   = 1;
% options.Resize='on';
% options.WindowStyle='modal';
% options.Interpreter='tex';
answer  = inputdlg(prompt,dlgtitle,lines,def);
numSamples = eval(answer{1}); % important parameter: increase up to 2000 -> it will consume 2 times of computational time (default:1000)
numIteration = eval(answer{2}); % important parameter: increase up to 20   -> it will consume 2 times of computational time (default:10)
% for detection
marx = 80;
mary = 30;

%%  region of interest
img=cineRead(pathName,fileName,1,info,LinLUT);

%% find background image (median image or you can use mean image)
tmp1 = uint16(zeros(height*width, numTrainFrames));
for i = 1 : numTrainFrames
    tmp2 = cineRead(pathName,fileName,bgStartFrame+i-1,info,LinLUT);
    tmp1(:,i) = tmp2(:);
end
bgImg = median(tmp1,2);
bgImg = reshape(bgImg, height, width);
display_image(bgImg, minVal, maxVal,1);
clear tmp1;clear tmp2;

%% annotate template
% find frames containing objects
flagObjFrame = false(numFrames,1);

tic
total = zeros(1,numFrames);
for i = 1:numFrames
% for i = 1:numFrames
    img = cineRead(pathName,fileName,i,info,LinLUT);
%     imgROI = img(iroi2:iroi4, iroi1:iroi3);
    forImg = img-bgImg;
    total(i) = sum(sum(corrcoef(mat2gray(bgImg),mat2gray(img))))/2-1;
    if total(i) < 0.995
        I = mat2gray(img);
        imwrite(I,[num2str(i),'.tif'])
    end
%     display_image(forImg, 0, 500,2);
%     pause
end
toc
