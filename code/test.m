cine_folder = 'C:\Users\Diego\Documents\MATLAB\JHU\HUR\asymmetricParticles\clips';
% cine_file = '370ul_min_1_67333-67281.cine';
cine_file = '916ul_min_1_98880.cine';


window_height = [1:128]; % vector of pixels at which to read data
window_length = 1280; % number of pixles 
window_origin = 0; % offset 
frames = [63 63]; % range of frames

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
importfile('LinLUT.mat'); %a conversion between packed 10bit data to real 10bit data
info = cineInfo(cine_folder,cine_file);

blank = (cineRead2(cine_folder,cine_file,[1:1],info,LinLUT,...
        window_height,window_length,window_origin));
    
for i = 63    
img = (cineRead2(cine_folder,cine_file,[i:i],info,LinLUT,...
        window_height,window_length,window_origin));
 
I = imgaussfilt(mat2gray(img-blank),2);

[Gmag,~] = imgradient(I);
bin = imbinarize(Gmag);
bin = imfill(bin,'holes');

% bin = imbinarize(I);

% bin = bwareaopen(imfill(edge(I),'holes'),64*4*0.5);

figure
subplot(2,1,1)
imshow(mat2gray(img))
subplot(2,1,2)
imshow(bin)
end