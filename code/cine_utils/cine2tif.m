cine_folder = 'C:\Users\Diego\Documents\MATLAB\JHU\HUR\asymmetricParticles\clips';
% cine_file = '370ul_min_1_67333-67281.cine';
cine_file = '916ul_min_1_98880.cine';


window_height = [1:128]; % vector of pixels at which to read data
window_length = 1280; % number of pixles 
window_origin = 0; % offset 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
importfile('LinLUT.mat'); %a conversion between packed 10bit data to real 10bit data
info = cineInfo(cine_folder,cine_file);
frames = -info.endFrame:-info.startFrame;

for i = 63:10:110
    img = (cineRead2(cine_folder,cine_file,[i:i],info,LinLUT,...
        window_height,window_length,window_origin));
    I = mat2gray(img);
    
    imwrite(I,['L-',num2str(frames(i)),'.tif'])
end