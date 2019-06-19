tic
from_folder = '\data\authentic\L\1466ul_min_2\';
to_folder = '\data\predicted\img\frames\';
path = 'C:\Users\Diego\Documents\MATLAB\JHU\HUR\asymmetricParticles\code\UNET';

img_files = dir([[path,from_folder] '*.tif']);
n_imgs = numel(img_files);

for i = 1:n_imgs
    img = imread([img_files(i).folder,'\',img_files(i).name]);
    for j = 1:10
        small = img(:,(128*(j-1)+1):(128*j));
        imwrite(small,[path,to_folder,num2str(i),'-',num2str(j-1),'.tif'])
    end
end
toc