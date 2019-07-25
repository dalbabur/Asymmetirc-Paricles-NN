
folder = '\data\synthetic\particles\masks';


path = 'C:\Users\Diego\Documents\MATLAB\JHU\HUR\asymmetricParticles\AsymParticles\code\UNET';
Upath = [path,'\data\synthetic\particles\U\'];
Lpath = [path,'\data\synthetic\particles\L\'];
UFOpath = [path,'\data\synthetic\particles\UFOs\'];

U = dir([Upath '*.png']);
L = dir([Lpath '*.png']);
UFO = dir([UFOpath '*.png']);
dirs = {U,L,UFO};

% load all images
for i = 1:length(dirs)
    for j=1:numel(dirs{i})
       [imgs{i,j}, ~, masks{i,j}] = imread([dirs{i}(j).folder,'/',dirs{i}(j).name]);
       if ~isempty(masks{i,j})
           switch i
           case 1
               imwrite(uint8(masks{i,j}>100),[path,folder,'\U\',num2str(j),'.tif'])
           case 2
               imwrite(uint8(masks{i,j}>100),[path,folder,'\L\',num2str(j),'.tif'] )
           case 3
               imwrite(uint8(masks{i,j}>100),[path,folder,'\UFOs\',num2str(j),'.tif'])
           end
       end
    end
end
