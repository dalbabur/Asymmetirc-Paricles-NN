%Folder for storing complements that are created
ShadowFolder = '\data\synthetic\Shadow\';

%Opens folders for letters and finds/counts objects within
path = 'C:\Users\Diego\Documents\MATLAB\JHU\HUR\asymmetricParticles\AsymParticles\code\UNET';
Upath = [path,'\data\synthetic\U\object\'];
Lpath = [path,'\data\synthetic\L\object\'];

%Lists elements in each folder and stores them in vars
L = dir(fullfile(Lpath,'*.png'));
U = dir(fullfile(Upath,'*.png'));
vars = {L,U};

%Creates complements of all images stored
for x = 1:length(vars)
    for k = 1:numel(vars{x})
        [images,~,masks] = imread([vars{x}(k).folder,'\',vars{x}(k).name]);
        Complement = imcomplement(images);
        imwrite(Complement,[path,ShadowFolder,num2str(x),num2str(k),'000.png'],'png','Alpha',masks)
    end
end
