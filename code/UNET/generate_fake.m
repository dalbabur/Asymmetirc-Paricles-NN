tic
generate = 20;
max_objs = 7;
folder = '\data\test';

path = 'C:\Users\Diego\Documents\MATLAB\JHU\HUR\asymmetricParticles\AsymParticles\code\UNET';
bgpath = [path,'\data\synthetic\background\'];
Upath = [path,'\data\synthetic\U\object\'];
Lpath = [path,'\data\synthetic\L\object\'];
UFOpath = [path,'\data\synthetic\UFOs\object\'];
Shadowpath = [path,'\data\synthetic\Shadow\'];

bg = dir([bgpath '*.tif']);
U = dir([Upath '*.png']);
L = dir([Lpath '*.png']);
UFO = dir([UFOpath '*.png']);
Shadow = dir([Shadowpath '*.png']);
dirs = {U,L,UFO,Shadow};

% load all images
for i = 1:numel(bg), bgs{i} = imread([bg(i).folder,'/',bg(i).name]);
end
for i = 1:length(dirs)
    for j=1:numel(dirs{i})
       [imgs{i,j}, ~, masks{i,j}] = imread([dirs{i}(j).folder,'/',dirs{i}(j).name]);
    end
end

dummy = ~cellfun(@isempty, imgs);
imgs(dummy) = cellfun(@rgb2gray, imgs(dummy), 'UniformOutput',false);
spacing = round(mean2(cellfun(@length,imgs))*1.2);


for g = 1:generate
    % pick background
    b = bgs{randi([1 numel(bg)])};

    % pick how many of each obj, and which
    dist = rand(1,length(dirs));
    dist = dist/sum(dist);
    total = randi([1,max_objs]);
    final = round(total*dist);
    id = cell(1,sum(final));
    indx = [];
    for i = 1:length(dirs)
        indx = [indx repmat(i,1,final(i))];
        id{i} = randi([1 numel(dirs{i})],1,final(i));
    end
    id = [id{:}];

    % pick location, angle
    angle = randi(360,1,sum(final));
    x = randi([30 60],1,sum(final)); % maybe relate to size(b)
    y = randsample(randi(spacing):spacing:1260,sum(final));

    b2 = b;
    bin = (zeros(size(b)));
    if ~isempty(id)
        for i = 1:sum(final)
        img = imgs{indx(i),id(i)};
        mask = masks{indx(i),id(i)};

        img = imrotate(img,angle(i));
        mask = imrotate(mask,angle(i));
        [m,n] = size(img);

        xend = (x(i)+m-1);
        yend = (y(i)+n-1);
        if xend>128, xend = 128; end
        if yend>1280, yend = 1280; end

        t = b2(x(i):xend,y(i):yend);
        [m,n] = size(t);

        img = (img(1:m,1:n));
        idx = mask(1:m,1:n) > 100;

%Modification of shadows to better match background
    if indx(i) == length(dirs)
              val = t(double(img).*double(idx) <= 100 & double(img).*double(idx) > 0);
              value = t(double(img).*double(idx) >= 100 & double(img).*double(idx) < 200);
              t(double(img).*double(idx) <= 100 & double(img).*double(idx) > 0) = val-15;
              t(double(img).*double(idx) >= 100 & double(img).*double(idx)<200) = value+20;
%All other objects remain unchanged
    else
        t(idx) = img(idx);
    end

        h = fspecial('motion',randi(4),-randi(360));
        t2 = imfilter(t,h,'replicate');
        b2(x(i):xend,y(i):yend) = t2;

        b2(x(i):xend,y(i):yend) = imfilter(b2(x(i):xend,y(i):yend), ones(3)/9,'replicate');
        b2(x(i):xend,y(i):yend) = imsharpen(b2(x(i):xend,y(i):yend),'Amount',2,'Radius',0.5);

%         idx = double(idx);
%         idx(idx==1) = indx(i);
        idx2 = bin(x(i):xend,y(i):yend);
        idx2(idx) = indx(i);
        bin(x(i):xend,y(i):yend) = idx2;
        end
    end
% figure
% imagesc(bin)

imwrite(b2,[path, folder,'/img/frames/',num2str(g),'.tif'])
imwrite(uint8(bin),[path, folder,'/mask/frames/',num2str(g),'.tif'])
end
toc
