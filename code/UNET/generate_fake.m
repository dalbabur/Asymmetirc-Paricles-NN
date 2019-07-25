tic
generate = 5120*3;
max_objs = 7;
folder = '\data\train';
noise = 0;
transform = 1;

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

spacing = round(mean2(cellfun(@length,imgs))*1.5);

class_bal = zeros(generate,length(dirs));
for g = 1:generate
    % pick background
    b = bgs{randi([1 numel(bg)])};

    if noise == 1, if rand(1) > 0.5, b = imnoise(b, 'poisson'); end, end
    if transform == 1, if rand(1) > 0.5, b = fliplr(b); end, end

    % pick how many of each obj, and which
    dist = rand(1,length(dirs));
    dist = dist/sum(dist);
    total = randi([1,max_objs]);
    final = round(total*dist);
    class_bal(g,:) = final;
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
        img = imgs{indx(i),id(i)}(:,:,1);
        mask = masks{indx(i),id(i)};

        if noise == 1, if rand(1) > 0.25, img = imnoise(img, 'poisson'); end, end
        if transform == 1
            if rand(1) > 0.10
                sc = 0.85 + (0.35).*rand(1,2);
                sh = rand(1,2)-0.5;
                shm = randn(1,2);
                img = imwarp(img, affine2d([sc(1) sh(1)*shm(1) 0; sh(2)*shm(2) sc(2) 0; 0 0 1]));
                mask = imwarp(mask, affine2d([sc(1) sh(1)*shm(1) 0; sh(2)*shm(2) sc(2) 0; 0 0 1]));
            end
        end

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

        % Modification of shadows to better match background
        if indx(i) == length(dirs)
                  val = t(double(img).*double(idx) <= 100 & double(img).*double(idx) > 0);
                  value = t(double(img).*double(idx) >= 100 & double(img).*double(idx) < 200);
                  t(double(img).*double(idx) <= 100 & double(img).*double(idx) > 0) = val-15;
                  t(double(img).*double(idx) >= 100 & double(img).*double(idx)<200) = value+20;
        % All other objects remain unchanged
        else
            t(idx) = img(idx);
        end

        h = fspecial('motion',randi(4),-randi(360));
        t2 = imfilter(t,h,'replicate');
        b2(x(i):xend,y(i):yend) = t2;

        b2(x(i):xend,y(i):yend) = imfilter(b2(x(i):xend,y(i):yend), ones(3)/9,'replicate');
        b2(x(i):xend,y(i):yend) = imsharpen(b2(x(i):xend,y(i):yend),'Amount',2,'Radius',0.5);

        idx2 = bin(x(i):xend,y(i):yend);
        idx2(idx) = indx(i);
        if indx(i) == length(dirs)
            idx2(idx) = 0;
        end
        bin(x(i):xend,y(i):yend) = idx2;
        end
    end

    [m,n] = size(b2);
    if transform == 1
        b2 = b2*(0.5 + rand(1));
        if rand(1) > 0.10
            q = 2.5*randn(1);
            b2 = imrotate(b2, q, 'crop');
            bin = imrotate(bin, q, 'crop');
        end
        if rand(1) > 0.10
            s = 1+rand(1)/4;
            b2 = imresize(b2,s,'OutputSize',[m,n]);
            bin = imresize(bin,s,'OutputSize',[m,n],'method','nearest');
        end
    end

% % figure
% % subplot(2,1,1)
% % imshow(b2)
% % subplot(2,1,2)
% % imshow(bin)

imwrite(b2,[path, folder,'/img/frames/Augmented2/',num2str(g),'.tif'])
imwrite(uint8(bin),[path, folder,'/mask/frames/Augmented2/',num2str(g),'.tif'])
end
toc
