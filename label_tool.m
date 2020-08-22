function [cor1, cor2] = getmask(img, img2)

% author: Patrick Doran
% Modified by: Qinbo Li

img = im2double(img);
img2 = im2double(img2);

img = permute(img,[2 1 3]);
img2 = permute(img2,[2 1 3]);

h = figure;

subplot(1, 2, 1);
imshow(img);
hold on

subplot(1, 2, 2);
imshow(img2);
hold on

% Loop, picking up the points.
disp('Left mouse button picks points. Right mouse button picks last point. Pick points on left and right in turn.')

% Initially, the list of points is empty.
xy = [];
xy2 = [];
n1 = 1;
n2 = 1;
flag = true;
but = 1;
while but == 1
    [xi, yi, but] = ginput(1);
    plot(xi, yi,'ro')

    if flag
        xy(:, n1) = [xi; yi];
        n1 = n1 + 1;
    else
        xy2(:, n2) = [xi; yi];
        n2 = n2 + 1;
    end
    flag = ~flag;
end

cor1 = int32(xy);
cor2 = int32(xy2);

hold off

close(h);
drawnow

end