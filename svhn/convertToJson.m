load digitStruct.mat

fid = fopen('svhn_annot.json','w');
fprintf(fid, '{\n\t"data": [');

for i = 1:length(digitStruct)
    im = imread([digitStruct(i).name]);
    [height, width, depth] = size(im);
    fprintf(fid, '\n\t\t{\n\t\t\t"name": "%s",\n\t\t\t"width": %d,\n\t\t\t"height": %d,\n\t\t\t"bbox": [', digitStruct(i).name, width, height);
    for j = 1:length(digitStruct(i).bbox)                
        fprintf(fid, '\n\t\t\t\t{\n\t\t\t\t\t"top": %d,\n\t\t\t\t\t"left": %d,\n\t\t\t\t\t"width": %d,\n\t\t\t\t\t"height": %d,\n\t\t\t\t\t"label": %d\n\t\t\t\t}', digitStruct(i).bbox(j).top, digitStruct(i).bbox(j).left, digitStruct(i).bbox(j).width, digitStruct(i).bbox(j).height, digitStruct(i).bbox(j).label);
        if (j ~= length(digitStruct(i).bbox))
            fprintf(fid, ',');
        end
    end
    fprintf(fid, "\n\t\t\t]\n\t\t}");
    if (i ~= length(digitStruct))
        fprintf(fid, ',');
    end
end

fprintf(fid, '\n\t]\n}\n\n');

fclose(fid);