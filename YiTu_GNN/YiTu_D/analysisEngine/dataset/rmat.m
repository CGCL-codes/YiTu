clear all
scale = 19;
edge_factor = 32;
rand ("seed", 103);

ijw = kronecker_generator (scale, edge_factor);

G = sparse(ijw(1,:) + 1, ijw(2,:) + 1, ones (1, size (ijw, 2)));

[ei,ej] = find(G>0);

dim = size(ei);
edge_num = dim(1);

filename = sprintf("rmat-%d-%d.txt",scale,edge_factor);
fp=fopen(filename,'w');

for i = 1 : edge_num
    fprintf(fp,"%d %d \n",ei(i) , ej(i));
end

fclose(fp)
