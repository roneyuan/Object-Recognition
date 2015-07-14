clear all
%%
% Name: Sung-Yuan Chen
% CPE-558 Computer Vision
% Final Project
% Object Recognition
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% REFERENCES
% 1. function [cim, r, c] = harris(im, sigma, thresh, radius, disp)
% Author: 
% Peter Kovesi   
% Department of Computer Science & Software Engineering
% The University of Western Australia
% pk@cs.uwa.edu.au  www.cs.uwa.edu.au/~pk
% March 2002
%
% 2. function sift_arr = find_sift(I, circles, enlarge_factor)
% Author:
% Lana Lazebnik
%
% 3. function n2 = dist2(x, c)
% Author:
% Ian T Nabney (1996-2001)
%
% 4. function ROCout=roc(varargin)
% Author:
% Cardillo G. (2008) ROC curve: compute a Receiver Operating Characteristics curve.
% http://www.mathworks.com/matlabcentral/fileexchange/19950
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Bike
%% Step 0: Read the all images
% The directory for the images is varied by system. Please change the
% directory for your system in order to read the images
dirname='D:\Final\Sung-Yuan_Chen_FinalProject\Train\bike';
files=dir([dirname,'\*.bmp']);
Num_file=numel(files);
data=cell(1,Num_file);

for k = 1:Num_file    
    data{k} = imread([dirname '\' files(k).name]);
end

%% Step 1: Feature extraction:
% It uses harris corner feature extraction 
for i = 1:Num_file
    image{i} = rgb2gray(data{i});
    [cim{i}, r{i}, c{i}] = harris(image{i}, 2,500,2,1);
end

%% Step 2: Feature Description:
% It uses find_sift function to implement the SIFT descriptor

% The iteratioin is used for locating the radius for all row coordinates
% of corner points of all images in Bike.
for i = 1:Num_file
    radius{i}(1:size(r{i}),1) = 2;
end

% The iteration implements find SIFT descriptors for all images in Bike
for i = 1:Num_file
   sift_arr{i} = find_sift(image{i},[c{i} r{i} radius{i}] ,3); 
end

% Find the size of SIFT descriptor for all images
for i = 1:Num_file
    [row{i} col{i}] = size(sift_arr{i});
end

% Define a matrix to save all images for each image category
KM = [];
temp_r = 0;

% The iteration put every image's SIFT descriptor in Bike into a matrix
for i = 1:Num_file
    for a = 1:col{i}
        for b = 1:row{i}
       
            KM(b+temp_r,a) = sift_arr{i}(b,a);
            
        end;
    end;
    [temp_r temp_c] = size(KM);
end


%% Cars
%% Step 0: Read the all images
% The directory for the images is varied by system. Please change the
% directory for your system in order to read the images
dirname2='D:\Final\Sung-Yuan_Chen_FinalProject\Train\cars';
files2=dir([dirname2,'\*.bmp']);
Num_file2=numel(files2);
data2=cell(1,Num_file2);

for k = 1:Num_file2    
    data2{k} = imread([dirname2 '\' files2(k).name]);
end

%% Step 1: Feature extraction:
% It uses harris corner feature extraction 
for i = 1:Num_file2
    image2{i} = rgb2gray(data2{i});
    [cim2{i}, r2{i}, c2{i}] = harris(image2{i}, 2,500,2,1);
end
%% Step 2: Feature Description:
% It uses find_sift function to implement the SIFT descriptor

% The iteratioin is used for locating the radius for all row coordinates
% of corner points of all images in Cars.
for i = 1:Num_file2
    radius2{i}(1:size(r2{i}),1) = 2;
end
% The iteration implements find SIFT descriptors for all images in Cars
for i = 1:Num_file2
   sift_arr2{i} = find_sift(image2{i},[c2{i} r2{i} radius2{i}] ,3); 
end
% Find the size of SIFT descriptor for all images
for i = 1:Num_file2
    [row2{i} col2{i}] = size(sift_arr2{i});
end
% Define a matrix to save all images for each image category
KM2 = [];
temp_r2 = 0;
% The iteration put every image's SIFT descriptor in Bike into a matrix
for i = 1:Num_file2
    for a = 1:col2{i}
        for b = 1:row2{i}
       
            KM2(b+temp_r2,a) = sift_arr2{i}(b,a);
            
        end;
    end;
    [temp_r2 temp_c2] = size(KM2);
end


%% Following categories are doing same process as above
%% Person

dirname3='D:\Final\Sung-Yuan_Chen_FinalProject\Train\person';
files3=dir([dirname3,'\*.bmp']);
Num_file3=numel(files3);
data3=cell(1,Num_file3);

for k = 1:Num_file3    
    data3{k} = imread([dirname3 '\' files3(k).name]);
end

for i = 1:Num_file3
    image3{i} = rgb2gray(data3{i});
    [cim3{i}, r3{i}, c3{i}] = harris(image3{i}, 2,500,2,1);
end

for i = 1:Num_file3
    radius3{i}(1:size(r3{i}),1) = 2;
end

for i = 1:Num_file3
   sift_arr3{i} = find_sift(image3{i},[c3{i} r3{i} radius3{i}] ,3); 
end

for i = 1:Num_file3
    [row3{i} col3{i}] = size(sift_arr3{i});
end

KM3 = [];
temp_r3 = 0;

for i = 1:Num_file3
    for a = 1:col3{i}
        for b = 1:row3{i}
       
            KM3(b+temp_r3,a) = sift_arr3{i}(b,a);
            
        end;
    end;
    [temp_r3 temp_c3] = size(KM3);
end

%% None

dirname4='D:\Final\Sung-Yuan_Chen_FinalProject\Train\none';
files4=dir([dirname4,'\*.bmp']);
Num_file4=numel(files4);
data4=cell(1,Num_file4);

for k = 1:Num_file4    
    data4{k} = imread([dirname4 '\' files4(k).name]);
end

for i = 1:Num_file4
    image4{i} = rgb2gray(data4{i});
    [cim4{i}, r4{i}, c4{i}] = harris(image4{i}, 2,500,2,1);
end

for i = 1:Num_file4
    radius4{i}(1:size(r4{i}),1) = 2;
end

for i = 1:Num_file4
   sift_arr4{i} = find_sift(image4{i},[c4{i} r4{i} radius4{i}] ,3); 
end

for i = 1:Num_file4
    [row4{i} col4{i}] = size(sift_arr4{i});
end

KM4 = [];
temp_r4 = 0;

for i = 1:Num_file4
    for a = 1:col4{i}
        for b = 1:row4{i}
       
            KM4(b+temp_r4,a) = sift_arr4{i}(b,a);
            
        end;
    end;
    [temp_r4 temp_c3] = size(KM4);
end

%% Step 3: Dictionary Computation
% Run All training feature for kmeans
KM_Total = [KM;KM2;KM3;KM4];
[IDX,C_k,sumd,D_k] = kmeans(KM_Total,500);
% Define codebook which is equal to all center of the clusters
codebook = C_k;

%% Step 4: Compute Image Representation
%% Bike

% The iteration compute the shortest every codevector in the dictionary
for  i = 1:Num_file
    n2{i} = dist2(sift_arr{i}, codebook);
end

% Find index of the nearest codevector in the dictionary
for i = 1:Num_file
    nearest_codevector{i} = min(n2{i},[],2);
    [roww{i} coll{i}] = size(nearest_codevector{i});
    for a = 1:roww{i}
        [min_row{i}(a,1) min_col{i}(a,1)] = find(n2{i} == nearest_codevector{i}(a,1));
    end
end

% Image Representation
H = [];
for i = 1:Num_file
        H{i} = hist(min_col{i},1:500);
end
HIST = [];
SET = 0;
for i = 1:Num_file
    for j = 1:500 % 500 clusters
        HIST(i,j) = H{i}(1,j);
    end
end
% Normalized the Histogram
A = 1/norm(HIST);
B = HIST * A;

%%%%%%%%%%%%
% Testing
%%%%%%%%%%%%
% % Define an empty matrix to save all indice of nearest codevector in Bike
% norm_hist = [];
% set1 = 0;
% % The iteration save all indice of nearest codevector to norm_hist
% for i = 1:Num_file
%     for j = 1:size(min_col{i})
%         norm_hist(j+set1,1) = min_col{1,i}(j,1);        
%     end;
%     [set1 set2] = size(norm_hist);
% end
% h = norm_hist;
% figure(1)
% hist(h,1:500);

%% Cars

% The iteration compute the shortest every codevector in the dictionary
for  i = 1:Num_file2
    n2_2{i} = dist2(sift_arr2{i}, codebook);
end

% Find index of the nearest codevector in the dictionary
for i = 1:Num_file2
    nearest_codevector2{i} = min(n2_2{i},[],2);
    [roww2{i} coll2{i}] = size(nearest_codevector2{i});
    for a2 = 1:roww2{i}
        [min_row2{i}(a2,1) min_col2{i}(a2,1)] = find(n2_2{i} == nearest_codevector2{i}(a2,1));
    end
end

% Image Representation
H2 = [];
for i = 1:Num_file2
        H2{i} = hist(min_col2{i},1:500);
end
HIST2 = [];
SET2 = 0;
for i = 1:Num_file2
    for j = 1:500 % 500 clusters
        HIST2(i,j) = H2{i}(1,j);
    end
end
% Normalized the Histogram
A2 = 1/norm(HIST2);
B2 = HIST2 * A2;

%%%%%%%%%%%%
% Testing
%%%%%%%%%%%%
% % Define an empty matrix to save all indice of nearest codevector in Bike
% norm_hist2 = [];
% set1_2 = 0;
% % The iteration save all indice of nearest codevector to norm_hist
% for i = 1:Num_file2
%     for j = 1:size(min_col2{i})
%         norm_hist2(j+set1_2,1) = min_col2{1,i}(j,1);        
%     end;
%     [set1_2 set2_2] = size(norm_hist2);
% end
% h2 = norm_hist2;
% figure(2)
% hist(h2,1:500);

%% Following processes is same as above
%% Person

for  i = 1:Num_file3
    n2_3{i} = dist2(sift_arr3{i}, codebook);
end

% index of the nearest codevector in the dictionary
for i = 1:Num_file3
    nearest_codevector3{i} = min(n2_3{i},[],2);
    [roww3{i} coll3{i}] = size(nearest_codevector3{i});
    for a3 = 1:roww3{i}
        [min_row3{i}(a3,1) min_col3{i}(a3,1)] = find(n2_3{i} == nearest_codevector3{i}(a3,1));
    end
end

% Image Representation
H3 = [];
for i = 1:Num_file3
        H3{i} = hist(min_col3{i},1:500);
end
HIST3 = [];
SET3 = 0;
for i = 1:Num_file3
    for j = 1:500 % 500 clusters
        HIST3(i,j) = H3{i}(1,j);
    end
end
% Normalized the Histogram
A3 = 1/norm(HIST3);
B3 = HIST3 * A3;

%%%%%%%%%%%
% Testing
%%%%%%%%%%%
% norm_hist3 = [];
% set1_3 = 0;
% for i = 1:Num_file3
%     for j = 1:size(min_col3{i})
%         norm_hist3(j+set1_3,1) = min_col3{1,i}(j,1);        
%     end;
%     [set1_3 set2_3] = size(norm_hist3);
% end
% h3 = norm_hist3;
% figure(3)
% hist(h3,1:500);

%% None

for  i = 1:Num_file4
    n2_4{i} = dist2(sift_arr4{i}, codebook);
end

% index of the nearest codevector in the dictionary
for i = 1:Num_file4
    nearest_codevector4{i} = min(n2_4{i},[],2);
    [roww4{i} coll4{i}] = size(nearest_codevector4{i});
    for a4 = 1:roww4{i}
        [min_row4{i}(a4,1) min_col4{i}(a4,1)] = find(n2_4{i} == nearest_codevector4{i}(a4,1));
    end
end

% Image Representation
H4 = [];
for i = 1:Num_file4
        H4{i} = hist(min_col4{i},1:500);
end
HIST4 = [];
SET4 = 0;
for i = 1:Num_file4
    for j = 1:500 % 500 clusters
        HIST4(i,j) = H4{i}(1,j);
    end
end
% Normalized the Histogram
A4 = 1/norm(HIST4);
B4 = HIST4 * A4;

%%%%%%%%%%
% Testing
%%%%%%%%%%
% norm_hist4 = [];
% set1_4 = 0;
% for i = 1:Num_file4
%     %figure(i)
%     %hist(min_col4{i})
%     for j = 1:size(min_col4{i})
%         norm_hist4(j+set1_4,1) = min_col4{1,i}(j,1);        
%     end;
%     [set1_4 set2_4] = size(norm_hist4);
% end
% h4 = hist(norm_hist4,1:500);
% figure(4)
% hist(h4,1:500);

%%
%%
%% Validation and TEST 
%% Change the name in dirname_v to Test or Validation to get different results
%%
%%
%% Following steps are same as Train

dirname_v='D:\Final\Sung-Yuan_Chen_FinalProject\Test\bike';
files_v=dir([dirname_v,'\*.bmp']);
Num_file_v=numel(files_v);
data_v=cell(1,Num_file_v);


for k = 1:Num_file_v    
    data_v{k} = imread([dirname_v '\' files_v(k).name]);
end

for i = 1:Num_file_v
    image_v{i} = rgb2gray(data_v{i});
    [cim_v{i}, r_v{i}, c_v{i}] = harris(image_v{i}, 2,500,2,1);
end

for i = 1:Num_file_v
    radius_v{i}(1:size(r_v{i}),1) = 2;
end

for i = 1:Num_file_v
   sift_arr_v{i} = find_sift(image_v{i},[c_v{i} r_v{i} radius_v{i}] ,3); 
end

for i = 1:Num_file_v
    [row_v{i} col_v{i}] = size(sift_arr_v{i});
end

% It uses the codebook from train to compute nearest codevector
for i = 1:Num_file_v
    n2_v{i} = dist2(sift_arr_v{i}, codebook);
end

% index of the nearest codevector in the dictionary
for i = 1:Num_file_v
    nearest_codevector_v{i} = min(n2_v{i},[],2);
    [roww_v{i} coll_v{i}] = size(nearest_codevector_v{i});
    for a = 1:roww_v{i}
        [min_row_v{i}(a,1) min_col_v{i}(a,1)] = find(n2_v{i} == nearest_codevector_v{i}(a,1));
    end
end

% Image Representation
H_V = [];
for i = 1:Num_file_v
        H_V{i} = hist(min_col_v{i},1:500);
end
HIST_V = [];
SET_V = 0;
for i = 1:Num_file_v
    for j = 1:500 % 500 clusters
        HIST_V(i,j) = H_V{i}(1,j);
    end
end
% Normalized the Histogram
A_V = 1/norm(HIST_V);
B_V = HIST_V * A_V;

%%%%%%%%%%%%
% Testing
%%%%%%%%%%%%
% norm_hist_v = [];
% set1_v = 0;
% for i = 1:Num_file_v
%     for j = 1:size(min_col_v{i})
%         norm_hist_v(j+set1_v,1) = min_col_v{1,i}(j,1);        
%     end;
%     [set1_v set2_v] = size(norm_hist_v);
% end
% h_v = norm_hist_v;
% figure(5)
% hist(h_v,1:500);


%% Cars

dirname_v2='D:\Final\Sung-Yuan_Chen_FinalProject\Test\cars';
files_v2=dir([dirname_v2,'\*.bmp']);
Num_file_v2=numel(files_v2);
data_v2=cell(1,Num_file_v2);

for k = 1:Num_file_v2    
    data_v2{k} = imread([dirname_v2 '\' files_v2(k).name]);
end

for i = 1:Num_file_v2
    image_v2{i} = rgb2gray(data_v2{i});
    [cim_v2{i}, r_v2{i}, c_v2{i}] = harris(image_v2{i}, 2,500,2,1);
end

for i = 1:Num_file_v2
    radius_v2{i}(1:size(r_v2{i}),1) = 2;
end

for i = 1:Num_file_v2
   sift_arr_v2{i} = find_sift(image_v2{i},[c_v2{i} r_v2{i} radius_v2{i}] ,3); 
end

for i = 1:Num_file_v2
    [row_v2{i} col_v2{i}] = size(sift_arr_v2{i});
end

for i = 1:Num_file_v2
    n2_v2{i} = dist2(sift_arr_v2{i}, codebook);
end

% index of the nearest codevector in the dictionary
for i = 1:Num_file_v2
    nearest_codevector_v2{i} = min(n2_v2{i},[],2);
    [roww_v2{i} coll_v2{i}] = size(nearest_codevector_v2{i});
    for a = 1:roww_v2{i}
        [min_row_v2{i}(a,1) min_col_v2{i}(a,1)] = find(n2_v2{i} == nearest_codevector_v2{i}(a,1));
    end
end

% Image Representation
H_V2 = [];
for i = 1:Num_file_v2
        H_V2{i} = hist(min_col_v2{i},1:500);
end
HIST_V2 = [];
SET_V2 = 0;
for i = 1:Num_file_v2
    for j = 1:500 % 500 clusters
        HIST_V2(i,j) = H_V2{i}(1,j);
    end
end
% Normalized the Histogram
A_V2 = 1/norm(HIST_V2);
B_V2 = HIST_V2 * A_V2;

%%%%%%%%%%%
% Testing
%%%%%%%%%%%
% norm_hist_v2 = [];
% set1_v2 = 0;
% for i = 1:Num_file_v2
%     for j = 1:size(min_col_v2{i})
%         norm_hist_v2(j+set1_v2,1) = min_col_v2{1,i}(j,1);        
%     end;
%     [set1_v2 set2_v2] = size(norm_hist_v2);
% end
% h_v2 = norm_hist_v2;
% figure(6)
% hist(h_v2,1:500);



%% Person

dirname_v3='D:\Final\Sung-Yuan_Chen_FinalProject\Test\person';
files_v3=dir([dirname_v3,'\*.bmp']);
Num_file_v3=numel(files_v3);
data_v3=cell(1,Num_file_v3);

for k = 1:Num_file_v3    
    data_v3{k} = imread([dirname_v3 '\' files_v3(k).name]);
end

for i = 1:Num_file_v3
    image_v3{i} = rgb2gray(data_v3{i});
    [cim_v3{i}, r_v3{i}, c_v3{i}] = harris(image_v3{i}, 2,500,2,1);
end

for i = 1:Num_file_v3
    radius_v3{i}(1:size(r_v3{i}),1) = 2;
end

for i = 1:Num_file_v3
   sift_arr_v3{i} = find_sift(image_v3{i},[c_v3{i} r_v3{i} radius_v3{i}] ,3); 
end

for i = 1:Num_file_v3
    [row_v3{i} col_v3{i}] = size(sift_arr_v3{i});
end

for i = 1:Num_file_v3
    n2_v3{i} = dist2(sift_arr_v3{i}, codebook);
end

% index of the nearest codevector in the dictionary
for i = 1:Num_file_v3
    nearest_codevector_v3{i} = min(n2_v3{i},[],2);
    [roww_v3{i} coll_v3{i}] = size(nearest_codevector_v3{i});
    for a = 1:roww_v3{i}
        [min_row_v3{i}(a,1) min_col_v3{i}(a,1)] = find(n2_v3{i} == nearest_codevector_v3{i}(a,1));
    end
end

% Image Representation
H_V3 = [];
for i = 1:Num_file_v3
        H_V3{i} = hist(min_col_v3{i},1:500);
end
HIST_V3 = [];
SET_V3 = 0;
for i = 1:Num_file_v3
    for j = 1:500 % 500 clusters
        HIST_V3(i,j) = H_V3{i}(1,j);
    end
end
% Normalized the Histogram
A_V3 = 1/norm(HIST_V3);
B_V3 = HIST_V3 * A_V3;

%%%%%%%%%
% Testing
%%%%%%%%%
% norm_hist_v3 = [];
% set1_v3 = 0;
% for i = 1:Num_file_v3
%     for j = 1:size(min_col_v3{i})
%         norm_hist_v3(j+set1_v3,1) = min_col_v3{1,i}(j,1);        
%     end;
%     [set1_v3 set2_v3] = size(norm_hist_v3);
% end
% h_v3 = norm_hist_v3;
% figure(7)
% hist(h_v3,1:500);

%% None

dirname_v4='D:\Final\Sung-Yuan_Chen_FinalProject\Test\none';
files_v4=dir([dirname_v4,'\*.bmp']);
Num_file_v4=numel(files_v4);
data_v4=cell(1,Num_file_v4);

for k = 1:Num_file_v4    
    data_v4{k} = imread([dirname_v4 '\' files_v4(k).name]);
end

for i = 1:Num_file_v4
    image_v4{i} = rgb2gray(data_v4{i});
    [cim_v4{i}, r_v4{i}, c_v4{i}] = harris(image_v4{i}, 2,500,2,1);
end

for i = 1:Num_file_v4
    radius_v4{i}(1:size(r_v4{i}),1) = 2;
end

for i = 1:Num_file_v4
   sift_arr_v4{i} = find_sift(image_v4{i},[c_v4{i} r_v4{i} radius_v4{i}] ,3); 
end

for i = 1:Num_file_v4
    [row_v4{i} col_v4{i}] = size(sift_arr_v4{i});
end

for i = 1:Num_file_v4
    n2_v4{i} = dist2(sift_arr_v4{i}, codebook);
end

% index of the nearest codevector in the dictionary
for i = 1:Num_file_v4
    nearest_codevector_v4{i} = min(n2_v4{i},[],2);
    [roww_v4{i} coll_v4{i}] = size(nearest_codevector_v4{i});
    for a = 1:roww_v4{i}
        [min_row_v4{i}(a,1) min_col_v4{i}(a,1)] = find(n2_v4{i} == nearest_codevector_v4{i}(a,1));
    end
end

% Image Representation
H_V4 = [];
for i = 1:Num_file_v4
        H_V4{i} = hist(min_col_v4{i},1:500);
end
HIST_V4 = [];
SET_V4 = 0;
for i = 1:Num_file_v4
    for j = 1:500 % 500 clusters
        HIST_V4(i,j) = H_V4{i}(1,j);
    end
end
% Normalized the Histogram
A_V4 = 1/norm(HIST_V4);
B_V4 = HIST_V4 * A_V4;
        
%%%%%%%%%%
% Testing
%%%%%%%%%%
% norm_hist_v4 = [];
% set1_v4 = 0;
% for i = 1:Num_file_v4
%     for j = 1:size(min_col_v4{i})
%         norm_hist_v4(j+set1_v4,1) = min_col_v4{1,i}(j,1);        
%     end;
%     [set1_v4 set2_v4] = size(norm_hist_v4);
% end
% h_v4 = norm_hist_v4;
% figure(8)
% hist(h_v4,1:500);

%% Classifier Training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Real Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Put all image representation in Validation or Test into a matrix
OutputTest = [B_V;B_V2;B_V3;B_V4];
InputTrain = [B;B2;B3;B4];
% Grouping for bike, cars, person, and none
Group = [];
Group(1:size(B),1) = 1;
Group(size(B)+1:size(B2)+size(B),1) = 2;
Group(size(B)+size(B2)+1:size(B3)+size(B)+size(B2),1) = 3;
Group(size(B)+size(B2)+size(B3)+1:size(B4)+size(B)+size(B2)+size(B3),1) = 4;

Class = knnclassify(OutputTest,InputTrain,Group,6,'cosine');

% Grouping for Test images
Group2 = [];
Group2(1:size(B_V),1) = 1;
Group2(size(B_V)+1:size(B_V2)+size(B_V),1) = 2;
Group2(size(B_V)+size(B_V2)+1:size(B_V3)+size(B_V)+size(B_V2),1) = 3;
Group2(size(B_V)+size(B_V2)+size(B_V3)+1:size(B_V4)+size(B_V)+size(B_V2)+size(B_V3),1) = 4;


% Confusion Matrix
ROC_M = [Class,Group2];

% Compute the accuray
TFPNR = zeros(size(ROC_M),1);
for i = 1:size(ROC_M)
    if ROC_M(i,1) == ROC_M(i,2)
        TFPNR(i,1) = 0;
    else
    TFPNR(i,1) = 1;
    end
end

% Plot the ROC curve
% Notice: This is just for testing, result is not guaranteed.
% ROC_M2 = [Group,TFPNR];
% ROCout=roc(ROC_M2);

% Acuuray rate
Res1 = size(find(TFPNR(:,1) == 1)); % Numbers of misclassified
Res2 = size(find(TFPNR(:,1) == 0)); % Numbers of accurate
Accuracy = Res2/(Res1+Res2); 

% Confusion Matrix
Confusion_Matrix_Test = [Group2,Class,TFPNR];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Notice: Following Step is for testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Put all image representation in Validation or Test into a matrix
% OutputTest = [h_v;h_v2;h_v3;h_v4];
% 
% % Normalized the image representation for each category in Train
% t = 1/norm(h);
% tr = h * t;
% t2 = 1/norm(h2);
% tr2 = h2*t2;
% t3 = 1/norm(h3);
% tr3 = h3*t3;
% t4 = 1/norm(h4);
% tr4 = h4*t4;
% 
% % Find the mean value of normalized data for each category in Train
% tv = mean(tr);
% tv2 = mean(tr2);
% tv3 = mean(tr3);
% tv4 = mean(tr4);
% 
% % Define group. 1: Bike     2: Cars    3. Person     4. None
% TG = [1;2;3;4];
% % Multiply 100000 to each mean value in the category of Train
% TF = [tv*100000;tv2*100000;tv3*100000;tv4*100000];
% %% Final Result
% Class = knnclassify(OutputTest,TF,TG);

% Define every image representation to right category
% Group2 = [];
% Group2(1:size(h_v),1) = 1;
% Group2(size(h_v)+1:size(h_v2)+size(h_v),1) = 2;
% Group2(size(h_v)+size(h_v2)+1:size(h_v3)+size(h_v)+size(h_v2),1) = 3;
% Group2(size(h_v)+size(h_v2)+size(h_v3)+1:size(h_v4)+size(h_v)+size(h_v2)+size(h_v3),1) = 4;
% This matrix can compare the actual category and classified category 
% ROC_M = [Class,Group2];

% Compute the accuray
% TFPNR = zeros(size(ROC_M),1);
% for i = 1:size(ROC_M)
%     if ROC_M(i,1) == ROC_M(i,2)
%         TFPNR(i,1) = 0;
%     else
%     TFPNR(i,1) = 1;
%     end
% end
% 
% % Plot the ROC curve
% % Notice: This is just for testing, result is not guaranteed.
% ROC_M2 = [Group2,TFPNR];
% ROCout=roc(ROC_M2);
% 
% % Acuuray rate
% Res1 = size(find(TFPNR(:,1) == 1)); % Numbers of misclassified
% Res1 = Res1/400; % Numbers of image misclassified
% Res2 = size(find(TFPNR(:,1) == 0)); % Numbers of accurate
% Res2 = Res2/400; % Numbers of image accurate
% Accuracy = Res2/(Res1+Res2); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% Archive
% [row_out col_out] = size(OutputTrain);
% Group = [];
% Group(1:size(h),1) = 1;
% Group(size(h)+1:size(h2)+size(h),1) = 2;
% Group(size(h)+size(h2)+1:size(h3)+size(h)+size(h2),1) = 3;
% Group(size(h)+size(h2)+size(h3)+1:size(h4)+size(h)+size(h2)+size(h3),1) = 4;
% 
% Class = knnclassify(OutputTrain, OutputTest, Group);

% Names1 = cell(size(h_v),1);
%  for i=1:size(h_v)
% Names1{i} = ['Bike'];
% end
% Names2 = cell(size(h_v2),1);
%  for i=1:size(h_v2)
% Names2{i} = ['Cars'];
% end
% 
%  Names3 = cell(size(h_v3),1);
%  for i=1:size(h_v3)
%  Names3{i} = ['Person'];
% end
% Names4 = cell(size(h_v4),1);
% for i=1:size(h_v4)
% Names4{i} = ['None'];
% end

% mdl1 = ClassificationKNN.fit(h_v,Names1);
% mdl2 = ClassificationKNN.fit(h_v2,Names2);
% mdl3 = ClassificationKNN.fit(h_v3,Names3);
% mdl4 = ClassificationKNN.fit(h_v4,Names4);

% Names1 = cell(size(h),1);
% for i=1:size(h)/2
% Names1{i} = ['Bike'];
% end
% for (i=size(h)/2)+1:size(h)
% Names1{i} = ['NotBike'];
% end
% Names2 = cell(size(h2),1);
% for i=1:size(h2)
% Names2{i} = ['Cars'];
% end
% for i=(size(h2)/2)+1:size(h2)
% Names2{i} = ['NotCars'];
% end
% Names3 = cell(size(h3),1);
% for i=1:size(h3)
% Names3{i} = ['Person'];
% end
% for i=(size(h3)/2)+1:size(h3)
% Names3{i} = ['NotPerson'];
% end
% Names4 = cell(size(h4),1);
% for i=1:size(h)
% Names4{i} = ['None'];
% end
% for i=(size(h4)/2)+1:size(h4)
% Names4{i} = ['NotNone'];
% end

% Group1(1:size(h),1) = 'Bike';
% Group2(1:size(h2,1) = 'Cars';
% Group3(1:size(h3,1) = 'Person';
% Group4(1:size(h4,1) = 'None';

% Group1 = zeros(size(h),1);
% Group2 = zeros(size(h2),1);
% Group3 = zeros(size(h3),1);
% Group4 = zeros(size(h4),1);
% 
% Group1(1:size(h)/2,1) = 1;
% Group2(1:size(h2)/2,1) = 1;
% Group3(1:size(h3)/2,1) = 1;
% Group4(1:size(h4)/2,1) = 1;

% Group1(1:size(h),1) = 1;
% Group2(1:size(h2),1) = 2;
% Group3(1:size(h3),1) = 3;
% Group4(1:size(h4),1) = 4;

% SVMStruct = svmtrain(h,Group1,'kernel_function','linear');
% SVMStruct2 = svmtrain(h2,Group2,'kernel_function','linear');
% SVMStruct3 = svmtrain(h3,Group3,'kernel_function','linear');
% SVMStruct4 = svmtrain(h4,Group4,'kernel_function','linear');

% SVMStruct = svmtrain(h,Group1);
% SVMStruct2 = svmtrain(h2,Group2);
% SVMStruct3 = svmtrain(h3,Group3);
% SVMStruct4 = svmtrain(h4,Group4);
% 
% Group1a = svmclassify(SVMStruct,h_v);
% Group1b = svmclassify(SVMStruct,h_v2);
% Group1c = svmclassify(SVMStruct,h_v3);
% Group1d = svmclassify(SVMStruct,h_v4);
% 
% Group2a = svmclassify(SVMStruct2,h_v);
% Group2b = svmclassify(SVMStruct2,h_v2);
% Group2c = svmclassify(SVMStruct2,h_v3);
% Group2d = svmclassify(SVMStruct2,h_v4);
% 
% Group3a = svmclassify(SVMStruct3,h_v);
% Group3b = svmclassify(SVMStruct3,h_v2);
% Group3c = svmclassify(SVMStruct3,h_v3);
% Group3d = svmclassify(SVMStruct3,h_v4);
% 
% Group4a = svmclassify(SVMStruct4,h_v);
% Group4b = svmclassify(SVMStruct4,h_v2);
% Group4c = svmclassify(SVMStruct4,h_v3);
% Group4d = svmclassify(SVMStruct4,h_v4);

