function [data, local_n, noise_data, Y] = data_generation(pms)

data = cell(pms.worker_num,1);
Y=[];

if pms.data_type == 0 % synthesize data
    thres = 200;
    beta =10;
    for iter = 1: pms.worker_num
        local_n(iter) = randi(50) + 50;
    end
    % local_n = ones(1, pms.worker_num)*pms.n/pms.worker_num;
    V_gt = cell(pms.worker_num,1);
    % V_gt{1} = orth([randn(local_n(1), local_n(1))]);
    for iter = 1: pms.worker_num
        V_gt{iter} =orth([randn(local_n(iter), local_n(iter))]);
    end
    U_gt = orth(randn(pms.m, pms.m));
    U_gt2 = orth(randn(pms.m, pms.m));
    
    sigma_gt = [diag(sort([thres; thres; thres/4; beta*randn(pms.m - 3,1)],'descend'))];
    
    idx = 1;
    pms.n = sum(local_n);
    
    for iter = 1: pms.n
        tmp = rand(pms.m, 1);
        tmp = tmp/norm(tmp);
        ss = sum(local_n(1:idx));
        if iter > ss
            idx = idx + 1;
        end
        if idx == 1
            data{idx}(:,iter) = (U_gt+U_gt2)*sigma_gt*tmp;
        else
            s = iter - sum(local_n(1:idx-1));
            data{idx}(:, s) = (U_gt+U_gt2)*sigma_gt*tmp;
        end
    end
elseif pms.data_type == 1 %real data
    %     load('C:\Users\sunchengjin\Desktop\hf\dataset\pca_dataset\2_ionosphere.mat')
    %     load('C:\Users\sunchengjin\Desktop\hf\dataset\pca_dataset\1_wbc.mat')
    %     load('C:\Users\sunchengjin\Desktop\hf\dataset\pca_dataset\4_cardio.mat')
    %% usps
    %     load('C:\Users\sunchengjin\Desktop\hf\dataset\usps\usps_data.mat')
    %
    %     while  1
    %         digit = randi(10,[2,1]);
    %         if digit(1) ~= digit(2)
    %             break;
    %         end
    %     end
    %     X=[usps(:,:, digit(1))  usps(:,:,digit(2))]';
    %     Y = [zeros(length(usps(:,:, 1)),1); ones(length(usps(:,:, 2)),1)];
    %% mnist
    load('mnist_all.mat')
    num_tmp  = pms.worker_num*280/2;
    X = [train0(1:num_tmp,:); train9(1:num_tmp,:)];    
    Y = [zeros(num_tmp,1); ones(num_tmp,1)];
%     Y = [zeros(length(test0),1); ones(length(test9),1)];
    
    
    [pms.n, pms.m] = size(X);
    % random
    idx_rand = randperm(pms.n);
    X = X(idx_rand,:);
    Y = Y(idx_rand);
    
    local_n = floor(pms.n/pms.worker_num)*ones(pms.worker_num,1);
    local_n(end) = pms.n - sum(local_n) + local_n(end);
    idx = 1;
    for iter = 1: pms.worker_num
        data{iter} = double(X(idx:idx+local_n(iter) - 1, :)');
        idx = idx + local_n(iter);
    end
end

noise_data = cell(pms.worker_num,1);
for iter = 1: pms.worker_num
    noise_data{iter} = data{iter};
    for fea_iter = 1: pms.m
        noise_data{iter}(fea_iter, :)  =  noise_data{iter}(fea_iter, :) + pms.noise_level*var(data{iter}(fea_iter, :))*rand(size(data{iter}(1,:))) ;
    end
end

