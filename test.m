clc; clear all; close all;
pms.m = 500;
variable_j = [20];
pms.k = 50; % rank of sigma
pms.target_k = 1;
pms.sigma = sqrt(pms.m)/0.01; %hyper-parameter of RBF
pms.noise_level = 0.0;
pms.data_type = 1;
max_repeat = 1;
test_result= [];
for j_iter = 1: length(variable_j)
    pms.worker_num = variable_j(j_iter);
    % ----------------compute the digragh from the undigraph--------------
    
    xi = cell(pms.worker_num,1);
    
    ee = eye(pms.worker_num);
    E = cell(pms.worker_num,1);
    nei_list = cell(pms.worker_num,1);
    adj_mat = eye(pms.worker_num);
    
    for iter = 1: pms.worker_num-1
        adj_mat(iter, iter +1) = 1;
        adj_mat(iter+1, iter) = 1;
        tmp = randi((pms.worker_num - iter), 1,1) + iter;
        adj_mat(iter, tmp) = ones(1, 1);
        adj_mat(tmp, iter) = ones(1, 1);
    end
    adj_mat(pms.worker_num, 1) = 1;
    adj_mat(1, pms.worker_num) = 1;
    for iter = 1: pms.worker_num
        nei_list{iter}= unique([iter find(adj_mat(iter, :)==1)], 'stable');
        
        for  nei_iter = 1:length(nei_list{iter})
            xi{iter} = [xi{iter} ee(:, nei_list{iter}(nei_iter))];
        end
        E{iter} = ones(1, length(nei_list{iter}));
    end
    
    for repeat = 1: max_repeat
        [ data, local_n, noise_data, label] = data_generation(pms);
        
        %% ------------compute the ground truth ------------------------
       
        data_total = [];
        pms.n = sum(local_n);
        for iter = 1: pms.worker_num
            data_total = [data_total data{iter}];
        end
        
        kernel_total = cell(pms.worker_num,1);
        alpha_gt = cell(pms.worker_num,1);
        for iter = 1: pms.worker_num
            kernel_total{iter} = cal_RBF(data{iter}, data_total, pms.sigma);
            kernel_total{iter} = centralize_kernel(kernel_total{iter});% centralization
%                                     [alpha_gt{iter}, ~, ~,~] = solve_global_svd(kernel_total{iter}*kernel_total{iter}', 1);% ground truth
        end
        
        tic
        kernel_tt = cal_RBF(data_total, data_total, pms.sigma);
        kernel_tt = centralize_kernel(kernel_tt);
        [alpha_ggt,~,~,~]= solve_global_svd(kernel_tt, 1);
        fprintf('time: %f s\n',toc);
%                         for iter = 1: pms.worker_num
%                             alpha_gt{iter}'*kernel_total{iter}*kernel_total{iter}'*alpha_gt{iter}/(alpha_gt{iter}'*kernel_mat{iter,iter,iter}*alpha_gt{iter})
%                             alpha_gt{iter} = kernel_inv{iter}*kernel_total{iter}*alpha_ggt;
%                             alpha_gt{iter}'*kernel_total{iter}*kernel_total{iter}'*alpha_gt{iter}/(alpha_gt{iter}'*kernel_mat{iter,iter,iter}*alpha_gt{iter})
%                         end
        
        % ----------------------gt: end-----------------
        
        %% distributed KPCA
        
        %% ---------------- data preparation --------------
        alpha_old = cell(pms.worker_num,1);
        eta_old = cell(pms.worker_num,1);
        eta = cell(pms.worker_num,1);
        alpha = cell(pms.worker_num,1);
        alpha_ini = cell(pms.worker_num,1);
        z_norm = ones(1, pms.worker_num);
        phi_z  = cell(pms.worker_num,pms.worker_num);
        phi_z_old  = cell(pms.worker_num,pms.worker_num);
        phi_z_xi = cell(pms.worker_num,1);
        ill_thres = 0.01;
        kernel_mat = cell(pms.worker_num, pms.worker_num, pms.worker_num);
        kernel_inv = cell(pms.worker_num, 1);
        for iter = 1: pms.worker_num
            eta{iter} = zeros(local_n(iter), size(xi{iter},2)*pms.target_k);
            for nei_iter = 1: length(nei_list{iter})
                nei_tmp1 =  nei_list{iter}(nei_iter);
                for nei_iter2 = nei_iter: length(nei_list{iter})
                    nei_tmp2 = nei_list{iter}(nei_iter2);
                    if nei_tmp1 == iter    && nei_tmp2 == iter
                        kernel_mat{nei_tmp1,nei_tmp2, iter} = cal_RBF(data{iter}, data{iter}, pms.sigma);
                    elseif nei_tmp1 == iter
                        % suppose the data from neighbor contain noise.
                        kernel_mat{nei_tmp1,nei_tmp2, iter} = cal_RBF(data{nei_tmp1}, noise_data{nei_tmp2}, pms.sigma);
                    elseif nei_tmp2 == iter
                        kernel_mat{nei_tmp1,nei_tmp2, iter} = cal_RBF(noise_data{nei_tmp1}, data{nei_tmp2}, pms.sigma);
                    else
                        kernel_mat{nei_tmp1,nei_tmp2, iter} = cal_RBF(noise_data{nei_tmp1}, noise_data{nei_tmp2}, pms.sigma);
                    end
                    [kernel_mat{nei_tmp1,nei_tmp2, iter}] = centralize_kernel(kernel_mat{nei_tmp1,nei_tmp2, iter}); % centralization
                    kernel_mat{nei_tmp2, nei_tmp1, iter} = kernel_mat{nei_tmp1,nei_tmp2, iter}';
                end
                phi_z_old{iter, nei_tmp1} = zeros(local_n(iter), 1);
            end
            [alpha_ini{iter}, ~, ~,ss] = solve_global_svd(kernel_mat{iter, iter, iter}, 1);% a  good initial point
            kernel_mat{iter,iter, iter} = kernel_mat{iter,iter,iter} + ill_thres*ss/local_n(iter)*ones(size(kernel_mat{iter,iter,iter}));
            % !!! directly take inv on K will lead to undesirable numerical problem
            [v, d] = eig(kernel_mat{iter,iter, iter});
            dd= diag(d);
            [idx_pos] = find(dd > 1e-3);
            dd(idx_pos) = 1./(dd(idx_pos));
            kernel_inv{iter} = v*diag(dd)*v';
            
            alpha_gt{iter} = kernel_inv{iter}*kernel_total{iter}*alpha_ggt;
        end
        
        kernel_nei =  cell(pms.worker_num,1);
        alpha_nei =  cell(pms.worker_num,1);
        for iter = 1: pms.worker_num
            kernel_nei{iter} = [];
            for ii = 1: length(nei_list{iter})
                kernel_nei{iter} = [kernel_nei{iter} kernel_mat{iter, nei_list{iter}(ii), iter}] ;
            end
            [alpha_nei{iter}, ~, ~, ~] = solve_global_svd(kernel_nei{iter}*kernel_nei{iter}', 1 );
        end
        
        % ---------------- data preparation: end --------------
        
        %% -------------------alpha initialization-----------------
        for iter = 1: pms.worker_num
            alpha{iter} = alpha_ini{iter};% initialization
        end
        % -------------------alpha initialization: end -----------------
        for iter = 1: pms.worker_num
            tmp2(iter) = alpha_ini{iter}'*kernel_total{iter}*kernel_total{iter}'*alpha_ini{iter}/(alpha_ini{iter}'*kernel_mat{iter,iter, iter}*alpha_ini{iter});
            tmp3(iter) = alpha_nei{iter}'*kernel_total{iter}*kernel_total{iter}'*alpha_nei{iter}/(alpha_nei{iter}'*kernel_mat{iter,iter, iter}*alpha_nei{iter});
            tmp5(iter) = alpha_ini{iter}'*alpha_gt{iter}/norm(alpha_ini{iter})/norm(alpha_gt{iter});
            tmp6(iter) = alpha_nei{iter}'*alpha_gt{iter}/norm(alpha_nei{iter})/norm(alpha_gt{iter});
        end
        %% ----debug: loss variables---------
        L_value = zeros(1500,1);
        obj_loss = zeros(1500,1);
        comm_loss = zeros(1500,1);
        stage = 1;
        cnt = 1;
        %-----debug: end----------------
        %
        rho =  cell(pms.worker_num,1);
        H_hat = cell(pms.worker_num,1);
        update_flag  = ones(pms.worker_num,1);
        UPDATE_THRES = 1e-3;
        STOP_FLAG = 1e-4;
        
        tic
        for ADMM_iter = 1: 50
            
            if stage == 1
                rho1 = 100;% Random initial: 100;
                rho2 = 1;% Random initial: 10;
            elseif stage == 2
                rho1 = 100;
                rho2 = 50;
            else
                rho1 = 100;
                rho2 = 100;
            end
            for iter = 1: pms.worker_num
                for nei_iter = 1: length(nei_list{iter})
                    nei_tmp1 = nei_list{iter}(nei_iter);
                    if nei_tmp1 == iter
                        rho{iter}(nei_iter) = rho1*local_n(nei_tmp1);%/pms.n;
                    else
                        rho{iter}(nei_iter) = rho2*local_n(nei_tmp1);%/pms.n;
                    end
                end
                H_hat{iter} = 1/(sum(rho{iter}));
            end
            
            alpha_flag = 0;
            z_flag = 0;
            %% ------------update phi' z---------------------
            
            for iter = 1: pms.worker_num
                
                if update_flag(iter) > UPDATE_THRES
                    for nei_iter = 1: length(nei_list{iter})
                        nei_tmp1 = nei_list{iter}(nei_iter);
                        phi_z{nei_tmp1, iter} = zeros(local_n(nei_tmp1), 1);
                        for nei_iter2 = 1: length(nei_list{iter})
                            nei_tmp2 = nei_list{iter}(nei_iter2);
                            idx_z = find(nei_list{nei_tmp2} == iter);
                            phi_z{nei_tmp1, iter}  = phi_z{nei_tmp1, iter}  + kernel_mat{nei_tmp1, nei_tmp2, iter}*...
                                (kernel_inv{nei_tmp2}*eta{nei_tmp2}(:,idx_z)+ rho{nei_tmp2}(idx_z)*alpha{nei_tmp2});
                        end
                        phi_z{nei_tmp1, iter} = H_hat{iter}*phi_z{nei_tmp1, iter};
                    end
                end
            end
            %------------ update phi' z: end ---------------------
            
            %% ------------ update z_norm ---------------------
            
            z_norm_old = z_norm;
            for iter = 1: pms.worker_num
                if update_flag(iter) > UPDATE_THRES
                    z_norm(iter) = 0;
                    % compute the norm
                    for nei_iter = 1: length(nei_list{iter})
                        nei_tmp1 = nei_list{iter}(nei_iter);
                        idx_z = find(nei_list{nei_tmp1} == iter);
                        inf_nei1 = (kernel_inv{nei_tmp1}*eta{nei_tmp1}(:,idx_z) + alpha{nei_tmp1}*rho{nei_tmp1}(idx_z))*H_hat{iter};
                        for nei_iter2 = 1: length(nei_list{iter})
                            nei_tmp2 = nei_list{iter}(nei_iter2);
                            idx_z = find(nei_list{nei_tmp2} == iter);
                            inf_nei2 = (kernel_inv{nei_tmp2}*eta{nei_tmp2}(:,idx_z) + alpha{nei_tmp2}*rho{nei_tmp2}(idx_z))*H_hat{iter};
                            z_norm(iter) = z_norm(iter) + inf_nei1'*kernel_mat{nei_tmp1, nei_tmp2, iter}*inf_nei2;
                        end
                    end
                    if z_norm(iter) < 1
                        z_norm(iter) = 1;
                    else
                        z_norm(iter) = sqrt(z_norm(iter));
                    end
                    % divide the norm.
                    for nei_iter = 1: length(nei_list{iter})
                        nei_tmp = nei_list{iter}(nei_iter);
                        phi_z{nei_tmp, iter} = phi_z{nei_tmp, iter}./z_norm(iter);
                    end
                end
            end
            
            % phi_j z_p should have the same direction as phi_j z_j.
            for iter = 1: pms.worker_num
                if update_flag(iter) > UPDATE_THRES
                    for nei_iter = 1: length(nei_list{iter})
                        nei_tmp = nei_list{iter}(nei_iter);
                        direction = phi_z{iter, nei_tmp}'*phi_z{iter, iter};
                        if direction<0
                            phi_z{iter, nei_tmp} = -phi_z{iter, nei_tmp};
                        end
                    end
                end
            end
            %------------ update z_norm : end ---------------------
            
            %% --------------- compute the convergence of z ---------
            delta_z = 0;
            for iter = 1: pms.worker_num
                for nei_iter = 1: length(nei_list{iter})
                    nei_tmp = nei_list{iter}(nei_iter);
                    delta_z = delta_z + norm(phi_z{iter, nei_tmp} - phi_z_old{iter, nei_tmp});
                end
            end
            if  delta_z < pms.worker_num*UPDATE_THRES % if z converges, then update eta.
                z_flag = 1;
            end
            phi_z_old =  phi_z;
            
            % --------------- convergence of z: end ----------------
            
            %% ------------ compute phi z xi --------------------
            for iter =  1: pms.worker_num
                phi_z_xi{iter}  = [phi_z{iter,iter}];
                for nei_iter = 1: length(nei_list{iter})
                    nei_tmp = nei_list{iter}(nei_iter);
                    if nei_tmp  ~= iter
                        phi_z_xi{iter} = [phi_z_xi{iter} phi_z{iter, nei_tmp}];
                    end
                end
            end
            % ------------------ phi_z_xi: end --------------------
            %% ------------ debug: compute the lagrange function value------------
            
            for iter = 1: pms.worker_num
                alpha_minus_z =(alpha{iter}*E{iter} - kernel_inv{iter}* phi_z_xi{iter} );
                obj_loss(cnt) = obj_loss(cnt) - norm(kernel_mat{iter,iter, iter}*alpha{iter}, 'fro')^2;
                comm_loss(cnt) = comm_loss(cnt) + trace(eta{iter}'*alpha_minus_z);
                comm_loss(cnt) = comm_loss(cnt) + 0.5*trace(alpha_minus_z'*alpha_minus_z*diag(rho{iter}));
            end
            L_value(cnt) = obj_loss(cnt) + comm_loss(cnt);
            cnt = cnt + 1;
            %------------debug: end-----------------------
            
            %% ------------update alpha---------------------
            delta_alpha = 0;
            for iter = 1: pms.worker_num
                alpha_old{iter} = alpha{iter};
                H_alpha = -2*kernel_mat{iter,iter,iter}*kernel_mat{iter,iter, iter} + E{iter}*diag(rho{iter})*E{iter}'*eye(local_n(iter));
                f_alpha = (kernel_inv{iter}*phi_z_xi{iter}*diag(rho{iter}) - eta{iter})*E{iter}';
                [v, d] = eig(H_alpha);
                d = diag(d);
                [idx_pos] = find(d > 1e-4);
                [idx_neg] = find(d < -1e-4);
                if idx_neg % if so, then the rho is too small
                    fprintf('rho is too small.\n');
                    return;
                end
                d(idx_pos) = 1./d(idx_pos);
                alpha{iter} = v*diag(d)*(v'*f_alpha);
                delta_alpha = delta_alpha + norm(alpha{iter} - alpha_old{iter});
            end
            if  delta_alpha < pms.worker_num*UPDATE_THRES % if alpha converges, then update eta.
                alpha_flag = 1;
            end
            
            %------------- update alpha: end ------------------
            %% ------------debug: compute the lagrange function value------------
            
            for iter = 1: pms.worker_num
                alpha_minus_z =(alpha{iter}*E{iter}- kernel_inv{iter}*phi_z_xi{iter});
                obj_loss(cnt) = obj_loss(cnt) - norm(kernel_mat{iter,iter, iter}*alpha{iter}, 'fro')^2;
                comm_loss(cnt) = comm_loss(cnt) + trace(eta{iter}'*alpha_minus_z);
                comm_loss(cnt) = comm_loss(cnt) + 0.5*trace(alpha_minus_z'*alpha_minus_z*diag(rho{iter}));
            end
            L_value(cnt) = obj_loss(cnt) + comm_loss(cnt);
            cnt = cnt + 1;
            %------------debug: end-----------------------
            
            %% ------------ update eta ---------------------
            if z_flag && alpha_flag
                eta_tmp=0;
                for iter = 1: pms.worker_num
                    eta_old{iter} = eta{iter};
                    eta{iter} = eta{iter}  + (alpha{iter}*E{iter} - kernel_inv{iter}*phi_z_xi{iter})*diag(rho{iter});
                    update_flag(iter) = sin(subspace(alpha{iter}*E{iter}, kernel_inv{iter}*phi_z_xi{iter}*diag(rho{iter})));
                    eta_tmp =  eta_tmp + norm(eta{iter});
                end
                if sum(update_flag) < pms.worker_num*STOP_FLAG
                    stage = stage + 1;
                    if stage == 4
                        break;
                    end
                end
%                 fprintf('%d\n',ADMM_iter, sum(update_flag));
                %----------- updata eta: end -----------------
            end
            %% ------------ compute the update flag of z -------------
            for iter = 1: pms.worker_num
                update_flag(iter) = sin(subspace(alpha{iter}*E{iter}, kernel_inv{iter}*phi_z_xi{iter}*diag(rho{iter})));
            end
            % ------------------- update flag of z: end --------------------
            %% ------------debug: compute the lagrange function value------------
            
            for iter = 1: pms.worker_num
                alpha_minus_z =(alpha{iter}*E{iter}- kernel_inv{iter}*phi_z_xi{iter});
                obj_loss(cnt) = obj_loss(cnt) - norm(kernel_mat{iter,iter, iter}*alpha{iter}, 'fro')^2;
                comm_loss(cnt) = comm_loss(cnt) + trace(eta{iter}'*alpha_minus_z);
                comm_loss(cnt) = comm_loss(cnt) + 0.5*trace(alpha_minus_z'*alpha_minus_z*diag(rho{iter}));
            end
            L_value(cnt) = obj_loss(cnt) + comm_loss(cnt);
            cnt = cnt + 1;
            %------------debug: end-----------------------
            
            %% test: compute the similarity with the ground truth.
            
            for iter = 1: pms.worker_num
                tmp1(iter) = alpha{iter}'*kernel_total{iter}*kernel_total{iter}'*alpha{iter}/(alpha{iter}'*kernel_mat{iter,iter, iter}*alpha{iter});
                tmp4(iter) = alpha{iter}'*alpha_gt{iter}/norm(alpha{iter})/norm(alpha_gt{iter});
            end
            test_result = [test_result sum(abs(tmp4))];
        end
        
        fprintf('time: %f s\n',toc);
    end
end
for iter = 1: pms.worker_num
    tmp1(iter) = alpha{iter}'*kernel_total{iter}*kernel_total{iter}'*alpha{iter}/(alpha{iter}'*kernel_mat{iter,iter, iter}*alpha{iter});
    tmp4(iter) = alpha{iter}'*alpha_gt{iter}/norm(alpha{iter})/norm(alpha_gt{iter});
end
test_result = [test_result sum(abs(tmp4))];
figure;
plot(test_result)
figure;
plot(L_value(1:cnt-1))
gt_value = alpha_ggt'*(kernel_tt*kernel_tt')*alpha_ggt/(alpha_ggt'*kernel_tt*alpha_ggt);

fprintf('Similarty: (our - initial)/gt: %.3d %%\n', 100*(sum(tmp1) - sum(tmp2))/(pms.worker_num*gt_value));
fprintf('Similarty: (our - neighbor)/gt: %.3d %%\n',100*(sum(tmp1) - sum(tmp3))/(pms.worker_num*gt_value));

