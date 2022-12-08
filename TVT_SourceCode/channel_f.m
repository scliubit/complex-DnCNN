function [H, H_At, H_Ar, H_D, An_t, An_r] = channel_f(Nt, Nr, K, N_carrier,BW, ch)
% generate channel model (ULA, UPA) in narrow band: H = H_Ar * H_D * H_At
% channel's basic information: ch{.fc, .Nc, .Np, .max_BS, .max_MS, .sigma, .lambda, .d}
% BS&MS_Path{.elevation, .azimuth, .gain}
% H_At/H_Ar:  AoDs/AoAs steering vector
% H_D: path gain
% An_t/An_r:  AoDs/AoAs of paths
% N:  the path of strongest gain
% H[a b c d]; a���Ӿ�����ά�ȣ�b���Ӿ����ά�ȣ�c�����ز�������d���û�����
%% Set parameters
if nargin <= 6
    ch.fc = 28e9; % Ƶ��
    ch.Nc = 6;    % ��  8
    ch.Np = 1;   % ÿ�����е�·������  10
    ch.max_BS = pi; % BS�Ƕȷ�Χ
    % ch.max_BS1 = pi * 2 / 3; % BS�Ƕȷ�Χ
    ch.max_BS1 = ch.max_BS; % BS�Ƕȷ�Χ
    ch.max_MS = pi; % users�Ƕȷ�Χ
    ch.sigma = 7.5;     % 7.5
    ch.lambda = 3e8 / ch.fc; % length of carrier wave
    ch.d = ch.lambda / 2;    % antenna spacing
    
    ch.fs=BW;
    ch.tau_max = (N_carrier / 2) / ch.fs;  % ·�����ʱ��
end

angle = ch.sigma * pi / 180;%��׼���������ļ���Ƕ�
k = 2 * pi * ch.d / ch.lambda;
total_Np = ch.Nc * ch.Np;
%%
switch (length(Nr) + length(Nt))
    case 2  % creat mmWave channel(ULA)
        H = zeros(Nr, Nt, N_carrier, K);
        H_At = zeros(Nt, total_Np, K);
        H_Ar = zeros(Nr, total_Np, K);
        H_D = zeros(total_Np, total_Np, K);
        for i = 1 : K
            N_t = (0 : (Nt - 1))';    % transmiter
            phi = (ch.max_BS - angle) * rand(ch.Nc, 1) * ones(1, ch.Np) + ...
                angle * (rand(ch.Nc, ch.Np) - 0.5 * ones(ch.Nc, ch.Np)) - ch.max_BS / 2 + angle / 2;
            %             phi = sort(phi, 2); % �����Ķ���ʵ�� 18/08/18
            
            phi = k * sin(reshape(phi',[1, total_Np]));
            a_BS = N_t * phi;
            At = exp(1i * a_BS) / sqrt(Nt);
            
            N_r = (0 : (Nr - 1))';    % receiver
            theta = (ch.max_MS - angle) * rand(ch.Nc, 1) * ones(1,ch. Np) + ...
                angle * (rand(ch.Nc, ch.Np) - 0.5 * ones(ch.Nc, ch.Np)) - ch.max_MS / 2 + angle / 2;
            %             theta = sort(theta, 2); % �����Ķ���ʵ�� 18/08/18
            
            theta = k * (reshape(theta',[1, total_Np]));
            a_MS = N_r * theta;
            Ar = exp(1i * a_MS) / sqrt(Nr);
            
            
            %----------------------------�����Ķ���ʵ�� 18/08/18 ------------------------------------
            %             D_am = (randn(1, ch.Nc) + 1i * randn(1, ch.Nc)) / sqrt(2);
            %             D_am = sort(D_am, 'descend');
            %             D_am = reshape(ones(ch.Np, 1) * D_am, 1, total_Np);
            %
            %             tau = ch.tau_max .* rand(1, ch.Nc);
            %             tau = sort(tau);
            %             tau = reshape(ones(ch.Np, 1) * tau, 1, total_Np);
            %-------------------------------------end-----------------------------------------------
            D_am = (randn(1, total_Np) + 1i * randn(1, total_Np)) / sqrt(2); % Path gain
            tau = ch.tau_max .* rand(1, total_Np);
            
            %             D_am = sort(D_am);
            %             tau = sort(tau, 'descend');
            for ii = 1 : ch.Nc
                D_am(((ii - 1) * ch.Np + 1) : (ii * ch.Np)) = sort(D_am(((ii - 1) * ch.Np + 1) : (ii * ch.Np)));
                tau(((ii - 1) * ch.Np + 1) : (ii * ch.Np)) = sort(tau(((ii - 1) * ch.Np + 1) : (ii * ch.Np)), 'descend');
            end
            %----------------------------------------------------------------------------------------
            
            miu_tau = -2 * pi * ch.fs * tau / N_carrier;
            
            for ii = 1 : N_carrier
                D = diag(D_am .* exp(1i * (ii - 1) * miu_tau)) * sqrt(Nt * Nr / total_Np);
                H(:, :, ii, i) = Ar * D * At';
            end
            H_At(:, :, i) = At;
            H_Ar(:, :, i) = Ar;
        end
    case 3 % BS(UPA) User(ULA)
        total_Nt = Nt(1) * Nt(2); total_Nr = Nr;
        H = zeros(total_Nr, total_Nt, N_carrier, K);
        H_At = zeros(total_Nt, total_Np, N_carrier, K);
        H_Ar = zeros(total_Nr, total_Np, N_carrier, K);
        H_D = zeros(total_Np, ch.Nc * ch.Np, N_carrier, K);
        An_t = zeros(1, ch.Nc * ch.Np, K);
        An_r = zeros(1, ch.Nc * ch.Np, K);
        for i = 1 : K
            n_t = (0 : Nt(1) - 1)';     % transmitter
            m_t = (0 : Nt(2) - 1)';
            phi1 = (ch.max_BS - angle) * rand(ch.Nc, 1) * ones(1, ch.Np) +...
                angle * (rand(ch.Nc, ch.Np) - 0.5 * ones(ch.Nc, ch.Np)) - ch.max_BS / 2 + angle / 2; % azimuth
            
            phi1 = reshape(phi1', [total_Np, 1]);
            theta1 = (ch.max_BS1 - angle) * rand(ch.Nc, 1) * ones(1, ch.Np) +...
                angle * (rand(ch.Nc, ch.Np) - 0.5 * ones(ch.Nc, ch.Np)) - ch.max_BS1 / 2 + angle / 2; % elevation
            
            theta1 = reshape(theta1', [total_Np, 1]);
            An_t(:, :, i) = theta1.';
            An_r(:, :, i) = phi1.';   %->��ȡ�Ƕ�
            A_t = zeros(total_Nt, total_Np);
            for path = 1 : total_Np
                e_a1 = exp(-1i * k * sin(phi1(path, 1)) * cos(theta1(path, 1)) * n_t); % channel model 2
                e_e1 = exp(-1i * k * sin(theta1(path, 1)) * m_t);
                A_t(:, path) = kron(e_a1, e_e1) / sqrt(total_Nt);
            end
            
            %             N_r = (0 : (total_Nr - 1))';    % receiver
            %             theta = (ch.max_MS - angle) * rand(ch.Nc, 1) * ones(1,ch. Np) + ...
            %                 angle * (rand(ch.Nc, ch.Np) - 0.5 * ones(ch.Nc, ch.Np)) - ch.max_MS / 2 + angle / 2;
            %
            %             theta = k * (reshape(theta',[1, total_Np]));
            %             a_MS = N_r * theta;
            %             A_r = exp(1i * a_MS) / sqrt(total_Nr);
            
            A_r = zeros(total_Nr, total_Np);
            for ii = 1 : total_Nr
                theta = (ch.max_MS - angle) * rand(ch.Nc, 1) * ones(1,ch. Np) + ...
                    angle * (rand(ch.Nc, ch.Np) - 0.5 * ones(ch.Nc, ch.Np)) - ch.max_MS / 2 + angle / 2;
                theta = k * (reshape(theta',[1, total_Np]));
                A_r(ii, :) = exp(1i * theta) / sqrt(total_Nr);
            end
            
            
            D_am = (randn(1, total_Np) + 1i * randn(1, total_Np)) / sqrt(2); % Path gain
            tau = ch.tau_max .* rand(1, total_Np);
            
            %             D_am = sort(D_am);
            %             tau = sort(tau, 'descend');
            for ii = 1 : ch.Nc
                D_am(((ii - 1) * ch.Np + 1) : (ii * ch.Np)) = sort(D_am(((ii - 1) * ch.Np + 1) : (ii * ch.Np)));
                tau(((ii - 1) * ch.Np + 1) : (ii * ch.Np)) = sort(tau(((ii - 1) * ch.Np + 1) : (ii * ch.Np)), 'descend');
            end
            
            miu_tau = -2 * pi * ch.fs * tau / N_carrier;
            
            for ii = 1 : N_carrier
                D = diag(D_am .* exp(1i * (ii - 1) * miu_tau)) * sqrt(total_Nt * total_Nr / total_Np);
                H(:, :, ii, i) = A_r * D * A_t';
                H_D(:, :, ii, i) = D;
            end
            
            H_At(:, :, i) = A_t;
            H_Ar(:, :, i) = A_r;
        end
    case 4  % creat mmWave channel(UPA)
        total_Nt = Nt(1) * Nt(2); 
        total_Nr = Nr(1) * Nr(2);
        H = zeros(total_Nr, total_Nt, N_carrier, K);
        H_At = zeros(total_Nt, total_Np, N_carrier, K);
        H_Ar = zeros(total_Nr, total_Np, N_carrier, K);
        H_D = zeros(total_Np, ch.Nc * ch.Np, N_carrier, K);
        An_t = zeros(1, ch.Nc * ch.Np, K);
        An_r = zeros(1, ch.Nc * ch.Np, K);
        for i = 1 : K
            n_t = (0 : Nt(1) - 1)';     % transmitter
            m_t = (0 : Nt(2) - 1)';
            phi1 = (ch.max_BS - angle) * rand(ch.Nc, 1) * ones(1, ch.Np) +...
                angle * (rand(ch.Nc, ch.Np) - 0.5 * ones(ch.Nc, ch.Np)) - ch.max_BS / 2 + angle / 2; % azimuth
            
            phi1 = reshape(phi1', [total_Np, 1]);
            theta1 = (ch.max_BS1 - angle) * rand(ch.Nc, 1) * ones(1, ch.Np) +...
                angle * (rand(ch.Nc, ch.Np) - 0.5 * ones(ch.Nc, ch.Np)) - ch.max_BS1 / 2 + angle / 2; % elevation
            
            theta1 = reshape(theta1', [total_Np, 1]);
            An_t(:, :, i) = theta1.';
            An_r(:, :, i) = phi1.';   %->��ȡ�Ƕ�
            A_t = zeros(total_Nt, total_Np);
            for path = 1 : total_Np
                e_a1 = exp(-1i * k * sin(phi1(path, 1)) * cos(theta1(path, 1)) * n_t); % channel model 2
                e_e1 = exp(-1i * k * sin(theta1(path, 1)) * m_t);
                A_t(:, path) = kron(e_a1, e_e1) / sqrt(total_Nt);
            end
            
            n_r = (0:(Nr(1) - 1))';     % receiver
            m_r = (0:(Nr(2) - 1))';
            phi2 = (ch.max_MS - angle) * rand(ch.Nc, 1) * ones(1, ch.Np) +...
                angle * (rand(ch.Nc, ch.Np) - 0.5 * ones(ch.Nc, ch.Np)) - ch.max_MS / 2 + angle / 2; % azimuth
            
            phi2 = reshape(phi2', [total_Np, 1]);
            theta2 = (ch.max_MS - angle) * rand(ch.Nc, 1) * ones(1, ch.Np) +...
                angle * (rand(ch.Nc, ch.Np) - 0.5 * ones(ch.Nc, ch.Np)) - ch.max_MS / 2 + angle / 2; % elevation
            
            theta2 = reshape(theta2', [total_Np, 1]);
            A_r = zeros(total_Nr, total_Np);
            for path = 1 : total_Np
                e_a2 = exp(-1i * k * sin(phi2(path, 1)) * cos(theta2(path, 1)) * n_r); % channel model 2
                e_e2 = exp(-1i * k * sin(theta2(path, 1)) * m_r);
                A_r(:,path) = kron(e_a2,e_e2) / sqrt(total_Nr);
            end
            
            D_am = (randn(1, total_Np) + 1i * randn(1, total_Np)) / sqrt(2); % Path gain
            tau = ch.tau_max .* rand(1, total_Np);
            
            %             D_am = sort(D_am);
            %             tau = sort(tau, 'descend');
            for ii = 1 : ch.Nc
                D_am(((ii - 1) * ch.Np + 1) : (ii * ch.Np)) = sort(D_am(((ii - 1) * ch.Np + 1) : (ii * ch.Np)));
                tau(((ii - 1) * ch.Np + 1) : (ii * ch.Np)) = sort(tau(((ii - 1) * ch.Np + 1) : (ii * ch.Np)), 'descend');
            end
            
            miu_tau = -2 * pi * ch.fs * tau / N_carrier;
            
            for ii = 1 : N_carrier
                D = diag(D_am .* exp(1i * (ii - 1) * miu_tau)) * sqrt(total_Nt * total_Nr / total_Np);
                H(:, :, ii, i) = A_r * D * A_t';
                H_D(:, :, ii, i) = D;
            end
            
            H_At(:, :, i) = A_t;
            H_Ar(:, :, i) = A_r;
        end
    otherwise, error('error in data structure: Nr or Nt');
end
