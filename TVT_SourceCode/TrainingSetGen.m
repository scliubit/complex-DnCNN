%% 1127 Generation of training data
% snr fix=10 15 20; channel 6 8; active array rate 20%;
% oversampling 2; MMV(32 carriers);
% 5dB 
% clear;
%% para
N_ms = [1 1];
% N_MS = N_ms(1)*N_ms(2);
N_MS = 1;
N_irs = [16 16];
N_IRS = N_irs(1)*N_irs(2);
% N_IRS = 64;
FFT_len = 256;
Nc = FFT_len; % sub carriers

BW = 90e6;
fs = BW;
fc = 28e9;          % frequency of carrier is 28GHz
lambda = 3e8/fc;
d_ant = lambda/2;   % interval of antennas
sigma_2_alpha = 1;  % variance of path gain
awgn_en = 1;        % 1: add noise; 0: no noise
N_bits = 3;         % number of quantized bits
N_Bits = 2^N_bits;
N_RF = 1;
Ns = N_RF;
% activeEleRateList=[0.01 0.02 0.05 0.1 0.2 0.5 1];
activeEleRateList=[0.25];
activeEleNumList=round(activeEleRateList*N_IRS);
identity_d = 4;   	% digital precoding/combining matrix is an identity matrix:  0(no); 1(yes); Make a reasonable assumption that Ns = N_RF;
oversampling = 2;   % dict
mode=1; % 0: phase shifters, 1: switchers

%% Training Set Generation setup
debug           =   1;
debugRate       =   0.75;
trainingSize    =   100;%12000-debug*debugRate*12000;
testSize        =   0;%4000-debug*debugRate*4000;
PNR_dBs         =   5;
iterMax         =   trainingSize+testSize;
NMSEvsActiveNum =   zeros(length(PNR_dBs),length(activeEleRateList));
% randomSelect    =   4;  % choose 4 per 4*32 channels=>batchsize should be n*4
trainingChannel =   zeros([trainingSize 2 N_IRS Nc]); % NCHW
testChannel     =   zeros([testSize 2 N_IRS Nc]);
trueTrainingChannel     =   zeros([trainingSize 2 N_IRS Nc]);
trueTestChannel         =   zeros([testSize 2 N_IRS Nc]);
%% dict over
N_2=N_MS;
m=(0:N_2-1).';
grid_M=N_2*oversampling;
virtual_ang_2=-1 : 2/grid_M : 1-2/grid_M;	% quantizing based on virtual angles for MS side
% virtual_ang_1=-pi/2:pi/grid_N:pi/2-pi/grid_N;	% quantizing based on virtual angles for MS side
% virtual_ang_2=-pi/2:pi/grid_M:pi/2-pi/grid_M;	% quantizing based on virtual angles for MS side

sub_N=N_irs(1);
sub=(0:sub_N-1).';
sub_grid_N=sub_N*oversampling;
sub_virtual_ang_1=-1 : 2/sub_grid_N : 1-2/sub_grid_N;
sub_N_tilde = exp(1i*pi*sub*sub_virtual_ang_1)/sqrt(sub_N); % IRS   U

sub_M=N_irs(2);
sub=(0:sub_M-1).';
sub_grid_M=sub_M*oversampling;
sub_virtual_ang_2=-1 : 2/sub_grid_M : 1-2/sub_grid_M;
sub_M_tilde = exp(1i*pi*sub*sub_virtual_ang_2)/sqrt(sub_M); % IRS   U

M_tilde = exp(1i*pi*m*virtual_ang_2)/sqrt(N_2); % MS    V
Psi=kron(M_tilde,kron(sub_M_tilde,sub_N_tilde));
Psi=kron(sub_M_tilde,sub_N_tilde);


%% sim start
for kk=1:length(activeEleNumList)
    activeEleNum = activeEleNumList(kk);
    M = activeEleNum;             % number of traning frames ( = number of time-slots ?)
    NMSE = zeros(2,length(PNR_dBs));
    for ii = 1:length(PNR_dBs)
        snr=PNR_dBs(ii);
        sigma2 = 10^(-(PNR_dBs(ii)/10));    % noise variance = epsilon
        sigma = sqrt(sigma2);
        for iter = 1:iterMax
            %% Channel
            % H_f = FSF_Channel_Model_uplink(N_ms, N_irs, fc, Lp, sigma_2_alpha, fs, K);
            H_f_o   = channel_f(N_ms,N_irs,1,FFT_len,BW);
            activeEleIndex  =   randperm(N_IRS);
            activeEleIndex  =   sort(activeEleIndex(1:activeEleNum));
            %% pilot Sig
            S_matrix = exp(-1i*2*pi*rand(Ns, activeEleNum));  % Consider M successive traning frames	sqrt(Ns)
            %% recv Sig
            % Generating combined received signal y_k_com
            y_k_com = zeros(activeEleNum*Ns,Nc);
            %% sensing Mat
            Phi = zeros(activeEleNum*Ns,N_IRS*N_MS);
            %% comm Proc
            for mm = 1:activeEleNum
                % Generating analog precoding/combining matrix, and quantized
                %% F(RF and BB)
                F_BB_m          = eye(N_RF,Ns);
                
                F_RF_m = exp(1j*2*pi*rand(N_MS,N_RF));
                %  F_RF_m = ones(N_MS,N_RF);
                F_RF_quan_phase_m = round((angle(F_RF_m)+pi)*N_Bits/(2*pi)) *2*pi/N_Bits;  %  - pi
                Quantized_F_RF_m = exp(1j*F_RF_quan_phase_m)/sqrt(N_MS);
                
                F_temp_m        = Quantized_F_RF_m*F_BB_m;
                norm_factor_m   = sqrt(N_RF)/norm(F_temp_m,'fro');	% The normalized factor of power. N_RF
                F_m             = norm_factor_m*F_temp_m;
                
                %% W(RF and BB)
                W_BB_m = exp(1j*2*pi*rand(N_RF,Ns));
                if mode==0
                    % 0: phase shifters
                    W_RF_m = exp(1j*2*pi*rand(N_IRS,N_RF));
                    W_RF_quan_phase_m = round((angle(W_RF_m)+pi)*N_Bits/(2*pi)) *2*pi/N_Bits;  %  - pi
                    Quantized_W_RF_m = exp(1j*W_RF_quan_phase_m)/sqrt(N_IRS);
                    W_m = Quantized_W_RF_m*W_BB_m;
                else
                    % 1: switchers
                    activeEle=activeEleIndex(mm);
                    IRSCombine=zeros(N_IRS,N_RF);
                    IRSCombine(activeEle)=1;
                    W_m=IRSCombine;
                end
                
                Phi((mm-1)*Ns+1:mm*Ns,:) = kron((F_m*S_matrix(:,mm)).',W_m');
                % signal transmission model
                for carrier=1:Nc
                    y_k_com((mm-1)*Ns+1:mm*Ns,carrier)=W_m'*H_f_o(:,:,carrier)*F_m*S_matrix(:,mm) + awgn_en*sigma*W_m'*(normrnd(0,1,N_IRS,1) + 1i*normrnd(0,1,N_IRS,1))/sqrt(2);
                end
                
            end
            
            %% OMP_Algorithm
            snreff=snr;
            epsilon = 10^(-(snreff/10));
            h_v_hat=zeros(N_IRS*N_MS,Nc);
            [ h_v_hat,iter_num ] = OMP_Algorithm_MMV( y_k_com,Phi,Psi,epsilon,eye(M),100);
            % return
            %% Reestablish the high-dimensional channel matrix based on estimated channal support and corresponding channel coefficients
            h_k_est_com = Psi*h_v_hat;
            H_f_est=reshape(h_k_est_com,[N_IRS,N_MS,Nc]);
            
            %% NMSE performance
            difference_channel_MSE = zeros(Nc,1);
            true_channel_MSE = zeros(Nc,1);
            for carrier = 1:Nc
                difference_channel_MSE(carrier) = norm(H_f_est(:,:,carrier) - H_f_o(:,:,carrier),'fro')^2;
                true_channel_MSE(carrier) = norm(H_f_o(:,:,carrier),'fro')^2;
            end
            NMSE_temp = sum(difference_channel_MSE)/sum(true_channel_MSE);
            NMSE(1,ii) = NMSE(1,ii) + NMSE_temp;
            disp(['Active Elements = ' num2str(M) ', SNR = ' num2str(PNR_dBs(ii)) ', iter_max = ' num2str(iterMax) ', iter_now = ' num2str(iter)...
                ', OMP_iter_num = ' num2str(iter_num)...
                ', NMSE = ' num2str(NMSE_temp) ...
                '  , NMSE_dB = ' num2str(10*log10(NMSE_temp)) ...
                'dB, total_NMSE = ' num2str(10*log10(NMSE(1,ii)/iter)) ...
                'dB']);
            if iter<=trainingSize
                temp=reshape(fft(fft(H_f_est,[],1),[],3),[256 256]);
                trainingChannel(iter,1,:,:,:)=real(temp);
                trainingChannel(iter,2,:,:,:)=imag(temp);
                temp=reshape(fft(fft(H_f_o,[],1),[],3),[256 256]);
                trueTrainingChannel(iter,1,:,:,:)=real(temp);
                trueTrainingChannel(iter,2,:,:,:)=imag(temp);
            else
                temp=reshape(fft(fft(H_f_est,[],1),[],3),[256 256]);
                testChannel(iter,1,:,:,:)=real(temp);
                testChannel(iter,2,:,:,:)=imag(temp);
                temp=reshape(fft(fft(H_f_o,[],1),[],3),[256 256]);
                trueTestChannel(iter,1,:,:,:)=real(temp);
                trueTestChannel(iter,2,:,:,:)=imag(temp);
            end
        end
        %%
        % NMSE(ii) = NMSE(ii)/iterMax;
        % toc
        disp(['Finished ',num2str(ii),'/', num2str(length(PNR_dBs)) ' , NMSE = ' num2str(NMSE(1,ii)/iterMax)]);
        NMSEvsActiveNum(ii,kk)=NMSE(1,ii);
    end
    
end
%%
save trueTrainingChannel trueTrainingChannel
save trainingChannel trainingChannel

% save channelMat channel3
% save trainingChannel15 trainingChannel
% save trueTrainingChannel15 trueTrainingChannel
% save testChannel15 testChannel
% save trueTestChannel15 trueTestChannel
% 4000 iters, = -17.4566dB of 15dB snr

