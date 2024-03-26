%Run "SimulateBasement.m" before running this file to generate the
%simulated basement magnetic field

clear; close all;
load('simulated_basement.mat');
load('hyperparams.mat');
p1=westwall:reshigh:eastwall;
p2=southwall:reshigh:northwall;
[P1,P2] = meshgrid(p1,p2);
mag_map=reshape(mag_map,[3,size(P1)]);
mag_norm_map(:,:)=sqrt(mag_map(1,:,:).^2+mag_map(2,:,:).^2+mag_map(3,:,:).^2);
mag_norm_mean=mean(mean(mag_norm_map));
mag_norm_map_c=mag_norm_map-mag_norm_mean;

Periodx=6;
Periody=15;
Periodz=2;
omegax=1./Periodx;
omegay=1./Periody;
omegaz=1./Periodz;
sigma_y=0.8719;

Q=2;


%Model SE disturbances
l_SE=1.4765;
sigma_SE=2.1382;
m1=70;
m2=56;
m3=3;
M_SE=m1*m2*m3;
lL1=westwall-2*l_SE;
uL1=eastwall+2*l_SE;
lL2=southwall-2*l_SE;
uL2=northwall+2*l_SE;
lL3=-2*l_SE;
uL3=2*l_SE;

experiments=20;
params=10;
N_meass=ceil(logspace(1,4,params));

Extrapolation_RMSE_PDSE=zeros(experiments,params);
Extrapolation_RMSE_PD=zeros(experiments,params);
Extrapolation_RMSE_SE=zeros(experiments,params);
Interpolation_RMSE_PDSE=zeros(experiments,params);
Interpolation_RMSE_PD=zeros(experiments,params);
Interpolation_RMSE_SE=zeros(experiments,params);

for experiment=1:experiments
    for param=1:params
        tic;
        LamSE=LambdaSE3D(m1,m2,m3,sigma_SE,l_SE,lL1,uL1,lL2,uL2,lL3,uL3);

        %Sample some measurements in the magnetic field
        x1min=5; x1max=35; x2min=-30; x2max=30;
        imin=(x1min-westwall)/reshigh;
        imax=(x1max-westwall)/reshigh;
        jmin=(x2min-southwall)/reshigh;
        jmax=(x2max-southwall)/reshigh;
        N_meas=N_meass(param);
        i_samp=floor(rand(N_meas,1).*(imax-imin)+imin);
        j_samp=floor(rand(N_meas,1).*(jmax-jmin)+jmin);
        x1_samp=p1(i_samp)';
        x2_samp=p2(j_samp)';
        x3_samp=zeros(N_meas,1);
        p=[x1_samp, x2_samp, x3_samp]';
        y_norm_train=zeros(N_meas,1);
        y_raw_train=zeros(N_meas,3);
        for t=1:N_meas
            y_norm_train(t)=mag_norm_map_c(j_samp(t),i_samp(t));
            y_raw_train(t,:)=mag_map(:,j_samp(t),i_samp(t)); 
        end
        y_c_train=mean(y_raw_train);
        y_train=y_raw_train-y_c_train;

        %Sample some interpolation testpoints in the magnetic field
        x1min=5; x1max=35; x2min=-30; x2max=30;
        imin=(x1min-westwall)/reshigh;
        imax=(x1max-westwall)/reshigh;
        jmin=(x2min-southwall)/reshigh;
        jmax=(x2max-southwall)/reshigh;
        N_test=1000;
        i_interp=floor(rand(N_test,1).*(imax-imin)+imin);
        j_interp=floor(rand(N_test,1).*(jmax-jmin)+jmin);
        x1_interp=p1(i_interp)';
        x2_interp=p2(j_interp)';
        x3_interp=zeros(N_test,1);
        p_interp=[x1_interp, x2_interp, x3_interp]';
        y_norm_interp=zeros(N_test,1);
        y_interp=zeros(N_test,3);
        for t=1:N_test
            y_norm_interp(t)=mag_norm_map_c(j_interp(t),i_interp(t));
            y_interp(t,:)=mag_map(:,j_interp(t),i_interp(t));
        end

        %Sample some extrapolation testpoints in the magnetic field
        x1min=-35; x1max=-5; x2min=-30; x2max=30;
        imin=(x1min-westwall)/reshigh;
        imax=(x1max-westwall)/reshigh;
        jmin=(x2min-southwall)/reshigh;
        jmax=(x2max-southwall)/reshigh;
        i_test=floor(rand(N_test,1).*(imax-imin)+imin);
        j_test=floor(rand(N_test,1).*(jmax-jmin)+jmin);
        x1_test=p1(i_test)';
        x2_test=p2(j_test)';
        x3_test=zeros(N_test,1);
        p_extra=[x1_test, x2_test, x3_test]';
        y_norm_extra=zeros(N_test,1);
        y_extra=zeros(N_test,3);
        for t=1:N_test
            y_norm_extra(t)=mag_norm_map_c(j_test(t),i_test(t));
            y_extra(t,:)=mag_map(:,j_test(t),i_test(t));
        end

        if N_meas==1000
            numbers=[p; p_interp; p_extra];
            entries=["Training x";"Training y";"Training z";"Interp test x";"Interp test y";"Interp test z";"Extrap test x";"Extrap test y";"Extrap test z"];
            thingy=[entries';numbers'];
            writematrix(thingy,'Locations.csv');
        end


        %Construct the test matrices
        %PhiExtraSE=Phi3D(m1,m2,m3,p_extra,lL1,uL1,lL2,uL2,lL3,uL3);
        %PhiInterpSE=Phi3D(m1,m2,m3,p_interp,lL1,uL1,lL2,uL2,lL3,uL3);
        %PhiSE=Phi3D(m1,m2,m3,p,lL1,uL1,lL2,uL2,lL3,uL3);
        %wSE=tempSE*PhiSE'*y_norm;
        %MuTestSEExtra=PhiExtraSE*wSE+mag_norm_mean;
        %MuTestSEInterp=PhiInterpSE*wSE+mag_norm_mean;

        NablaPhiExtraSE=NablaPhi3D(m1,m2,m3,p_extra,lL1,uL1,lL2,uL2,lL3,uL3);
        NablaPhiInterpSE=NablaPhi3D(m1,m2,m3,p_interp,lL1,uL1,lL2,uL2,lL3,uL3);
        NablaPhiSE=NablaPhi3D(m1,m2,m3,p,lL1,uL1,lL2,uL2,lL3,uL3);

        tempSE=NablaPhiSE'*NablaPhiSE+sigma_y^2.*inv(diag(LamSE));
        tempSE=inv(tempSE);
        wSE=tempSE*NablaPhiSE'*y_train(:);
        MuTestSEExtra=reshape(NablaPhiExtraSE*wSE,[N_test 3])+y_c_train;
        MuTestSEInterp=reshape(NablaPhiInterpSE*wSE,[N_test 3])+y_c_train;      

        %VarTestSEExtra=reshape(diag(sigma_y^2*NablaPhiExtraSE*tempSE*NablaPhiExtraSE'),[N_test 3]);
        %VarTestSEInterp=reshape(diag(sigma_y^2*NablaPhiInterpSE*tempSE*NablaPhiInterpSE'),[N_test 3]);

        %LambdaPer=LambdaPeriodicxyz(sigma_per,omegax, omegay, omegaz, l_perx, l_pery, l_perz, Q);
        LambdaPer=(mixture_weights)';
        NablaPhiExtraPer=NablaPhiPeriodicxyz(p_extra,omegax,omegay,omegaz,Q);
        NablaPhiInterpPer=NablaPhiPeriodicxyz(p_interp,omegax,omegay,omegaz,Q);
        NablaPhiPer=NablaPhiPeriodicxyz(p,omegax,omegay,omegaz,Q);

        NablaPhiExtra=[NablaPhiExtraSE, NablaPhiExtraPer];
        NablaPhiInterp=[NablaPhiInterpSE, NablaPhiInterpPer];
        NablaPhiAll=[NablaPhiSE, NablaPhiPer];
        Lambda=[LamSE; LambdaPer];

        temp=NablaPhiAll'*NablaPhiAll+sigma_y^2.*inv(diag(Lambda));
        temp=inv(temp);
        w=temp*NablaPhiAll'*y_train(:);

        %Now, estimate the magnetic field potential
        MuTestExtra=reshape(NablaPhiExtra*w,[N_test 3])+y_c_train;
        MuTestInterp=reshape(NablaPhiInterp*w,[N_test 3])+y_c_train; 

        Extrapolation_RMSE_PDSE(experiment,param)=sqrt(mean((y_extra(:)-MuTestExtra(:)).^2));
        Extrapolation_RMSE_SE(experiment,param)=sqrt(mean((y_extra(:)-MuTestSEExtra(:)).^2));

        Interpolation_RMSE_PDSE(experiment,param)=sqrt(mean((y_interp(:)-MuTestInterp(:)).^2));
        Interpolation_RMSE_SE(experiment,param)=sqrt(mean((y_interp(:)-MuTestSEInterp(:)).^2));
        toc;
    end
    disp(experiment);
end

%% Plot MC results
colors=viridis();
linewidth=1.5;
fontsize=15;

figure; clf;
errorbar(N_meass,mean(Extrapolation_RMSE_PDSE),std(Extrapolation_RMSE_PDSE),'Color',colors(60,:),'LineWidth',linewidth);
hold on;
errorbar(N_meass,mean(Extrapolation_RMSE_SE),std(Extrapolation_RMSE_SE),'Color',colors(170,:),'LineWidth',linewidth);
set(gca, 'XScale', 'log');
set(gca,'fontname','times');
set(gca,'XTick',[10 100 1000 10000],'TickLabelInterpreter','latex','Fontsize',fontsize);
set(gca,'YTick',0:2:10,'TickLabelInterpreter','latex','Fontsize',fontsize);
set(gca,'linewidth',1.5);
box off;
xlabel('Number of measurements','Interpreter','Latex','FontSize',fontsize);
ylabel('RMSE','Interpreter','Latex','FontSize',fontsize);
legend({'Pattern detection map','Squared exponential map'},'Interpreter','latex');
grid off; box off;
exportgraphics(gca,'Figures/ExtrapolationRMSEs.png','Resolution','500');

figure; clf;
errorbar(N_meass,mean(Interpolation_RMSE_PDSE),std(Interpolation_RMSE_PDSE),'Color',colors(60,:),'LineWidth',linewidth);
hold on;
errorbar(N_meass,mean(Interpolation_RMSE_SE),std(Interpolation_RMSE_SE),'Color',colors(170,:),'LineWidth',linewidth);
set(gca, 'XScale', 'log');
set(gca,'fontname','times');
set(gca,'XTick',[10 100 1000 10000],'TickLabelInterpreter','latex','Fontsize',fontsize);
set(gca,'YTick',0:2:10,'TickLabelInterpreter','latex','Fontsize',fontsize);
set(gca,'linewidth',1.5);
box off;
xlabel('Number of measurements','Interpreter','Latex','FontSize',fontsize);
ylabel('RMSE','Interpreter','Latex','FontSize',fontsize);
legend({'Pattern detection map','Squared exponential map'},'Interpreter','latex');
grid off; box off;
exportgraphics(gca,'Figures/InterpolationRMSEs.png','Resolution','500');

%% Store results in file

i_pdse_mean=mean(Interpolation_RMSE_PDSE);
i_pdse_std=std(Interpolation_RMSE_PDSE);
i_se_mean=mean(Interpolation_RMSE_SE);
i_se_std=std(Interpolation_RMSE_SE);
e_pdse_mean=mean(Extrapolation_RMSE_PDSE);
e_pdse_std=std(Extrapolation_RMSE_PDSE);
e_se_mean=mean(Extrapolation_RMSE_SE);
e_se_std=std(Extrapolation_RMSE_SE);

numbers=[N_meass; i_pdse_mean; i_pdse_std; i_se_mean; i_se_std; e_pdse_mean; e_pdse_std; e_se_mean; e_se_std];
entries=["NumberOfMeasurements","InterpolationPDSEMean","InterpolationPDSEStd","InterpolationSEMean","InterpolationSEStd","ExtrapolationPDSEMean","ExtrapolationPDSEStd","ExtrapolationSEMean","ExtrapolationSEStd"]';
thingy=[entries';numbers'];
writematrix(thingy,'ResultsForPlot.csv');


