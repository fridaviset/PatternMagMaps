clear; close all;


reshigh=0.05;
westwall=-50;
eastwall=50;
southwall=-40;
northwall=40;
linewidth=1.2;
centersx=linspace(-30,30,5);
centersy=linspace(-45,45,16);
Nx=length(centersx);
Ny=length(centersy);

width_mag=1; height_mag=8;
width_column=3; height_column=10;


%axis off;
xlim([westwall, eastwall]);
ylim([southwall, northwall]);

dipoles=[];
moments=[];

res_column=1;
column_x=-height_mag/2:res_column:height_mag/2; Nx_column=length(column_x);
column_y=-width_mag/2:res_column:width_mag/2; Ny_column=length(column_y);

for i=1:Nx
    for j=1:Ny
        for i_column=1:Nx_column
            for j_column=1:Ny_column
                dipole=[centersy(j)+column_y(j_column), centersx(i)+column_x(i_column), -2];
                dipoles=[dipoles; dipole];
                moment=[0, 0, 100];
                moments=[moments; moment];
            end
        end
    end
end

n_rand=1000;
for i=1:n_rand
    x=rand()*(eastwall-westwall)+westwall;
    y=rand()*(northwall-southwall)+southwall;
    dipole=[y, x, -1.5];
    dipoles=[dipoles; dipole];
    moment=[0, 0, 25];
    moments=[moments; moment];
end

B=[10; 15; 3];

px=southwall:reshigh:northwall;
py=westwall:reshigh:eastwall;
[PY,PX]=meshgrid(py,px);
PZ=0.*PY;
positions=[PY(:),PX(:),PZ(:)]';
NVec=size(positions,2);
mag_norm_map=zeros(NVec,1);
mag_map=zeros(3,NVec);
for k=1:NVec
    p=positions(:,k);
    [mag_field]=H(p,dipoles,moments,B);
    mag_norm_map(k)=norm(mag_field);
    mag_map(:,k)=mag_field;
end

save('simulated_basement.mat','westwall','eastwall','northwall','southwall','reshigh','mag_map');
