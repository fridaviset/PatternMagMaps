function NablaPhiPerxyz=NablaPhiPeriodicxyz(x,omegax,omegay,omegaz,Q)

phiper1=PhiPeriodic(x(1,:),omegax,Q);
phiper2=PhiPeriodic(x(2,:),omegay,Q);
phiper3=PhiPeriodic(x(3,:),omegaz,Q);
dphiper1=dPhiPeriodic(x(1,:),omegax,Q);
dphiper2=dPhiPeriodic(x(2,:),omegay,Q);
dphiper3=dPhiPeriodic(x(3,:),omegaz,Q);
N=size(x,2);
M=(2*Q+1)*(2*Q+1)*(2*Q+1);
NablaPhiPerxyz=zeros(3*N,M);
for i1=1:(2*Q+1)
    for i2=1:(2*Q+1)
        for i3=1:(2*Q+1)
            NablaPhi1=dphiper1(:,i1).*phiper2(:,i2).*phiper3(:,i3);
            NablaPhi2=phiper1(:,i1).*dphiper2(:,i2).*phiper3(:,i3);
            NablaPhi3=phiper1(:,i1).*phiper2(:,i2).*dphiper3(:,i3);
            i=i3+(i2-1)*(2*Q+1)+(i1-1)*(2*Q+1)*(2*Q+1);
            NablaPhiPerxyz(:,i)=[NablaPhi1; NablaPhi2; NablaPhi3];
        end
    end
end


end


function PhiPer=PhiPeriodic(x,omega,Q)

N=size(x,2);
PhiQCos=zeros(N,Q);
PhiQSin=zeros(N,Q);
const=ones(N,1);
for t=1:N
    for q=1:Q
        PhiQCos(t,q)=cos(2*pi*q.*x(t).*omega);
        PhiQSin(t,q)=sin(2*pi*q.*x(t)*omega);
    end
end
PhiPer=[const, PhiQCos, PhiQSin];


end

function dPhiPer=dPhiPeriodic(x,omega,Q)

N=size(x,2);
dPhiQCos=zeros(N,Q);
dPhiQSin=zeros(N,Q);
dconst=zeros(N,1);
for t=1:N
    for q=1:Q
        dPhiQCos(t,q)=-(2*q*pi*omega).*sin(2*pi*q.*x(t).*omega);
        dPhiQSin(t,q)=(2*q*pi*omega).*cos(2*pi*q.*x(t)*omega);
    end
end
dPhiPer=[dconst, dPhiQCos, dPhiQSin];

end