function PhiPerxyz=PhiPeriodicxyz(x,omegax,omegay,omegaz,Q)

Phi1Per=PhiPeriodic(x(1,:),omegax,Q);
Phi2Per=PhiPeriodic(x(2,:),omegay,Q);
Phi3Per=PhiPeriodic(x(3,:),omegaz,Q);
N=size(x,2);
M=(2*Q+1)*(2*Q+1)*(2*Q+1);
PhiPerxyz=zeros(N,M);
for t=1:N
    PhiPerxy=kron(Phi1Per(t,:),Phi2Per(t,:));
    PhiPerxyz(t,:)=kron(PhiPerxy,Phi3Per(t,:));
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