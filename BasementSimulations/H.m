function [mag_field, Jacobian]=H(p,dipoles,moments,B)
%p - the current position in world frame, 3x1
%dipoles - the dipoles position in world frame, n_dipolesx3
%moments - the dipoles moments in world frame, n_dipolesx3
%moments - the magnetic moment of the dipoles

mag_field=B;
n_dipoles=size(dipoles,1);
Jacobian=0.*B*B';
for i=1:n_dipoles
    dipole=dipoles(i,:)';
    m=moments(i,:)';
    r=p-dipole;
    n_r=norm(r);
    a=(m'*r).*r./n_r^5;
    b=m./n_r^3;
    mag_field=mag_field+1./(4*pi)*(3*a-b);
    
    Ja=r*m'./n_r^5+m'*r*eye(3)./n_r^5-5*(m'*r).*r*r'./n_r^7;
    Jb=-3*(m*r')./n_r^5;
    Jacobian=Jacobian+1./(4*pi)*(3*Ja-Jb);
end

end