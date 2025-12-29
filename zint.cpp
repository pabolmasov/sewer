// C++ headers
#include <cmath>      // sqrt()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <fstream>    // ASCII file IO
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include<valarray> // for valarray functions

#include <filesystem> // sending commands to the system
namespace fs = std::filesystem;

std::valarray<double> uniform(double, double, int);
std::string namedir(void);

// TODO: injection!!

// boolean switches:
bool ifMaxwell = true;
bool ifmatter = false;
bool ifadaptivedt = true;

// globals:
double acoeff = 5.;
double omega = 40.;
double Bxbgd = 5.;

double n0 = 1.; // comoving density is everywhere just equal to 1

double dtCFL = 0.1; // CFL factor TODO: adaptive step!!!
double dtout = 0.1/omega;

int nz = 1024;
int numberofperiods = 3, packpower = 12;
double zlen = 2.*M_PI / omega * (double)numberofperiods; // assuming the BC are periodic, and the wavenumber fits the box "numberofperiods" times
double zmin = -zlen/2., zmax = zlen/2.;
double dz = zlen / (double)nz;

std::valarray<double> z0 = uniform(zmin, zmax, nz);

double tpack = -1.;
double tstart = 3. * tpack;

std::string outdir = namedir();

// std::cerr << " ";
// getchar();

// outdir name:
std::string namedir(void){
  std::stringstream ss;
  std::string s;
  ss << "A" << (int)rint(acoeff) << "B" <<  (int)rint(Bxbgd) ;
  if (!ifmatter){
    ss << "nofeed";
  }
  getline(ss, s);

  std::cerr << "output directory: " <<  s << "\n";
  fs::create_directory(s);

  std::cerr << "created\n";
  
  return s;
}

//
std::valarray<double> zwrap(std::valarray<double> z){
  std::valarray<double> znew = z;
  double t;
  
  for (int k = 0; k<nz;k++){
    if ((z[k] < zmin)||(z[k]>zmax)){
      t = std::floor((z[k]-zmin)/zlen);
      // t=0 if z is within the original z range
      znew[k] = z[k] - t * zlen;
    }
  }
  return znew;
}

// interpolating fields onto a non-regular grid
std::valarray<double> fieldtoz(std::valarray<double> f, std::valarray<double> z){
  int index = 0;
  double ddz = 0.;

  std::valarray<double> fnew(nz);
  
  for(int k = 0; k<nz; k++){ // w/o loop??
    index = (int)floor((z[k] - zmin) / (zmax - zmin) * (double)nz);
    index = index % nz;
    ddz = std::fmod((z[k] - zmin),dz);
    if (ddz < 0.) {
      ddz += dz;
    }
    if (index < 0) {
      index += nz; // not sure how modulo works here
    }
    fnew[k] = f[index] + (f.cshift(1)[index]-f[index]) * ddz/dz;
  }
  
  return fnew;
}

// aux: setting a uniform array
std::valarray<double> uniform(double x1, double x2, int nx){
  std::valarray<double> x(nx);

  // std::cerr << "x = " << x1 << ".." << x2 << "\n";

  // getchar();
  
  for (int k = 0; k<nx; k++){
    x[k] = (double)k/(double)nx * (x2-x1) + x1;
    // std::cerr << x[k] << "\n";
  }
  return x;
}

// scalar field characteristics: vector potential (y), B, and E

double Avec(double xi){
  if (tpack > 0.){
    return acoeff * std::cos(omega * xi) * std::pow(std::cos(0.5 *omega/(double)numberofperiods * xi), (double)packpower);
      // std::exp(-0.5 * ((xi+tstart)/tpack)*((xi+tstart)/tpack));
  }else{
    return acoeff * std::cos(omega * xi);
  }
}

std::valarray<double>  Avec(std::valarray<double>  xi){
  if (tpack > 0.){
    return acoeff * std::cos(omega * xi) * std::pow(std::cos(0.5 * omega/(double)numberofperiods * xi), (double)packpower);
      // std::cos(omega * (xi+tstart)) * std::exp(-0.5 * ((xi+tstart)/tpack)*((xi+tstart)/tpack));
  }else{
    return acoeff * std::cos(omega * xi);
  }
}


double Ey(double xi){ // Ey = - dAvec/dt
  if (tpack > 0.){
    return acoeff * (std::sin(omega * xi) * std::pow(std::cos(0.5 * omega/(double)numberofperiods * xi), (double)packpower)
		     + std::cos(omega * xi) * 0.5 * (double)packpower/(double)numberofperiods * std::pow(std::cos(0.5 * omega/(double)numberofperiods * xi), (double)packpower-1) * std::sin(0.5 * omega/(double)numberofperiods * xi));
      // acoeff * ((xi+tstart)/tpack/tpack * std::cos(omega * (xi+tstart))+omega * std::sin(omega * (xi+tstart))) * std::exp(-0.5 * ((xi+tstart)/tpack)*((xi+tstart)/tpack));
  }else{
    return acoeff * omega * std::sin(omega * xi);
  }
  //  return acoeff * omega * std::sin(omega * xi);
}

double Bx(double xi){ // Bx = - dAvec/dz
  if (tpack > 0.){
    return -acoeff * (std::sin(omega * xi) * std::pow(std::cos(0.5 * omega/(double)numberofperiods * xi), (double)packpower)
		     + std::cos(omega * xi) * 0.5 * (double)packpower/(double)numberofperiods * std::pow(std::cos(0.5 * omega/(double)numberofperiods * xi), (double)packpower-1) * std::sin(0.5 * omega/(double)numberofperiods * xi)) + Bxbgd;
      //acoeff * ((xi+tstart)/tpack/tpack * std::cos(omega * (xi+tstart))+omega * std::sin(omega * (xi+tstart))) * std::exp(-0.5 * ((xi+tstart)/tpack)*((xi+tstart)/tpack)) + Bxbgd;
  }else{
    return - acoeff * omega * std::sin(omega * xi);
  }
  //   return -acoeff * omega * std::sin(omega * xi) + Bxbgd;
}

// valarray fields
std::valarray<double> Ey(std::valarray<double> xi){ // Ey = - dAvec/dt
  if (tpack > 0.){
    return acoeff * (std::sin(omega * xi) * std::pow(std::cos(0.5 * omega/(double)numberofperiods * xi), (double)packpower)
		     + std::cos(omega * xi) * 0.5 * (double)packpower/(double)numberofperiods * std::pow(std::cos(0.5 * omega/(double)numberofperiods * xi), (double)packpower-1) * std::sin(0.5 * omega/(double)numberofperiods * xi));
      // acoeff * ((xi+tstart)/tpack/tpack * std::cos(omega * (xi+tstart))+omega * std::sin(omega * (xi+tstart))) * std::exp(-0.5 * ((xi+tstart)/tpack)*((xi+tstart)/tpack));
  }else{
    return acoeff * omega * std::sin(omega * xi);
  }
  //  return acoeff * omega * std::sin(omega * xi);
}

std::valarray<double> Bx(std::valarray<double> xi){ // Bx = - dAvec/dz
  if (tpack > 0.){
    return -acoeff * (std::sin(omega * xi) * std::pow(std::cos(0.5 * omega/(double)numberofperiods * xi), (double)packpower)
		     + std::cos(omega * xi) * 0.5 * (double)packpower/(double)numberofperiods * std::pow(std::cos(0.5 * omega/(double)numberofperiods * xi), (double)packpower-1) * std::sin(0.5 * omega/(double)numberofperiods * xi));
      // acoeff * ((xi+tstart)/tpack/tpack * std::cos(omega * (xi+tstart))+omega * std::sin(omega * (xi+tstart))) * std::exp(-0.5 * ((xi+tstart)/tpack)*((xi+tstart)/tpack));
  }else{
    return -acoeff * omega * std::sin(omega * xi);
  }
  // return -acoeff * omega * std::sin(omega * xi) + Bxbgd;
}

// setting initial fields:
std::valarray<double> setE(std::valarray<double> z){
  // assuming t = 0; grid shifted by dz/2 to the right
  return Ey(z+dz/2.); 
}

std::valarray<double> setB(std::valarray<double> z){
  // assuming t = 0
  return Bx(z);
}

double gamma(double uy, double uz){
  return std::sqrt(1.+uy*uy+uz*uz);
}

std::valarray<double> gamma(std::valarray<double> uy, std::valarray<double> uz){
  return std::sqrt(1.+uy*uy+uz*uz);
}

// accelerations:
double ay(double uy, double uz, double z, double t){
  double xi = z-t;
  return Ey(xi) + uz/gamma(uy, uz) * Bx(xi);
}

double az(double uy, double uz, double z, double t){
  double xi = z-t;
  return - uy/gamma(uy, uz) * Bx(xi);
}

// accelerations for valarray arguments (ifMaxwell = false)
std::valarray<double> ay(std::valarray<double> uy, std::valarray<double>  uz, std::valarray<double> z, double t){
  return Ey(z-t) + uz/std::sqrt(1.+uy*uy+uz*uz) * (Bx(z-t)+Bxbgd);
}

std::valarray<double> az(std::valarray<double> uy, std::valarray<double>  uz, std::valarray<double> z, double t){
  return - uy/std::sqrt(1.+uy*uy+uz*uz) * (Bx(z-t)+Bxbgd);
}

// accelerations for valarray arguments and arbitrary fields (ifMaxwell = true)
std::valarray<double> ay_F(std::valarray<double> uy, std::valarray<double>  uz, std::valarray<double> z, double t, std::valarray<double> ear, std::valarray<double>  bar){
  // return ear + uz/std::sqrt(1.+uy*uy+uz*uz) * bar;
  return fieldtoz(ear, z-dz/2.) + uz/std::sqrt(1.+uy*uy+uz*uz) * (fieldtoz(bar,z)+Bxbgd); 
}

std::valarray<double> az_F(std::valarray<double> uy, std::valarray<double>  uz, std::valarray<double> z, double t, std::valarray<double> bar){
  // return  - uy/std::sqrt(1.+uy*uy+uz*uz) * bar;
  return - uy/std::sqrt(1.+uy*uy+uz*uz) * (fieldtoz(bar, z)+Bxbgd);
}

// Maxwell equations
std::valarray<double> MaxE(std::valarray<double>  bar, std::valarray<double> vy, std::valarray<double> z){
  // we need to interpolate vy from z to z0+dz/2.; and compression is going to be huge!

  if(!ifmatter){
    return (bar.cshift(1)-bar)/dz;
  }else{
  
    std::valarray<double> jy(nz);
    std::valarray<double> kern = z * 0.;

    for (int k=0; k< nz; k++){
      // kernel:
      double zc = z0[k]+dz/2.;
      // kernel = std::max(std::min(z-zc, zc-z)/dz +1.,0.);
      kern = 1. - std::abs(zwrap(z)-zc);
      kern = (kern + std::abs(kern))/2. ;  // doing the same as the commented line above without pairwise min/max
      jy[k] = (n0 * vy * kern).sum(); // current in this particular point
    }
    
    return (bar.cshift(1)-bar)/dz - jy; // with current
  }
}

std::valarray<double> MaxB(std::valarray<double>  ear){
  return (ear-ear.cshift(-1))/dz;
}

// single particle evolution
void oneparticle(){

  double t = 0., dt = 1e-2 * std::min(1./omega, 1./std::max(acoeff * omega, Bxbgd)), tmax = std::max(1e3 * acoeff / omega, tstart*acoeff*acoeff), tstore = 0. ;
  double uy = Avec(0.)*0., uz = uy*uy/2., z = 0.;

  double az1, az2, az3, az4, ay1, ay2, ay3, ay4, dz1, dz2, dz3, dz4;
  double aout;

  // dt selection
  std::cout << "omega cycle is " << 2. * M_PI / omega << "\n";
  std::cout << "shortest Larmor cycle = " << 2. * M_PI / std::max(acoeff * omega, Bxbgd) << "\n";
  std::cout << "dz = " << dz << "\n";
  
  dt = std::min(dtCFL * dz, 0.01 / omega);
  
  std::cerr << "tmax = " << tmax << "\n";
  
  std::cout << "#  t  z  uy  uz Avec(z-t) Avec(z-t)**2/2 \n";
  
  while((t < tmax) && ((t-z)<std::max(tstart*5., tmax))){
    // RK4 implementation
    az1 = az(uy, uz, z, t); 
    ay1 = ay(uy, uz, z, t);
    dz1 = uz/gamma(uy, uz);
    az2 = az(uy+ay1*dt/2., uz+az1*dt/2., z+dz1*dt/2., t+dt/2.); 
    ay2 = ay(uy+ay1*dt/2., uz+az1*dt/2., z+dz1*dt/2., t+dt/2.);
    dz2 = (uz+az1*dt/2.)/gamma(uy+ay1*dt/2., uz+az1*dt/2.);
    az3 = az(uy+ay2*dt/2., uz+az2*dt/2., z+dz2*dt/2., t+dt/2.); 
    ay3 = ay(uy+ay2*dt/2., uz+az2*dt/2., z+dz2*dt/2., t+dt/2.);
    dz3 = (uz+az2*dt/2.)/gamma(uy+ay2*dt/2., uz+az2*dt/2.);
    az4 = az(uy+ay3*dt, uz+az3*dt, z+dz3*dt, t+dt); 
    ay4 = ay(uy+ay3*dt, uz+az3*dt, z+dz3*dt, t+dt);
    dz4 = (uz+az3*dt)/gamma(uy+ay3*dt, uz+az3*dt);

    // advance:
    uz += (az1 + 2. * az2 + 2. * az3 + az4) * dt / 6.;
    uy += (ay1 + 2. * ay2 + 2. * ay3 + ay4) * dt / 6.;
    z += (dz1 + 2. * dz2 + 2. * dz3 + dz4) * dt / 6.;
    t += dt;

    // printout:
    if (t > tstore){
      aout = Avec(z-t);
      std::cout << omega * t / (2.*M_PI) << " " << omega * z / (2.*M_PI) << " " << uy << " " << uz << " " << aout << " " << aout*aout/2. <<   "\n";
      tstore += dtout;
    }
  } 
}

// ASCII output for a snapshot
void ascout(double t, std::valarray<double> z, std::valarray<double> uy, std::valarray<double> uz, int ctr){

  std::ofstream fout;
  std::stringstream ss;
  std::string s;
  ss << outdir << "/asc" << ctr << ".dat";
  getline(ss, s);
  // std::cout << "snapshot to be written to " << s << "\n";
  fout.open(s);
  fout << "# t z uy uz \n";
  for (int k  = 0 ; k < nz; k++){
    fout << t << " " << z[k] << " " << uy[k] << " " << uz[k] << "\n";    
  }
  fout.close();
  // std::cout << "snapshot written to " << s << "\n";
}

void maxout(double t, std::valarray<double> ear,  std::valarray<double> bar, int ctr){

  std::ofstream fout;
  std::stringstream ss;
  std::string s;
  ss << outdir << "/ascmax" << ctr << ".dat";
  getline(ss, s);
  fout.open(s);
  fout << "# t z ear bar \n";
  for (int k  = 0 ; k < nz; k++){
    fout << t << " " << z0[k] * omega / (2.*M_PI) << " " << ear[k] << " " << bar[k]+Bxbgd << "\n";    
  }
  fout.close();
  // std::cout << "fields written to " << s << "\n";
}

void onemesh(std::valarray<double> z){
  double t = 0., dt = 1e-2 * std::min(1./omega, 1./std::max(acoeff * omega, Bxbgd)),
    tmax = std::max(100. * acoeff / omega, 3.*tstart), tstore = 0. ;

  // dt selection
  std::cout << "omega cycle is " << 2. * M_PI / omega << "\n";
  std::cout << "shortest Larmor cycle = " << 2. * M_PI / std::max(acoeff * omega, Bxbgd) << "\n";
  std::cout << "dz = " << dz << "\n";
  
  dt = std::min(dtCFL * dz, 0.01 / omega);

  std::valarray<double> uz = uniform(0., 0., nz);
  std::valarray<double> uy = uniform(0., 0., nz);

  uy = Avec(z);
  uz = uy*uy/2.;

  std::valarray<double> thegamma = uniform(0., 0., nz);
  std::valarray<double> vy = uniform(0., 0., nz);

  std::valarray<double> az1 = uniform(0., 0., nz);
  std::valarray<double> ay1 = uniform(0., 0., nz);
  std::valarray<double> dz1 = uniform(0., 0., nz);
 
  std::valarray<double> az2 = uniform(0., 0., nz);
  std::valarray<double> ay2 = uniform(0., 0., nz);
  std::valarray<double> dz2 = uniform(0., 0., nz);
  
  std::valarray<double> az3 = uniform(0., 0., nz);
  std::valarray<double> ay3 = uniform(0., 0., nz);
  std::valarray<double> dz3 = uniform(0., 0., nz);

  std::valarray<double> az4 = uniform(0., 0., nz);
  std::valarray<double> ay4 = uniform(0., 0., nz);
  std::valarray<double> dz4 = uniform(0., 0., nz);

  //   if (ifMaxwell){
  // setting initial EM fields
  std::valarray<double> ear = setE(z);
  std::valarray<double> bar = setB(z);
  std::valarray<double> dear1 = ear;
  std::valarray<double> dbar1 = bar;
  std::valarray<double> dear2 = ear;
  std::valarray<double> dbar2 = bar;
  std::valarray<double> dear3 = ear;
  std::valarray<double> dbar3 = bar;
  std::valarray<double> dear4 = ear;
  std::valarray<double> dbar4 = bar;    
  // }
  
  int ctr = 0;
  double adt = 0.;
  
  std::cout << "#  t  z  uy  uz\n";
  
  while(t < tmax){
    // RK4 implementation
    if (ifMaxwell){
      // fields evolving self-consistently
      az1 = az_F(uy, uz, z, t, bar); // uses fieldtoz, interpolates fields from z0 to z
      ay1 = ay_F(uy, uz, z, t, ear, bar);
      thegamma = gamma(uy, uz);
      dz1 = uz/thegamma;
      vy = uy/thegamma;
      dear1 = MaxE(bar, vy, z);
      dbar1 = MaxB(ear);

      if(ifadaptivedt){
	adt = 1./std::max(ay1.max(), az1.max());
	dt = std::min( std::min(dtCFL * dz, 0.01 / omega), 0.1*adt);
      }
      az2 = az_F(uy+ay1*dt/2., uz+az1*dt/2., z+dz1*dt/2., t+dt/2., bar+dbar1*dt/2.); 
      ay2 = ay_F(uy+ay1*dt/2., uz+az1*dt/2., z+dz1*dt/2., t+dt/2., ear+dear1*dt/2., bar+dbar1*dt/2.);
      thegamma = gamma(uy+ay1*dt/2., uz+az1*dt/2.);
      dz2 = (uz+az1*dt/2.)/thegamma;
      vy = (uy+ay1*dt/2.)/thegamma;
      dear2 = MaxE(bar+dbar1*dt/2., vy, z+dz1*dt/2.);
      dbar2 = MaxB(ear+dear1*dt/2.);
      az3 = az_F(uy+ay2*dt/2., uz+az2*dt/2., z+dz2*dt/2., t+dt/2., bar+dbar2*dt/2.); 
      ay3 = ay_F(uy+ay2*dt/2., uz+az2*dt/2., z+dz2*dt/2., t+dt/2., ear+dear2*dt/2., bar+dbar2*dt/2.);
      thegamma = gamma(uy+ay2*dt/2., uz+az2*dt/2.);
      dz3 = (uz+az2*dt/2.)/thegamma;
      vy = (uy+ay2*dt/2.)/thegamma;
      dear3 = MaxE(bar+dbar2*dt/2., vy, z+dz2*dt/2.);
      dbar3 = MaxB(ear+dear2*dt/2.);
      az4 = az_F(uy+ay3*dt, uz+az3*dt, z+dz3*dt, t+dt, bar+dbar3*dt); 
      ay4 = ay_F(uy+ay3*dt, uz+az3*dt, z+dz3*dt, t+dt, ear+dear3*dt, bar+dbar3*dt);
      thegamma = gamma(uy+ay3*dt, uz+az3*dt);
      dz4 = (uz+az3*dt)/thegamma;
      vy = (uy+ay3*dt)/thegamma;
      dear4 = MaxE(bar+dbar3*dt, vy, z+dz3*dt);
      dbar4 = MaxB(ear+dear3*dt);
    }else{
      // fields set by the global vector potential    
      az1 = az(uy, uz, z, t); 
      ay1 = ay(uy, uz, z, t);
      dz1 = uz/gamma(uy, uz);
      if(ifadaptivedt){
	adt = 1./std::max(ay1.max(), az1.max());
	dt = std::min( std::min(dtCFL * dz, 0.01 / omega), 0.1*adt);
      }
      az2 = az(uy+ay1*dt/2., uz+az1*dt/2., z+dz1*dt/2., t+dt/2.); 
      ay2 = ay(uy+ay1*dt/2., uz+az1*dt/2., z+dz1*dt/2., t+dt/2.);
      dz2 = (uz+az1*dt/2.)/gamma(uy+ay1*dt/2., uz+az1*dt/2.);
      az3 = az(uy+ay2*dt/2., uz+az2*dt/2., z+dz2*dt/2., t+dt/2.); 
      ay3 = ay(uy+ay2*dt/2., uz+az2*dt/2., z+dz2*dt/2., t+dt/2.);
      dz3 = (uz+az2*dt/2.)/gamma(uy+ay2*dt/2., uz+az2*dt/2.);
      az4 = az(uy+ay3*dt, uz+az3*dt, z+dz3*dt, t+dt); 
      ay4 = ay(uy+ay3*dt, uz+az3*dt, z+dz3*dt, t+dt);
      dz4 = (uz+az3*dt)/gamma(uy+ay3*dt, uz+az3*dt);
    }
    
    // advance:
    uz += (az1 + 2. * az2 + 2. * az3 + az4) * dt / 6.;
    uy += (ay1 + 2. * ay2 + 2. * ay3 + ay4) * dt / 6.;
    z += (dz1 + 2. * dz2 + 2. * dz3 + dz4) * dt / 6.; 
    t += dt;
    
    if (ifMaxwell){ // updating fields
      ear += (dear1 + 2. * dear2 + 2. * dear3 + dear4) * dt / 6.;
      bar += (dbar1 + 2. * dbar2 + 2. * dbar3 + dbar4) * dt / 6.;
    }

    // printout:
    if (t > tstore){
      if (ctr%10 == 0){
	std::cerr << ctr << ": omega t / (2 pi) = " << omega * t / (2. * M_PI) << "\n";
	if (ifadaptivedt){
	  std::cerr << "dt = " << dt << "\n";
	  std::cerr << "  dt(CFL) = " << dtCFL*dz << "\n";
	}
      }
      maxout(omega * t / (2. * M_PI), ear, bar, ctr);
      ascout(omega * t / (2. * M_PI), omega * z / (2. * M_PI), uy, uz, ctr);
      tstore += dtout;
      ctr++;
    }
  } 
}

 
int main(){
  // std::cout << std::fmod(5., 2.) << "\n";
  // std::cout << std::fmod(-5., 2.) << "\n";
  // std::cout << (-5 % 7) << "\n";
  // std::cout << std::fmod(-5, 7) << "\n";
  // getchar();
  std::cout << "zlen = " << zlen << "\n";
  std::cout << "z0 = " << z0[0] << ".." << z0[nz-1] << "\n";
  
  if(nz <= 0){
    oneparticle();
  }else{
    //    dz = zlen/(double)nz;
    std::valarray<double> z = z0; // uniform(-zlen/2., zlen/2., nz);
    onemesh(z);
  }
}
