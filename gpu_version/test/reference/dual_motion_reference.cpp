// Independent motion fixture generator. Compile with the unmodified upstream
// src/source files listed in README.md. No DualSPHysics code is vendored here.
#include "JMotion.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

static const double pi = std::acos(-1.0);

static void configure(JMotion& m, const std::string& name) {
    const tdouble3 origin = TDouble3(0), axis = TDouble3(0,0,1);
    if (name == "03") {
        m.ObjAdd(1,0,-1);
        m.ObjAdd(2,1,2);
        m.MovAddRectilinear(1,1,0,3,TDouble3(0.2,0,0.1));
        m.EventAdd(1,1,0);
        m.MovAddRotation(2,1,0,3,false,origin,axis,0.4);
        m.EventAdd(2,1,0);
        m.Prepare();
        return;
    }
    m.ObjAdd(1,0,2);
    if (name == "01") {
        m.MovAddRectilinear(1,1,2,1,TDouble3(1,0,0));
        m.MovAddWait(1,2,3,1);
        m.MovAddRectilinear(1,3,0,1,TDouble3(0,0,1));
    } else if (name == "02") {
        m.MovAddRectilinear(1,1,2,1,TDouble3(2,0,0));
        m.MovAddRectilinear(1,2,0,2,TDouble3(0,0,1));
    } else if (name == "04") {
        m.MovAddRectilinearAce(1,1,0,3,TDouble3(2,0,-1),TDouble3(1,0,0),false);
    } else if (name == "05") {
        m.MovAddRotation(1,1,0,3,false,origin,axis,0.4);
    } else if (name == "06rot") {
        m.MovAddRotationAce(1,1,0,3,false,origin,axis,0.2,0.3,false);
    } else if (name == "06cir") {
        m.MovAddCircularAce(1,1,0,3,false,origin,axis,TDouble3(2,0,1),0.2,0.3,false);
    } else if (name == "07rect") {
        m.MovAddRecSinu(1,1,0,3,false,TDouble3(0.5,0.25,0.75),TDouble3(1,0.5,0.25),TDouble3(0.2,0.4,0.1),false);
    } else if (name == "07rot") {
        m.MovAddRotSinu(1,1,0,3,false,origin,axis,0.5,0.8,0.3,false);
    } else if (name == "07cir") {
        m.MovAddCirSinu(1,1,0,3,false,origin,axis,TDouble3(2,0,1),0.5,0.8,0.3,false);
    } else if (name == "08") {
        m.MovAddRectilinearFile(1,1,0,3,"translation.dat",4,0,1,2,3);
    } else if (name == "09deg") {
        m.MovAddRotationFile(1,1,0,3,true,origin,axis,"degrees.dat");
    } else if (name == "09rad") {
        m.MovAddRotationFile(1,1,0,3,false,origin,axis,"radians.dat");
    }
    m.EventAdd(1,1,0);
    m.Prepare();
}

int main() {
    std::ofstream("translation.dat") << "10 5 2 1\n11 7 3 0\n13 8 0 2\n";
    std::ofstream("degrees.dat") << "10 30\n11 75\n13 20\n";
    std::ofstream radians("radians.dat");
    radians << std::setprecision(17) << "10 " << 30*pi/180 << "\n11 " << 75*pi/180 << "\n13 " << 20*pi/180 << "\n";
    radians.close();
    std::cout << "pattern,time,x,y,z,increment_vx,increment_vy,increment_vz\n" << std::setprecision(17);
    for (const auto& name : {"01","02","03","04","05","06rot","06cir","07rect","07rot","07cir","08","09deg","09rad"}) {
        JMotion m;
        configure(m,name);
        tdouble3 position = TDouble3(2,0,1);
        const double dt = 0.125;
        std::cout << name << ",0,2,0,1,0,0,0\n";
        for (unsigned step=0; step<32; step++) {
            const tdouble3 previous = position;
            m.ProcesTime(step*dt,dt);
            for (unsigned i=0; i<m.GetMovCount(); i++) {
                unsigned ref;
                tdouble3 displacement;
                JMatrix4d matrix;
                if (m.GetMov(i,ref,displacement,matrix)) position = position + displacement;
                else position = matrix.MulPoint(position);
            }
            const tdouble3 velocity = (position-previous)/dt;
            std::cout << name << ',' << (step+1)*dt << ',' << position.x << ',' << position.y << ',' << position.z
                      << ',' << velocity.x << ',' << velocity.y << ',' << velocity.z << '\n';
        }
    }
}
