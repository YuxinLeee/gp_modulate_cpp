#include "locally_modulated_ds/linear_velocity_fields.h"
#include "locally_modulated_ds/gp_modulated_ds.h"
#include "lpvDS.h"  
#include <matio.h>
#include <vector>
#include <iostream>
#include <random>
#include <string>


using Mat = LinearVelocityField::Mat;
using Vec = LinearVelocityField::Vec;


int main(){
  /**
  // 1) 打开 MAT 文件
  mat_t *matfp = Mat_Open("/Users/macpro/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/969743b82696fc3d2e7630b589d93917/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/File/project/Manipulability-master/ds_control.mat", MAT_ACC_RDONLY);
  if(!matfp){
    std::cerr << "无法打开 ds_control.mat" << std::endl;
    return -1;
  }
  // 2) 读取 ds_gmm 结构体
  matvar_t *gmm_var = Mat_VarRead(matfp, "ds_gmm");
  if(!gmm_var || gmm_var->class_type!=MAT_C_STRUCT){
    std::cerr << "找不到 ds_gmm struct" << std::endl;
    Mat_Close(matfp);
    return -1;
  }
  // 3) Priors
  matvar_t *priors_var = Mat_VarGetStructFieldByName(gmm_var, "Priors", 0);
  size_t K = priors_var->dims[1];                // 1×K
  double *pr_data = static_cast<double*>(priors_var->data);
  Eigen::MatrixXd Priors(1, K);
  for(size_t i=0; i<K; ++i) {
    Priors(0, i) = pr_data[i];
  }

  // 打印 Priors
  std::cout << "Priors (1x" << K << "):" << std::endl;
  std::cout << Priors << std::endl;

  // 4) Mu
  matvar_t *mu_var = Mat_VarGetStructFieldByName(gmm_var, "Mu", 0);
  size_t M = mu_var->dims[0], K2 = mu_var->dims[1];
  if(K2!=K){ std::cerr<<"Mu 列数与 Priors 不匹配"<<std::endl; }
  double *mu_data = static_cast<double*>(mu_var->data);
  Eigen::MatrixXd Mu(M, K);
  for(size_t m=0; m<M; ++m) {
    for(size_t k=0; k<K; ++k) {
      Mu(m, k) = mu_data[m + k*M];
    }
  }

  // 打印 Mu
  std::cout << "Mu (" << M << "x" << K << "):" << std::endl;
  std::cout << Mu << std::endl;

  // 5) Sigma (3D: M×M×K，在 MAT-file 中一般展平为 (K*M)×M)
  matvar_t *sigma_var = Mat_VarGetStructFieldByName(gmm_var, "Sigma", 0);
  size_t sigma_rows = sigma_var->dims[0], sigma_cols = sigma_var->dims[1];
  // 通常 sigma_rows == K*M, sigma_cols == M
  double *sigma_data = static_cast<double*>(sigma_var->data);
  Eigen::MatrixXd Sigma(K*M, M);
  for(size_t k=0; k<K; ++k) {
    for(size_t i=0; i<M; ++i) {
      for(size_t j=0; j<M; ++j) {
        Sigma(k*M + i, j) = sigma_data[i + j*M + k*M*M];
      }
    }
  }

  // 打印 Sigma
  std::cout << "Sigma (" << sigma_rows << "x" << sigma_cols << "):" << std::endl;
  std::cout << Sigma << std::endl;

  // 6) A_k （3D 数组: D×D×K）
  matvar_t *A_var = Mat_VarRead(matfp, "A_k");
  size_t D = A_var->dims[0], D2 = A_var->dims[1], K3 = A_var->dims[2];
  if(D!=D2 || K3!=K){ std::cerr<<"A_k 维度不符"<<std::endl; }
  double *A_data = static_cast<double*>(A_var->data);
  Eigen::MatrixXd A(K*D, D);
  for(size_t k=0; k<K; ++k) {
    for(size_t i=0; i<D; ++i) {
      for(size_t j=0; j<D; ++j) {
        A(k*D + i, j) = A_data[i + j*D + k*D*D];
      }
    }
  }

  // 打印 A
  std::cout << "A (" << K*D << "x" << D << "):" << std::endl;
  std::cout << A << std::endl;

  // 7) Attractor (att)
  matvar_t *att_var = Mat_VarRead(matfp, "att");
  Eigen::VectorXd att;
  
  if(!att_var) {
    std::cerr << "找不到 att 变量，使用默认值" << std::endl;
    att.resize(M);
    att.setZero();
  } else {
    size_t att_dim = att_var->dims[0];
    double *att_data = static_cast<double*>(att_var->data);
    att.resize(att_dim);
    for(size_t i=0; i<att_dim; ++i) {
      att(i) = att_data[i];
    }

    // 打印 Attractor
    std::cout << "Attractor (" << att_dim << "x1):" << std::endl;
    std::cout << att << std::endl;
  }
  
  


    //   7) 构造 lpvDS 对象
    //  我们这里用 vector<double> 版本的构造器：
//      lpvDS(int K, int M, const vector<double>& Priors,
//            const vector<double>& Mu, const vector<double>& Sigma,
//            const vector<double>& A);
//   lpvDS lpvds_obj(static_cast<int>(K),
//                   static_cast<int>(D),  // M == D
//                   Priors,
//                   Mu,
//                   Sigma,
//                   A);

  // std::cout << "成功构造 lpvDS 对象！\n";

  // 记得清理
//   Mat_VarFree(priors_var);
//   Mat_VarFree(mu_var);
//   Mat_VarFree(sigma_var);
//   Mat_VarFree(A_var);
//   Mat_VarFree(gmm_var);
//   Mat_Close(matfp);
**/

// // Step 1: Define your DS
int Dd = 3;
Mat Aa = -0.4*Mat::Identity(Dd,Dd);
Vec target(Dd);
LinearVelocityField ds_field(target,Aa,5);

// lpvDS lpvds_obj(K, 3, Priors, Mu, Sigma, A, att);


Eigen::Matrix<double, 3, 1> currpos(-0.523821, 0.158865, 0.387647);
// Vec vel2 = lpvds_obj(currpos);
// std::cout << "Velcocity of lpvDS: " << vel2 << std::endl;


// to evaluate the system:
// Eigen::Matrix<double, 3, 1> currpos(1.3, 1.3, 1.3);
Vec vel = ds_field(currpos);
std::cout << "Velocity of ds_field (before): " << vel << std::endl;


// Step 2: Define GP
GaussianProcessModulatedDS<double> gpmds_(ds_field); // provide the original dynamics to the constructor
gpmds_.get_gpr()->SetHyperParams(0.1, 1.0, 0.0001);


// GaussianProcessModulatedDS<double> gpmds_(lpvds_obj); // provide the original dynamics to the constructor
// gpmds_.get_gpr()->SetHyperParams(0.1, 1.0, 0.0001);


// Step 3: Provide Modulation Data
// Eigen::Matrix<double, 3,1> training_pos, training_vel;
// training_pos << 1,1,1;
// training_vel << 10.0,10.0,0.0;
// gpmds_.AddData(training_pos, training_vel); // add a single training point,


// Eigen::MatrixXd training_pos(3, 2), training_vel(3, 2);
// training_pos << 1,1,1,
//                 1.2, 1.2, 1.2;
 
// training_vel <<  -1.0, -2.0, 0.0,
//                  -1.0, -5.0, 0.0; 


const int dim = 3;        // dimensionality (e.g., x, y, z)
const int num_samples = 20; // number of samples
double stddev = 0.1; // standard deviation of the Gaussian noise

// std::cout << num_samples << std::endl; 


Eigen::VectorXd mean_pos(dim);
mean_pos << 1.0, 1.2, 1.2;

Eigen::VectorXd mean_vel(dim);
mean_vel << -1.0, -2.0, 0.0;

std::vector<Vec> pos_vecs, vel_vecs;

// Random number generation with standard normal distribution
// std::random_device rd;
// std::mt19937 gen(rd());
// std::normal_distribution<> normal_dist(0.0, 0.05); // stddev = 0.05

Eigen::MatrixXd training_pos(dim, num_samples);
Eigen::MatrixXd training_vel(dim, num_samples);


// Random number generator
std::random_device rd;
std::mt19937 gen(rd());
std::normal_distribution<> dist(0.0, stddev);
std::normal_distribution<> dist2(0.0, 0.01);



for (int i = 0; i < num_samples; ++i) {
    training_pos.col(i) = mean_pos.unaryExpr([&](double m) { return m + dist(gen); });
    training_vel.col(i) = mean_vel.unaryExpr([&](double m) { return m + dist2(gen); });
}
// training_vel = mean_vel.replicate(1, num_samples);

gpmds_.AddDataBatch(training_pos, training_vel); // add a single training point,


// gpmds_.AddDataBatch(training_pos, training_vel); // add a single training point,
// Eigen::Matrix<double, 3, 1> test_pos(1, 1, 1);

auto output_vel = gpmds_.GetOutput(currpos);
std::cout << output_vel << std::endl;

return 0;
}