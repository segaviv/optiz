#include <Optiz/Optiz.h>

int main() {
    Optiz::Problem problem(Eigen::MatrixXd::Random(10,1));
    problem.add_element_energy(10, [&](int i, auto& x) {
      std::cout << "i: " << i << std::endl;
      return Optiz::sqr(x(i) - i);
    });
    std::cout << problem.optimize().x() << std::endl;
    return 0;
}
