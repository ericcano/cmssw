#ifndef DataFormats_PortableTestObjects_interface_TestSoA_h
#define DataFormats_PortableTestObjects_interface_TestSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace portabletest {

  using Matrix = Eigen::Matrix<double, 3, 6>;
  // SoA layout with x, y, z, id fields
  GENERATE_SOA_LAYOUT(TestSoALayout,
                      // columns: one value per element
                      SOA_COLUMN(double, x),
                      SOA_COLUMN(double, y),
                      SOA_COLUMN(double, z),
                      SOA_COLUMN(int32_t, id),
                      // scalars: one value for the whole structure
                      SOA_SCALAR(double, r),
                      // Eigen columns
                      // the typedef is needed because commas confuse macros
                      SOA_EIGEN_COLUMN(Matrix, m))

  using TestSoA = TestSoALayout<>;

  GENERATE_SOA_LAYOUT(TestSoALayout2,
                      // columns: one value per element
                      SOA_COLUMN(double, x2),
                      SOA_COLUMN(double, y2),
                      SOA_COLUMN(double, z2),
                      SOA_COLUMN(int32_t, id2),
                      // scalars: one value for the whole structure
                      SOA_SCALAR(double, r2),
                      // Eigen columns
                      // the typedef is needed because commas confuse macros
                      SOA_EIGEN_COLUMN(Matrix, m2))

  using TestSoA2 = TestSoALayout2<>;

  GENERATE_SOA_LAYOUT(TestSoALayout3,
                      // columns: one value per element
                      SOA_COLUMN(double, x3),
                      SOA_COLUMN(double, y3),
                      SOA_COLUMN(double, z3),
                      SOA_COLUMN(int32_t, id3),
                      // scalars: one value for the whole structure
                      SOA_SCALAR(double, r3),
                      // Eigen columns
                      // the typedef is needed because commas confuse macros
                      SOA_EIGEN_COLUMN(Matrix, m3))

  using TestSoA3 = TestSoALayout3<>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_TestSoA_h
