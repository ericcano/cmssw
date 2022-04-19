#ifndef CUDADataFormats_Common_test_SoAGeneratedToROOT_h
#define CUDADataFormats_Common_test_SoAGeneratedToROOT_h

#include "CUDADataFormats/Common/interface/SoALayout.h"
#include "CUDADataFormats/Common/interface/SoAView.h"
#include <Eigen/Dense>


GENERATE_SOA_LAYOUT_AND_VIEW(SoALayoutTemplate,
                             SoAViewTemplate,
                             // predefined static scalars
                             // size_t size;
                             // size_t alignment;

                             // columns: one value per element
                             SOA_COLUMN(double, x),
                             SOA_COLUMN(double, y),
                             SOA_COLUMN(double, z)/*,
                             SOA_EIGEN_COLUMN(Eigen::Vector3d, a),
                             SOA_EIGEN_COLUMN(Eigen::Vector3d, b),
                             SOA_EIGEN_COLUMN(Eigen::Vector3d, r),
                             // scalars: one value for the whole structure
                             SOA_SCALAR(const char*, description),
                             SOA_SCALAR(uint32_t, someNumber)*/)
        
#endif // ndef CUDADataFormats_Common_test_SoAGeneratedToROOT_h