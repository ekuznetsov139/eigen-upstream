// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TYPE_CASTING_GPU_H
#define EIGEN_TYPE_CASTING_GPU_H

namespace Eigen {

namespace internal {

#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
  (defined(EIGEN_HAS_HIP_FP16) && defined(EIGEN_HIP_DEVICE_COMPILE))

template <>
struct type_casting_traits<Eigen::half, float> {
  enum {
    VectorizedCast = 1,
    SrcCoeffRatio = 1,
    TgtCoeffRatio = 2
  };
};

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pcast<half8, float4>(const half8& a) {
  return float4{__half2float(a[0]), __half2float(a[1]), __half2float(a[2]), __half2float(a[3])};
}

template <>
struct type_casting_traits<float, Eigen::half> {
  enum {
    VectorizedCast = 1,
    SrcCoeffRatio = 2,
    TgtCoeffRatio = 1
  };
};


template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half8 pcast<float4, half8>(const float4& a, const float4& b) {
    return half8{__float2half(a.x), __float2half(a.y), __float2half(a.z), __float2half(a.w),
      __float2half(b.x), __float2half(b.y), __float2half(b.z), __float2half(b.w)
    };
}

#endif

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_TYPE_CASTING_GPU_H
