// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET_MATH_HALF_GPU_H
#define EIGEN_PACKET_MATH_HALF_GPU_H


namespace Eigen {
namespace internal {

// Most of the following operations require arch >= 3.0
#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDACC) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
  (defined(EIGEN_HAS_HIP_FP16) && defined(EIGEN_HIPCC))

typedef _Float16 half8 __attribute__((ext_vector_type(8)));

template<> struct is_arithmetic<half8> { enum { value = true }; };

template<> struct packet_traits<Eigen::half> : default_packet_traits
{
  typedef half8 type;
  typedef half8 half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size=8,
    HasHalfPacket = 0,
    HasAdd    = 1,
    HasMul    = 1,
    HasDiv    = 1,
    HasSqrt   = 1,
    HasRsqrt  = 1,
    HasExp    = 1,
    HasExpm1  = 1,
    HasLog    = 1,
    HasLog1p  = 1,
    HasBlend = 1,
  };
};

template<> struct unpacket_traits<half8> { typedef Eigen::half type; enum {size=8, alignment=Aligned16}; typedef half8 half; };

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half8 pset1<half8>(const Eigen::half& from_) {
  const _Float16& from = reinterpret_cast<const _Float16&>(from_);
  return half8{from, from, from, from, from, from, from, from};
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half8 pload<half8>(const Eigen::half* from_) {
  const half8* from = reinterpret_cast<const half8*>(from_);
  return *from;
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half8 ploadu<half8>(const Eigen::half* from_) {
  const _Float16* from = reinterpret_cast<const _Float16*>(from_);
  return half8{from[0], from[1], from[2], from[3], from[4], from[5], from[6], from[7]};
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half8 ploaddup<half8>(const Eigen::half* from_) {
  const _Float16* from = reinterpret_cast<const _Float16*>(from_);
  return half8{from[0], from[0], from[1], from[1], from[2], from[2], from[3], from[3]};
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstore<Eigen::half>(Eigen::half* to, const half8& from) {
  *reinterpret_cast<half8*>(to) = from;
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstoreu<Eigen::half>(Eigen::half* to_, const half8& from) {
  _Float16* to = reinterpret_cast<_Float16*>(to_);
  to[0] = from[0];
  to[1] = from[1];
  to[2] = from[2];
  to[3] = from[3];
  to[4] = from[4];
  to[5] = from[5];
  to[6] = from[6];
  to[7] = from[7];
}

template<>
 EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE half8 ploadt_ro<half8, Aligned>(const Eigen::half* from) {
  return pload<half8>(from);
}

template<>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE half8 ploadt_ro<half8, Unaligned>(const Eigen::half* from) {
  if (reinterpret_cast<uintptr_t>(from) & 15)
    return ploadu<half8>(from);
  return pload<half8>(from);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half8 pgather<Eigen::half, half8>(const Eigen::half* from_, Index stride) {
  const _Float16* from = reinterpret_cast<const _Float16*>(from_);
  return half8{from[0*stride], from[1*stride], from[2*stride], from[3*stride],
               from[4*stride], from[5*stride], from[6*stride], from[7*stride]
      };
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pscatter<Eigen::half, half8>(Eigen::half* to_, const half8& from, Index stride) {
  _Float16* to = reinterpret_cast<_Float16*>(to_);
  to[stride*0] = from[0];
  to[stride*1] = from[1];
  to[stride*2] = from[2];
  to[stride*3] = from[3];
  to[stride*4] = from[4];
  to[stride*5] = from[5];
  to[stride*6] = from[6];
  to[stride*7] = from[7];
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half pfirst<half8>(const half8& from_) {
  const _Float16 from0 = from_[0];
  const Eigen::half& from = reinterpret_cast<const Eigen::half&>(from0);
  return from;
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half8 pabs<half8>(const half8& a) {
  _Float16 result[8];
  const uint64_t* ai = reinterpret_cast<const uint64_t*>(&(a));
  uint64_t* ri = reinterpret_cast<uint64_t*>(result);
  ri[0] = ai[0] & 0x7FFF7FFF7FFF7FFFull;
  ri[1] = ai[1] & 0x7FFF7FFF7FFF7FFFull;
  return half8(*result);
}

/*
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void
ptranspose(PacketBlock<half2,2>& kernel) {
  __half a1 = __low2half(kernel.packet[0]);
  __half a2 = __high2half(kernel.packet[0]);
  __half b1 = __low2half(kernel.packet[1]);
  __half b2 = __high2half(kernel.packet[1]);
  kernel.packet[0] = __halves2half2(a1, b1);
  kernel.packet[1] = __halves2half2(a2, b2);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 plset<half2>(const Eigen::half& a) {
#if defined(EIGEN_HIP_DEVICE_COMPILE)
  
  return __halves2half2(a, __hadd(a, __float2half(1.0f)));
  
#else  // EIGEN_CUDA_ARCH

#if EIGEN_CUDA_ARCH >= 530
  return __halves2half2(a, __hadd(a, __float2half(1.0f)));
#else
  float f = __half2float(a) + 1.0f;
  return __halves2half2(a, __float2half(f));
#endif

#endif
}
*/
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half8 padd<half8>(const half8& a, const half8& b) {
  return a+b;
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half8 psub<half8>(const half8& a, const half8& b) {
  return a-b;
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half8 pnegate(const half8& a) {
  return -a;
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half8 pconj(const half8& a) { return a; }

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half8 pmul<half8>(const half8& a, const half8& b) {
  return a*b;
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half8 pmadd<half8>(const half8& a, const half8& b, const half8& c) {
  return a * b + c;
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half8 pdiv<half8>(const half8& a, const half8& b) {
  return a / b;
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half8 pmin<half8>(const half8& a, const half8& b) {
  half8 result;
  for (int i=0; i<8; i++)
    result[i] = a[i] < b[i] ? a[i] : b[i];
  return result;
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half8 pmax<half8>(const half8& a, const half8& b) {
  half8 result;
  for (int i=0; i<8; i++)
    result[i] = a[i] > b[i] ? a[i] : b[i];
  return result;
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux<half8>(const half8& a) {
  _Float16 result = a[0];
  for (int i=1; i<8; i++)
    result += a[i];
  return reinterpret_cast<const Eigen::half&>(result);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux_max<half8>(const half8& a) {
  _Float16 result = a[0];
  for (int i=1; i<8; i++)
    result = (result>a[i]) ? result : a[i];
  return reinterpret_cast<const Eigen::half&>(result);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux_min<half8>(const half8& a) {
  _Float16 result = a[0];
  for (int i=1; i<8; i++)
    result = (result<a[i]) ? result : a[i];
  return reinterpret_cast<const Eigen::half&>(result);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux_mul<half8>(const half8& a) {
  _Float16 result = a[0];
  for (int i=1; i<8; i++)
    result *= a[i];
  return reinterpret_cast<const Eigen::half&>(result);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half8 pblend(const Selector<8>& ifPacket, const half8& thenPacket, const half8& elsePacket) {
  half8 result;
  for (int i=0; i<8; i++) {
    result[i] = ifPacket.select[i] ? thenPacket[i] : elsePacket[i];
  }
  return result;
}

#define CWISE_OP(X, Y) \
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half8 X<half8>(const half8& a) {       \
  half8 result;                                                                         \
  for (int i=0; i<8; i++)                                                               \
    result[i] = _Float16(Y(float(a[i])));                                               \
  return result;                                                                        \
}

#define CWISE_OP_NO_TEMPLATE(X, Y) \
__host__ __device__ EIGEN_STRONG_INLINE half8 X(const half8& a) {                         \
  half8 result;                                                                         \
  for (int i=0; i<8; i++)                                                               \
    result[i] = _Float16(Y(float(a[i])));                                               \
  return result;                                                                        \
}


CWISE_OP(plog1p, log1pf)
CWISE_OP(pexpm1, expm1f)
CWISE_OP(plog, logf)
CWISE_OP(pexp, expf)
CWISE_OP(psqrt, sqrtf)
CWISE_OP(prsqrt, rsqrtf)
CWISE_OP(pfloor, floorf)
CWISE_OP_NO_TEMPLATE(floor, floorf)

#elif defined EIGEN_VECTORIZE_AVX512

typedef struct {
  __m256i x;
} Packet16h;


template<> struct is_arithmetic<Packet16h> { enum { value = true }; };

template <>
struct packet_traits<half> : default_packet_traits {
  typedef Packet16h type;
  // There is no half-size packet for Packet16h.
  typedef Packet16h half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 16,
    HasHalfPacket = 0,
    HasAdd    = 1,
    HasSub    = 1,
    HasMul    = 1,
    HasNegate = 1,
    HasAbs    = 0,
    HasAbs2   = 0,
    HasMin    = 0,
    HasMax    = 0,
    HasConj   = 0,
    HasSetLinear = 0,
    HasDiv = 0,
    HasSqrt = 0,
    HasRsqrt = 0,
    HasExp = 0,
    HasLog = 0,
    HasBlend = 0
  };
};


template<> struct unpacket_traits<Packet16h> { typedef Eigen::half type; enum {size=16, alignment=Aligned32}; typedef Packet16h half; };

template<> EIGEN_STRONG_INLINE Packet16h pset1<Packet16h>(const Eigen::half& from) {
  Packet16h result;
  result.x = _mm256_set1_epi16(from.x);
  return result;
}

template<> EIGEN_STRONG_INLINE Eigen::half pfirst<Packet16h>(const Packet16h& from) {
  return half_impl::raw_uint16_to_half(static_cast<unsigned short>(_mm256_extract_epi16(from.x, 0)));
}

template<> EIGEN_STRONG_INLINE Packet16h pload<Packet16h>(const Eigen::half* from) {
  Packet16h result;
  result.x = _mm256_load_si256(reinterpret_cast<const __m256i*>(from));
  return result;
}

template<> EIGEN_STRONG_INLINE Packet16h ploadu<Packet16h>(const Eigen::half* from) {
  Packet16h result;
  result.x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(from));
  return result;
}

template<> EIGEN_STRONG_INLINE void pstore<half>(Eigen::half* to, const Packet16h& from) {
  _mm256_store_si256((__m256i*)to, from.x);
}

template<> EIGEN_STRONG_INLINE void pstoreu<half>(Eigen::half* to, const Packet16h& from) {
  _mm256_storeu_si256((__m256i*)to, from.x);
}

template<> EIGEN_STRONG_INLINE Packet16h
ploaddup<Packet16h>(const Eigen::half*  from) {
  Packet16h result;
  unsigned short a = from[0].x;
  unsigned short b = from[1].x;
  unsigned short c = from[2].x;
  unsigned short d = from[3].x;
  unsigned short e = from[4].x;
  unsigned short f = from[5].x;
  unsigned short g = from[6].x;
  unsigned short h = from[7].x;
  result.x = _mm256_set_epi16(h, h, g, g, f, f, e, e, d, d, c, c, b, b, a, a);
  return result;
}

template<> EIGEN_STRONG_INLINE Packet16h
ploadquad(const Eigen::half* from) {
  Packet16h result;
  unsigned short a = from[0].x;
  unsigned short b = from[1].x;
  unsigned short c = from[2].x;
  unsigned short d = from[3].x;
  result.x = _mm256_set_epi16(d, d, d, d, c, c, c, c, b, b, b, b, a, a, a, a);
  return result;
}

EIGEN_STRONG_INLINE Packet16f half2float(const Packet16h& a) {
#ifdef EIGEN_HAS_FP16_C
  return _mm512_cvtph_ps(a.x);
#else
  EIGEN_ALIGN64 half aux[16];
  pstore(aux, a);
  float f0(aux[0]);
  float f1(aux[1]);
  float f2(aux[2]);
  float f3(aux[3]);
  float f4(aux[4]);
  float f5(aux[5]);
  float f6(aux[6]);
  float f7(aux[7]);
  float f8(aux[8]);
  float f9(aux[9]);
  float fa(aux[10]);
  float fb(aux[11]);
  float fc(aux[12]);
  float fd(aux[13]);
  float fe(aux[14]);
  float ff(aux[15]);

  return _mm512_set_ps(
      ff, fe, fd, fc, fb, fa, f9, f8, f7, f6, f5, f4, f3, f2, f1, f0);
#endif
}

EIGEN_STRONG_INLINE Packet16h float2half(const Packet16f& a) {
#ifdef EIGEN_HAS_FP16_C
  Packet16h result;
  result.x = _mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
  return result;
#else
  EIGEN_ALIGN64 float aux[16];
  pstore(aux, a);
  half h0(aux[0]);
  half h1(aux[1]);
  half h2(aux[2]);
  half h3(aux[3]);
  half h4(aux[4]);
  half h5(aux[5]);
  half h6(aux[6]);
  half h7(aux[7]);
  half h8(aux[8]);
  half h9(aux[9]);
  half ha(aux[10]);
  half hb(aux[11]);
  half hc(aux[12]);
  half hd(aux[13]);
  half he(aux[14]);
  half hf(aux[15]);

  Packet16h result;
  result.x = _mm256_set_epi16(
      hf.x, he.x, hd.x, hc.x, hb.x, ha.x, h9.x, h8.x,
      h7.x, h6.x, h5.x, h4.x, h3.x, h2.x, h1.x, h0.x);
  return result;
#endif
}

template<> EIGEN_STRONG_INLINE Packet16h pnegate(const Packet16h& a) {
  // FIXME we could do that with bit manipulation
  Packet16f af = half2float(a);
  Packet16f rf = pnegate(af);
  return float2half(rf);
}

template<> EIGEN_STRONG_INLINE Packet16h padd<Packet16h>(const Packet16h& a, const Packet16h& b) {
  Packet16f af = half2float(a);
  Packet16f bf = half2float(b);
  Packet16f rf = padd(af, bf);
  return float2half(rf);
}

template<> EIGEN_STRONG_INLINE Packet16h psub<Packet16h>(const Packet16h& a, const Packet16h& b) {
  Packet16f af = half2float(a);
  Packet16f bf = half2float(b);
  Packet16f rf = psub(af, bf);
  return float2half(rf);
}

template<> EIGEN_STRONG_INLINE Packet16h pmul<Packet16h>(const Packet16h& a, const Packet16h& b) {
  Packet16f af = half2float(a);
  Packet16f bf = half2float(b);
  Packet16f rf = pmul(af, bf);
  return float2half(rf);
}

template<> EIGEN_STRONG_INLINE half predux<Packet16h>(const Packet16h& from) {
  Packet16f from_float = half2float(from);
  return half(predux(from_float));
}

template<> EIGEN_STRONG_INLINE half predux_mul<Packet16h>(const Packet16h& from) {
  Packet16f from_float = half2float(from);
  return half(predux_mul(from_float));
}

template<> EIGEN_STRONG_INLINE Packet16h preduxp<Packet16h>(const Packet16h* p) {
  Packet16f pf[16];
  pf[0] = half2float(p[0]);
  pf[1] = half2float(p[1]);
  pf[2] = half2float(p[2]);
  pf[3] = half2float(p[3]);
  pf[4] = half2float(p[4]);
  pf[5] = half2float(p[5]);
  pf[6] = half2float(p[6]);
  pf[7] = half2float(p[7]);
  pf[8] = half2float(p[8]);
  pf[9] = half2float(p[9]);
  pf[10] = half2float(p[10]);
  pf[11] = half2float(p[11]);
  pf[12] = half2float(p[12]);
  pf[13] = half2float(p[13]);
  pf[14] = half2float(p[14]);
  pf[15] = half2float(p[15]);
  Packet16f reduced = preduxp<Packet16f>(pf);
  return float2half(reduced);
}

template<> EIGEN_STRONG_INLINE Packet16h preverse(const Packet16h& a)
{
  __m128i m = _mm_setr_epi8(14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1);
  Packet16h res;
  res.x = _mm256_insertf128_si256(
                    _mm256_castsi128_si256(_mm_shuffle_epi8(_mm256_extractf128_si256(a.x,1),m)),
                                           _mm_shuffle_epi8(_mm256_extractf128_si256(a.x,0),m), 1);
  return res;
}

template<> EIGEN_STRONG_INLINE Packet16h pinsertfirst(const Packet16h& a, Eigen::half b)
{
  Packet16h res;
  res.x = _mm256_insert_epi16(a.x,b.x,0);
  return res;
}

template<> EIGEN_STRONG_INLINE Packet16h pinsertlast(const Packet16h& a, Eigen::half b)
{
  Packet16h res;
  res.x = _mm256_insert_epi16(a.x,b.x,15);
  return res;
}

template<> EIGEN_STRONG_INLINE Packet16h pgather<Eigen::half, Packet16h>(const Eigen::half* from, Index stride)
{
  Packet16h result;
  result.x = _mm256_set_epi16(
      from[15*stride].x, from[14*stride].x, from[13*stride].x, from[12*stride].x,
      from[11*stride].x, from[10*stride].x, from[9*stride].x, from[8*stride].x,
      from[7*stride].x, from[6*stride].x, from[5*stride].x, from[4*stride].x,
      from[3*stride].x, from[2*stride].x, from[1*stride].x, from[0*stride].x);
  return result;
}

template<> EIGEN_STRONG_INLINE void pscatter<half, Packet16h>(half* to, const Packet16h& from, Index stride)
{
  EIGEN_ALIGN64 half aux[16];
  pstore(aux, from);
  to[stride*0].x = aux[0].x;
  to[stride*1].x = aux[1].x;
  to[stride*2].x = aux[2].x;
  to[stride*3].x = aux[3].x;
  to[stride*4].x = aux[4].x;
  to[stride*5].x = aux[5].x;
  to[stride*6].x = aux[6].x;
  to[stride*7].x = aux[7].x;
  to[stride*8].x = aux[8].x;
  to[stride*9].x = aux[9].x;
  to[stride*10].x = aux[10].x;
  to[stride*11].x = aux[11].x;
  to[stride*12].x = aux[12].x;
  to[stride*13].x = aux[13].x;
  to[stride*14].x = aux[14].x;
  to[stride*15].x = aux[15].x;
}

EIGEN_STRONG_INLINE void
ptranspose(PacketBlock<Packet16h,16>& kernel) {
  __m256i a = kernel.packet[0].x;
  __m256i b = kernel.packet[1].x;
  __m256i c = kernel.packet[2].x;
  __m256i d = kernel.packet[3].x;
  __m256i e = kernel.packet[4].x;
  __m256i f = kernel.packet[5].x;
  __m256i g = kernel.packet[6].x;
  __m256i h = kernel.packet[7].x;
  __m256i i = kernel.packet[8].x;
  __m256i j = kernel.packet[9].x;
  __m256i k = kernel.packet[10].x;
  __m256i l = kernel.packet[11].x;
  __m256i m = kernel.packet[12].x;
  __m256i n = kernel.packet[13].x;
  __m256i o = kernel.packet[14].x;
  __m256i p = kernel.packet[15].x;

  __m256i ab_07 = _mm256_unpacklo_epi16(a, b);
  __m256i cd_07 = _mm256_unpacklo_epi16(c, d);
  __m256i ef_07 = _mm256_unpacklo_epi16(e, f);
  __m256i gh_07 = _mm256_unpacklo_epi16(g, h);
  __m256i ij_07 = _mm256_unpacklo_epi16(i, j);
  __m256i kl_07 = _mm256_unpacklo_epi16(k, l);
  __m256i mn_07 = _mm256_unpacklo_epi16(m, n);
  __m256i op_07 = _mm256_unpacklo_epi16(o, p);

  __m256i ab_8f = _mm256_unpackhi_epi16(a, b);
  __m256i cd_8f = _mm256_unpackhi_epi16(c, d);
  __m256i ef_8f = _mm256_unpackhi_epi16(e, f);
  __m256i gh_8f = _mm256_unpackhi_epi16(g, h);
  __m256i ij_8f = _mm256_unpackhi_epi16(i, j);
  __m256i kl_8f = _mm256_unpackhi_epi16(k, l);
  __m256i mn_8f = _mm256_unpackhi_epi16(m, n);
  __m256i op_8f = _mm256_unpackhi_epi16(o, p);

  __m256i abcd_03 = _mm256_unpacklo_epi32(ab_07, cd_07);
  __m256i abcd_47 = _mm256_unpackhi_epi32(ab_07, cd_07);
  __m256i efgh_03 = _mm256_unpacklo_epi32(ef_07, gh_07);
  __m256i efgh_47 = _mm256_unpackhi_epi32(ef_07, gh_07);
  __m256i ijkl_03 = _mm256_unpacklo_epi32(ij_07, kl_07);
  __m256i ijkl_47 = _mm256_unpackhi_epi32(ij_07, kl_07);
  __m256i mnop_03 = _mm256_unpacklo_epi32(mn_07, op_07);
  __m256i mnop_47 = _mm256_unpackhi_epi32(mn_07, op_07);

  __m256i abcd_8b = _mm256_unpacklo_epi32(ab_8f, cd_8f);
  __m256i abcd_cf = _mm256_unpackhi_epi32(ab_8f, cd_8f);
  __m256i efgh_8b = _mm256_unpacklo_epi32(ef_8f, gh_8f);
  __m256i efgh_cf = _mm256_unpackhi_epi32(ef_8f, gh_8f);
  __m256i ijkl_8b = _mm256_unpacklo_epi32(ij_8f, kl_8f);
  __m256i ijkl_cf = _mm256_unpackhi_epi32(ij_8f, kl_8f);
  __m256i mnop_8b = _mm256_unpacklo_epi32(mn_8f, op_8f);
  __m256i mnop_cf = _mm256_unpackhi_epi32(mn_8f, op_8f);

  __m256i abcdefgh_01 = _mm256_unpacklo_epi64(abcd_03, efgh_03);
  __m256i abcdefgh_23 = _mm256_unpackhi_epi64(abcd_03, efgh_03);
  __m256i ijklmnop_01 = _mm256_unpacklo_epi64(ijkl_03, mnop_03);
  __m256i ijklmnop_23 = _mm256_unpackhi_epi64(ijkl_03, mnop_03);
  __m256i abcdefgh_45 = _mm256_unpacklo_epi64(abcd_47, efgh_47);
  __m256i abcdefgh_67 = _mm256_unpackhi_epi64(abcd_47, efgh_47);
  __m256i ijklmnop_45 = _mm256_unpacklo_epi64(ijkl_47, mnop_47);
  __m256i ijklmnop_67 = _mm256_unpackhi_epi64(ijkl_47, mnop_47);
  __m256i abcdefgh_89 = _mm256_unpacklo_epi64(abcd_8b, efgh_8b);
  __m256i abcdefgh_ab = _mm256_unpackhi_epi64(abcd_8b, efgh_8b);
  __m256i ijklmnop_89 = _mm256_unpacklo_epi64(ijkl_8b, mnop_8b);
  __m256i ijklmnop_ab = _mm256_unpackhi_epi64(ijkl_8b, mnop_8b);
  __m256i abcdefgh_cd = _mm256_unpacklo_epi64(abcd_cf, efgh_cf);
  __m256i abcdefgh_ef = _mm256_unpackhi_epi64(abcd_cf, efgh_cf);
  __m256i ijklmnop_cd = _mm256_unpacklo_epi64(ijkl_cf, mnop_cf);
  __m256i ijklmnop_ef = _mm256_unpackhi_epi64(ijkl_cf, mnop_cf);

  // NOTE: no unpacklo/hi instr in this case, so using permute instr.
  __m256i a_p_0 = _mm256_permute2x128_si256(abcdefgh_01, ijklmnop_01, 0x20);
  __m256i a_p_1 = _mm256_permute2x128_si256(abcdefgh_23, ijklmnop_23, 0x20);
  __m256i a_p_2 = _mm256_permute2x128_si256(abcdefgh_45, ijklmnop_45, 0x20);
  __m256i a_p_3 = _mm256_permute2x128_si256(abcdefgh_67, ijklmnop_67, 0x20);
  __m256i a_p_4 = _mm256_permute2x128_si256(abcdefgh_89, ijklmnop_89, 0x20);
  __m256i a_p_5 = _mm256_permute2x128_si256(abcdefgh_ab, ijklmnop_ab, 0x20);
  __m256i a_p_6 = _mm256_permute2x128_si256(abcdefgh_cd, ijklmnop_cd, 0x20);
  __m256i a_p_7 = _mm256_permute2x128_si256(abcdefgh_ef, ijklmnop_ef, 0x20);
  __m256i a_p_8 = _mm256_permute2x128_si256(abcdefgh_01, ijklmnop_01, 0x31);
  __m256i a_p_9 = _mm256_permute2x128_si256(abcdefgh_23, ijklmnop_23, 0x31);
  __m256i a_p_a = _mm256_permute2x128_si256(abcdefgh_45, ijklmnop_45, 0x31);
  __m256i a_p_b = _mm256_permute2x128_si256(abcdefgh_67, ijklmnop_67, 0x31);
  __m256i a_p_c = _mm256_permute2x128_si256(abcdefgh_89, ijklmnop_89, 0x31);
  __m256i a_p_d = _mm256_permute2x128_si256(abcdefgh_ab, ijklmnop_ab, 0x31);
  __m256i a_p_e = _mm256_permute2x128_si256(abcdefgh_cd, ijklmnop_cd, 0x31);
  __m256i a_p_f = _mm256_permute2x128_si256(abcdefgh_ef, ijklmnop_ef, 0x31);

  kernel.packet[0].x = a_p_0;
  kernel.packet[1].x = a_p_1;
  kernel.packet[2].x = a_p_2;
  kernel.packet[3].x = a_p_3;
  kernel.packet[4].x = a_p_4;
  kernel.packet[5].x = a_p_5;
  kernel.packet[6].x = a_p_6;
  kernel.packet[7].x = a_p_7;
  kernel.packet[8].x = a_p_8;
  kernel.packet[9].x = a_p_9;
  kernel.packet[10].x = a_p_a;
  kernel.packet[11].x = a_p_b;
  kernel.packet[12].x = a_p_c;
  kernel.packet[13].x = a_p_d;
  kernel.packet[14].x = a_p_e;
  kernel.packet[15].x = a_p_f;
}

EIGEN_STRONG_INLINE void
ptranspose(PacketBlock<Packet16h,8>& kernel) {
  EIGEN_ALIGN64 half in[8][16];
  pstore<half>(in[0], kernel.packet[0]);
  pstore<half>(in[1], kernel.packet[1]);
  pstore<half>(in[2], kernel.packet[2]);
  pstore<half>(in[3], kernel.packet[3]);
  pstore<half>(in[4], kernel.packet[4]);
  pstore<half>(in[5], kernel.packet[5]);
  pstore<half>(in[6], kernel.packet[6]);
  pstore<half>(in[7], kernel.packet[7]);

  EIGEN_ALIGN64 half out[8][16];

  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      out[i][j] = in[j][2*i];
    }
    for (int j = 0; j < 8; ++j) {
      out[i][j+8] = in[j][2*i+1];
    }
  }

  kernel.packet[0] = pload<Packet16h>(out[0]);
  kernel.packet[1] = pload<Packet16h>(out[1]);
  kernel.packet[2] = pload<Packet16h>(out[2]);
  kernel.packet[3] = pload<Packet16h>(out[3]);
  kernel.packet[4] = pload<Packet16h>(out[4]);
  kernel.packet[5] = pload<Packet16h>(out[5]);
  kernel.packet[6] = pload<Packet16h>(out[6]);
  kernel.packet[7] = pload<Packet16h>(out[7]);
}

EIGEN_STRONG_INLINE void
ptranspose(PacketBlock<Packet16h,4>& kernel) {
  EIGEN_ALIGN64 half in[4][16];
  pstore<half>(in[0], kernel.packet[0]);
  pstore<half>(in[1], kernel.packet[1]);
  pstore<half>(in[2], kernel.packet[2]);
  pstore<half>(in[3], kernel.packet[3]);

  EIGEN_ALIGN64 half out[4][16];

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      out[i][j] = in[j][4*i];
    }
    for (int j = 0; j < 4; ++j) {
      out[i][j+4] = in[j][4*i+1];
    }
    for (int j = 0; j < 4; ++j) {
      out[i][j+8] = in[j][4*i+2];
    }
    for (int j = 0; j < 4; ++j) {
      out[i][j+12] = in[j][4*i+3];
    }
  }

  kernel.packet[0] = pload<Packet16h>(out[0]);
  kernel.packet[1] = pload<Packet16h>(out[1]);
  kernel.packet[2] = pload<Packet16h>(out[2]);
  kernel.packet[3] = pload<Packet16h>(out[3]);
}


#elif defined EIGEN_VECTORIZE_AVX

typedef struct {
  __m128i x;
} Packet8h;


template<> struct is_arithmetic<Packet8h> { enum { value = true }; };

template <>
struct packet_traits<Eigen::half> : default_packet_traits {
  typedef Packet8h type;
  // There is no half-size packet for Packet8h.
  typedef Packet8h half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 8,
    HasHalfPacket = 0,
    HasAdd    = 1,
    HasSub    = 1,
    HasMul    = 1,
    HasNegate = 1,
    HasAbs    = 0,
    HasAbs2   = 0,
    HasMin    = 0,
    HasMax    = 0,
    HasConj   = 0,
    HasSetLinear = 0,
    HasDiv = 0,
    HasSqrt = 0,
    HasRsqrt = 0,
    HasExp = 0,
    HasLog = 0,
    HasBlend = 0
  };
};


template<> struct unpacket_traits<Packet8h> { typedef Eigen::half type; enum {size=8, alignment=Aligned16}; typedef Packet8h half; };

template<> EIGEN_STRONG_INLINE Packet8h pset1<Packet8h>(const Eigen::half& from) {
  Packet8h result;
  result.x = _mm_set1_epi16(from.x);
  return result;
}

template<> EIGEN_STRONG_INLINE Eigen::half pfirst<Packet8h>(const Packet8h& from) {
  return half_impl::raw_uint16_to_half(static_cast<unsigned short>(_mm_extract_epi16(from.x, 0)));
}

template<> EIGEN_STRONG_INLINE Packet8h pload<Packet8h>(const Eigen::half* from) {
  Packet8h result;
  result.x = _mm_load_si128(reinterpret_cast<const __m128i*>(from));
  return result;
}

template<> EIGEN_STRONG_INLINE Packet8h ploadu<Packet8h>(const Eigen::half* from) {
  Packet8h result;
  result.x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(from));
  return result;
}

template<> EIGEN_STRONG_INLINE void pstore<Eigen::half>(Eigen::half* to, const Packet8h& from) {
  _mm_store_si128(reinterpret_cast<__m128i*>(to), from.x);
}

template<> EIGEN_STRONG_INLINE void pstoreu<Eigen::half>(Eigen::half* to, const Packet8h& from) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(to), from.x);
}

template<> EIGEN_STRONG_INLINE Packet8h
ploaddup<Packet8h>(const Eigen::half*  from) {
  Packet8h result;
  unsigned short a = from[0].x;
  unsigned short b = from[1].x;
  unsigned short c = from[2].x;
  unsigned short d = from[3].x;
  result.x = _mm_set_epi16(d, d, c, c, b, b, a, a);
  return result;
}

template<> EIGEN_STRONG_INLINE Packet8h
ploadquad<Packet8h>(const Eigen::half* from) {
  Packet8h result;
  unsigned short a = from[0].x;
  unsigned short b = from[1].x;
  result.x = _mm_set_epi16(b, b, b, b, a, a, a, a);
  return result;
}

EIGEN_STRONG_INLINE Packet8f half2float(const Packet8h& a) {
#ifdef EIGEN_HAS_FP16_C
  return _mm256_cvtph_ps(a.x);
#else
  EIGEN_ALIGN32 Eigen::half aux[8];
  pstore(aux, a);
  float f0(aux[0]);
  float f1(aux[1]);
  float f2(aux[2]);
  float f3(aux[3]);
  float f4(aux[4]);
  float f5(aux[5]);
  float f6(aux[6]);
  float f7(aux[7]);

  return _mm256_set_ps(f7, f6, f5, f4, f3, f2, f1, f0);
#endif
}

EIGEN_STRONG_INLINE Packet8h float2half(const Packet8f& a) {
#ifdef EIGEN_HAS_FP16_C
  Packet8h result;
  result.x = _mm256_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
  return result;
#else
  EIGEN_ALIGN32 float aux[8];
  pstore(aux, a);
  Eigen::half h0(aux[0]);
  Eigen::half h1(aux[1]);
  Eigen::half h2(aux[2]);
  Eigen::half h3(aux[3]);
  Eigen::half h4(aux[4]);
  Eigen::half h5(aux[5]);
  Eigen::half h6(aux[6]);
  Eigen::half h7(aux[7]);

  Packet8h result;
  result.x = _mm_set_epi16(h7.x, h6.x, h5.x, h4.x, h3.x, h2.x, h1.x, h0.x);
  return result;
#endif
}

template<> EIGEN_STRONG_INLINE Packet8h pconj(const Packet8h& a) { return a; }

template<> EIGEN_STRONG_INLINE Packet8h pnegate(const Packet8h& a) {
  // FIXME we could do that with bit manipulation
  Packet8f af = half2float(a);
  Packet8f rf = pnegate(af);
  return float2half(rf);
}

template<> EIGEN_STRONG_INLINE Packet8h padd<Packet8h>(const Packet8h& a, const Packet8h& b) {
  Packet8f af = half2float(a);
  Packet8f bf = half2float(b);
  Packet8f rf = padd(af, bf);
  return float2half(rf);
}

template<> EIGEN_STRONG_INLINE Packet8h psub<Packet8h>(const Packet8h& a, const Packet8h& b) {
  Packet8f af = half2float(a);
  Packet8f bf = half2float(b);
  Packet8f rf = psub(af, bf);
  return float2half(rf);
}

template<> EIGEN_STRONG_INLINE Packet8h pmul<Packet8h>(const Packet8h& a, const Packet8h& b) {
  Packet8f af = half2float(a);
  Packet8f bf = half2float(b);
  Packet8f rf = pmul(af, bf);
  return float2half(rf);
}

template<> EIGEN_STRONG_INLINE Packet8h pgather<Eigen::half, Packet8h>(const Eigen::half* from, Index stride)
{
  Packet8h result;
  result.x = _mm_set_epi16(from[7*stride].x, from[6*stride].x, from[5*stride].x, from[4*stride].x, from[3*stride].x, from[2*stride].x, from[1*stride].x, from[0*stride].x);
  return result;
}

template<> EIGEN_STRONG_INLINE void pscatter<Eigen::half, Packet8h>(Eigen::half* to, const Packet8h& from, Index stride)
{
  EIGEN_ALIGN32 Eigen::half aux[8];
  pstore(aux, from);
  to[stride*0].x = aux[0].x;
  to[stride*1].x = aux[1].x;
  to[stride*2].x = aux[2].x;
  to[stride*3].x = aux[3].x;
  to[stride*4].x = aux[4].x;
  to[stride*5].x = aux[5].x;
  to[stride*6].x = aux[6].x;
  to[stride*7].x = aux[7].x;
}

template<> EIGEN_STRONG_INLINE Eigen::half predux<Packet8h>(const Packet8h& a) {
  Packet8f af = half2float(a);
  float reduced = predux<Packet8f>(af);
  return Eigen::half(reduced);
}

template<> EIGEN_STRONG_INLINE Eigen::half predux_max<Packet8h>(const Packet8h& a) {
  Packet8f af = half2float(a);
  float reduced = predux_max<Packet8f>(af);
  return Eigen::half(reduced);
}

template<> EIGEN_STRONG_INLINE Eigen::half predux_min<Packet8h>(const Packet8h& a) {
  Packet8f af = half2float(a);
  float reduced = predux_min<Packet8f>(af);
  return Eigen::half(reduced);
}

template<> EIGEN_STRONG_INLINE Eigen::half predux_mul<Packet8h>(const Packet8h& a) {
  Packet8f af = half2float(a);
  float reduced = predux_mul<Packet8f>(af);
  return Eigen::half(reduced);
}

template<> EIGEN_STRONG_INLINE Packet8h preduxp<Packet8h>(const Packet8h* p) {
  Packet8f pf[8];
  pf[0] = half2float(p[0]);
  pf[1] = half2float(p[1]);
  pf[2] = half2float(p[2]);
  pf[3] = half2float(p[3]);
  pf[4] = half2float(p[4]);
  pf[5] = half2float(p[5]);
  pf[6] = half2float(p[6]);
  pf[7] = half2float(p[7]);
  Packet8f reduced = preduxp<Packet8f>(pf);
  return float2half(reduced);
}

template<> EIGEN_STRONG_INLINE Packet8h preverse(const Packet8h& a)
{
  __m128i m = _mm_setr_epi8(14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1);
  Packet8h res;
  res.x = _mm_shuffle_epi8(a.x,m);
  return res;
}

template<> EIGEN_STRONG_INLINE Packet8h pinsertfirst(const Packet8h& a, Eigen::half b)
{
  Packet8h res;
  res.x = _mm_insert_epi16(a.x,int(b.x),0);
  return res;
}

template<> EIGEN_STRONG_INLINE Packet8h pinsertlast(const Packet8h& a, Eigen::half b)
{
  Packet8h res;
  res.x = _mm_insert_epi16(a.x,int(b.x),7);
  return res;
}

template<int Offset>
struct palign_impl<Offset,Packet8h>
{
  static EIGEN_STRONG_INLINE void run(Packet8h& first, const Packet8h& second)
  {
    if (Offset!=0)
      first.x = _mm_alignr_epi8(second.x,first.x, Offset*2);
  }
};

EIGEN_STRONG_INLINE void
ptranspose(PacketBlock<Packet8h,8>& kernel) {
  __m128i a = kernel.packet[0].x;
  __m128i b = kernel.packet[1].x;
  __m128i c = kernel.packet[2].x;
  __m128i d = kernel.packet[3].x;
  __m128i e = kernel.packet[4].x;
  __m128i f = kernel.packet[5].x;
  __m128i g = kernel.packet[6].x;
  __m128i h = kernel.packet[7].x;

  __m128i a03b03 = _mm_unpacklo_epi16(a, b);
  __m128i c03d03 = _mm_unpacklo_epi16(c, d);
  __m128i e03f03 = _mm_unpacklo_epi16(e, f);
  __m128i g03h03 = _mm_unpacklo_epi16(g, h);
  __m128i a47b47 = _mm_unpackhi_epi16(a, b);
  __m128i c47d47 = _mm_unpackhi_epi16(c, d);
  __m128i e47f47 = _mm_unpackhi_epi16(e, f);
  __m128i g47h47 = _mm_unpackhi_epi16(g, h);

  __m128i a01b01c01d01 = _mm_unpacklo_epi32(a03b03, c03d03);
  __m128i a23b23c23d23 = _mm_unpackhi_epi32(a03b03, c03d03);
  __m128i e01f01g01h01 = _mm_unpacklo_epi32(e03f03, g03h03);
  __m128i e23f23g23h23 = _mm_unpackhi_epi32(e03f03, g03h03);
  __m128i a45b45c45d45 = _mm_unpacklo_epi32(a47b47, c47d47);
  __m128i a67b67c67d67 = _mm_unpackhi_epi32(a47b47, c47d47);
  __m128i e45f45g45h45 = _mm_unpacklo_epi32(e47f47, g47h47);
  __m128i e67f67g67h67 = _mm_unpackhi_epi32(e47f47, g47h47);

  __m128i a0b0c0d0e0f0g0h0 = _mm_unpacklo_epi64(a01b01c01d01, e01f01g01h01);
  __m128i a1b1c1d1e1f1g1h1 = _mm_unpackhi_epi64(a01b01c01d01, e01f01g01h01);
  __m128i a2b2c2d2e2f2g2h2 = _mm_unpacklo_epi64(a23b23c23d23, e23f23g23h23);
  __m128i a3b3c3d3e3f3g3h3 = _mm_unpackhi_epi64(a23b23c23d23, e23f23g23h23);
  __m128i a4b4c4d4e4f4g4h4 = _mm_unpacklo_epi64(a45b45c45d45, e45f45g45h45);
  __m128i a5b5c5d5e5f5g5h5 = _mm_unpackhi_epi64(a45b45c45d45, e45f45g45h45);
  __m128i a6b6c6d6e6f6g6h6 = _mm_unpacklo_epi64(a67b67c67d67, e67f67g67h67);
  __m128i a7b7c7d7e7f7g7h7 = _mm_unpackhi_epi64(a67b67c67d67, e67f67g67h67);

  kernel.packet[0].x = a0b0c0d0e0f0g0h0;
  kernel.packet[1].x = a1b1c1d1e1f1g1h1;
  kernel.packet[2].x = a2b2c2d2e2f2g2h2;
  kernel.packet[3].x = a3b3c3d3e3f3g3h3;
  kernel.packet[4].x = a4b4c4d4e4f4g4h4;
  kernel.packet[5].x = a5b5c5d5e5f5g5h5;
  kernel.packet[6].x = a6b6c6d6e6f6g6h6;
  kernel.packet[7].x = a7b7c7d7e7f7g7h7;
}

EIGEN_STRONG_INLINE void
ptranspose(PacketBlock<Packet8h,4>& kernel) {
  EIGEN_ALIGN32 Eigen::half in[4][8];
  pstore<Eigen::half>(in[0], kernel.packet[0]);
  pstore<Eigen::half>(in[1], kernel.packet[1]);
  pstore<Eigen::half>(in[2], kernel.packet[2]);
  pstore<Eigen::half>(in[3], kernel.packet[3]);

  EIGEN_ALIGN32 Eigen::half out[4][8];

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      out[i][j] = in[j][2*i];
    }
    for (int j = 0; j < 4; ++j) {
      out[i][j+4] = in[j][2*i+1];
    }
  }

  kernel.packet[0] = pload<Packet8h>(out[0]);
  kernel.packet[1] = pload<Packet8h>(out[1]);
  kernel.packet[2] = pload<Packet8h>(out[2]);
  kernel.packet[3] = pload<Packet8h>(out[3]);
}


// Disable the following code since it's broken on too many platforms / compilers.
//#elif defined(EIGEN_VECTORIZE_SSE) && (!EIGEN_ARCH_x86_64) && (!EIGEN_COMP_MSVC)
#elif 0

typedef struct {
  __m64 x;
} Packet4h;


template<> struct is_arithmetic<Packet4h> { enum { value = true }; };

template <>
struct packet_traits<Eigen::half> : default_packet_traits {
  typedef Packet4h type;
  // There is no half-size packet for Packet4h.
  typedef Packet4h half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 4,
    HasHalfPacket = 0,
    HasAdd    = 0,
    HasSub    = 0,
    HasMul    = 0,
    HasNegate = 0,
    HasAbs    = 0,
    HasAbs2   = 0,
    HasMin    = 0,
    HasMax    = 0,
    HasConj   = 0,
    HasSetLinear = 0,
    HasDiv = 0,
    HasSqrt = 0,
    HasRsqrt = 0,
    HasExp = 0,
    HasLog = 0,
    HasBlend = 0
  };
};


template<> struct unpacket_traits<Packet4h> { typedef Eigen::half type; enum {size=4, alignment=Aligned16}; typedef Packet4h half; };

template<> EIGEN_STRONG_INLINE Packet4h pset1<Packet4h>(const Eigen::half& from) {
  Packet4h result;
  result.x = _mm_set1_pi16(from.x);
  return result;
}

template<> EIGEN_STRONG_INLINE Eigen::half pfirst<Packet4h>(const Packet4h& from) {
  return half_impl::raw_uint16_to_half(static_cast<unsigned short>(_mm_cvtsi64_si32(from.x)));
}

template<> EIGEN_STRONG_INLINE Packet4h pconj(const Packet4h& a) { return a; }

template<> EIGEN_STRONG_INLINE Packet4h padd<Packet4h>(const Packet4h& a, const Packet4h& b) {
  __int64_t a64 = _mm_cvtm64_si64(a.x);
  __int64_t b64 = _mm_cvtm64_si64(b.x);

  Eigen::half h[4];

  Eigen::half ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64));
  Eigen::half hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64));
  h[0] = ha + hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 16));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 16));
  h[1] = ha + hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 32));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 32));
  h[2] = ha + hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 48));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 48));
  h[3] = ha + hb;
  Packet4h result;
  result.x = _mm_set_pi16(h[3].x, h[2].x, h[1].x, h[0].x);
  return result;
}

template<> EIGEN_STRONG_INLINE Packet4h pmul<Packet4h>(const Packet4h& a, const Packet4h& b) {
  __int64_t a64 = _mm_cvtm64_si64(a.x);
  __int64_t b64 = _mm_cvtm64_si64(b.x);

  Eigen::half h[4];

  Eigen::half ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64));
  Eigen::half hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64));
  h[0] = ha * hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 16));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 16));
  h[1] = ha * hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 32));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 32));
  h[2] = ha * hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 48));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 48));
  h[3] = ha * hb;
  Packet4h result;
  result.x = _mm_set_pi16(h[3].x, h[2].x, h[1].x, h[0].x);
  return result;
}

template<> EIGEN_STRONG_INLINE Packet4h pload<Packet4h>(const Eigen::half* from) {
  Packet4h result;
  result.x = _mm_cvtsi64_m64(*reinterpret_cast<const __int64_t*>(from));
  return result;
}

template<> EIGEN_STRONG_INLINE Packet4h ploadu<Packet4h>(const Eigen::half* from) {
  Packet4h result;
  result.x = _mm_cvtsi64_m64(*reinterpret_cast<const __int64_t*>(from));
  return result;
}

template<> EIGEN_STRONG_INLINE void pstore<Eigen::half>(Eigen::half* to, const Packet4h& from) {
  __int64_t r = _mm_cvtm64_si64(from.x);
  *(reinterpret_cast<__int64_t*>(to)) = r;
}

template<> EIGEN_STRONG_INLINE void pstoreu<Eigen::half>(Eigen::half* to, const Packet4h& from) {
  __int64_t r = _mm_cvtm64_si64(from.x);
  *(reinterpret_cast<__int64_t*>(to)) = r;
}

template<> EIGEN_STRONG_INLINE Packet4h
ploadquad<Packet4h>(const Eigen::half* from) {
  return pset1<Packet4h>(*from);
}

template<> EIGEN_STRONG_INLINE Packet4h pgather<Eigen::half, Packet4h>(const Eigen::half* from, Index stride)
{
  Packet4h result;
  result.x = _mm_set_pi16(from[3*stride].x, from[2*stride].x, from[1*stride].x, from[0*stride].x);
  return result;
}

template<> EIGEN_STRONG_INLINE void pscatter<Eigen::half, Packet4h>(Eigen::half* to, const Packet4h& from, Index stride)
{
  __int64_t a = _mm_cvtm64_si64(from.x);
  to[stride*0].x = static_cast<unsigned short>(a);
  to[stride*1].x = static_cast<unsigned short>(a >> 16);
  to[stride*2].x = static_cast<unsigned short>(a >> 32);
  to[stride*3].x = static_cast<unsigned short>(a >> 48);
}

EIGEN_STRONG_INLINE void
ptranspose(PacketBlock<Packet4h,4>& kernel) {
  __m64 T0 = _mm_unpacklo_pi16(kernel.packet[0].x, kernel.packet[1].x);
  __m64 T1 = _mm_unpacklo_pi16(kernel.packet[2].x, kernel.packet[3].x);
  __m64 T2 = _mm_unpackhi_pi16(kernel.packet[0].x, kernel.packet[1].x);
  __m64 T3 = _mm_unpackhi_pi16(kernel.packet[2].x, kernel.packet[3].x);

  kernel.packet[0].x = _mm_unpacklo_pi32(T0, T1);
  kernel.packet[1].x = _mm_unpackhi_pi32(T0, T1);
  kernel.packet[2].x = _mm_unpacklo_pi32(T2, T3);
  kernel.packet[3].x = _mm_unpackhi_pi32(T2, T3);
}

#endif

}
}

#endif // EIGEN_PACKET_MATH_HALF_GPU_H
