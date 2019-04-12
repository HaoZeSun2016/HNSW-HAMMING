
#pragma once
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>

#define  __builtin_popcount(t) __popcnt(t)
#else

#include <x86intrin.h>

#endif


#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif


#include "hnswlib.h"

///////////////////////////////
//	Hamming space by Haoze Sun
//  2018.9.13 @ sogou-inc
///////////////////////////////

namespace hnswlib {
	static float
		HammingDistUInt8(const void * pVect1, const void * pVect2, const void * qty_ptr) {
			size_t qty = *((size_t *)qty_ptr) / (sizeof(unsigned int)* 8); // use int32 to speed up counting for hamming distance
			int res = 0;
			for (size_t i = 0; i < qty; i++) {
				res += __builtin_popcount(((unsigned int *)pVect1)[i] ^ ((unsigned int *)pVect2)[i]);
			}
			return ((float)res);  // convert to float32
		};

	class HMSpace : public SpaceInterface<float> {
		DISTFUNC<float> fstdistfunc_;
		size_t data_size_;
		size_t dim_;
	public:
		HMSpace(size_t dim) {
			/*
			if ((dim % 32) != 0) {
			throw runtime_error("Dim size not supported!");
			}
			**/
			fstdistfunc_ = HammingDistUInt8;
			dim_ = dim;
			data_size_ = dim / 8; // n bits / 8 = bytes
		}

		size_t get_data_size() {
			return data_size_;
		}

		DISTFUNC<float> get_dist_func() {
			return fstdistfunc_;
		}

		void *get_dist_func_param() {
			return &dim_;
		}

	};
}