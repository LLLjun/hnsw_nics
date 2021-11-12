#pragma once
#include "hnswlib.h"
#include "config.h"

namespace hnswlib {

    static float
    L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr, const void *pFlag1 = nullptr, const void *pFlag2 = nullptr) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

        float res = 0;
        for (size_t i = 0; i < qty; i++) {
            float t = *pVect1 - *pVect2;
            pVect1++;
            pVect2++;
            res += t * t;
        }
        return (res);
    }

#if defined(USE_AVX)

    // Favor using AVX if available.
    static float
    L2SqrSIMD16Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr, const void *pFlag1 = nullptr, const void *pFlag2 = nullptr) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);
        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty16 = qty >> 4;

        const float *pEnd1 = pVect1 + (qty16 << 4);

        __m256 diff, v1, v2;
        __m256 sum = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }

        _mm256_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
    }

#elif defined(USE_SSE)

    static float
    L2SqrSIMD16Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr, const void *pFlag1 = nullptr, const void *pFlag2 = nullptr) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);
        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty16 = qty >> 4;

        const float *pEnd1 = pVect1 + (qty16 << 4);

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }

        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }
#endif

#if defined(USE_SSE) || defined(USE_AVX)
    static float
    L2SqrSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr, const void *pFlag1 = nullptr, const void *pFlag2 = nullptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty16 = qty >> 4 << 4;
        float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
        float *pVect1 = (float *) pVect1v + qty16;
        float *pVect2 = (float *) pVect2v + qty16;

        size_t qty_left = qty - qty16;
        float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
        return (res + res_tail);
    }
#endif


#ifdef USE_SSE
    static float
    L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr, const void *pFlag1 = nullptr, const void *pFlag2 = nullptr) {
        float PORTABLE_ALIGN32 TmpRes[8];
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);


        size_t qty4 = qty >> 2;

        const float *pEnd1 = pVect1 + (qty4 << 2);

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }

    static float
    L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr, const void *pFlag1 = nullptr, const void *pFlag2 = nullptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty4 = qty >> 2 << 2;

        float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4);
        size_t qty_left = qty - qty4;

        float *pVect1 = (float *) pVect1v + qty4;
        float *pVect2 = (float *) pVect2v + qty4;
        float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

        return (res + res_tail);
    }
#endif

    class L2Space : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        // DISTFUNCFLAG<float> xx;
        size_t data_size_;
        size_t dim_;
    public:
        L2Space(size_t dim) {
            fstdistfunc_ = L2Sqr;
        #if defined(USE_SSE) || defined(USE_AVX)
            if (dim % 16 == 0)
                fstdistfunc_ = L2SqrSIMD16Ext;
            else if (dim % 4 == 0)
                fstdistfunc_ = L2SqrSIMD4Ext;
            else if (dim > 16)
                fstdistfunc_ = L2SqrSIMD16ExtResiduals;
            else if (dim > 4)
                fstdistfunc_ = L2SqrSIMD4ExtResiduals;
        #endif
            dim_ = dim;
            data_size_ = dim * sizeof(float);
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        // DISTFUNCFLAG<float> get_dist_func_flag() {
        //     exit(1);
        //     return xx;
        // }

        void *get_dist_func_param() {
            return &dim_;
        }

        ~L2Space() {}
    };

    static int
    L2SqrI4x(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr,
                const void *__restrict pFlag1 = nullptr, const void *__restrict pFlag2 = nullptr) {

        size_t qty = *((size_t *) qty_ptr);
        int res = 0;
        unsigned char *a = (unsigned char *) pVect1;
        unsigned char *b = (unsigned char *) pVect2;

        qty = qty >> 2;
        for (size_t i = 0; i < qty; i++) {

            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
        }
        return (res);
    }

    static int L2SqrI(const void* __restrict pVect1, const void* __restrict pVect2, const void* __restrict qty_ptr,
                        const void *__restrict pFlag1 = nullptr, const void *__restrict pFlag2 = nullptr) {
        size_t qty = *((size_t*)qty_ptr);
        int res = 0;
        unsigned char* a = (unsigned char*)pVect1;
        unsigned char* b = (unsigned char*)pVect2;

        for(size_t i = 0; i < qty; i++)
        {
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
        }
        return (res);
    }

    class L2SpaceI : public SpaceInterface<int> {

        DISTFUNC<int> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        L2SpaceI(size_t dim) {
            if(dim % 4 == 0) {
                fstdistfunc_ = L2SqrI4x;
            }
            else {
                fstdistfunc_ = L2SqrI;
            }
            dim_ = dim;
            data_size_ = dim * sizeof(unsigned char);
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<int> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &dim_;
        }

        ~L2SpaceI() {}
    };


    /*
        SSD距离计算
    */
    template<typename DTval, typename DTdiff, typename DTres>
    static DTres
    L2SqrSSD4x(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr,
                const void *__restrict pFlag1 = nullptr, const void *__restrict pFlag2 = nullptr) {

        size_t qty = *((size_t *) qty_ptr);
        DTdiff diff = 0;
        DTres res = 0;
        DTval *pv1 = (DTval *) pVect1;
        DTval *pv2 = (DTval *) pVect2;

        qty = qty >> 2;
        for (size_t i = 0; i < qty; i++) {
            diff = (DTdiff)((DTdiff)(*pv1) - (DTdiff)(*pv2));
            res += (DTres)((DTdiff)diff * (DTdiff)diff);
            pv1++;
            pv2++;

            diff = (DTdiff)((DTdiff)(*pv1) - (DTdiff)(*pv2));
            res += (DTres)((DTdiff)diff * (DTdiff)diff);
            pv1++;
            pv2++;
            
            diff = (DTdiff)((DTdiff)(*pv1) - (DTdiff)(*pv2));
            res += (DTres)((DTdiff)diff * (DTdiff)diff);
            pv1++;
            pv2++;

            diff = (DTdiff)((DTdiff)(*pv1) - (DTdiff)(*pv2));
            res += (DTres)((DTdiff)diff * (DTdiff)diff);
            pv1++;
            pv2++;
        }
        return (res);
    }

    class L2SpaceSSD : public SpaceInterface<FCP64> {

        DISTFUNC<FCP64> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        L2SpaceSSD(size_t dim) {
            if(dim % 4 == 0) {
                fstdistfunc_ = L2SqrSSD4x<FCP16, FCP32, FCP64>;
            }
            else {
                printf("Error, no support\n");
                exit(1);
                // fstdistfunc_ = L2SqrI;
            }
            dim_ = dim;
            data_size_ = dim * sizeof(DTFSSD);
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<FCP64> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &dim_;
        }

        ~L2SpaceSSD() {}
    };

    /*
        flag编解码 距离计算
    */
    uint8_t one_bit_to_value[8] = {128, 64, 32, 16, 8, 4, 2, 1};
    // 已有粗粒度表格，返回比例倍数，用于计算
    void getFactorListByFlag(size_t dims, const uint8_t *CoarseTable, FCP8 *FactorList){
        
        for (size_t cur_pos = 0; cur_pos < dims; cur_pos++){
            uint8_t flag_id = (uint8_t) (cur_pos / 8);
            uint8_t bit_id = (uint8_t) (cur_pos % 8);
            uint8_t flag_add = one_bit_to_value[bit_id];

            uint8_t final_flag = CoarseTable[flag_id] & flag_add;

            if (final_flag == 0){
                FactorList[cur_pos] = 1;
            } else if(final_flag == flag_add){
                FactorList[cur_pos] = PORP;
            } else {
                printf("error in Table Id ToProportionList\n");
                printf("table flag: %d\n", final_flag);
                exit(1);
            }
        }
    }

    static FCP32
    L2SqrIntFlag4x(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr,
                    const void *__restrict pFlag1, const void *__restrict pFlag2) {

        size_t qty = *((size_t *) qty_ptr);
        FCP16 diff = 0;
        FCP32 res = 0;
        FCP8 *pv1 = (FCP8 *) pVect1;
        FCP8 *pv2 = (FCP8 *) pVect2;

        FCP8 *pFactor1 = new FCP8[qty]();
        FCP8 *pFactor2 = new FCP8[qty]();
        getFactorListByFlag(qty, (uint8_t *)pFlag1, pFactor1);
        getFactorListByFlag(qty, (uint8_t *)pFlag2, pFactor2);
        FCP8 *pf1 = (FCP8 *) pFactor1;
        FCP8 *pf2 = (FCP8 *) pFactor2;

        qty = qty >> 2;
        for (size_t i = 0; i < qty; i++) {
            diff = (FCP16)((FCP16)(*pv1) * (FCP16)(*pf1) - (FCP16)(*pv2) * (FCP16)(*pf2));
            res += (FCP32)((FCP32)diff * (FCP32)diff);
            pv1++;
            pv2++;
            pf1++;
            pf2++;

            diff = (FCP16)((FCP16)(*pv1) * (FCP16)(*pf1) - (FCP16)(*pv2) * (FCP16)(*pf2));
            res += (FCP32)((FCP32)diff * (FCP32)diff);
            pv1++;
            pv2++;
            pf1++;
            pf2++;

            diff = (FCP16)((FCP16)(*pv1) * (FCP16)(*pf1) - (FCP16)(*pv2) * (FCP16)(*pf2));
            res += (FCP32)((FCP32)diff * (FCP32)diff);
            pv1++;
            pv2++;
            pf1++;
            pf2++;
            
            diff = (FCP16)((FCP16)(*pv1) * (FCP16)(*pf1) - (FCP16)(*pv2) * (FCP16)(*pf2));
            res += (FCP32)((FCP32)diff * (FCP32)diff);
            pv1++;
            pv2++;
            pf1++;
            pf2++;
        }
        delete[] pFactor1;
        delete[] pFactor2;
        return (res);
    }

    class L2SpaceIntFlag : public SpaceInterface<FCP32> {

        // DISTFUNCFLAG<FCP32> fstdistfunc_;
        DISTFUNC<FCP32> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        L2SpaceIntFlag(size_t dim) {
            if(dim % 4 == 0) {
                fstdistfunc_ = L2SqrIntFlag4x;
                // xx = L2SqrSSD4x<FCP8, FCP16, FCP32>;
            }
            else {
                printf("Error, no support\n");
                exit(1);
                // fstdistfunc_ = L2SqrI;
            }
            dim_ = dim;
            data_size_ = dim * sizeof(FCP8);
        }

        size_t get_data_size() {
            return data_size_;
        }

        // DISTFUNCFLAG<FCP32> get_dist_func_flag() {
        //     return fstdistfunc_;
        // }

        DISTFUNC<FCP32> get_dist_func() {
            // exit(1);
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &dim_;
        }

        ~L2SpaceIntFlag() {}
    };

}