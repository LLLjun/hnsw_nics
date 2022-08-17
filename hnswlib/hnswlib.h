#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#ifndef NO_MANUAL_VECTORIZATION
#ifdef __SSE__
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
#else
#include <x86intrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif
#endif

#include <queue>
#include <vector>
#include <iostream>
#include <string.h>
#include "config.h"

namespace hnswlib {
    typedef unsigned int tableint;
    typedef size_t labeltype;

    template <typename T>
    class pairGreater {
    public:
        bool operator()(const T& p1, const T& p2) {
            return p1.first > p2.first;
        }
    };

    template<typename T>
    static void writeBinaryPOD(std::ostream &out, const T &podRef) {
        out.write((char *) &podRef, sizeof(T));
    }

    template<typename T>
    static void readBinaryPOD(std::istream &in, T &podRef) {
        in.read((char *) &podRef, sizeof(T));
    }

    template<typename DTres, typename DTset=float>
    using DISTFUNC = DTres(*)(const void *, const void *, const void *);


    template<typename DTres, typename DTset=float>
    class SpaceInterface {
    public:
        //virtual void search(void *);
        virtual size_t get_data_size() = 0;

        virtual DISTFUNC<DTres, DTset> get_dist_func() = 0;

        virtual void *get_dist_func_param() = 0;

        virtual ~SpaceInterface() {}
    };

    template<typename dist_t>
    class AlgorithmInterface {
    public:
        virtual void addPoint(const void *datapoint, labeltype label)=0;
        virtual std::priority_queue<std::pair<dist_t, labeltype >> searchKnn(const void *, size_t) const = 0;

        // Return k nearest neighbor in the order of closer fist
        virtual std::vector<std::pair<dist_t, labeltype>>
            searchKnnCloserFirst(const void* query_data, size_t k) const;

        virtual void saveIndex(const std::string &location)=0;
        virtual ~AlgorithmInterface(){
        }
    };

    template<typename dist_t>
    std::vector<std::pair<dist_t, labeltype>>
    AlgorithmInterface<dist_t>::searchKnnCloserFirst(const void* query_data, size_t k) const {
        std::vector<std::pair<dist_t, labeltype>> result;

        // here searchKnn returns the result in the order of further first
        auto ret = searchKnn(query_data, k);
        {
            size_t sz = ret.size();
            result.resize(sz);
            while (!ret.empty()) {
                result[--sz] = ret.top();
                ret.pop();
            }
        }

        return result;
    }


    /*
        detail infomation
    */
    class QueryStats {
    public:
        // 获取硬件模拟的时间,
        double hw_us;
        // 这一部分的所有变量都是单个step的结果
        double hlc_us;
        double rank_us;
        double sort_us;
        double visited_us;
        size_t n_DC_max;
        size_t n_DC_total;

        // 这一部分的所有变量都是统计结果
        double all_hlc_us;
        double all_rank_us;
        double all_sort_us;
        double all_visited_us;
        size_t all_n_DC_max;
        size_t all_n_DC_total;
        size_t all_n_hops;

        QueryStats() {
            Reset();
        }

        void Reset() {
            all_hlc_us = 0;
            all_rank_us = 0;
            all_sort_us = 0;
            all_visited_us = 0;
            all_n_DC_max = 0;
            all_n_DC_total = 0;
            all_n_hops = 0;

            hw_us = 0;
            ReflushTime();
        }

        inline void ReflushTime() {
            hlc_us = 0;
            rank_us = 0;
            sort_us = 0;
            visited_us = 0;
            n_DC_max = 0;
            n_DC_total = 0;
        }

        inline void AccumulateAll() {
            all_hlc_us += hlc_us;
            all_rank_us += rank_us;
            all_sort_us += sort_us;
            all_visited_us += visited_us;
            all_n_DC_max += n_DC_max;
            all_n_DC_total += n_DC_total;
        }

        inline void UpdateHardwareTime() {
            double hw_rank_us = 0;
            if (n_DC_total != 0) {
                hw_rank_us = rank_us * n_DC_max / n_DC_total;
            }
#if (OPT_SORT && OPT_VISITED)
            double max_para_us = std::max(hw_rank_us, sort_us);
            max_para_us = std::max(max_para_us, visited_us);
#elif OPT_SORT
            double max_para_us = std::max(hw_rank_us, sort_us);
            max_para_us += visited_us;
#elif OPT_VISITED
            double max_para_us = std::max(hw_rank_us, visited_us);
            max_para_us += sort_us;
#else
            double max_para_us = hw_rank_us + sort_us + visited_us;
#endif
            hw_us += (hlc_us + max_para_us);
            AccumulateAll();
            ReflushTime();
        }
    };

    // 硬件搜索中的信息
    class InfoHardwareSearch {
    public:
        void* query_data;
        // p_queue_min 始终指向当前 retset 中最靠前的且 flag == true 的点的位置
        int p_queue_min;
        int cur_queue_size;
        // is_end 是指queue中的元素已满, 且全部都是false
        // is_done 是指搜索过程是否终止
        bool is_end;
        bool is_done;
        int l_search;

        InfoHardwareSearch() {
            Reset(nullptr);
        }

        void Reset(void* query) {
            p_queue_min = 0;
            cur_queue_size = 0;
            is_end = false;
            is_done = false;
            query_data = query;
        }

        void SetEfs(int efs) {
            l_search = efs;
        }
    };

#if WALK
    class RandomWalk {
    public:
        RandomWalk(int nums, int steps) {
            qsize_ = nums;
            steps_ = steps;
            cur_qid_ = 0;

            initTable();
            initList();
            printf("[walk] qsize: %d, steps: %d\n", qsize_, steps_);
        }

        // 基于定长的步数，显示信息量的变化情况
        void variationInfoContent() {
            // int n_item = 10;
            // int item = steps_ / n_item;
            std::vector<float> ICMean(steps_, 0);
            std::vector<float> ICStdev(steps_, 0);

            std::vector<std::vector<float>> ICTablePerStep;
            transPosition(ICTablePerStep);

            for (int si = 0; si < steps_; si++) {
                float mean = getMean(ICTablePerStep[si]);
                if (mean < 0) {
                    printf("Walk Error, mean: %.3f less than 0\n", mean); exit(1);
                }
                ICMean[si] = mean;
                ICStdev[si] = getStdev(mean, ICTablePerStep[si]);
            }

            // printf
            printf("Step\t ICMean\t ICStdev\n");
            for (int si = 0; si < steps_; si++) {
                printf("%d\t %.3f\t %.3f\n", si, ICMean[si], ICStdev[si]);
            }
            printf("\n");
        }
        void addICInQuery(float value) {
            ICListQuery.push_back(value);
        }
        void endICInQuery() {
            InfoContentTable[cur_qid_].swap(ICListQuery);
            std::vector<float>().swap(ICListQuery);
            if (InfoContentTable[cur_qid_].size() != steps_) {
                printf("RandomWalk Error, out step: %lu\n", InfoContentTable[cur_qid_].size()); exit(1);
            }

            cur_qid_++;
        }

    private:
        int qsize_, steps_;
        int cur_qid_;
        std::vector<float> ICListQuery;
        // std::vector<float> ICListTotal;
        std::vector<std::vector<float>> InfoContentTable;

        void initTable() {
            InfoContentTable.resize(qsize_);
        }
        void initList() {
            std::vector<float>().swap(ICListQuery);
            // ICListTotal.resize(steps_, 0);
        }
        void transPosition(std::vector<std::vector<float>>& transedTable) {
            transedTable.resize(steps_);

            for (int si = 0; si < steps_; si++) {
                transedTable[si].resize(qsize_);
                for (int qi = 0; qi < qsize_; qi++) {
                    transedTable[si][qi] = InfoContentTable[qi][si];
                }
            }
        }
        // 根据mean和vector计算标准差
        float getMean(std::vector<float>& List) {
            float sum = std::accumulate(List.begin(), List.end(), 0.0);
            float mean = sum / List.size();
            return mean;
        }
        float getStdev(float mean, std::vector<float>& List) {
            float accum = 0;
            for (float& v: List)
                accum += (v - mean) * (v - mean);
            float stdev = std::sqrt(accum / (List.size() - 1));
            return stdev;
        }
    };

#endif

}

#include "space_l2.h"
#include "space_ip.h"
#include "bruteforce.h"
#include "hnswalg.h"
