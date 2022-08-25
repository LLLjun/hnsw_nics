#pragma once
#include <unordered_set>
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

#include <cstdio>
#include <queue>
#include <vector>
#include <numeric>
#include <algorithm>
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

        /*
            分析 hotdata
        */
#if HOTDATA
    // internalId-times
    struct Idtimes{
        tableint id;
        size_t   times;

        Idtimes() = default;
        Idtimes(tableint id, size_t times) : id(id), times(times) {}

        inline bool operator<(const Idtimes &other) const {
            return times < other.times;
        }
    };

    class HotData {
    public:
        HotData(size_t nums_point) {
            nums_point_ = nums_point;
        }

        void AddTimes(tableint id) {
            if (isTrain)
                AccessTimesTrain[id]++;
            else
                AccessTimesTest[id]++;
        }

        void endQuery() {
            qi_cur++;
            if (isTrain && qi_cur >= train_size) {
                isTrain = false;
            }
        }

        // For Training, 目前的做法是将query set 按照1:1的比例切分
        void initTrainSplit(int qsize, float train_ratio=0.5) {
            usingSample = false;
            setTrainStats(true);
            qi_cur = 0;
            train_size = qsize * train_ratio;
            test_size = qsize - train_size;
        }

        void initTrainSample(int sample_size, int qsize) {
            usingSample = true;
            setTrainStats(true);
            train_size = sample_size;
            test_size = qsize;
        }

        bool isSplitQuery() {
            return (!usingSample);
        }

        void setTrainStats(bool is_train) {
            isTrain = is_train;
            if (isTrain) {
                std::vector<size_t>().swap(AccessTimesTrain);
                AccessTimesTrain.resize(nums_point_, 0);
            } else {
                std::vector<size_t>().swap(AccessTimesTest);
                AccessTimesTest.resize(nums_point_, 0);
            }
        }

        void processTrain() {
            std::vector<Idtimes> AccessTdtimesTrain;
            std::vector<Idtimes> AccessTdtimesTest;
            transTimesToTdtimes(AccessTimesTrain, AccessTdtimesTrain);
            transTimesToTdtimes(AccessTimesTest, AccessTdtimesTest);

            size_t accessTotalTrain = std::accumulate(AccessTimesTrain.begin(), AccessTimesTrain.end(), 0);
            size_t accessTotalTest = std::accumulate(AccessTimesTest.begin(), AccessTimesTest.end(), 0);
            float ratio_max = 0.5;
            int n_steps = 10;
            size_t interval = ratio_max * nums_point_ / n_steps;

            size_t accessCurrent = 0;

            printf("Points(%%)\t Access(%%)\t Freq.Avg\t Max\t Min\t Match(%%)\t M.Access(%%)\n");
            for (int si = 0; si < n_steps; si++) {
                int begin = si * interval;
                int end = begin + interval;

                // 分析 Train 数据信息
                size_t accessTmpTrain = 0;
                for (int i = begin; i < end; i++) {
                    accessTmpTrain += AccessTdtimesTrain[i].times;
                }
                accessCurrent += accessTmpTrain;
                // printf
                printf("%.1f%%\t %.1f%%\t %.5f\t %.5f\t %.5f\t ",
                                100.0 * end / nums_point_,
                                100.0 * accessCurrent / accessTotalTrain,
                                1.0 * accessTmpTrain / interval / train_size,
                                1.0 * AccessTdtimesTrain[begin].times / train_size,
                                1.0 * AccessTdtimesTrain[end-1].times / train_size);

                // 比较 Test 和 Train 的一致程度
                int n_hit = 0;
                size_t accessTmpTest = 0;
                std::unordered_set<tableint> pointSet;
                for (int i = 0; i < end; i++) {
                    pointSet.emplace(AccessTdtimesTrain[i].id);
                }
                for (int i = 0; i < end; i++) {
                    tableint ptest = AccessTdtimesTest[i].id;
                    if (pointSet.find(ptest) != pointSet.end()) {
                        n_hit++;
                        accessTmpTest += AccessTdtimesTest[i].times;
                    }
                }
                printf("%.1f%%\t %.1f%%\n",
                                100.0 * n_hit / end,
                                100.0 * accessTmpTest / accessTotalTest);

                if (AccessTdtimesTrain[end-1].times == 0)
                    break;
            }
            printf("\n");
        }

    private:
        size_t nums_point_;

        // For Training
        std::vector<size_t> AccessTimesTrain;
        std::vector<size_t> AccessTimesTest;
        bool usingSample, isTrain;
        int qi_cur;
        int train_size, test_size;

        void transTimesToTdtimes(std::vector<size_t>& timesList, std::vector<Idtimes>& timesPair) {
            timesPair.resize(nums_point_, Idtimes(0, 0));

            for (int i = 0; i < nums_point_; i++) {
                timesPair[i].id = i;
                timesPair[i].times = timesList[i];
            }

#if DDEBUG
            printf("Id\tTimes\n");
            for (int i = 0; i < 5; i++)
                printf("%d\t%lu\n", timesPair[i].id, timesPair[i].times);
#endif
            sort(timesPair.rbegin(), timesPair.rend());
#if DDEBUG
            printf("Id\tTimes\n");
            for (int i = 0; i < 5; i++)
                printf("%d\t%lu\n", timesPair[i].id, timesPair[i].times);
#endif
        }
    };
#endif

}

#include "space_l2.h"
#include "space_ip.h"
#include "bruteforce.h"
#include "hnswalg.h"
