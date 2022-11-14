#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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
#include <unordered_set>
#include "config.h"
#include "dataset.h"
#include "tool.h"

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


#if QTRACE
    class QueryTrace {
    public:
        QueryTrace(int qsize, int nbor_size, int efs) {
            query_size = qsize;
            max_nbor_size = nbor_size;
            num_step = efs;
            initQueryTrace();
        }

        void addSearchPoint(tableint id) {
            QueryPointSet[qi_cur][sti_cur] = id;
        }
        void addNeighborBeHash(tableint id) {
            QueryTraceBeHashSet[qi_cur][sti_cur].push_back(id);
        }
        void addNeighborAfHash(tableint id) {
            QueryTraceAfHashSet[qi_cur][sti_cur].push_back(id);
        }
        void endStep() {
            sti_cur++;
        }
        void endQuery() {
            qi_cur++;
            sti_cur = 0;
        }

        // 根据Query的Trace 输出 相关信息
        void outputStatInfo(std::string path_file) {
            std::ofstream writer(path_file.c_str());
            // global average #nbor
            float nbor_avg_base = nborGlobalAvg(QueryTraceBeHashSet);
            float nbor_avg_hash = nborGlobalAvg(QueryTraceAfHashSet);
            float hit_rate = 1.0 - nbor_avg_hash / nbor_avg_base;
            float rank_valid = rankValidGlobal(QueryTraceBeHashSet, 8);

            // output
            writer << "#round " << num_step << "\n";
            writer << "#nbor_per_round " << nbor_avg_base << "\n";
            writer << "#r_for_8 " << rank_valid << "\n";
            writer << "hit_rate " << hit_rate << "\n";

            writer.close();
        }
        /*
            search_point num_visited
            list of neighbor before hash
            list of neighbor after hash
        */
        void writeQueryTrace(std::string path_qt) {
            std::ofstream writer(path_qt.c_str());
            writer << "# " << query_size << " " << max_nbor_size << " " << num_step << "\n";

            for (int qi = 0; qi < query_size; qi++) {
                writer << "# qid=" << qi << "\n";
                if (QueryPointSet[qi].size() != num_step) {
                    printf("Error, search point size is: %lu\n", QueryPointSet[qi].size()); exit(1);
                }

                for (int sti = 0; sti < num_step; sti++) {
                    int num_visited = QueryTraceBeHashSet[qi][sti].size() - QueryTraceAfHashSet[qi][sti].size();
                    if (num_visited < 0) {
                        printf("Error, num_visited is: %d\n", num_visited); exit(1);
                    }

                    writer << QueryPointSet[qi][sti] << " " << num_visited << "\n";
                    for (tableint id: QueryTraceBeHashSet[qi][sti])
                        writer << id << " ";
                    writer << "\n";
                    for (tableint id: QueryTraceAfHashSet[qi][sti])
                        writer << id << " ";
                    writer << "\n";
                }
            }
            writer.close();

            printf("write query trace to %s done\n", path_qt.c_str());
        }
    private:
        int query_size, max_nbor_size, num_step;
        int qi_cur, sti_cur;

        // Query trace
        std::vector<std::vector<std::vector<tableint>>> QueryTraceBeHashSet;
        std::vector<std::vector<std::vector<tableint>>> QueryTraceAfHashSet;
        std::vector<std::vector<tableint>> QueryPointSet;

        void initQueryTrace() {
            qi_cur = 0;
            sti_cur = 0;
            QueryTraceBeHashSet.resize(query_size);
            QueryTraceAfHashSet.resize(query_size);
            QueryPointSet.resize(query_size);
            for (int i = 0; i < query_size; i++) {
                QueryTraceBeHashSet[i].resize(num_step);
                QueryTraceAfHashSet[i].resize(num_step);
                QueryPointSet[i].resize(num_step);
            }
        }

        float nborGlobalAvg(std::vector<std::vector<std::vector<tableint>>>& QueryTraceSet) {
            float nbor_avg = 0;
            std::vector<float> nbor_avg_query;
            for (std::vector<std::vector<tableint>>& QT_query: QueryTraceSet) {
                size_t nbor_all_query = 0;
                for (std::vector<tableint>& QT_step: QT_query)
                    nbor_all_query += QT_step.size();
                nbor_avg_query.push_back(1.0 * nbor_all_query / num_step);
            }
            nbor_avg = std::accumulate(nbor_avg_query.begin(), nbor_avg_query.end(), 0.0) / query_size;
            return nbor_avg;
        }
        float rankValidGlobal(std::vector<std::vector<std::vector<tableint>>>& QueryTraceSet, int num_rank) {
            // todo: internal or external?
            size_t NDC_total = 0, NDC_rank_max = 0;
            for (std::vector<std::vector<tableint>>& QT_query: QueryTraceSet) {
                for (std::vector<tableint>& QT_step: QT_query) {
                    NDC_total += QT_step.size();
                    std::vector<size_t> tmp_NDC(num_rank, 0);
                    for (tableint& id: QT_step)
                        tmp_NDC[rankMapping(id, num_rank)]++;
                    int pos_max = std::max_element(tmp_NDC.begin(), tmp_NDC.end()) - tmp_NDC.begin();
                    NDC_rank_max += tmp_NDC[pos_max];
                }
            }
            float rank_valid = 1.0 * NDC_total / NDC_rank_max;
            return rank_valid;
        }
        // mapping strategy: mod
        inline int rankMapping(tableint id, int num_rank) {
            return (int)(id % num_rank);
        }
    };
#endif

}

#include "space_l2.h"
#include "space_ip.h"
#include "bruteforce.h"
#include "hnswalg.h"
