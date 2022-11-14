#pragma once
#include "mem.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <sys/stat.h>
#include <utility>
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
    template<typename data_t>
    class QueryTrace {
    public:
        QueryTrace(int qsize, int feature_dim, int nbor_size, int efs) {
            query_size = qsize;
            max_nbor_size = nbor_size;
            num_step = efs;
            initQueryTrace();
            Trace = new MemTrace(feature_dim * sizeof(data_t), nbor_size);
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
        void outputStatInfo(std::string path_output, int batch_size) {
            std::string dir_trace = path_output + "/traces/" + "bs" + to_string(batch_size);
            createDir(dir_trace);
            std::string path_file = dir_trace + "/0_stat.csv";
            std::ofstream writer(path_file.c_str());
            // global average #nbor
            float nbor_avg_base = nborGlobalAvg(QueryTraceBeHashSet);
            float nbor_avg_hash = nborGlobalAvg(QueryTraceAfHashSet);
            float hit_rate = 1.0 - nbor_avg_hash / nbor_avg_base;
            float rank_valid = rankValidGlobal(QueryTraceBeHashSet, batch_size, 8);

            // output
            writer << "#round " << num_step << "\n";
            writer << "#nbor_per_round " << nbor_avg_base << "\n";
            writer << "#r_for_8 " << rank_valid << "\n";
            writer << "hit_rate " << hit_rate << "\n";

            writer.close();
        }
        void ouputMemTrace(std::string path_output, int batch_size) {
            std::string dir_trace = path_output + "/traces/" + "bs" + to_string(batch_size);
            createDir(dir_trace);

            // rankToMemTrace(Trace, dir_trace, batch_size, 8);
            rankSelectTrace(Trace, dir_trace, batch_size, 8);
        }
        /*
            for fpga
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
        MemTrace* Trace = nullptr;

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
        float rankValidGlobal(std::vector<std::vector<std::vector<tableint>>>& QueryTraceSet, int batch_size, int num_rank) {
            size_t NDC_total = 0, NDC_rank_max = 0;
            for (int begin = 0; begin < query_size; begin += batch_size) {
                int end = std::min(query_size, begin + batch_size);

                size_t len_ft = 0;
                for (int si = 0; si < num_step; si++) {
                    std::vector<size_t> tmp_NDC(num_rank, 0);
                    for (int qi = begin; qi < end; qi++) {
                        std::vector<tableint>& nbor_list = QueryTraceSet[qi][si];
                        NDC_total += nbor_list.size();
                        for (tableint& id: nbor_list)
                            tmp_NDC[rankMapping(id, num_rank)]++;
                    }
                    int pos_max = std::max_element(tmp_NDC.begin(), tmp_NDC.end()) - tmp_NDC.begin();
                    NDC_rank_max += tmp_NDC[pos_max];
                }
            }
            float rank_valid = 1.0 * NDC_total / NDC_rank_max;
            return rank_valid;
        }
        // select batch query's trace
        void rankSelectTrace(MemTrace* trace, std::string dir_trace, int batch_size, int num_rank) {
            // select avg max min
            int batch_num = query_size / batch_size;
            std::vector<std::vector<int>> rank_id_round(batch_num, std::vector<int>(num_step, -1));
            std::vector<size_t> len_all_round(batch_num, 0);

            for (int bi = 0; bi < batch_num; bi++) {
                int begin = bi * batch_size;
                int end = begin + batch_size;

                size_t len_ft = 0;
                for (int si = 0; si < num_step; si++) {
                    std::vector<size_t> tmp_NDC(num_rank, 0);
                    for (int qi = begin; qi < end; qi++) {
                        std::vector<tableint>& nbor_list = QueryTraceAfHashSet[qi][si];
                        for (tableint& id: nbor_list)
                            tmp_NDC[rankMapping(id, num_rank)]++;
                    }
                    int pos_max = std::max_element(tmp_NDC.begin(), tmp_NDC.end()) - tmp_NDC.begin();
                    rank_id_round[bi][si] = pos_max;
                    len_ft += tmp_NDC[pos_max];
                }
                len_all_round[bi] = len_ft;
            }

            // output info
            std::string file_info = dir_trace + "/0_info.csv";
            float t_avg = 1.0 * std::accumulate(len_all_round.begin(), len_all_round.end(), 0) / len_all_round.size();
            int t_avg_i = findNearest(len_all_round, t_avg);
            int t_max_i = std::max_element(len_all_round.begin(), len_all_round.end()) - len_all_round.begin();
            int t_min_i = std::min_element(len_all_round.begin(), len_all_round.end()) - len_all_round.begin();
            ofstream info_writer(file_info.c_str());
            info_writer << "Type,Value,Range" << "\n";
            info_writer << "Avg," << len_all_round[t_avg_i] << "," << to_string(t_avg_i * batch_size) + "-" + to_string((t_avg_i+1) * batch_size) << "," << t_avg << "\n";
            info_writer << "Max," << len_all_round[t_max_i] << "," << to_string(t_max_i * batch_size) + "-" + to_string((t_max_i+1) * batch_size) << "\n";
            info_writer << "Min," << len_all_round[t_min_i] << "," << to_string(t_min_i * batch_size) + "-" + to_string((t_min_i+1) * batch_size) << "\n";
            info_writer.close();
            // output trace
            std::vector<std::pair<std::string, int>> type_pair(3);
            type_pair[0] = std::make_pair("avg", t_avg_i);
            type_pair[1] = std::make_pair("max", t_max_i);
            type_pair[2] = std::make_pair("min", t_min_i);
            for (std::pair<std::string, int>& type: type_pair) {
                std::string file_trace = dir_trace + "/" + type.first + ".txt";
                int cur_batch = type.second;
                trace->ResetTrace();
                for (int si = 0; si < num_step; si++) {
                    int cur_rank = rank_id_round[cur_batch][si];
                    for (int qi = cur_batch * batch_size; qi < cur_batch * batch_size + batch_size; qi++) {
                        std::vector<tableint>& nbor_list = QueryTraceAfHashSet[qi][si];
                        for (tableint& id: nbor_list) {
                            if (rankMapping(id, num_rank) == cur_rank)
                                trace->AddTraceFeature(0, idMapping(id, num_rank));
                        }
                    }
                }
                trace->WriteTrace(file_trace);
            }
        }
        // all batch query's trace
        void rankToMemTrace(MemTrace* trace, std::string dir_trace, int batch_size, int num_rank) {
            for (int begin = 0; begin < query_size; begin += batch_size) {
                int end = std::min(query_size, begin + batch_size);
                std::string file_trace = dir_trace + "/" + to_string(begin) + "-" + to_string(end);
                createDir(file_trace);
                for (int si = 0; si < num_step; si++) {
                    std::string trace_name = file_trace + "/" + to_string(si) + ".txt";
                    trace->ResetTrace();
                    std::vector<std::vector<tableint>> rankAllocer(num_rank);
                    for (int qi = begin; qi < end; qi++) {
                        std::vector<tableint>& nbor_list = QueryTraceAfHashSet[qi][si];
                        for (tableint& id: nbor_list)
                            rankAllocer[rankMapping(id, num_rank)].push_back(id);
                    }
                    std::vector<size_t> tmp_NDC(num_rank, 0);
                    for (int i = 0; i < num_rank; i++)
                        tmp_NDC[i] = rankAllocer[i].size();
                    int pos_max = std::max_element(tmp_NDC.begin(), tmp_NDC.end()) - tmp_NDC.begin();
                    // trace
                    for (tableint& internal_id: rankAllocer[pos_max])
                        trace->AddTraceFeature(pos_max, idMapping(internal_id, num_rank));
                    trace->WriteTrace(trace_name);
                }
            }
        }
        // mapping strategy: mod
        inline int rankMapping(tableint id, int num_rank) {
            return (int)(id % num_rank);
        }
        inline tableint idMapping(tableint id, int num_rank) {
            return (id / num_rank);
        }
        int findNearest(std::vector<size_t>& list, float value) {
            int fid = 0;
            float diff = std::numeric_limits<float>::max();
            for (size_t i = 0; i < list.size(); i++) {
                float tmp_diff = std::abs(value - list[i]);
                if (tmp_diff < diff) {
                    diff = tmp_diff;
                    fid = i;
                }
            }
            return fid;
        }
    };
#endif

}

#include "space_l2.h"
#include "space_ip.h"
#include "bruteforce.h"
#include "hnswalg.h"
