#pragma once
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
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

#if PARTGRAPH
    class PartGraph {
    public:
        PartGraph(std::string dataname, int vecsize, int qsize, int n_pg, int efs) {
            data_name = dataname;
            base_size = vecsize;
            query_size = qsize;
            num_partgraph = n_pg;
            num_step = efs;
            query_cur = 0;
            step_cur = 0;
            reserve_size = 100;

            SearchPointTable.resize(query_size);
            for (std::vector<tableint>& SearchPointList: SearchPointTable)
                SearchPointList.resize(num_step);
            CalcuNeighTable.resize(query_size);
            for (std::vector<std::vector<tableint>>& CalcuNeighList: CalcuNeighTable) {
                CalcuNeighList.resize(num_step);
                for (std::vector<tableint>& CalcuNeighPoint: CalcuNeighList)
                    CalcuNeighPoint.reserve(reserve_size);
            }
        }

        void addSearchPoint(tableint id) {
            SearchPointTable[query_cur][step_cur] = id;
        }

        void addCalcuNeighbor(tableint id) {
            CalcuNeighTable[query_cur][step_cur].push_back(id);
        }

        void endStep() {
            step_cur++;
        }

        void endQuery() {
            query_cur++;
            step_cur = 0;
        }

        void testPartGraph(int part_size) {
            // test random
            printf("Part-graph Test [random]:\n");
            std::vector<int> IdToPartitionRandom(base_size);
            srand(time(0));
            for (int i = 0; i < base_size; i++)
                IdToPartitionRandom[i] = rand() % part_size;
            evalCommucation(IdToPartitionRandom);

            // test METIS
            printf("Part-graph Test [METIS]:\n");
            std::vector<int> IdToPartition = getIdToPartMap(part_size);
            evalCommucation(IdToPartition);
        }

        std::string getFileName(int num_part_graph) {
            int size_million = base_size / 1e6;
            std::string file_partition = "../output/part-graph/" + data_name + std::to_string(size_million) +
                                         "m.txt.part." + std::to_string(num_part_graph);
            printf("Partition file: %s\n", file_partition.c_str());
            return file_partition;
        }

        // Transfer search
        void initTransferSearch() {
            request_threshold = 40;
            IdToPartitionMap = getIdToPartMap(num_partgraph);

            qc_transfer_request.resize(num_partgraph);
            initTransInfo();

            transfer_size = 0;
            partgraph_computation.resize(num_partgraph, 0);
        }

        bool keepIdInGraph(tableint id) {
            bool keep = true;
            int pg_id = IdToPartitionMap[id];
            if (pg_id != qc_local_graph) {
                qc_transfer_request[pg_id].push_back(id);
                keep = false;
            }
            return keep;
        }

        int statTransferBySize() {
            int stat = -1;
            int max_size = 0;
            for (int i = 0; i < num_partgraph; i++) {
                int rq_size = qc_transfer_request[i].size();
                if (rq_size >= request_threshold && rq_size > max_size) {
                    stat = i;
                    max_size = rq_size;
                }
            }
            return stat;
        }

        int statTransferByStep(int step_cur) {
            int stat = -1;
            if ((step_cur + 1) % 8 == 0) {
                int max_size = 0;
                for (int i = 0; i < num_partgraph; i++) {
                    int rq_size = qc_transfer_request[i].size();
                    if (rq_size > max_size) {
                        stat = i;
                        max_size = rq_size;
                    }
                }
            }
            return stat;
        }

        void setLocalGraph(int pg_id) {
            qc_local_graph = pg_id;
        }

        int getLocalGraph() {
            return qc_local_graph;
        }

        std::vector<tableint> popRequest(int pg) {
            std::vector<tableint> popRq;
            popRq.swap(qc_transfer_request[pg]);

            transfer_size++;
            return popRq;
        }

        void initTransInfo() {
            qc_local_graph = IdToPartitionMap[0];
            for (std::vector<tableint>& rq: qc_transfer_request)
                std::vector<tableint>().swap(rq);
        }

        void collectComputation(tableint id) {
            int pg = IdToPartitionMap[id];
            partgraph_computation[pg]++;
        }

        // 统计transfer次数
        size_t transfer_size;
        std::vector<size_t> partgraph_computation;

        void printfTransferStat() {
            printf("Transfer.Num\n");
            printf("%.1f\n", 1.0 * transfer_size / query_size);
            printf("\n");
        }

        size_t getMaxComputation() {
            size_t max_size = *(std::max_element(partgraph_computation.begin(), partgraph_computation.end()));
            return max_size;
        }

        int getNumPartGraph() {
            return num_partgraph;
        }


    private:
        int base_size, query_size, num_partgraph, num_step;
        std::vector<std::vector<tableint>> SearchPointTable;
        std::vector<std::vector<std::vector<tableint>>> CalcuNeighTable;

        int query_cur, step_cur;
        int reserve_size;

        std::string data_name;

        std::vector<int> getIdToPartMap(int part_size) {
            std::string partitionMapFile = getFileName(part_size);
            std::vector<int> IdToPartition(base_size, part_size);

            std::ifstream reader(partitionMapFile.c_str());
            if (reader) {
                for (int i = 0; i < base_size; i++) {
                    std::string line;
                    std::getline(reader, line);
                    IdToPartition[i] = std::stoi(line);
                }
                reader.close();
            } else {
                printf("Error, file unexist: %s\n", partitionMapFile.c_str());
                exit(1);
            }
            for (int i = 0; i < 10; i++)
                printf("%d\t", IdToPartition[i]);
            printf("\n");
            for (int pt: IdToPartition) {
                if (pt >= part_size) {
                    printf("Error, read partition error \n"); exit(1);
                }
            }
            return IdToPartition;
        }

        void evalCommucation(std::vector<int>& IdToPartition) {
            size_t num_search_point = query_size * num_step;
            size_t num_calcu_neighbor = 0;
            for (std::vector<std::vector<tableint>>& CalcuNeighList: CalcuNeighTable) {
                for (std::vector<tableint>& CalcuNeighPoint: CalcuNeighList)
                    num_calcu_neighbor += CalcuNeighPoint.size();
            }

            // 只考虑单个query，定义local graph
            int graph_local = IdToPartition[0];
            size_t commu_search_point = 0;
            size_t commu_calcu_neighbor = 0;

            for (int qi = 0; qi < query_size; qi++) {
                for (int sti = 0; sti < num_step; sti++) {
                    tableint sp_cur = SearchPointTable[qi][sti];
                    if (IdToPartition[sp_cur] != graph_local)
                        commu_search_point++;
                    for (tableint neigh: CalcuNeighTable[qi][sti]) {
                        if (IdToPartition[neigh] != graph_local)
                            commu_calcu_neighbor++;
                    }
                }
            }

            // printf
            printf("Search point\t Calcu neighbor\t Commu.Point\t Commu.Neighbor\n");
            printf("%lu\t %lu\t %.1f%%\t %.1f%%\n",
                            num_search_point, num_calcu_neighbor,
                            100.0 * commu_search_point / num_search_point,
                            100.0 * commu_calcu_neighbor / num_calcu_neighbor);
        }

        // to support transfer search
        // 超参数
        int request_threshold;

        std::vector<int> IdToPartitionMap;
        // 存储单个query的transfer 请求
        int qc_local_graph;
        // 可能有重复
        std::vector<std::vector<tableint>> qc_transfer_request;



    };
#endif

}

#include "space_l2.h"
#include "space_ip.h"
#include "bruteforce.h"
#include "hnswalg.h"
