#pragma once
#include <cstdio>
#include <cstdlib>
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

    struct SubPoint{
        int graph;
        int ingraph_id;

        SubPoint() = default;
        SubPoint(int graph, int ingraph_id) : graph{graph}, ingraph_id{ingraph_id} {}
    };

    // 分配整张图的点策略
    class AllocSubGraph {
    public:
        AllocSubGraph(string name, int nums_point, int nums_graph) {
            dataname = name;
            n_point_total = nums_point;
            n_subgraph = nums_graph;

            OriginToSubPoint.resize(n_point_total);
            SubPointToOrigin.resize(n_subgraph);
            SubCenter.resize(n_subgraph, 0);
            printf("[SubGraph] num_graph: %d, n_point_total: %d\n", n_subgraph, n_point_total);

#if SG_METIS
            MetisAlloc();
#else
            RandomAlloc();
#endif
        }

        // 根据分配后的mapping关系，得到对应subgraph的中心点
        // 支持no-balance 分配
        template<typename data_T>
        void computSubgCenter(const data_T *data_m, uint32_t dims) {
            for (int i_sg = 0; i_sg < n_subgraph; i_sg++) {
                int size = SubPointToOrigin[i_sg].size();
                data_T* data_s = new data_T[size * dims]();
                for (int i_ig = 0; i_ig < size; i_ig++) {
                    int origin_id = SubPointToOrigin[i_sg][i_ig];
                    memcpy(data_s + dims * i_ig, data_m + dims * origin_id, dims * sizeof(data_T));
                }
                int ingraph_id = compArrayCenter<data_T>(data_s, size, dims);
                SubCenter[i_sg] = SubPointToOrigin[i_sg][ingraph_id];
                // 放在0号位置
                swapMapByIngraphId(i_sg, 0, ingraph_id);
                delete[] data_s;
            }
            printf("ComputSubgCenter successed\n");
        }

        int getSubgCenter(int subg) {
            return SubCenter[subg];
        }

        int getOriginId(int subg_i, int ingraph_i) {
            return SubPointToOrigin[subg_i][ingraph_i];
        }

        size_t getSubgSize(int subg) {
            return SubPointToOrigin[subg].size();
        }

    private:
        string dataname;
        int n_point_total, n_point_subg;
        int n_subgraph;
        std::vector<SubPoint> OriginToSubPoint;
        std::vector<std::vector<int>> SubPointToOrigin;
        std::vector<int> SubCenter;

        // 随机分配
        void RandomAlloc() {
            if (n_point_total % n_subgraph != 0) {
                printf("Error, unsupport the n_subgraph\n"); exit(1);
            }
            n_point_subg = n_point_total / n_subgraph;
            for (std::vector<int>& subgraph: SubPointToOrigin)
                subgraph.resize(n_point_subg, 0);

            srand((unsigned)time(NULL));
            std::vector<int> random_list(n_point_total);
            for (int i = 0; i < n_point_total; i++)
                random_list[i] = i;
            shuffle_vector<int>(random_list);

            for (int i_sg = 0; i_sg < n_subgraph; i_sg++) {
                for (int i_ig = 0; i_ig < n_point_subg; i_ig++) {
                    int origin_id = random_list[i_sg * n_point_subg + i_ig];
                    SubPointToOrigin[i_sg][i_ig] = origin_id;
                    OriginToSubPoint[origin_id].graph = i_sg;
                    OriginToSubPoint[origin_id].ingraph_id = i_ig;
                }
            }
            printf("RandomAlloc successed\n");
        }

        // 根据METIS的clustering结果分配
        void MetisAlloc() {
            vector<int> IdToPart = getIdToPart();
            vector<int> part_i(n_subgraph, 0);
            for (int i = 0; i < n_point_total; i++) {
                int graph = IdToPart[i];
                OriginToSubPoint[i].graph = graph;
                OriginToSubPoint[i].ingraph_id = part_i[graph];
                SubPointToOrigin[graph].push_back(i);
                part_i[graph]++;
            }

            for (int pi = 0; pi < n_subgraph; pi++) {
                if (part_i[pi] != SubPointToOrigin[pi].size()) {
                    printf("Error, size is error\n"); exit(1);
                }
            }
            printf("MetisAlloc successed\n");
        }

        vector<int> getIdToPart() {
            int size_million = n_point_total / 1e6;
            string path_txt = "/home/ljun/self_data/hnsw_nics/output/part-graph/" + dataname + to_string(size_million) + "m.txt";
            string metis_file = path_txt + ".part." + to_string(n_subgraph);

            vector<int> IdToPart(n_point_total);

            std::ifstream reader(metis_file.c_str());
            if (reader) {
                for (int i = 0; i < n_point_total; i++) {
                    std::string line;
                    std::getline(reader, line);
                    IdToPart[i] = std::stoi(line);
                }
                reader.close();
            } else {
                printf("Error, file unexist: %s\n", metis_file.c_str());
                exit(1);
            }
            return IdToPart;
        }

        void swapMapByIngraphId(int subg, int ingraph_id_x, int ingraph_id_y) {
            int origin_x = SubPointToOrigin[subg][ingraph_id_x];
            int origin_y = SubPointToOrigin[subg][ingraph_id_y];

            OriginToSubPoint[origin_x].ingraph_id = ingraph_id_y;
            OriginToSubPoint[origin_y].ingraph_id = ingraph_id_x;
            SubPointToOrigin[subg][ingraph_id_x] = origin_y;
            SubPointToOrigin[subg][ingraph_id_y] = origin_x;
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
            if (times == other.times)
                return id > other.id;
            else
                return times < other.times;
        }
    };

    class HotData {
    public:
        HotData(size_t nums_range, size_t nums_point) {
            range_size = nums_range;
            base_size = nums_point;
            printf("[Hotdata] range: %lu, base size: %lu\n", range_size, base_size);
        }

        void AddTimes(tableint id) {
            if (isTrain)
                AccessTimesTrain[id]++;
            else
                AccessTimesTest[id]++;
        }

        void initTrainSample(size_t sample_size, size_t qsize) {
            setTrainStats(true);
            train_size = sample_size;
            test_size = qsize;
        }

        void setTrainStats(bool is_train) {
            isTrain = is_train;
            if (isTrain) {
                std::vector<size_t>().swap(AccessTimesTrain);
                AccessTimesTrain.resize(range_size, 0);
            } else {
                std::vector<size_t>().swap(AccessTimesTest);
                AccessTimesTest.resize(range_size, 0);
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
            size_t interval = ratio_max * base_size / n_steps;

            size_t accessCurrent = 0;

            printf("Points(%%)\t Access(%%)\t Freq.Avg\t Max\t Min\t Match(%%)\t M.Access(%%)\n");
            for (int si = 0; si < n_steps; si++) {
                size_t begin = si * interval;
                size_t end = begin + interval;

                // 分析 Train 数据信息
                size_t accessTmpTrain = 0;
                for (size_t i = begin; i < end; i++) {
                    accessTmpTrain += AccessTdtimesTrain[i].times;
                }
                accessCurrent += accessTmpTrain;
                // printf
                printf("%.1f%%\t %.1f%%\t %.5f\t %.5f\t %.5f\t ",
                                100.0 * end / base_size,
                                100.0 * accessCurrent / accessTotalTrain,
                                1.0 * accessTmpTrain / interval / train_size,
                                1.0 * AccessTdtimesTrain[begin].times / train_size,
                                1.0 * AccessTdtimesTrain[end-1].times / train_size);

                // 比较 Test 和 Train 的一致程度
                size_t n_hit = 0;
                size_t accessTmpTest = 0;
                std::unordered_set<tableint> pointSet;
                for (size_t i = 0; i < end; i++) {
                    pointSet.emplace(AccessTdtimesTrain[i].id);
                }
                for (size_t i = 0; i < end; i++) {
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

        // 根据AccessTimesTrain 输出 hot/cold data
        void writeHotColdId(std::string path_hc, float hot_r=0.3) {
            std::vector<Idtimes> AccessTdtimesTrain;
            transTimesToTdtimes(AccessTimesTrain, AccessTdtimesTrain);

            size_t total_size = base_size;

            // output
            std::ofstream writer(path_hc.c_str());
            writer << "# " << total_size << "\n";
            for (size_t i = 0; i < total_size; i++)
                writer << AccessTdtimesTrain[i].id << "\n";
            writer.close();

            printf("write hot/cold data to %s done\n", path_hc.c_str());
        }

    private:
        size_t range_size, base_size;

        // For Training
        std::vector<size_t> AccessTimesTrain;
        std::vector<size_t> AccessTimesTest;
        bool isTrain;
        size_t qi_cur;
        size_t train_size, test_size;

        void transTimesToTdtimes(std::vector<size_t>& timesList, std::vector<Idtimes>& timesPair) {
            timesPair.resize(range_size, Idtimes(0, 0));

            for (size_t i = 0; i < range_size; i++) {
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

#if QTRACE
    class QueryTrace {
    public:
        QueryTrace(int qsize, int nbor_size, int efs) {
#if HOTDATA
            printf("Error, During analysis hot data, can't generate query trace\n"); exit(1);
#endif
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
    };
#endif

}

#include "space_l2.h"
#include "space_ip.h"
#include "bruteforce.h"
#include "hnswalg.h"
