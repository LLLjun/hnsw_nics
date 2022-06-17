#ifndef MEM_H
#define MEM_H

#include <stdio.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <fstream>
#include <stdint.h>
#include <algorithm>

#define MEM_ALIGNED     64
#define RANK_SIZE_GB    4

#define QUERY_SIZE_KB   1
#define ALLOC_SIZE_KB   1
#define GATHER_SIZE_KB  1

typedef uint64_t        DTaddr;

using namespace std;

/*
    分析 rank_opt 代码中HNSW的访存记录
    硬件地址(dram access trace)
        |- feature group: channel0所对应的全部rank，分散分配，最大支持8ranks
        |- neighbor group: channel0所对应的全部rank，按照顺序填充
    其他(visited list和queue记录次数)
*/


struct trace {
    char type;          // l for load, s for save
    DTaddr addr;
};
struct TraceVisitedList {
    size_t n_read;
    size_t n_write;
};
struct TraceQueue {
    size_t n_insert;
    size_t n_readmin;
    size_t n_readmax;
};

class MemTrace {
private:
    bool track_trace_detail = true;
    vector<trace> MemTraceRank;
    size_t num_read_trace, num_write_trace;
    TraceVisitedList MemStatVisitedList;
    TraceQueue MemStatQueue;

    // 目前邻居的存储方式采用规整存储的方式，最开始的4bytes是有效邻居的大小。
    size_t v_size, v_size_aligned;
    size_t v_line, v_maxinum_rank;
    size_t nl_size, nl_size_aligned;
    size_t nl_line, nl_maxinum_rank;

    // 每个rank都给query预留一部分空间
    int channel_feature = 0;
    int channel_neighbor = 1;
    // 一些最大限制
    DTaddr RANK_ADDR_LIMIT = (DTaddr)1 << 36;

public:
    MemTrace(int featureSize, int neighborMaxNum) {
        v_size = featureSize;
        v_line = (size_t) ceil(1.0 * v_size / MEM_ALIGNED);
        v_size_aligned = v_line * MEM_ALIGNED;
        v_maxinum_rank = ((size_t)RANK_SIZE_GB * 1024 * 1024 * 1024) / v_size_aligned;

        nl_size = sizeof(int) * (1 + neighborMaxNum);
        nl_line = (size_t) ceil(1.0 * nl_size / MEM_ALIGNED);
        nl_size_aligned = nl_line * MEM_ALIGNED;
        nl_maxinum_rank = ((size_t)RANK_SIZE_GB * 1024 * 1024 * 1024) / nl_size_aligned;

        ResetStat();

        printf("\nMemory Trace Analysis ...\n");
        printf("[rank-level] maxinum vector: %lu, maxinum neighbor: %lu\n", v_maxinum_rank, nl_maxinum_rank);
        printf("[point-level] vector line: %lu, neighbor line: %lu\n", v_line, nl_line);
    }
    ~MemTrace() {
        vector<trace>().swap(MemTraceRank);
    }

    DTaddr TransAddrFormat(int channelId, int rankId, DTaddr addrInput);
    void AddTraceQuery(int rankId);
    void AddTraceFeature(int rankId, int innerId);
    void AddTraceNeighbor(int internalId);

    void ResetStat();
    void AddStatQueue(char name, int nums=1);
    void AddStatVistedList(char name, int nums=1);

    int write_file(std::string& filePrefix, size_t max_num_trace = (size_t)(-1));
    size_t count_trace(char mode = 'a'); // a:all, l:load, s:save
};

DTaddr MemTrace::TransAddrFormat(int channelId, int rankId, DTaddr addrInput) {
    DTaddr addrOutput;
    DTaddr a_byte = addrInput % MEM_ALIGNED;
    DTaddr a_channel = (DTaddr) channelId;
    DTaddr a_column = (addrInput / MEM_ALIGNED) % 128;
    DTaddr a_rank = (DTaddr) rankId;
    DTaddr a_high = (addrInput / MEM_ALIGNED) >> 7;
    addrOutput = (a_high << 17) + (a_rank << 14) + (a_column << 7)
                + (a_channel << 6) + a_byte;
    if (addrOutput > RANK_ADDR_LIMIT){
        printf("trans address: out rank range\n"); exit(1);
    }

    return addrOutput;
}

void MemTrace::AddTraceQuery(int rankId) {
    trace this_trace;
    this_trace.type = 's';
    for (int i = 0; i < v_line; i++){
        this_trace.addr = TransAddrFormat(channel_feature, rankId, (DTaddr)(i * MEM_ALIGNED));
        MemTraceRank.push_back(this_trace);
    }
}

void MemTrace::AddTraceFeature(int rankId, int innerId) {
    if (innerId >= v_maxinum_rank){
        printf("feature: out rank range\n"); exit(1);
    }
    trace this_trace;
    this_trace.type = 'l';

    DTaddr start = innerId * v_line * MEM_ALIGNED;
    for (int i = 0; i < v_line; i++){
        this_trace.addr = TransAddrFormat(channel_feature, rankId, (start + i * MEM_ALIGNED));
        MemTraceRank.push_back(this_trace);
    }
}

void MemTrace::AddTraceNeighbor(int internalId) {
    trace this_trace;
    this_trace.type = 'l';

    int rankid = internalId / nl_maxinum_rank;
    int innerid = internalId % nl_maxinum_rank;
    DTaddr start = innerid * nl_line * MEM_ALIGNED;
    for (int i = 0; i < nl_line; i++) {
        this_trace.addr = TransAddrFormat(channel_neighbor, rankid, (start + i * MEM_ALIGNED));
        MemTraceRank.push_back(this_trace);
    }
}

void MemTrace::ResetStat() {
    MemStatQueue.n_insert = 0;
    MemStatQueue.n_readmax = 0;
    MemStatQueue.n_readmin = 0;
    MemStatVisitedList.n_read = 0;
    MemStatVisitedList.n_write = 0;
}

void MemTrace::AddStatQueue(char name, int nums) {
    if (name == 'i')
        MemStatQueue.n_insert += nums;
    else if (name == 'x')
        MemStatQueue.n_readmax += nums;
    else if (name == 'n')
        MemStatQueue.n_readmin += nums;
    else {
        printf("StstQueue: no such name\n"); exit(1);
    }
}

void MemTrace::AddStatVistedList(char name, int nums) {
    if (name == 'r')
        MemStatVisitedList.n_read += nums;
    else if (name == 'w')
        MemStatVisitedList.n_write += nums;
    else {
        printf("StatVistedList: no such name\n"); exit(1);
    }
}

int MemTrace::write_file(std::string& filePrefix, size_t max_num_trace) {
    string path_trace = filePrefix + "_trace.txt";
    std::ofstream file_out(path_trace.c_str());
    const size_t endpos = MemTraceRank.size();
    for (size_t i = 0; i < endpos; ++i) {
        file_out << MemTraceRank[i].type << " " << "0x" << std::setfill('0') << std::setw(9) << std::hex << MemTraceRank[i].addr << '\n';
    }
    file_out.close();
    std::cout << "write " << std::dec << endpos << " trace" << std::endl;
    if(track_trace_detail==false)
        std::cout << "did not write all trace" << std::endl;
    std::cout << std::endl;

    string path_stat = filePrefix + "_stat.txt";
    std::ofstream file_stat(path_stat.c_str());
    file_stat << "memory trace configuration:\n";
    file_stat << "n_ranks: " << NUM_RANKS << "\n";
    file_stat << "opt_visited: " << OPT_VISITED << "\n";
    file_stat << "\n";

    file_stat << "[visited] n_read: " << MemStatVisitedList.n_read << "\n";
    file_stat << "[visited] n_write: " << MemStatVisitedList.n_write << "\n";
    file_stat << "[queue] n_insert: " << MemStatQueue.n_insert << "\n";
    file_stat << "[queue] n_readmax: " << MemStatQueue.n_readmax << "\n";
    file_stat << "[queue] n_readmin: " << MemStatQueue.n_readmin << "\n";
    file_stat.close();

    return 0;
}

size_t MemTrace::count_trace(char mode) {
    long long count = 0;
    if (track_trace_detail) {
        switch (mode) {
        case 'a':
            count = MemTraceRank.size();
            break;
        case 'l':
            for (auto it = MemTraceRank.begin(); it != MemTraceRank.end(); ++it) {
                count = count + (it->type == 'l');
            }
            break;
        case 's':
            for (auto it = MemTraceRank.begin(); it != MemTraceRank.end(); ++it) {
                count = count + (it->type == 's');
            }
            break;
        default:
            std::cout << "usage: mode = {a:all, l:load, s:save}" << std::endl;
            count = -1;
            break;
        }
    }
    else {
        switch (mode) {
        case 'a':
            count = num_read_trace + num_write_trace;
            break;
        case 'l':
            count = num_read_trace;
            break;
        case 's':
            count = num_write_trace;
            break;
        default:
            std::cout << "usage: mode = {a:all, l:load, s:save}" << std::endl;
            count = -1;
            break;
        }
    }
    return count;
}


#endif /* MEM_H */
