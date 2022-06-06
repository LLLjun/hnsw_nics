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
#define RANK_ADDR_LIMIT (1 << 36)

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
    char type; // l for load, s for save
    DTaddr addr;
};

class MemTrace {
private:
    bool track_trace_detail = true;
    vector<trace> mem_trace;
    size_t num_read_trace, num_write_trace;

    // 目前邻居的存储方式采用规整存储的方式，最开始的4bytes是有效邻居的大小。
    size_t v_size, v_size_aligned;
    size_t v_line, v_maxinum_rank;
    size_t nl_size, nl_size_aligned;
    size_t nl_line, nl_maxinum_rank;

    // 每个rank都给query预留一部分空间
    int channel_feature = 0;
    int channel_neighbor = 1;
    size_t s_query_set = 1024;
    DTaddr f_addr_start = (DTaddr) s_query_set;


public:
    MemTrace(int featureSize, int neighborMaxNum) {
        v_size = featureSize;
        v_line = (size_t) ceil(1.0 * v_size / MEM_ALIGNED);
        v_size_aligned = v_line * MEM_ALIGNED;
        v_maxinum_rank = ((RANK_SIZE_GB << 9) - s_query_set) / v_size_aligned;

        nl_size = sizeof(int) * (1 + neighborMaxNum);
        nl_line = (size_t) ceil(1.0 * nl_size / MEM_ALIGNED);
        nl_size_aligned = nl_line * MEM_ALIGNED;
        nl_maxinum_rank = (RANK_SIZE_GB << 9) / nl_size_aligned;
    }
    ~MemTrace() {
        vector<trace>().swap(mem_trace);
    }

    void AddTraceQuery(int rankId);
    void AddTraceFeature(int rankId, int innerId);
    void AddTraceNeighbor(int internalId);

    std::map<std::string, long long> mem_offest_sw;
    std::map<std::string, long long> mem_offest_hw;

    std::vector<unsigned int> offset_addrs;
    void print_offset_info();
    int add_trace(const char* begin, const char* end, long long offset_software, long long offset_hardware, char type);
    int add_trace(const char* begin, const char* end, std::string name_region, char type);
    int add_trace_mid(const char* begin, const char* end, long long offset, char type);
    int write_file(std::string& file, size_t max_num_trace = (size_t)(-1));
    size_t count_trace(char mode = 'a'); // a:all, l:load, s:save
    long long int max_phisical_addr = 0;
    bool autoswitch_track_detail(const unsigned long int max_runtime_trace);
};

DTaddr TransAddrFormat(int channelId, int rankId, DTaddr addrInput) {
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
        mem_trace.push_back(this_trace);
    }
}

void MemTrace::AddTraceFeature(int rankId, int innerId) {
    if (innerId >= v_maxinum_rank){
        printf("feature: out rank range\n"); exit(1);
    }
    trace this_trace;
    this_trace.type = 'l';

    for (int i = 0; i < v_line; i++){
        this_trace.addr = TransAddrFormat(channel_feature, rankId, (DTaddr)(i * MEM_ALIGNED));
        mem_trace.push_back(this_trace);
    }
    DTaddr start = innerId * v_line * MEM_ALIGNED + f_addr_start;
    for (int i = 0; i < v_line; i++){
        this_trace.addr = TransAddrFormat(channel_feature, rankId, (start + i * MEM_ALIGNED));
        mem_trace.push_back(this_trace);
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
        mem_trace.push_back(this_trace);
    }
}

int MemTrace::add_trace(const char* begin, const char* end, long long offset_software, long long offset_hardware, char type) {
    assert((long long) begin - offset_software >= 0);

    long long begin_hw = (long long) begin - offset_software + offset_hardware;
    long long end_hw = (long long) end - offset_software + offset_hardware;
    long long begin_aligned = floor((long long) begin_hw / MEM_ALIGNED) * MEM_ALIGNED;
    long long end_aligned = floor((long long) end_hw / MEM_ALIGNED) * MEM_ALIGNED;
    size_t round = ceil((1.0f * end_aligned - begin_aligned) / MEM_ALIGNED);
    // bool two_word_track = true;
    if (track_trace_detail) {
        trace this_trace;
        // for (const char* i = begin; i != end; i++) {
        for (size_t i = 0; i < round; i++){
            long long addr = begin_aligned + i * MEM_ALIGNED;
            this_trace.type = type;
            this_trace.addr = addr;
            mem_trace.push_back(this_trace);
        }
    }
    else {
        if (type == 'l') {
            num_read_trace += (end - begin + 1) >> 1;
        }
        else {
            num_write_trace += (end - begin + 1) >> 1;
        }
    }
    return 0;
};


int MemTrace::add_trace(const char* begin, const char* end, std::string name_region, char type) {
    if (mem_offest_sw.find(name_region) == mem_offest_sw.end() ||
        mem_offest_hw.find(name_region) == mem_offest_hw.end()){
        printf("unfind this name: %s\n", name_region.c_str());
        exit(1);
    }
    long long offset_software = mem_offest_sw[name_region];
    long long offset_hardware = mem_offest_hw[name_region];
    assert((long long) begin - offset_software >= 0);

    long long begin_hw = (long long) begin - offset_software + offset_hardware;
    long long end_hw = (long long) end - offset_software + offset_hardware;
    long long begin_aligned = floor((long long) begin_hw / MEM_ALIGNED) * MEM_ALIGNED;
    long long end_aligned = floor((long long) end_hw / MEM_ALIGNED) * MEM_ALIGNED;
    size_t round = ceil((1.0f * end_aligned - begin_aligned) / MEM_ALIGNED);
    // bool two_word_track = true;
    if (track_trace_detail) {
        trace this_trace;
        // for (const char* i = begin; i != end; i++) {
        for (size_t i = 0; i < round; i++){
            long long addr = begin_aligned + i * MEM_ALIGNED;
            this_trace.type = type;
            this_trace.addr = addr;
            mem_trace.push_back(this_trace);
        }
    }
    else {
        if (type == 'l') {
            num_read_trace += (end - begin + 1) >> 1;
        }
        else {
            num_write_trace += (end - begin + 1) >> 1;
        }
    }
    return 0;
};


int MemTrace::add_trace_mid(const char* begin, const char* end, long long offset, char type) {
    bool two_word_track = true;
    if (track_trace_detail) {
        trace this_trace;
        for (const char* i = begin; i != end; i++) {
            if (two_word_track) {
                this_trace.type = type;
                assert(i - offset >= 0);
                this_trace.addr = (((long long)i - offset) << 1) & 0x00000000FFFFFFF8;
                mem_trace.push_back(this_trace);
            }
        }
    }
    else {
        if (type == 'l') {
            num_read_trace ++;
        }
        else {
            num_write_trace ++;
        }
    }
    return 0;
};


int MemTrace::write_file(std::string& file, size_t max_num_trace) {
    std::ofstream file_out(file.c_str());
    const size_t endpos = mem_trace.size();
    for (size_t i = 0; i < endpos; ++i) {
        file_out << mem_trace[i].type << " " << "0x" << std::setfill('0') << std::setw(8) << std::hex << mem_trace[i].addr << '\n';
    }
    file_out.close();
    std::cout << "write " << std::dec << endpos << " trace" << std::endl;
    if(track_trace_detail==false)
        std::cout << "did not write all trace" << std::endl;
    std::cout << std::endl << std::endl;
    return 0;
}


size_t MemTrace::count_trace(char mode) {
    long long count = 0;
    if (track_trace_detail) {
        switch (mode) {
        case 'a':
            count = mem_trace.size();
            break;
        case 'l':
            for (auto it = mem_trace.begin(); it != mem_trace.end(); ++it) {
                count = count + (it->type == 'l');
            }
            break;
        case 's':
            for (auto it = mem_trace.begin(); it != mem_trace.end(); ++it) {
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


void MemTrace::print_offset_info() {
    // the smallest element as offset_init
    unsigned int offset_init = offset_addrs[0];
    std::cout << "********** offset_info **********" << std::endl;
    for (auto it = offset_addrs.begin(); it != offset_addrs.end(); ++it) {
        std::cout << "0x" << std::setfill('0') << std::setw(8) << std::hex << (*it - offset_init) << std::endl;
    }
    std::cout << "max_phisical_addr: " << "0x" << std::setfill('0') << std::setw(8) << std::hex << max_phisical_addr << std::endl;
    std::cout << "*********************************" << std::endl;
    std::cout << std::dec;
}


bool MemTrace::autoswitch_track_detail(const unsigned long int max_runtime_trace) {
    if (track_trace_detail) {
        if (mem_trace.size() >= max_runtime_trace) {
            num_read_trace = count_trace('l');
            num_write_trace = count_trace('a') - num_read_trace;
            std::cout << "SWITCH to simple track at trace: " << count_trace('a') << std::endl;
            track_trace_detail = false;
        }
    }
    return track_trace_detail;
}


#endif /* MEM_H */
