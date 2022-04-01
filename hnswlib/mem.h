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

#define MEM_ALIGNED 64

template<typename T>
struct trace {
    char type; // l for load, s for save
    long long addr;
};

template<class T>
class mem {
private:
    bool track_trace_detail = true;
    std::vector<trace<T>> mem_trace;
    unsigned long long num_read_trace;
    unsigned long long num_write_trace;
public:
    std::map<std::string, long long> mem_offest_sw;
    std::map<std::string, long long> mem_offest_hw;

    std::vector<unsigned int> offset_addrs;
    void print_offset_info();
    int add_trace(const T* begin, const T* end, long long offset_software, long long offset_hardware, char type);
    int add_trace(const T* begin, const T* end, std::string name_region, char type);
    int add_trace_mid(const T* begin, const T* end, long long offset, char type);
    int write_file(std::string& file, long unsigned int max_num_trace = (long unsigned int)(-1));
    unsigned long long count_trace(char mode = 'a'); // a:all, l:load, s:save
    long long int max_phisical_addr = 0;
    bool autoswitch_track_detail(const unsigned long int max_runtime_trace);
};

template<class T>
int mem<T>::add_trace(const T* begin, const T* end, long long offset_software, long long offset_hardware, char type) {
    assert((long long) begin - offset_software >= 0);

    long long begin_hw = (long long) begin - offset_software + offset_hardware;
    long long end_hw = (long long) end - offset_software + offset_hardware;
    long long begin_aligned = floor((long long) begin_hw / MEM_ALIGNED) * MEM_ALIGNED;
    long long end_aligned = floor((long long) end_hw / MEM_ALIGNED) * MEM_ALIGNED;
    size_t round = ceil((1.0f * end_aligned - begin_aligned) / MEM_ALIGNED);
    // bool two_word_track = true;
    if (track_trace_detail) {
        trace<T> this_trace;
        // for (const T* i = begin; i != end; i++) {
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

template<class T>
int mem<T>::add_trace(const T* begin, const T* end, std::string name_region, char type) {
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
        trace<T> this_trace;
        // for (const T* i = begin; i != end; i++) {
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

template<class T>
int mem<T>::add_trace_mid(const T* begin, const T* end, long long offset, char type) {
    bool two_word_track = true;
    if (track_trace_detail) {
        trace<T> this_trace;
        for (const T* i = begin; i != end; i++) {
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

template<class T>
int mem<T>::write_file(std::string& file, long unsigned int max_num_trace) {
    std::ofstream file_out(file.c_str());
    const long unsigned int endpos = mem_trace.size();
    for (long unsigned int i = 0; i < endpos; ++i) {
        file_out << mem_trace[i].type << " " << "0x" << std::setfill('0') << std::setw(8) << std::hex << mem_trace[i].addr << '\n';
    }
    file_out.close();
    std::cout << "write " << std::dec << endpos << " trace" << std::endl;
    if(track_trace_detail==false)
        std::cout << "did not write all trace" << std::endl;
    std::cout << std::endl << std::endl;
    return 0;
}

template<class T>
unsigned long long mem<T>::count_trace(char mode) {
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

template<class T>
void mem<T>::print_offset_info() {
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

template<class T>
bool mem<T>::autoswitch_track_detail(const unsigned long int max_runtime_trace) {
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
