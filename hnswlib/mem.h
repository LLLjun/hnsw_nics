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

/*
    priority queue
*/
struct pair_q{
    float       dist;
    unsigned    id;
};

class PriorityQueue{
    public:
    PriorityQueue(size_t max_size, bool full_sort = false);
    void initMem(mem<char>* queue_mem, long long offest_hw);
    ~PriorityQueue();
    void CreateHeap();
    void AdjustHeapDown(int node_cur, int length);
    void AdjustHeapUp(int node_cur);
    void HeapSort();

    void emplace(const float& dist, const unsigned& id);
    std::pair<float, unsigned> top();
    size_t size();
    void pop();
    bool empty();
    int check();

    int cur_size_;

#if MEMTRACE
    mem<char>* queue_mem_;
    long long offest_sw_, offest_hw_;
#endif

    private:
    size_t  max_size_;
    bool    full_sort_;
    pair_q* data_;

};

PriorityQueue::PriorityQueue(size_t max_size, bool full_sort){
    max_size_ = max_size;
    full_sort_ = full_sort;
    data_ = new pair_q[max_size_+1]();
    cur_size_ = 0;
}

PriorityQueue::~PriorityQueue(){
    delete[] data_;
}

#if MEMTRACE
void PriorityQueue::initMem(mem<char>* queue_mem, long long offest_hw){
    queue_mem_ = queue_mem;
    offest_sw_ = (long long) data_;
    offest_hw_ = offest_hw;
}
#endif

void PriorityQueue::CreateHeap(){
    for (int i = cur_size_/2 - 1; i >= 0; i--)
        AdjustHeapDown(i, cur_size_ - 1);
}

void PriorityQueue::AdjustHeapDown(int node_cur, int length){
    // 暂存当前节点
    pair_q root_c = data_[node_cur];
#if MEMTRACE
    queue_mem_->add_trace((char *)(&data_[node_cur]),
                    (char *)(&data_[node_cur+1]), offest_sw_, offest_hw_, 'l');
#endif

    for (int child = 2*node_cur + 1; child < length; child = child * 2 + 1){
        // 选择值最大的孩子
#if MEMTRACE
        queue_mem_->add_trace((char *)(&data_[child]),
                        (char *)(&data_[child+2]), offest_sw_, offest_hw_, 'l');
#endif
        if (child < (length-1) && data_[child].dist < data_[child + 1].dist)
            child++;

        if (root_c.dist > data_[child].dist)
            break;
#if MEMTRACE
        queue_mem_->add_trace((char *)(&data_[node_cur]),
                        (char *)(&data_[node_cur+1]), offest_sw_, offest_hw_, 's');
#endif

        data_[node_cur] = data_[child];
        node_cur = child;
    }
    data_[node_cur] = root_c;

#if MEMTRACE
    queue_mem_->add_trace((char *)(&data_[node_cur]),
                    (char *)(&data_[node_cur+1]), offest_sw_, offest_hw_, 's');
#endif
}

void PriorityQueue::AdjustHeapUp(int node_cur){
    // 暂存当前节点
    pair_q node_c = data_[node_cur];
#if MEMTRACE
    queue_mem_->add_trace((char *)(&data_[node_cur]),
                    (char *)(&data_[node_cur+1]), offest_sw_, offest_hw_, 'l');
#endif

    for (int parent = (node_cur-1)/2; parent >= 0; parent = (parent-1)/2){
#if MEMTRACE
        queue_mem_->add_trace((char *)(&data_[parent]),
                        (char *)(&data_[parent+1]), offest_sw_, offest_hw_, 'l');
#endif
        if (data_[parent].dist > node_c.dist)
            break;
        else{
#if MEMTRACE
            queue_mem_->add_trace((char *)(&data_[node_cur]),
                            (char *)(&data_[node_cur+1]), offest_sw_, offest_hw_, 's');
#endif

            data_[node_cur] = data_[parent];
            node_cur = parent;
        }
        if (parent == 0)
            break;
    }
    data_[node_cur] = node_c;
#if MEMTRACE
    queue_mem_->add_trace((char *)(&data_[node_cur]),
                    (char *)(&data_[node_cur+1]), offest_sw_, offest_hw_, 's');
#endif
}

void PriorityQueue::HeapSort(){
    CreateHeap();
    for (int i = cur_size_ - 1; i > 0; i--){
        pair_q root_c = data_[0];
        data_[0] = data_[i];
        data_[i] = root_c;

        AdjustHeapDown(0, i - 1);
    }
}

void PriorityQueue::emplace(const float& dist, const unsigned& id){
    if (full_sort_){
        if (cur_size_ == 0){
            data_[0].dist = dist;
            data_[0].id = id;
            cur_size_++;
#if MEMTRACE
            queue_mem_->add_trace((char *)(&data_[0]),
                            (char *)(&data_[1]), offest_sw_, offest_hw_, 's');
#endif
        } else{
            // 二分法, mid的位置以左半部分不动作为参考
#if MEMTRACE
            queue_mem_->add_trace((char *)(&data_[0]),
                            (char *)(&data_[1]), offest_sw_, offest_hw_, 'l');
#endif
            if (data_[0].dist <= dist){
                int low = 0;
                int high = cur_size_ - 1;
                int mid;
                while (low <= high){
                    mid = (low + high) / 2;
#if MEMTRACE
                    queue_mem_->add_trace((char *)(&data_[mid]),
                                    (char *)(&data_[mid+1]), offest_sw_, offest_hw_, 'l');
#endif
                    if (dist >= data_[mid].dist)
                        low = mid + 1;
                    else
                        high = mid - 1;
                }
                if (low <= cur_size_ && data_[low-1].dist > dist)
                    exit(1);
                if (low < cur_size_ && data_[low].dist < dist)
                    exit(1);

                if (cur_size_ < max_size_){
#if MEMTRACE
                    for (int i = cur_size_-1; i >= low; i--){
                        queue_mem_->add_trace((char *)(&data_[i]),
                                        (char *)(&data_[i+1]), offest_sw_, offest_hw_, 'l');
                        queue_mem_->add_trace((char *)(&data_[i+1]),
                                        (char *)(&data_[i+2]), offest_sw_, offest_hw_, 's');
                    }
                    queue_mem_->add_trace((char *)(&data_[low]),
                                    (char *)(&data_[low+1]), offest_sw_, offest_hw_, 's');
#endif
                    for (int i = cur_size_-1; i >= low; i--)
                        data_[i+1] = data_[i];
                    cur_size_++;
                    data_[low].dist = dist;
                    data_[low].id = id;
                } else {
#if MEMTRACE
                    for (int i = 0; i <= low; i++){
                        queue_mem_->add_trace((char *)(&data_[i+1]),
                                        (char *)(&data_[i+2]), offest_sw_, offest_hw_, 'l');
                        queue_mem_->add_trace((char *)(&data_[i]),
                                        (char *)(&data_[i+1]), offest_sw_, offest_hw_, 's');
                    }
                    queue_mem_->add_trace((char *)(&data_[low-1]),
                                    (char *)(&data_[low]), offest_sw_, offest_hw_, 's');
#endif
                    for (int i = 0; i <= low; i++)
                        data_[i] = data_[i+1];
                    data_[low-1].dist = dist;
                    data_[low-1].id = id;
                }

                if (check() != -1)
                    exit(1);
            } else{
                if (cur_size_ < max_size_){
#if MEMTRACE
                    for (int i = cur_size_; i >= 1; i--){
                        queue_mem_->add_trace((char *)(&data_[i-1]),
                                        (char *)(&data_[i]), offest_sw_, offest_hw_, 'l');
                        queue_mem_->add_trace((char *)(&data_[i]),
                                        (char *)(&data_[i+1]), offest_sw_, offest_hw_, 's');
                    }
                    queue_mem_->add_trace((char *)(&data_[0]),
                                    (char *)(&data_[1]), offest_sw_, offest_hw_, 's');
#endif
                    for (int i = cur_size_; i >= 1; i--)
                        data_[i] = data_[i-1];
                    data_[0].dist = dist;
                    data_[0].id = id;
                    cur_size_++;
                }
                if (check() != -1)
                    exit(1);
            }
        }
    } else {
        if (cur_size_ >= max_size_){
            if (data_[0].dist > dist){
#if MEMTRACE
            queue_mem_->add_trace((char *)(&data_[0]),
                            (char *)(&data_[1]), offest_sw_, offest_hw_, 's');
#endif
                data_[0].dist = dist;
                data_[0].id   = id;
                AdjustHeapDown(0, cur_size_);
            }
        } else {
#if MEMTRACE
            queue_mem_->add_trace((char *)(&data_[cur_size_]),
                            (char *)(&data_[cur_size_+1]), offest_sw_, offest_hw_, 's');
#endif
            data_[cur_size_].dist = dist;
            data_[cur_size_].id   = id;
            cur_size_++;
            if (cur_size_ > 1)
                AdjustHeapUp(cur_size_-1);
        }
    }

    if (check() != -1)
        exit(1);
}

void PriorityQueue::pop(){
    cur_size_--;

    if (!full_sort_ && cur_size_ > 0){
#if MEMTRACE
        queue_mem_->add_trace((char *)(&data_[cur_size_]),
                        (char *)(&data_[cur_size_+1]), offest_sw_, offest_hw_, 'l');
        queue_mem_->add_trace((char *)(&data_[0]),
                        (char *)(&data_[1]), offest_sw_, offest_hw_, 's');
#endif
        data_[0] = data_[cur_size_];
        AdjustHeapDown(0, cur_size_);
    }

    if (check() != -1)
        exit(1);
}

size_t PriorityQueue::size(){
    return cur_size_;
}

std::pair<float, unsigned> PriorityQueue::top(){
    std::pair<float, unsigned> data_max;
    int pos = 0;
    if (full_sort_)
        pos = cur_size_ - 1;

    data_max.first  = data_[pos].dist;
    data_max.second = data_[pos].id;
#if MEMTRACE
    queue_mem_->add_trace((char *)(&data_[pos]), (char *)(&data_[pos+1]), offest_sw_, offest_hw_, 'l');
#endif

    return data_max;
}

bool PriorityQueue::empty(){
    if (cur_size_ > 0)
        return false;
    else
        return true;
}

int PriorityQueue::check(){
    int pass = -1;
    if (cur_size_ > 1){
        if (full_sort_){
            float last = data_[0].dist;
            for (int i = 1; i < cur_size_; i++){
                if (data_[i].dist < last)
                    return i;
                last = data_[i].dist;
            }
        } else {
            for (int i = 0; i <= cur_size_/2-1; i++){
                int child = i * 2 + 1;
                if (data_[child].dist > data_[i].dist)
                    return i;
                child++;
                if (child < cur_size_ && data_[child].dist > data_[i].dist)
                    return i;
            }
        }
    }
    return pass;
}

#endif /* MEM_H */
