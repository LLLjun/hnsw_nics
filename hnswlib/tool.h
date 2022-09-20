#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <map>
#include <vector>
#include <random>
#include <algorithm>
#include "config.h"

using namespace std;

// load file. store format: (uint32_t)num, (uint32_t)dim, (data_T)num * dim.
template<typename data_T>
void LoadBinToArray(std::string& file_path, data_T *data_m,
                    uint32_t nums, size_t dims, bool non_header = false){
    std::ifstream file_reader(file_path.c_str(), ios::binary);
    if (!non_header){
        uint32_t nums_r, dims_r;
        file_reader.read((char *) &nums_r, sizeof(uint32_t));
        file_reader.read((char *) &dims_r, sizeof(uint32_t));
        if ((nums != nums_r) || (dims != dims_r)){
            printf("Error, file %s is error, nums_r: %u, dims_r: %u\n", file_path.c_str(), nums_r, dims_r);
            exit(1);
        }
    }

    uint32_t readsize = dims * sizeof(data_T);
    for (int i = 0; i < nums; i++) {
        file_reader.read((char *) (data_m + dims * i), readsize);
        if (file_reader.gcount() != readsize) {
            printf("Read Error\n"); exit(1);
        }
    }
    file_reader.close();
    printf("Load %u * %u Data from %s done.\n", nums, dims, file_path.c_str());
}

template<typename data_T>
void LoadBinToVector(std::string& file_path, std::vector<std::vector<data_T>>& data_m,
                    uint32_t nums, size_t dims, bool non_header = false){
    std::ifstream file_reader(file_path.c_str(), ios::binary);
    if (!non_header){
        uint32_t nums_r, dims_r;
        file_reader.read((char *) &nums_r, sizeof(uint32_t));
        file_reader.read((char *) &dims_r, sizeof(uint32_t));
        if ((nums != nums_r) || (dims != dims_r)){
            printf("Error, file %s is error, nums_r: %u, dims_r: %u\n", file_path.c_str(), nums_r, dims_r);
            exit(1);
        }
    }

    data_m.resize(nums);
    int readsize = sizeof(data_T);
    for (int i = 0; i < nums; i++) {
        data_m[i].resize(dims, 0);
        for (int j = 0; j < dims; j++) {
            file_reader.read((char *) (&data_m[i][j]), readsize);
            if (file_reader.gcount() != readsize) {
                printf("Read Error\n"); exit(1);
            }
        }
    }
    file_reader.close();
    printf("Load %u * %u Data from %s done.\n", nums, dims, file_path.c_str());
}

// store file. store format: (uint32_t)num, (uint32_t)dim, (data_T)num * dim.
template<typename data_T>
void WriteBinToArray(std::string& file_path, const data_T *data_m,
                    uint32_t nums, size_t dims, bool non_header = false){
    std::ofstream file_writer(file_path.c_str(), ios::binary);
    if (!non_header){
        file_writer.write((char *) &nums, sizeof(uint32_t));
        file_writer.write((char *) &dims, sizeof(uint32_t));
    }

    uint32_t writesize = dims * sizeof(data_T);
    for (int i = 0; i < nums; i++) {
        file_writer.write((char *) (data_m + dims * i), writesize);
        if (file_writer.fail() || file_writer.bad()) {
            printf("Write Error\n"); exit(1);
        }
    }
    file_writer.close();
    printf("Write %u * %u data to %s done.\n", nums, dims, file_path.c_str());
}


template<typename data_T>
uint32_t compArrayCenter(const data_T *data_m, uint32_t nums, size_t dims){
    cout << "Comput the center point: ";
    float *sum_m = new float[dims]();
    float *avg_m = new float[dims]();
    for (size_t i = 0; i < nums; i++) {
        for (size_t j = 0; j < dims; j++)
            sum_m[j] += (float) data_m[i * dims + j];
    }
    for (size_t j = 0; j < dims; j++)
        avg_m[j] = sum_m[j] / nums;

    float cur_max = std::numeric_limits<float>::max();
    uint32_t center_pt_id = 0;

    for (size_t i = 0; i < nums; i++){
        float tmp_sum = 0;
        for (size_t j = 0; j < dims; j++)
            tmp_sum += powf(((float) data_m[i*dims+j] - avg_m[j]), 2);

        if (tmp_sum < cur_max){
            cur_max = tmp_sum;
            center_pt_id = i;
        }
    }
    cout << center_pt_id << "\n";
    delete[] sum_m;
    delete[] avg_m;
    return center_pt_id;
}

// 创建文件夹
inline void createDir(string& dir) {
    if (access(dir.c_str(), R_OK|W_OK)){
        string command = "mkdir -p " + dir;
        int stat = system(command.c_str());
        if (stat != 0) {
            printf("Error, dir %s create failed \n", dir.c_str());
            exit(1);
        }
    }
}

// 随机排序容器内元素
template <typename T>
void shuffle_vector(std::vector<T> &v) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(v.begin(), v.end(), rng);
}

int selectNearAvgPos(vector<float>& values) {
    int pos = 0;
    int nums = values.size();
    if (nums < 2){
        printf("Error, vector size can't less than 2\n"); exit(1);
    }
    float avg = accumulate(values.begin(), values.end(), 0.0) / nums;
    float accum = 0;
    for (float& v : values)
        accum += (v - avg) * (v - avg);
    float stdev = sqrt(accum / (nums - 1));
    float upper = avg + stdev;
    float lower = avg - stdev;

    float sum = 0;
    int n_re = 0;
    for (float& v : values) {
        if (v > lower && v < upper) {
            sum += v;
            n_re++;
        }
    }
    float avg_re = sum / n_re;

    float diff_min = numeric_limits<float>::max();
    for (int i = 0; i < nums; i++){
        float diff = abs(values[i] - avg_re);
        if (diff < diff_min){
            diff_min = diff;
            pos = i;
        }
    }

    return pos;
}